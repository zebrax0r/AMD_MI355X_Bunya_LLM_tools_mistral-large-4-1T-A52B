"""Conversation affinity for LiteLLM in front of several vLLM replicas.

A coding agent resends its whole, growing conversation every turn. vLLM keeps
that prefix in its KV cache, but only on the replica that served the previous
turn: if the next turn lands on the other replica, the full context is
prefilled again (measured on Bunya: ~9 s at 100k tokens vs ~2 s with the hit).
LiteLLM's own `prompt_caching` routing keys on Anthropic cache_control
breakpoints, which OpenAI-style clients such as Mistral Vibe never send.

A conversation is identified by its head: the system prompt(s) plus the first
user message, which stay fixed for the life of an agent session.

  * New conversation (no assistant turn yet): pinned to the replica with the
    fewest conversations active in the last ACTIVE_S seconds (ties: round
    robin). Keeps agents spread out, which matters on MI355X because the AMD
    specialized decode serves one request's DSpark verify step per launch.
  * Known conversation: back to its pinned replica (prefix-cache hit).
  * Unknown conversation already under way (e.g. the gateway restarted):
    rendezvous hash of the head, so it at least stays put from then on.
  * Pinned replica unhealthy: re-pinned to the least-loaded healthy one.

State is in-process, so serve-ml4.sh runs LiteLLM with one worker when this is
enabled. Loaded via litellm_settings.callbacks: ml4_affinity.affinity
"""
import hashlib
import time
from collections import OrderedDict

from litellm.integrations.custom_logger import CustomLogger

ACTIVE_S = 900          # a conversation counts as active this long after its last turn
MAX_PINS = 50000


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text"):
                    out.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    out.append(hashlib.sha256(str(url).encode()).hexdigest())
        return "".join(out)
    return ""


def _as_dict(m):
    return m if isinstance(m, dict) else (getattr(m, "model_dump", lambda: {})() or {})


def conversation_head(messages):
    """System prompt(s) up to and including the first user message."""
    head = []
    for m in messages or []:
        m = _as_dict(m)
        role = m.get("role", "")
        head.append(f"{role}\x1f{_text(m.get('content'))}")
        if role == "user":
            break
    return "\x1e".join(head).encode()


def is_new_conversation(messages):
    return not any(_as_dict(m).get("role") == "assistant" for m in messages or [])


def pick(deployments, key):
    """Rendezvous (highest-random-weight) choice of one deployment for key."""
    def weight(d):
        did = str((d.get("model_info") or {}).get("id", ""))
        return hashlib.sha256(key + b"\x00" + did.encode()).digest()
    return max(deployments, key=weight)


class ConversationAffinity(CustomLogger):
    def __init__(self, clock=time.monotonic):
        super().__init__()
        self.pins = OrderedDict()      # head digest -> (deployment id, last seen)
        self.rr = 0
        self.clock = clock

    def _active_counts(self, ids, now):
        counts = dict.fromkeys(ids, 0)
        for did, seen in self.pins.values():
            if did in counts and now - seen < ACTIVE_S:
                counts[did] += 1
        return counts

    def _least_loaded(self, ids, now):
        counts = self._active_counts(ids, now)
        low = min(counts.values())
        tied = sorted(i for i, c in counts.items() if c == low)
        self.rr += 1
        return tied[self.rr % len(tied)]

    def choose(self, healthy_deployments, messages):
        if not messages or len(healthy_deployments) < 2:
            return healthy_deployments
        by_id = {str((d.get("model_info") or {}).get("id", "")): d for d in healthy_deployments}
        head = conversation_head(messages)
        key = hashlib.sha256(head).digest()
        now = self.clock()
        pinned = self.pins.get(key)
        if pinned and pinned[0] in by_id:
            did = pinned[0]
        elif is_new_conversation(messages) or pinned:
            did = self._least_loaded(by_id, now)
        else:
            did = str(pick(healthy_deployments, head)["model_info"]["id"])
        self.pins[key] = (did, now)
        self.pins.move_to_end(key)
        while len(self.pins) > MAX_PINS:
            self.pins.popitem(last=False)
        return [by_id[did]]

    async def async_filter_deployments(self, model, healthy_deployments, messages,
                                       request_kwargs=None, parent_otel_span=None):
        return self.choose(healthy_deployments, messages)


affinity = ConversationAffinity()
