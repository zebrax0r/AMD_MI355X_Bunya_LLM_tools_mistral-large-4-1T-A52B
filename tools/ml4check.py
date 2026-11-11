#!/usr/bin/env python3
"""Two-turn tool-call round trip against a Mistral Large 4 endpoint.

Turn 1 asks for a tool call; turn 2 returns a value the model cannot guess and
replays the COMPLETE assistant message (content + tool_calls + reasoning), as
an agent client must. Works against vLLM directly or through LiteLLM.

  --vibe    speak Mistral Vibe's exact wire format (api_style "openai"):
            streaming with stream_options.include_usage, tool calls assembled
            from deltas, reasoning read from and replayed under
            --reasoning-field. Also checks that the server accepts Vibe's
            other style ("reasoning": thinking content blocks) on replay.

Exit 0 = all checks pass, 1 = a check failed, 2 = request failed.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SECRET = 61.4
NODE = "bun161"
TOOLS = [{"type": "function", "function": {
    "name": "get_gpu_temperature",
    "description": "Return the current GPU temperature in Celsius for a named Bunya compute node.",
    "parameters": {"type": "object",
                   "properties": {"node": {"type": "string", "description": "Node hostname, e.g. bun161"}},
                   "required": ["node"]}}}]

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--key", default="")
p.add_argument("--model", required=True)
p.add_argument("--effort", default="low", help="low|medium|high, or '' to omit")
p.add_argument("--vibe", action="store_true")
p.add_argument("--reasoning-field", default="reasoning")
p.add_argument("--max-tokens", type=int, default=8192)
a = p.parse_args()

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(label)


class RequestFailed(Exception):
    pass


def request(payload, stream=False, soft=False):
    if a.effort:
        payload["reasoning_effort"] = a.effort
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        a.base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + a.key})
    try:
        r = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors='replace')[:600]
        if soft:
            raise RequestFailed(f"HTTP {e.code}: {body}")
        print(f"  HTTP {e.code}: {body}")
        sys.exit(2)
    except Exception as e:  # noqa: BLE001
        print(f"  request failed: {e}\n  Is the server up? ./serve-ml4.sh status")
        sys.exit(2)
    if not stream:
        with r:
            return json.load(r)
    # Assemble a streamed response the way Vibe does.
    msg = {"role": "assistant", "content": "", "reasoning": "", "tool_calls": {}}
    finish, usage, first = None, None, None
    t0 = time.time()
    with r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get(a.reasoning_field)):
                    first = time.time() - t0
                msg["content"] += d.get("content") or ""
                msg["reasoning"] += d.get(a.reasoning_field) or ""
                for tc in d.get("tool_calls") or []:
                    slot = msg["tool_calls"].setdefault(tc.get("index", 0),
                                                        {"id": None, "type": "function",
                                                         "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    slot["function"]["name"] += fn.get("name") or ""
                    slot["function"]["arguments"] += fn.get("arguments") or ""
                finish = ch.get("finish_reason") or finish
    calls = [dict(v, index=k) for k, v in sorted(msg["tool_calls"].items())]
    return {"choices": [{"message": {"role": "assistant", "content": msg["content"],
                                     a.reasoning_field: msg["reasoning"], "tool_calls": calls},
                         "finish_reason": finish}],
            "usage": usage, "_ttft": first}


def reasoning_of(m):
    return m.get(a.reasoning_field) or m.get("reasoning") or m.get("reasoning_content") or ""


style = f"Vibe wire format, streaming, reasoning field '{a.reasoning_field}'" if a.vibe else "non-streaming"
print(f"Endpoint {a.base}  model {a.model}  effort {a.effort or 'server default'}  ({style})\n")
msgs = [{"role": "user", "content": f"What is the current GPU temperature on {NODE}? Use the tool, then state the number."}]
print("Turn 1 — does the model emit a tool call?")
t = time.time()
r1 = request({"model": a.model, "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
              "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 0.95}, stream=a.vibe)
m1, fin = r1["choices"][0]["message"], r1["choices"][0]["finish_reason"]
calls = m1.get("tool_calls") or []
reasoning = reasoning_of(m1)
check(bool(reasoning), "reasoning separated from content",
      f"{len(reasoning)} chars" if reasoning else "empty: wrong reasoning parser/field, or the model skipped thinking")
check("[THINK]" not in (m1.get("content") or ""), "no raw [THINK] markers in content")
check(bool(calls), "model returned tool_calls",
      f"finish_reason={fin}, {time.time() - t:.1f}s" + ("" if calls else f", content={(m1.get('content') or '')[:160]!r}"))
if not calls:
    if fin == "length":
        print("  finish_reason=length: it never finished thinking. Raise --max-tokens or lower --effort.")
    print("  A raw [TOOL_CALLS] in content means the tool parser does not match (TOOL_PARSER=mistral).")
    sys.exit(1)
fn = calls[0].get("function", {})
check(fn.get("name") == "get_gpu_temperature", "correct function name", repr(fn.get("name")))
raw = fn.get("arguments")
try:
    args = json.loads(raw) if isinstance(raw, str) else raw
    ok_json = isinstance(args, dict)
except Exception as e:  # noqa: BLE001
    args, ok_json = None, False
    print(f"        arguments did not parse: {e}")
check(ok_json, "arguments are valid JSON", repr(raw)[:160])
check(bool(args) and NODE in str(args.get("node", "")), "argument value carried through", repr(args.get("node") if args else None))
check(bool(calls[0].get("id")), "tool_call has an id", repr(calls[0].get("id")))

print("\nTurn 2 — does the model use the tool result (full assistant message replayed)?")
asst = {"role": "assistant", "content": m1.get("content") or "", "tool_calls": calls}
if reasoning:
    asst[a.reasoning_field] = reasoning
tool_msg = {"role": "tool", "tool_call_id": calls[0].get("id"), "name": "get_gpu_temperature",
            "content": json.dumps({"node": NODE, "celsius": SECRET})}
r2 = request({"model": a.model, "messages": msgs + [asst, tool_msg], "tools": TOOLS,
              "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 0.95}, stream=a.vibe)
m2 = r2["choices"][0]["message"]
final = (m2.get("content") or "").strip()
check(bool(final), "final answer has content", f"finish_reason={r2['choices'][0]['finish_reason']}")
check(str(SECRET) in final, f"final answer contains the tool's value ({SECRET})")
print(f"\n  final answer: {final[:300]}")
if r2.get("usage"):
    print(f"  usage turn 2: {r2['usage']}")

if a.vibe:
    print("\nTurn 2b — Vibe's 'reasoning' api_style replay (thinking content blocks) is accepted?")
    blocks = [{"type": "thinking", "thinking": [{"type": "text", "text": reasoning or "…"}]}]
    if m1.get("content"):
        blocks.append({"type": "text", "text": m1["content"]})
    asst_b = {"role": "assistant", "content": blocks, "tool_calls": calls}
    # Informational only: vibe-setup.sh configures api_style "openai" because
    # this replay shape is what the model card's api_style "reasoning" sends.
    try:
        r3 = request({"model": a.model, "messages": msgs + [asst_b, tool_msg], "tools": TOOLS,
                      "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 0.95},
                     stream=True, soft=True)
        f3 = (r3["choices"][0]["message"].get("content") or "")
        print(f"  [INFO] thinking-block replay {'works' if str(SECRET) in f3 else 'answered without the tool value'}"
              " (api_style 'reasoning')")
    except RequestFailed as e:
        print(f"  [INFO] thinking-block replay REJECTED by the server ({e}) —"
              " api_style 'reasoning' would fail from the second tool turn; use 'openai'")

print()
if fails:
    print(f"ROUND TRIP FAILED: {len(fails)} check(s) — {', '.join(fails)}")
    sys.exit(1)
print("TOOL-CALL ROUND TRIP OK" + (" — Mistral Vibe with this provider config should work." if a.vibe else "."))
