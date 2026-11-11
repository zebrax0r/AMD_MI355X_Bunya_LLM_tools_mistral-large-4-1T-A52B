# Clients: Mistral Vibe, LiteLLM, anything OpenAI-compatible

[Documentation home](../README.md) | [Runbook](runbook.md)

The endpoint is OpenAI-compatible at `http://<node>:<PORT>/v1` (default port
8000). The model name is `mistral-large-4`. Requests need
`Authorization: Bearer <key>`:

| Mode | What listens on PORT | Key clients use |
|---|---|---|
| `GATEWAY=direct` | vLLM | serving key, `$STATE_DIR/ml4-api-key` |
| `GATEWAY=litellm` | LiteLLM (vLLM stays on 127.0.0.1) | LiteLLM master key, `$STATE_DIR/ml4-litellm-key` |

The banner printed at startup shows the node, port and key file.

## Request conventions (Mistral Large 4)

- **Thinking depth**: set the top-level `"reasoning_effort"` field to `"low"`,
  `"medium"` or `"high"`. The default is `high`. `"none"` is rejected by the
  release tokenizer. Use `low` for quick turns and `high` for hard or agentic
  work.
- **Sampling**: `temperature 1.0` and `top_p 0.95` at every effort level.
  These are also the server defaults, from `generation_config.json`.
- **Reasoning** comes back separately from `content`. vLLM calls the field
  `reasoning`; LiteLLM calls it `reasoning_content`. During tool use, send the
  whole assistant message back on the next turn: content, tool_calls and
  reasoning. `./serve-ml4.sh toolcheck` checks exactly this.
- `max_tokens` counts reasoning tokens too. Leave room: 8k or more for `high`.
- **Images**: the vision tower is served. Send OpenAI `image_url` parts.
- **Structured output**: `response_format: {"type": "json_schema", ...}`.

## From a laptop: SSH tunnel

```bash
ssh -N -L 8000:<node>:8000 <user>@bunya1.rcc.uq.edu.au
```

Then use `http://localhost:8000/v1`. The node name changes with every Slurm
job (`squeue -u $USER`), so the tunnel does too. The local port can stay the
same.

## Mistral Vibe

Run `vibe-setup.sh` from this repo on the laptop:

```bash
./vibe-setup.sh --via direct  --host localhost --port 8000 --api-key "$(cat key)"   # vLLM
./vibe-setup.sh --via litellm --host localhost --port 8000 --api-key "$(cat key)"   # on-node LiteLLM
./vibe-setup.sh --via litellm --base https://gateway.example/v1 --model <name-on-gateway> --api-key "$KEY"
vibe
```

It writes a managed block to `~/.vibe/config.toml` (provider `bunya-ml4`,
model alias `mistral-large-4-bunya`, thinking `high`, levels
low/medium/high, temperature 1.0, images on, context = `CONTEXT_LEN`). It also
sets `active_model`, unless you pass `--no-activate`, and stores the key as
`BUNYA_ML4_API_KEY` in `~/.vibe/.env` (mode 600). Re-running replaces the
block. `--remove` takes everything out again.

**The model card's Vibe snippet uses `api_style = "reasoning"`. This repo uses
`"openai"`.** Vibe's `reasoning` style reads thinking only from Mistral-API
content blocks (`{"type": "thinking"}`). vLLM and LiteLLM return thinking in a
separate field, so with the card's setting Vibe would never see the thinking
or send it back. With `api_style = "openai"` and `reasoning_field_name` set to
`reasoning` (vLLM) or `reasoning_content` (LiteLLM), it does both.
`./serve-ml4.sh vibecheck` replays Vibe's exact wire format against the running
server (streaming, tool calls assembled from deltas, reasoning field).

## LiteLLM on the node (`GATEWAY=litellm`)

`GATEWAY=litellm ./serve-ml4.sh serve --detach` (or
`./submit-ml4.sh mi355x --gateway litellm`) does the following:

- starts vLLM on `127.0.0.1:8101` (and `8102` for a second replica), out of
  reach from other nodes;
- starts LiteLLM `v1.108.5-lts` (pinned) on `0.0.0.0:PORT`, with the master key
  from `$STATE_DIR/ml4-litellm-key`;
- on a full MI355X node, runs **two TP4 replicas** with **conversation
  affinity** ([`litellm/ml4_affinity.py`](../litellm/ml4_affinity.py)):
  - A new conversation goes to the replica with fewer active sessions.
  - Every later turn of that conversation returns to the same replica, so it
    keeps its prefix cache.

  Without affinity, LiteLLM's balancers scatter turns. A turn that lands on the
  other replica re-prefills the whole context: measured 8.9 s instead of
  1.8 s at 100k tokens. LiteLLM's built-in `prompt_caching` routing does not
  help here, because it keys on Anthropic `cache_control` markers that
  OpenAI-style clients never send. The pin table lives in the LiteLLM process,
  so this mode runs one LiteLLM worker. That is plenty, since the gateway only
  forwards.

The generated config is `$STATE_DIR/litellm/ml4-litellm.yaml`. Change gateway
settings without reloading the engines:
`LITELLM_ROUTING=simple-shuffle ./serve-ml4.sh restart-gateway`. It runs without
a database, so there are no virtual keys or spend tracking. Point a
full-featured gateway at this node if you need those.

## Your own LiteLLM gateway (elsewhere)

Serve with `GATEWAY=direct`, then add this node to the gateway:

```bash
./serve-ml4.sh litellm-snippet                       # prints a model_list entry
cat "$STATE_DIR/litellm-external-snippet.yaml"       # written at every direct start
```

```yaml
model_list:
  - model_name: mistral-large-4
    litellm_params:
      model: hosted_vllm/mistral-large-4
      api_base: http://<node>:8000/v1        # or your tunnel's address
      api_key: os.environ/ML4_BACKEND_KEY     # = $STATE_DIR/ml4-api-key
      allowed_openai_params: ["reasoning_effort"]
```

The gateway has to reach `<node>:8000`. Inside the cluster it can connect
directly. From outside, use an SSH tunnel and set `api_base` to the tunnel's
local end. The node changes per job, so update `api_base` (or the tunnel) when
you resubmit.

## Other clients

Any OpenAI-compatible tool works (opencode, Continue, the `openai` SDK):
base URL `http://localhost:8000/v1`, model `mistral-large-4`, the key above.
Make sure the client sends `reasoning_effort` and replays the reasoning field.
