# Configuration

[Documentation home](../README.md) | [Runbook](runbook.md)

Values come from `ml4.env` (or `$ML4_ENV`). Variables exported in the shell
win. `./serve-ml4.sh plan` prints the resolved recipe, environment and argv
without touching a GPU. Run it after any change.

## Run shape (chosen at launch)

| Knob | Values | Default | Meaning |
|---|---|---|---|
| `GPU_ARCH` | `auto` `mi355x` `mi300x` | `auto` | read from `/sys/class/kfd` (90500 = gfx950, 90402 = gfx942) |
| `WEIGHTS` | `auto` `mxfp4` `fp8` | `auto` | MI355X → MXFP4, MI300X → FP8. MXFP4 on MI300X is refused |
| `ENGINE` | `auto` `day0` `branch` | `auto` | MI355X → day-0 image, MI300X → nightly + ML4 overlay ([why](engine-choice.md)) |
| `GATEWAY` | `direct` `litellm` | `direct` | vLLM on PORT, or LiteLLM on PORT with vLLM on 127.0.0.1 |
| `AGENTS` | integer | 4 | concurrent sessions; picks `PROFILE` and raises `MAX_NUM_SEQS` |
| `PROFILE` | `auto` `tp4` `tp8` | `auto` | `tp4` when MXFP4 and AGENTS ≤ 4, else `tp8` |
| `REPLICAS` | `auto` `1` `2` | `auto` | `2` = two TP4 engines (needs `GATEWAY=litellm` and 8 GPUs) |
| `SPECULATIVE` | `dspark` `none` | `dspark` | the checkpoint's own DSpark head, 3 draft tokens |
| `MONOKERNEL` | `auto` `0` `1` | `auto` | AMD specialized decode: on for TP4 + MXFP4 + gfx950 |

### The layouts

| Layout | When | Engine flags |
|---|---|---|
| MI355X `tp4` + monokernel | AGENTS ≤ 4 (default) | TP4, `--monokernel`, DSpark, FP8 KV, 16 seqs, 8k batched tokens, `VLLM_USE_V2_MODEL_RUNNER=1` |
| MI355X `tp4` x 2 | `GATEWAY=litellm` on 8 GPUs | the same, twice, on GPUs 0-3 and 4-7 |
| MI355X `tp8` | AGENTS > 4 | TP8, DSpark, FP8 KV, max(64, AGENTS) seqs, 16k batched tokens |
| MI355X FP8 | `WEIGHTS=fp8` | TP8, DSpark, FP8 KV, mem 0.88 |
| MI300X | default there | TP8, DSpark, BF16 KV, 32 seqs, mem 0.90, branch engine |

The specialized decode runs every MoE layer as one persistent FlyDSL kernel
launch. It handles pure decode steps of 1, 2 or 4 requests, or one request's
DSpark verify block of 4 rows. Bigger or mixed steps fall back to vLLM's own
layers, which is why the launcher switches to TP8 past four agents. The decode
refuses anything outside its envelope at startup and lists every reason. The
launcher checks the same conditions first: TP4, gfx950 with 256 CUs, MXFP4
experts, a bf16 or FP8 latent cache, DSpark or no speculation, V2 model runner,
and no DP/EP/PP/CP.

## Engine knobs (empty = profile value)

| Knob | Profile value | Notes |
|---|---|---|
| `CONTEXT_LEN` | 262144 | the model supports 1,048,576. MXFP4 TP4 has KV room for it |
| `MAX_NUM_SEQS` | 16 / 64 / 32 | raised to `AGENTS` when larger |
| `MAX_BATCHED_TOKENS` | 8192 / 16384 | chunked-prefill chunk |
| `MEM_UTIL` | 0.85 / 0.88 / 0.90 | `--gpu-memory-utilization` |
| `KV_CACHE_DTYPE` | `fp8` (MI355X), `auto` = bf16 (MI300X) | FP8 sparse MLA on gfx942 (fnuz) is not yet qualified |
| `TOOL_PARSER` / `REASONING_PARSER` | `mistral` / `mistral` | `none` disables |
| `WEIGHT_LOAD_THREADS` | 16 | `--model-loader-extra-config enable_multithread_load` |
| `PREWARM` | 0 | 1 = read shards in parallel before launch. Useless on Bunya's GPFS (~30 GB/s reads, 64 GB pagepool) |
| `EXTRA_ENGINE_ARGS` | | appended to `vllm serve` |
| `EXTRA_ENV` | | `NAME=VALUE ...` into the vLLM container |

## Endpoint

| Knob | Default | |
|---|---|---|
| `PORT` | 8000 | the port clients use, in either mode |
| `BACKEND_PORT_BASE` | 8100 | litellm mode: replicas on 127.0.0.1:8101, 8102 |
| `SERVED_MODEL_NAME` | `mistral-large-4` | |
| `ML4_API_KEY` | generated → `$STATE_DIR/ml4-api-key` | always protects vLLM |
| `LITELLM_MASTER_KEY` | generated → `$STATE_DIR/ml4-litellm-key` | litellm mode |
| `LITELLM_ROUTING` | `least-busy` | `simple-shuffle`, `latency-based-routing`, `usage-based-routing-v2` |
| `LITELLM_WORKERS` | 4 | uvicorn workers (forced to 1 when affinity routes 2 replicas) |
| `LITELLM_AFFINITY` | 1 | pin each conversation to one replica so it keeps its prefix cache; new conversations go to the less-busy replica |

## Paths and pins

| Knob | Default |
|---|---|
| `STATE_DIR` | `/scratch/user/$USER/models/mistral-large-4` (images, caches, keys, logs) |
| `FP8_PATH` / `MXFP4_PATH` | the shared store ([weights](weights.md)) |
| `ENGINE_IMAGE` | day0: `vllm/vllm-openai-rocm@sha256:536d1ef1…`; branch: `…@sha256:55968af1…` |
| `BRANCH_REPO` / `BRANCH_SHA` | `NickLucche/vllm` @ `28a4495d` |
| `LITELLM_IMAGE` | `litellm/litellm@sha256:008a5916…` (v1.108.5-lts) |
| `ROCM_MODE` | `auto` (probes `devices`, then `rocm`; cached per node + image) |
