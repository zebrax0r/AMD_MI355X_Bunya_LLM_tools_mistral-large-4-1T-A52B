# Compatibility, verification status, measurements

[Documentation home](../README.md) | [Engine choice](engine-choice.md)

## Pins

| Component | Value | Notes |
|---|---|---|
| FP8 weights | `mistralai/Mistral-Large-4-1T-A52B` @ `af26e786` | 54 shards, 1060.6 GB; verified vs the Hub 11 Nov 2026 |
| MXFP4 weights | `amd/Mistral-Large-4-1T-A52B-Quark-MXFP4` @ `d084803a` | 54 shards, 577.9 GB; verified vs the Hub 11 Nov 2026 |
| day0 engine | `vllm/vllm-openai-rocm@sha256:536d1ef1…` (`:mistral-large-4`, 29 Oct) | vLLM `0.1.dev23730+g64053a10b`, torch 2.13.0+rocm10.0.0 (HIP 7.15), aiter, FlyDSL 0.4.0, mistral_common 1.13.0, Python 3.12; kernels **gfx950 only** |
| branch engine | `vllm/vllm-openai-rocm@sha256:55968af1…` (`nightly-rocm72-83c078d9`, 23 Oct) + `NickLucche/vllm@28a4495d` Python | nightly built for gfx90a/gfx942/gfx950 |
| LiteLLM | `litellm/litellm@sha256:008a5916…` (v1.108.5-lts) | |
| Bunya MI355X | bun161: ROCm 10.1.0, amdgpu 7.1.9, 8 × MI355X (256 CUs, 288 GB), GPFS pagepool 64 GB | |
| Bunya MI300X | bun144-145: 8 × MI300X (304 CUs, 192 GB) | see status below |

## Verification status

| Path | Status |
|---|---|
| MI355X, MXFP4, TP4 + monokernel + DSpark, direct | **verified** 11 Nov 2026: toolcheck, vibecheck, agentstress 170k clean |
| MI355X, MXFP4, 2 × TP4 behind LiteLLM | **verified** 11 Nov 2026: toolcheck, vibecheck (`reasoning_content`), 4 concurrent agent sessions clean with conversation affinity |
| MI355X, MXFP4, TP8 + DSpark (AGENTS > 4) | **verified** 11 Nov 2026: toolcheck, 8 concurrent agent sessions clean, bench to concurrency 16 |
| MI355X, FP8, TP8, BF16 KV, branch engine (the MI300X software stack and profile, on gfx950) | **verified** 11 Nov 2026: toolcheck, vibecheck, agentstress 108k clean |
| MI300X, FP8, TP8, branch engine | **not yet run on MI300X hardware**: AMD's ML4 branch qualifies the MI350 series only. The ROCm sparse-MLA backend and the ML4 indexer carry gfx942 / fnuz paths, and the branch engine is rehearsed on MI355X |

## Bunya measurements

bun161, 11 Nov 2026, the pinned images and weights. Logs are in `logs/`. Each
figure is from a single run, with no error bars.

### MI355X, MXFP4, TP4 + specialized decode + DSpark (the default, GPUs 0-3)

| Startup | |
|---|---|
| Cold, first ever (empty JIT caches) | 631 s to healthy. Weights 54 shards in ~5 min (multithreaded, GPFS), specialized-decode repack 4.17 GiB/rank, graph capture 49 s, one-off aiter MXFP4-MoE JIT |
| KV pool | 96 GiB/GPU → **2,114,954 FP8 tokens** (8 x 262k sessions) |
| Specialized decode | engaged: 71 layer launches (nope 9, nope_kvs 9, rope 25, rope_shared 28), verify rows 4 |

**Real agentic text** (`agentstress.py`: one growing session, 20k-token turns,
effort `low`, temperature 1.0; `logs/stress-mxfp4-tp4-mono-dspark-150k.log`):

| Context | Single-stream decode | TTFT for a new 20k turn |
|---|---|---|
| 0–60k | **160 tok/s** | 2.1–3.6 s |
| 60–120k | **143 tok/s** | 2.0–2.3 s |
| 120–170k | **123 tok/s** | 2.6–2.7 s |

All 8 needles were found and nothing was corrupted. The server-side prefix
cache hit rate was 77.8%, and TTFT stayed flat as the context grew eightfold.
DSpark acceptance on this text was **2.6–3.4** tokens per verify.

**Random-prompt bench** (`bench-ml4.sh`, 1024 in / 512 out, n=32, effort low;
`logs/bench-mxfp4-tp4-mono-sweep.log`). With `--ignore-eos`, random prompts
drag DSpark acceptance down to 1.1–1.3, so this **understates** real decode:

| Concurrency | Output tok/s | Median TTFT | Median TPOT |
|---|---|---|---|
| 1 | 83 | 385 ms | 10.9 ms |
| 2 | 136 | 401 ms | 13.6 ms |
| 4 | 222 | 426 ms | 16.6 ms |
| 8 | 314 | 625 ms | 21.2 ms |

### MI355X, MXFP4, 2 × TP4 behind LiteLLM (`GATEWAY=litellm`, 8 GPUs)

| | |
|---|---|
| Startup (JIT caches warm) | both replicas healthy after 490 s (loading in parallel), LiteLLM up in seconds |
| Through LiteLLM | `reasoning_effort` passed through, reasoning as `reasoning_content`, `prompt_tokens_details.cached_tokens` reported |

Four concurrent agent sessions, each different (`AGENTSTRESS_SEED` 21–24),
growing to 107k tokens in 20k turns:

| Routing | Prefix-cache hits after turn 1 | Sessions per replica | Per-session decode | Notes |
|---|---|---|---|---|
| `least-busy`, 4 workers, no affinity | misses when a turn changes replica | uneven | — | a missed turn at 100k took **8.9 s** TTFT, vs 1.8 s with the hit |
| rendezvous hash only | 20/20 | 1 / 3 | 158–189 (alone), 25–90 (shared) | the hash spreads sessions only statistically |
| **sticky + least-active (shipped)** | **20/20** | **2 / 2** | **70–175 tok/s** (~480 aggregate) | TTFT 1.3–3.7 s throughout |

Two sessions on one replica sometimes decode in the same step. That falls back
from the single-request specialized decode to vLLM's batched layers, which is
why per-session speed varies within a run.

### MI355X, MXFP4, TP8 + DSpark (`AGENTS=16`)

| | |
|---|---|
| Startup (JIT caches warm) | 220 s to healthy |
| KV pool | **3,430,665 FP8 tokens** (13 x 262k) |
| 8 concurrent agent sessions (seeds 31–38, 20k-token turns to 80k) | 32/32 turns clean. Every round, all 8 sessions submit a new 20k-token turn at once (~160k tokens of prefill), giving TTFT median 6.1 s, max 15.6 s, and per-turn decode median 38 tok/s |
| Random bench, concurrency 4 / 16 | **235 / 615** output tok/s, median TPOT 15.0 / 20.7 ms |

### MI355X, FP8, TP8, BF16 KV, branch engine (MI300X rehearsal)

`ENGINE=branch WEIGHTS=fp8 KV_CACHE_DTYPE=auto MAX_NUM_SEQS=32 MEM_UTIL=0.90`
is the MI300X profile run on gfx950. It exercises the overlay, the
ROCm-7.2 nightly, the FP8 block-scaled path (AITER FP8 GEMM and MoE) and
BF16 sparse MLA.

| | |
|---|---|
| Startup (cold, new image) | 710 s. Weights 531 s (1.06 TB), **126.55 GiB/GPU** |
| KV pool | 2,551,660 BF16 tokens on 288 GB GPUs. On 192 GB MI300X the same weights leave ~34 GiB/GPU, roughly 300k tokens |
| Checks | toolcheck, vibecheck pass. The thinking-block replay is rejected, as on day0 |
| `agentstress` 20k turns to 108k | clean. Single-stream decode **153 tok/s** (0–60k), **147 tok/s** (60–100k). DSpark acceptance 2.6–3.5. Prefix-cache hits on every turn |

What this does not cover on MI300X: gfx942 kernel selection (AITER and Triton
pick different kernels there), FP8 `fnuz` conversion of the checkpoint, and
the 192 GB memory fit. Run the qualification sequence in
[runbook](runbook.md) step 4 on bun144/145 before relying on it.

### Client protocol checks

| Check | Result |
|---|---|
| `toolcheck` (non-streaming, reasoning replayed as `reasoning`) | pass: reasoning split, tool call, value used on turn 2 |
| `vibecheck` (Vibe `api_style="openai"`, streaming, `reasoning_field_name="reasoning"`) | pass |
| Vibe `api_style="reasoning"` replay (thinking content blocks, as in the model card) | **HTTP 400** from vLLM: `sequence item 0: expected str instance, list found`. This is why `vibe-setup.sh` uses `"openai"` |
