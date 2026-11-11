# Mistral Large 4 on Bunya (MI355X and MI300X)

Serve [mistralai/Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)
(1T parameters, 52B active, MLA + DSA sparse attention, 1M context, built-in
DSpark draft head) with **vLLM** in Apptainer on one Bunya GPU node. The
endpoint is OpenAI-compatible. Clients can use it directly, or through a
**LiteLLM** gateway that the launcher starts on the same node. Mistral Vibe
works either way.

This repo follows [AMD_MI355X_Bunya_LLM_tools_KimiK3.1](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_KimiK3.1).
It reuses that repo's Bunya plumbing: GPU passthrough probing, the rocminfo
shim, the aiter JIT and `/tmp/aiter_configs` binds, scratch-only caches, API
keys and detach. Two things change: the engine is vLLM, and the node class is
chosen when the job starts.

## Start here

| Task | Guide |
|---|---|
| Allocate, launch, verify, stop | [Operating runbook](docs/runbook.md) |
| Pick a layout, understand a knob | [Configuration](docs/configuration.md) |
| Connect Mistral Vibe, LiteLLM, other clients | [Clients](docs/clients.md) |
| Shared weights store (for other users) | [Weights](docs/weights.md) |
| Pins, what was verified, measurements | [Compatibility](docs/compatibility.md) |
| A launch or request failed | [Troubleshooting](docs/troubleshooting.md) |
| Why vLLM and not SGLang | [Engine choice](docs/engine-choice.md) |

## Quick start

```bash
# from a login node: queue a job (the endpoint banner lands in logs/ml4-<jobid>.out)
./submit-ml4.sh mi355x                       # 8 x MI355X
./submit-ml4.sh mi355x --gpus 4              # half a node: MXFP4 TP4 + AMD monokernel
./submit-ml4.sh mi355x --gateway litellm     # 2 x TP4 replicas behind LiteLLM
./submit-ml4.sh mi300x                       # 8 x MI300X, FP8

# or interactively, inside an allocation
cp -n ml4-env.example ml4.env && chmod 600 ml4.env
./serve-ml4.sh plan                          # what would run here (no GPU touched)
./serve-ml4.sh serve --detach                # GATEWAY=litellm ./serve-ml4.sh serve --detach
./serve-ml4.sh toolcheck && ./serve-ml4.sh vibecheck
```

On your laptop:

```bash
ssh -N -L 8000:<node>:8000 <user>@bunya1.rcc.uq.edu.au &
./vibe-setup.sh --via direct --host localhost --port 8000 --api-key "<key>"   # or --via litellm
vibe
```

## What runs where

The launcher reads the GPU architecture from the node (`GPU_ARCH=auto`) and
picks the weights and engine for it:

| Node | Weights (shared, read-only) | Engine | Layout |
|---|---|---|---|
| MI355X (bun159-161) | AMD Quark **MXFP4**, 539 GB | vLLM day-0 ROCm image | **TP4 + AMD specialized decode** ("monokernel") + DSpark, for up to 4 agents; TP8 + DSpark for more |
| MI355X | official **FP8**, 1.06 TB (`WEIGHTS=fp8`) | same | TP8 + DSpark |
| MI300X (bun144-145) | official **FP8**, 1.06 TB | vLLM ROCm nightly + ML4 branch overlay | TP8 + DSpark, BF16 KV |

`GATEWAY=direct` serves vLLM on `0.0.0.0:8000`, protected by an API key.
`GATEWAY=litellm` keeps vLLM on `127.0.0.1` and puts LiteLLM on `:8000` with its
own master key. On a full MI355X node this mode runs two TP4 replicas. A small
LiteLLM callback keeps each agent conversation on one replica, so its prefix
cache stays warm, and spreads new conversations across the two replicas. In either mode, the launcher writes a
`model_list` snippet you can paste into an external LiteLLM gateway
(`./serve-ml4.sh litellm-snippet`).

## Layout

```
serve-ml4.sh          launcher: serve plan pull overlay gpucheck check flagcheck download
                      verify-weights toolcheck vibecheck litellm-snippet restart-gateway
                      status stop loadstat
litellm/ml4_affinity.py   LiteLLM callback: same conversation -> same replica (prefix cache)
submit-ml4.sh         sbatch front end (node class, GPU count, gateway at submit time)
serve-ml4.sbatch      batch wrapper
vibe-setup.sh         Mistral Vibe provider/model config (run on the client machine)
bench-ml4.sh          vLLM bench client (sweep/latency/throughput/longcontext)
agentstress.py        growing-conversation agentic stress test (real text, reasoning replay)
tools/                ml4check.py (tool-call round trip), verify_weights.py
tests/                CPU-only checks:  python3 -m unittest discover -s tests -v
```
