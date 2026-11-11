# Troubleshooting

[Documentation home](../README.md) | [Configuration](configuration.md)

Start with `./serve-ml4.sh status`, then the end of
`$STATE_DIR/logs/ml4-r0.log` (and `ml4-litellm.log`). Every log begins with
the recipe, image, environment and redacted argv.

| Symptom | Cause | Fix |
|---|---|---|
| `apptainer not found` | You are on a login node | Run inside the allocation (`srun --jobid … --pty bash -l`), or use `submit-ml4.sh` |
| `Conflicting visibility of agent-N between HIP_VISIBLE_DEVICES and GPU_DEVICE_ORDINAL`, then abort | Slurm exports `GPU_DEVICE_ORDINAL=0..7`; a replica sees a 4-GPU slice | Fixed in the launcher, which sets all four visibility variables per replica. Only appears if you start vLLM by hand |
| gpucheck: `devices: FAIL … No CUDA GPUs are available` | The vLLM images need the node's ROCm user libraries bound in | Expected. `rocm` mode is chosen automatically (cached per node + image) |
| `ENGINE=day0 cannot run on MI300X` | The day-0 image's vLLM kernels are gfx950-only | Use the default `ENGINE=branch` on MI300X |
| `Mistral Large 4 specialized decode unsupported: …` | Monokernel requested outside its envelope (TP4, gfx950, MXFP4, bf16/fp8 KV, DSpark or none, V2 runner, no DP/EP/PP/CP) | `MONOKERNEL=0`, or go back to the profile defaults. The launcher refuses the known cases before launch |
| `PermissionError … /tmp/aiter_configs/*.lock` | Another user owns `/tmp/aiter_configs` on the node | The launcher binds a per-job directory over it. If that bind fails, ask RCC for `mkdir -m 1777 /tmp/aiter_configs`, or move nodes |
| `Read-only file system: …/aiter/jit/…` | The aiter JIT dir is not writable | The launcher seeds `$STATE_DIR/cache-<image>/aiter-jit`. Set `AITER_JIT_TARGET` if detection failed |
| `… weights … are missing or incomplete` | Shared checkpoint incomplete, or a download still running | `./serve-ml4.sh verify-weights <fp8|mxfp4>`, then `download` |
| HTTP 400 `reasoning_effort=none is not supported` | The release tokenizer only has low/medium/high | Send `low`/`medium`/`high`, or omit the field (= `high`) |
| Client sees no thinking, or the agent loses track after tool calls | The client reads or replays the wrong reasoning field | vLLM: `reasoning`, LiteLLM: `reasoning_content`. In Vibe use `api_style = "openai"` ([clients](clients.md#mistral-vibe)) |
| `finish_reason=length` with empty content | The thinking used the whole `max_tokens` budget | Raise `max_tokens` (it counts reasoning), or use a lower effort |
| HiP OOM during KV allocation or graph capture | `MEM_UTIL` too high for the layout | Lower `MEM_UTIL` (e.g. 0.80) or `CONTEXT_LEN` |
| LiteLLM: `UnsupportedParamsError … reasoning_effort` | The gateway drops or refuses the param | The generated config sets `allowed_openai_params: ["reasoning_effort"]`. Add it to external gateways too |
| LiteLLM exits at start | Bad key env or port in use | `$STATE_DIR/logs/ml4-litellm.log`. Check `PORT` is free |
| `Unknown vLLM environment variable detected: VLLM_…` | A host variable with a `VLLM_` prefix leaked into the container | Harmless. The launcher's own knobs avoid the prefix |
