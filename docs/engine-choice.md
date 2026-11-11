# Engine choice: vLLM, not SGLang

[Documentation home](../README.md) | [Compatibility](compatibility.md)

This repo started as an SGLang port of the Kimi K3.1 tools. It moved to vLLM
after a survey on 11 Nov 2026:

| | vLLM | SGLang |
|---|---|---|
| Mistral Large 4 in `main` | no | no |
| Model support | branch [vllm-project/vllm#64534](https://github.com/vllm-project/vllm/pull/64534) ("NVIDIA and AMD ROCm (MI350 Series) kernels") | open PR [sgl-project/sglang#47271](https://github.com/sgl-project/sglang/pull/47271) (Python only, validated on NVIDIA) |
| ROCm image with the model | **`vllm/vllm-openai-rocm:mistral-large-4`** (day 0, 29 Oct) | none (`lmsysorg/sglang:mistral-large-4` is CUDA only) |
| AMD Quark MXFP4 checkpoint | AMD validated on vLLM (4 x MI355X, `--monokernel`, DSpark) | not covered by the PR |
| AMD-specific decode path | TP4 FlyDSL specialized decode for ML4 on gfx950 | none for ML4 |
| MI300X (gfx942) | nightly images are built for gfx942. ROCm sparse-MLA and the ML4 indexer have gfx942 / fnuz code paths | `rocm/sgl-dev …-mi30x` nightlies + the PR overlay would be needed |

AMD's MXFP4 checkpoint, AMD's decode kernels and the only ROCm image all
point at vLLM.

## What "the day-0 image" and "the branch overlay" are

- **day0** (default on MI355X): `vllm/vllm-openai-rocm@sha256:536d1ef1…`
  (vLLM `0.1.dev23730+g64053a10b`, torch 2.13 + ROCm 10.0 wheels, aiter, FlyDSL
  0.4.0, mistral_common 1.13.0). Its vLLM kernels are compiled for **gfx950
  only** (`PYTORCH_ROCM_ARCH=gfx950`; `_C.abi3.so` carries only gfx950 code
  objects), so it cannot run on MI300X.
- **branch** (default on MI300X): `vllm/vllm-openai-rocm@sha256:55968af1…`
  (`nightly-rocm72-83c078d9`, 23 Oct, built for gfx90a/gfx942/gfx950). The ML4
  branch is two commits on top of `main@2d40c060`, and this is the closest
  nightly still published (base + 8 commits). The launcher copies the image's
  `vllm` package, including the compiled ops, puts the branch's `vllm/` Python
  at `BRANCH_SHA` on top, adds `mistral_common==1.13.0`, and bind-mounts the
  result over the image's copy. The branch's only native change
  (`csrc/hisparse_kernels.cu`, HiSparse host offload) is not used here. AMD
  used the same pattern: their MXFP4 card says they tested a nightly "on a
  development branch for Mistral Large 4".

When either project merges Mistral Large 4 and publishes a ROCm release image,
set `ENGINE_IMAGE` to it and `ENGINE=day0`. That removes the need for the
overlay.
