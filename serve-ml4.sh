#!/usr/bin/env bash
#
# serve-ml4.sh — one-click Mistral Large 4 (mistralai/Mistral-Large-4-1T-A52B)
# serving on Bunya's AMD GPU nodes with vLLM inside Apptainer, either facing
# clients directly or behind a LiteLLM gateway on the same node.
#
#   node class          weights (shared store)              layout
#   MI355X (gfx950)     AMD Quark MXFP4, 539 GB  (default)   TP4 + AMD specialized decode
#                                                            ("monokernel"), DSpark; or TP8
#   MI355X (gfx950)     official FP8, 1.06 TB                TP8, DSpark
#   MI300X (gfx942)     official FP8, 1.06 TB                TP8, DSpark
#
# The architecture is read from the GPUs at launch (GPU_ARCH=auto), so the same
# checkout serves on bun159-161 (MI355X) and bun144-145 (MI300X).
#
# Why vLLM and not SGLang: as of 11 Nov 2026 neither engine's main branch has
# Mistral Large 4, but vLLM publishes a day-0 ROCm image built from its ML4
# branch (vllm-project/vllm#64534), which carries AMD MI350-series kernels, the
# Quark MXFP4 path and the TP4 specialized decode. SGLang's support (#47271) is
# an unmerged PR with no ROCm image.
#
# Usage:
#   ./serve-ml4.sh [serve] [--detach]   start (vLLM replicas, plus LiteLLM if GATEWAY=litellm)
#   ./serve-ml4.sh plan                 print the resolved recipe + argv (no GPU, no apptainer)
#   ./serve-ml4.sh pull                 build the .sif image(s) (once)
#   ./serve-ml4.sh gpucheck             can the image reach this node's GPUs?
#   ./serve-ml4.sh check                does the image know Mistral Large 4 + the levers?
#   ./serve-ml4.sh flagcheck            does the image accept every flag of this recipe?
#   ./serve-ml4.sh download [fp8|mxfp4|all]   fetch pinned weights into the shared store
#   ./serve-ml4.sh verify-weights [fp8|mxfp4|all]   are the shared weights complete?
#   ./serve-ml4.sh toolcheck            two-turn tool call with reasoning replay (via PORT)
#   ./serve-ml4.sh vibecheck            the same, in Mistral Vibe's exact wire format
#   ./serve-ml4.sh litellm-snippet      print a model_list entry for an external LiteLLM
#   ./serve-ml4.sh status | stop | loadstat
#
# Configuration: ml4.env next to this script (or $ML4_ENV); see ml4-env.example.
# Variables exported in the shell win over the file.

set -euo pipefail

# The whole script is one { ... } block: bash parses it completely before
# running any of it, so editing or git-pulling this file while a server
# started from it is still attached cannot make bash resume mid-file.
{
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="serve"; MODE_ARG=""; DETACH=0; mode_set=0
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *) if (( mode_set )); then MODE_ARG="$arg"; else MODE="$arg"; mode_set=1; fi ;;
    esac
done

log()  { printf '\033[1;34m[ml4]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ml4 WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ml4 ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
quiet_mode() { [[ "$MODE" == plan || "$MODE" == litellm-snippet ]]; }

# ── Load config ─────────────────────────────────────────────────────────────

ENV_FILE="${ML4_ENV:-$SCRIPT_DIR/ml4.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    quiet_mode || log "Loaded config from $ENV_FILE"
else
    quiet_mode || warn "No config file at $ENV_FILE (cp ml4-env.example ml4.env); using defaults + environment."
fi

# ── Pins ────────────────────────────────────────────────────────────────────
#
# vllm/vllm-openai-rocm:mistral-large-4 (29 Oct 2026), the day-0 ROCm build of
# the ML4 branch. Pinned by the index digest so a re-push cannot change the
# engine under you; rocm/vllm-dev:mistral-large-4-20261029 is the same amd64
# image (sha256:4a898441).
DEFAULT_VLLM_IMAGE="docker://vllm/vllm-openai-rocm@sha256:536d1ef1c9ef0bcd619cbe0281d262f6df193e3ded68c90ccb83720a8e36d5f6"
# LiteLLM v1.108.5-lts (10 Nov 2026).
DEFAULT_LITELLM_IMAGE="docker://litellm/litellm@sha256:008a5916c0a0ca6f4f6ad64969f6a4ef6f03ce885a639f3d97fd53665754ce48"
# The day-0 image's vLLM kernels are built for gfx950 ONLY (PYTORCH_ROCM_ARCH=
# gfx950), so MI300X needs the 'branch' engine: vLLM's ROCm nightly of 23 Oct
# (main@83c078d9, built for gfx942+gfx950; the ML4 branch's base 2d40c060 + 8
# commits, the closest nightly still published) with the ML4 branch's Python
# overlaid at a pinned commit. The branch changes no compiled op that serving
# uses (its only csrc change is HiSparse host offload).
DEFAULT_BRANCH_BASE_IMAGE="docker://vllm/vllm-openai-rocm@sha256:55968af15ce4c1535ad20f95919738900004e99bbde9c0debe43891fee022128"
BRANCH_REPO="${BRANCH_REPO:-NickLucche/vllm}"           # vllm-project/vllm#64534
BRANCH_SHA="${BRANCH_SHA:-28a4495de85c64050b3aad39e946d02791710579}"   # 10 Nov 2026
ENGINE="${ENGINE:-auto}"
ENGINE_IMAGE="${ENGINE_IMAGE:-}"
LITELLM_IMAGE="${LITELLM_IMAGE:-$DEFAULT_LITELLM_IMAGE}"

FP8_REPO="mistralai/Mistral-Large-4-1T-A52B"
FP8_REVISION="${FP8_REVISION:-af26e7869af601441d5cdf75a1272469ddc3d059}"   # 2 Nov 2026 (weights unchanged since 29 Oct)
FP8_PATH="${FP8_PATH:-/scratch/opendata/models/gguf/mistral-large-4}"
FP8_GB=1061
MXFP4_REPO="amd/Mistral-Large-4-1T-A52B-Quark-MXFP4"
MXFP4_REVISION="${MXFP4_REVISION:-d084803a4a2bfac7274058937f6665750741d650}" # initial release; later commits README only
MXFP4_PATH="${MXFP4_PATH:-/scratch/opendata/models/huggingface/amd-Mistral-Large-4-1T-A52B-Quark-MXFP4}"
MXFP4_GB=579

STATE_DIR="${STATE_DIR:-/scratch/user/${USER:-$(id -un)}/models/mistral-large-4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistral-large-4}"
PORT="${PORT:-8000}"
BACKEND_PORT_BASE="${BACKEND_PORT_BASE:-8100}"

GPU_ARCH="${GPU_ARCH:-auto}"
WEIGHTS="${WEIGHTS:-auto}"
GATEWAY="${GATEWAY:-direct}"
AGENTS="${AGENTS:-4}"
PROFILE="${PROFILE:-auto}"
REPLICAS="${REPLICAS:-auto}"
SPECULATIVE="${SPECULATIVE-dspark}"
MONOKERNEL="${MONOKERNEL:-auto}"
DSPARK_TOKENS=3                     # the draft block depth; the checkpoint's head proposes exactly 3

CONTEXT_LEN="${CONTEXT_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"
MEM_UTIL="${MEM_UTIL:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
TOOL_PARSER="${TOOL_PARSER:-mistral}"
REASONING_PARSER="${REASONING_PARSER:-mistral}"
ENABLE_AITER="${ENABLE_AITER:-1}"
WEIGHT_LOAD_THREADS="${WEIGHT_LOAD_THREADS:-16}"
PREWARM="${PREWARM:-0}"
PREWARM_THREADS="${PREWARM_THREADS:-32}"
LITELLM_ROUTING="${LITELLM_ROUTING:-least-busy}"
LITELLM_WORKERS="${LITELLM_WORKERS:-4}"
LITELLM_AFFINITY="${LITELLM_AFFINITY:-1}"   # pin each conversation to one replica (litellm/ml4_affinity.py)
# (the pin table lives in one process, so affinity with 2 replicas runs 1 worker)
ROCM_MODE="${ROCM_MODE:-auto}"
ROCMINFO_SHIM="${ROCMINFO_SHIM:-auto}"
READY_TIMEOUT="${READY_TIMEOUT:-7200}"
EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
# How the image spells the specialized-decode switch. auto: '--monokernel' if
# the image's kernel config has it, else the model's own additional-config key.
MONO_ARGS="${MONO_ARGS:-auto}"

# ── Validation ──────────────────────────────────────────────────────────────

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_num()  { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }
opt_uint() { [[ -z "${!1}" ]] || is_uint "${!1}" || die "$1 must be a non-negative integer or empty, got '${!1}'."; }
opt_num()  { [[ -z "${!1}" ]] || is_num  "${!1}" || die "$1 must be a number or empty, got '${!1}'."; }

case "$GPU_ARCH"  in auto|mi355x|mi300x) ;; *) die "GPU_ARCH must be auto, mi355x or mi300x, got '$GPU_ARCH'." ;; esac
case "$WEIGHTS"   in auto|mxfp4|fp8) ;;     *) die "WEIGHTS must be auto, mxfp4 or fp8, got '$WEIGHTS'." ;; esac
case "$GATEWAY"   in direct|litellm) ;;     *) die "GATEWAY must be direct or litellm, got '$GATEWAY'." ;; esac
case "$PROFILE"   in auto|tp4|tp8) ;;       *) die "PROFILE must be auto, tp4 or tp8, got '$PROFILE'." ;; esac
case "$REPLICAS"  in auto|1|2) ;;           *) die "REPLICAS must be auto, 1 or 2, got '$REPLICAS'." ;; esac
case "$MONOKERNEL" in auto|0|1) ;;          *) die "MONOKERNEL must be auto, 0 or 1, got '$MONOKERNEL'." ;; esac
case "$SPECULATIVE" in dspark) ;; ""|none) SPECULATIVE=none ;;
    *) die "SPECULATIVE must be 'dspark' or 'none', got '$SPECULATIVE'. Mistral Large 4 ships its own DSpark draft head; there is no separate draft model." ;;
esac
case "$ENGINE"    in auto|day0|branch) ;;   *) die "ENGINE must be auto, day0 or branch, got '$ENGINE'." ;; esac
case "$PREWARM"   in auto|0|1) ;;           *) die "PREWARM must be auto, 0 or 1, got '$PREWARM'." ;; esac
case "$ROCM_MODE" in auto|rocm|devices) ;;  *) die "ROCM_MODE must be auto, rocm or devices, got '$ROCM_MODE'." ;; esac
case "$ROCMINFO_SHIM" in auto|off|force) ;; *) die "ROCMINFO_SHIM must be auto, off or force, got '$ROCMINFO_SHIM'." ;; esac
case "$LITELLM_ROUTING" in simple-shuffle|least-busy|latency-based-routing|usage-based-routing-v2) ;;
    *) die "LITELLM_ROUTING must be simple-shuffle, least-busy, latency-based-routing or usage-based-routing-v2." ;;
esac
is_uint "$AGENTS" && (( AGENTS >= 1 )) || die "AGENTS must be a positive integer, got '$AGENTS'."
for v in PORT BACKEND_PORT_BASE LITELLM_WORKERS LITELLM_AFFINITY READY_TIMEOUT PREWARM_THREADS; do
    is_uint "${!v}" || die "$v must be a non-negative integer, got '${!v}'."
done
for v in CONTEXT_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS WEIGHT_LOAD_THREADS; do opt_uint "$v"; done
opt_num MEM_UTIL
case "$KV_CACHE_DTYPE" in ""|auto|bfloat16|fp8|fp8_e4m3) ;;
    *) die "KV_CACHE_DTYPE must be empty, auto, bfloat16, fp8 or fp8_e4m3, got '$KV_CACHE_DTYPE'." ;;
esac

# ── Hardware: which GPUs, how many ──────────────────────────────────────────
# gfx_target_version in the KFD topology is readable without ROCm userspace:
# 90500 = gfx950 (MI350X/MI355X), 90402 = gfx942 (MI300X/MI325X).

kfd_gfx_versions() {
    local f
    for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [[ -r "$f" ]] && awk '$1=="gfx_target_version" && $2!=0 {print $2}' "$f"
    done 2>/dev/null
}
kfd_cus() {   # CUs of the first GPU (4 SIMDs per CU on CDNA3/4)
    cat /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null \
        | awk '$1=="simd_count" && $2!=0 {print $2/4; exit}'
}

ARCH_SRC="GPU_ARCH=$GPU_ARCH"
NODE_GFX="$(kfd_gfx_versions | sort -u | head -n1)"
if [[ "$GPU_ARCH" == "auto" ]]; then
    case "$NODE_GFX" in
        90500) GPU_ARCH=mi355x; ARCH_SRC="detected gfx950" ;;
        90402) GPU_ARCH=mi300x; ARCH_SRC="detected gfx942" ;;
        "")
            case "$(hostname -s 2>/dev/null)" in
                bun144|bun145) GPU_ARCH=mi300x; ARCH_SRC="hostname (no GPU visible)" ;;
                *)             GPU_ARCH=mi355x; ARCH_SRC="assumed (no GPU visible; set GPU_ARCH)" ;;
            esac ;;
        *) die "Unsupported GPU (gfx_target_version $NODE_GFX). This launcher targets MI355X (gfx950) and MI300X (gfx942)." ;;
    esac
elif [[ -n "$NODE_GFX" ]]; then
    case "$GPU_ARCH:$NODE_GFX" in
        mi355x:90500|mi300x:90402) ;;
        *) [[ "$MODE" == plan ]] || die "GPU_ARCH=$GPU_ARCH but this node's GPUs report gfx_target_version $NODE_GFX." ;;
    esac
fi
case "$GPU_ARCH" in mi355x) GFX=gfx950 ;; mi300x) GFX=gfx942 ;; esac

visible_devices_ok() { [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; }
# The allocation's GPU list (indices as ROCr sees them on the host).
ALLOC_LIST="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}}"
if [[ -z "$ALLOC_LIST" ]]; then
    n_kfd="$(kfd_gfx_versions | wc -l)"
    if (( n_kfd > 0 )); then ALLOC_LIST="$(seq -s, 0 $(( n_kfd - 1 )))"
    elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]] && is_uint "$SLURM_GPUS_ON_NODE"; then
        ALLOC_LIST="$(seq -s, 0 $(( SLURM_GPUS_ON_NODE - 1 )))"
    fi
fi
visible_devices_ok "${ALLOC_LIST:-x}" || ALLOC_LIST=""
IFS=, read -r -a ALLOC_GPUS <<<"$ALLOC_LIST"
NGPUS="${#ALLOC_GPUS[@]}"
GPUS_KNOWN=1
(( NGPUS > 0 )) || { NGPUS=8; GPUS_KNOWN=0; ALLOC_GPUS=(0 1 2 3 4 5 6 7); }

# ── Engine build ────────────────────────────────────────────────────────────
if [[ "$ENGINE" == auto ]]; then
    [[ "$GPU_ARCH" == mi355x ]] && ENGINE=day0 || ENGINE=branch
fi
if [[ "$ENGINE" == day0 && "$GPU_ARCH" == mi300x ]]; then
    die "ENGINE=day0 cannot run on MI300X: the day-0 image's vLLM kernels are compiled for gfx950 only. Use ENGINE=branch (the default there)."
fi
if [[ -z "$ENGINE_IMAGE" ]]; then
    [[ "$ENGINE" == day0 ]] && ENGINE_IMAGE="$DEFAULT_VLLM_IMAGE" || ENGINE_IMAGE="$DEFAULT_BRANCH_BASE_IMAGE"
fi

# ── Resolve the recipe ──────────────────────────────────────────────────────

if [[ "$WEIGHTS" == "auto" ]]; then
    [[ "$GPU_ARCH" == mi355x ]] && WEIGHTS=mxfp4 || WEIGHTS=fp8
fi
if [[ "$WEIGHTS" == mxfp4 && "$GPU_ARCH" == mi300x ]]; then
    die "WEIGHTS=mxfp4 needs MI355X: gfx942 has no MXFP4 matrix cores, and AMD qualified this
  checkpoint on MI350/MI355 only. Use WEIGHTS=fp8 (the default on MI300X)."
fi
case "$WEIGHTS" in
    mxfp4) MODEL_PATH="$MXFP4_PATH"; MODEL_REPO="$MXFP4_REPO"; MODEL_REVISION="$MXFP4_REVISION"; WEIGHTS_GB=$MXFP4_GB ;;
    fp8)   MODEL_PATH="$FP8_PATH";   MODEL_REPO="$FP8_REPO";   MODEL_REVISION="$FP8_REVISION";   WEIGHTS_GB=$FP8_GB ;;
esac

if [[ "$PROFILE" == "auto" ]]; then
    # MXFP4 fits four MI355X with room for ~2M tokens of FP8 KV, and four is
    # what the specialized decode is built for. FP8 needs eight GPUs on either
    # node class (1.06 TB). Past a handful of agents, one TP8 engine batches
    # better than a TP4 one.
    if [[ "$WEIGHTS" == mxfp4 && "$AGENTS" -le 4 ]]; then PROFILE=tp4; else PROFILE=tp8; fi
fi
TP_SIZE=${PROFILE#tp}
if [[ "$WEIGHTS" == fp8 && "$TP_SIZE" == 4 ]]; then
    die "PROFILE=tp4 with WEIGHTS=fp8: the FP8 checkpoint is 1.06 TB and needs all 8 GPUs (TP8)."
fi

# Specialized decode: TP4 on gfx950 with MXFP4 experts (vLLM refuses anything
# else at startup with one ValueError listing every reason, so check here).
USE_MONO=0
mono_ok=1; mono_why=""
[[ "$GPU_ARCH" == mi355x ]] || { mono_ok=0; mono_why+=" needs gfx950;"; }
[[ "$TP_SIZE" == 4 ]]       || { mono_ok=0; mono_why+=" needs TP4;"; }
[[ "$WEIGHTS" == mxfp4 ]]   || { mono_ok=0; mono_why+=" needs MXFP4 experts;"; }
case "$MONOKERNEL" in
    auto) USE_MONO=$mono_ok ;;
    1) (( mono_ok )) || die "MONOKERNEL=1:${mono_why% ;} (got arch=$GPU_ARCH tp=$TP_SIZE weights=$WEIGHTS)."; USE_MONO=1 ;;
    0) USE_MONO=0 ;;
esac

if [[ "$REPLICAS" == auto ]]; then
    if [[ "$GATEWAY" == litellm ]] && (( NGPUS / TP_SIZE >= 2 )); then REPLICAS=2; else REPLICAS=1; fi
fi
if (( REPLICAS > 1 )) && [[ "$GATEWAY" != litellm ]]; then
    die "REPLICAS=$REPLICAS needs GATEWAY=litellm: two engines need something in front to share one endpoint."
fi
NEED_GPUS=$(( TP_SIZE * REPLICAS ))
if (( GPUS_KNOWN )) && (( NEED_GPUS > NGPUS )); then
    [[ "$MODE" == plan ]] || die "This recipe needs $NEED_GPUS GPUs (TP$TP_SIZE x $REPLICAS); the allocation has $NGPUS ($ALLOC_LIST)."
fi
IDLE_GPUS=$(( NGPUS - NEED_GPUS ))

# Profile values (empty user knobs take these).
p_ctx=262144
case "$GPU_ARCH:$WEIGHTS:$TP_SIZE" in
    mi355x:mxfp4:4) p_seqs=16; p_batched=8192;  p_mem=0.85; p_kv=fp8 ;;
    mi355x:mxfp4:8) p_seqs=64; p_batched=16384; p_mem=0.85; p_kv=fp8 ;;
    mi355x:fp8:8)   p_seqs=64; p_batched=16384; p_mem=0.88; p_kv=fp8 ;;
    # MI300X: 132 GB of weights per 192 GB GPU. BF16 KV until FP8 (fnuz) sparse
    # MLA is qualified on gfx942 — see docs/compatibility.md.
    mi300x:fp8:8)   p_seqs=32; p_batched=8192;  p_mem=0.90; p_kv=auto ;;
    *) die "internal: no profile for $GPU_ARCH/$WEIGHTS/TP$TP_SIZE" ;;
esac
(( AGENTS > p_seqs )) && p_seqs=$AGENTS
: "${CONTEXT_LEN:=$p_ctx}"
: "${MAX_NUM_SEQS:=$p_seqs}"
: "${MAX_BATCHED_TOKENS:=$p_batched}"
: "${MEM_UTIL:=$p_mem}"
: "${KV_CACHE_DTYPE:=$p_kv}"
if (( USE_MONO )) && [[ "$KV_CACHE_DTYPE" != fp8 && "$KV_CACHE_DTYPE" != fp8_e4m3 && "$KV_CACHE_DTYPE" != bfloat16 ]]; then
    die "The specialized decode reads an FP8 (E4M3) or bf16 latent cache; KV_CACHE_DTYPE=$KV_CACHE_DTYPE is refused. Use fp8 or MONOKERNEL=0."
fi
if (( MAX_NUM_SEQS * REPLICAS < AGENTS )); then
    warn "MAX_NUM_SEQS=$MAX_NUM_SEQS x $REPLICAS replica(s) is below AGENTS=$AGENTS — extra agents will queue."
fi

# Engine env (the image's own defaults otherwise).
R_ENV=(OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1)
[[ "$ENABLE_AITER" == 1 ]] && R_ENV+=(VLLM_ROCM_USE_AITER=1)
# The specialized decode refuses the V1 model runner.
(( USE_MONO )) && R_ENV+=(VLLM_USE_V2_MODEL_RUNNER=1)
# shellcheck disable=SC2206
extra_env=($EXTRA_ENV)
for kv in ${extra_env[@]+"${extra_env[@]}"}; do
    [[ "$kv" == [A-Za-z_]*=* ]] || die "EXTRA_ENV entries must be NAME=VALUE, got '$kv'."
    R_ENV+=("$kv")
done

# Ports: direct = one replica on 0.0.0.0:PORT. litellm = replicas on
# 127.0.0.1:BACKEND_PORT_BASE+1.. (never reachable off-node), LiteLLM on PORT.
replica_host() { [[ "$GATEWAY" == litellm ]] && echo 127.0.0.1 || echo 0.0.0.0; }
replica_port() { [[ "$GATEWAY" == litellm ]] && echo $(( BACKEND_PORT_BASE + 1 + $1 )) || echo "$PORT"; }
replica_gpus() { local i="$1"; local s=$(( i * TP_SIZE )); local IFS=,; echo "${ALLOC_GPUS[*]:$s:$TP_SIZE}"; }

recipe_summary() {
    local s="arch=$GPU_ARCH engine=$ENGINE weights=$WEIGHTS tp=$TP_SIZE replicas=$REPLICAS"
    s+=" mono=$USE_MONO spec=$SPECULATIVE"
    [[ "$SPECULATIVE" == dspark ]] && s+="($DSPARK_TOKENS)"
    s+=" kv=$KV_CACHE_DTYPE ctx=$CONTEXT_LEN max-seqs=$MAX_NUM_SEQS gateway=$GATEWAY"
    printf '%s' "$s"
}

# ── Engine argv (per replica) ───────────────────────────────────────────────

MONO_RESOLVED=()
resolve_mono_args() {   # uses image help when available
    MONO_RESOLVED=()
    (( USE_MONO )) || return 0
    if [[ "$MONO_ARGS" != auto ]]; then
        # shellcheck disable=SC2206
        MONO_RESOLVED=($MONO_ARGS); return 0
    fi
    if [[ -n "${IMAGE_HELP:-}" ]] && ! grep -q -- '--monokernel' <<<"$IMAGE_HELP"; then
        MONO_RESOLVED=(--additional-config '{"decode_backend": "latency_specialized"}')
    else
        MONO_RESOLVED=(--monokernel)
    fi
}

build_cmd() {   # <replica index> -> cmd[]
    local i="$1"
    # shellcheck disable=SC2206
    local extra=($EXTRA_ENGINE_ARGS)
    cmd=(vllm serve "$MODEL_PATH"
         --served-model-name "$SERVED_MODEL_NAME"
         --tokenizer-mode mistral --config-format mistral --load-format mistral
         --tensor-parallel-size "$TP_SIZE"
         --max-model-len "$CONTEXT_LEN"
         --max-num-seqs "$MAX_NUM_SEQS"
         --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
         --gpu-memory-utilization "$MEM_UTIL")
    [[ "$KV_CACHE_DTYPE" != auto ]] && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    # prompt_tokens_details.cached_tokens in usage: agent clients and LiteLLM
    # report prefix-cache hits from it (vLLM omits it by default).
    cmd+=(--enable-prefix-caching --enable-prompt-tokens-details)
    [[ -n "$TOOL_PARSER" && "$TOOL_PARSER" != none ]] && cmd+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
    [[ -n "$REASONING_PARSER" && "$REASONING_PARSER" != none ]] && cmd+=(--reasoning-parser "$REASONING_PARSER")
    [[ "$SPECULATIVE" == dspark ]] && cmd+=(--speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": $DSPARK_TOKENS}")
    cmd+=(${MONO_RESOLVED[@]+"${MONO_RESOLVED[@]}"})
    if [[ -n "$WEIGHT_LOAD_THREADS" && "$WEIGHT_LOAD_THREADS" != 0 ]]; then
        cmd+=(--model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": $WEIGHT_LOAD_THREADS}")
    fi
    cmd+=(--host "$(replica_host)" --port "$(replica_port "$i")" --api-key "$API_KEY_ARG")
    cmd+=(${extra[@]+"${extra[@]}"})
}

# ── State paths ─────────────────────────────────────────────────────────────

# STATE_DIR is on shared scratch, so one user can run servers on several
# nodes at once (e.g. bun161 and bun144): runtime state is per node.
NODE_TAG="$(hostname -s 2>/dev/null || echo node)"
LOG_DIR="$STATE_DIR/logs"
PID_FILE="$STATE_DIR/ml4-server-$NODE_TAG.pids"
LOADTIMES_FILE="$STATE_DIR/ml4-loadtimes.log"
API_KEY_FILE="$STATE_DIR/ml4-api-key"
LITELLM_KEY_FILE="$STATE_DIR/ml4-litellm-key"
LITELLM_DIR="$STATE_DIR/litellm-$NODE_TAG"
digest12() { local d="${1##*@sha256:}"; [[ "$d" == "$1" ]] && d="$(printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_')"; printf '%s' "${d:0:12}"; }
SIF_PATH="${SIF_PATH:-$STATE_DIR/vllm-$ENGINE-$(digest12 "$ENGINE_IMAGE").sif}"
OVERLAY_DIR="$STATE_DIR/overlay-${BRANCH_SHA:0:12}-on-$(digest12 "$ENGINE_IMAGE")"
LITELLM_SIF="${LITELLM_SIF:-$STATE_DIR/litellm-$(digest12 "$LITELLM_IMAGE").sif}"
replica_log() { echo "$LOG_DIR/ml4-$NODE_TAG-r$1.log"; }
LITELLM_LOG="$LOG_DIR/ml4-$NODE_TAG-litellm.log"

read_key() { [[ -r "$1" ]] && cat "$1" || true; }
client_key() {   # the key a client uses on PORT
    if [[ "${ML4_GATEWAY_RUNNING:-$GATEWAY}" == litellm ]]; then
        printf '%s' "${LITELLM_MASTER_KEY:-$(read_key "$LITELLM_KEY_FILE")}"
    else
        printf '%s' "${ML4_API_KEY:-$(read_key "$API_KEY_FILE")}"
    fi
}

pids_alive() {   # prints "name pid" for live entries
    [[ -f "$PID_FILE" ]] || return 0
    local name pid
    while read -r name pid _; do
        [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null && echo "$name $pid"
    done < "$PID_FILE"
}
server_running() { [[ -n "$(pids_alive)" ]]; }

stop_all() {
    local name pid any=0
    while read -r name pid; do
        [[ -n "${pid:-}" ]] || continue
        any=1; log "Stopping $name (pid $pid) ..."
        kill "$pid" 2>/dev/null || true
    done < <(pids_alive)
    for _ in $(seq 1 30); do server_running || break; sleep 1; done
    while read -r name pid; do kill -9 "$pid" 2>/dev/null || true; done < <(pids_alive)
    (( any )) || warn "No running server recorded in $PID_FILE."
    # Strays of this user only (an engine left by a killed launcher).
    pkill -u "$(id -u)" -f 'vllm serve /scratch/opendata/models' 2>/dev/null || true
    pkill -u "$(id -u)" -f 'litellm --config .*ml4-litellm' 2>/dev/null || true
    rm -f "$PID_FILE"
}

# ── LiteLLM config (also used by litellm-snippet) ───────────────────────────

litellm_model_entries() {   # <api_base...>; keys come from the environment
    local base n=0
    for base in "$@"; do
        cat <<EOF
  - model_name: $SERVED_MODEL_NAME
    litellm_params:
      model: hosted_vllm/$SERVED_MODEL_NAME
      api_base: $base
      api_key: os.environ/ML4_BACKEND_KEY
      timeout: 3600
      stream_timeout: 3600
      # reasoning_effort is Mistral Large 4's thinking control (low|medium|high)
      allowed_openai_params: ["reasoning_effort"]
    model_info:
      id: ml4-r$n
      mode: chat
      max_input_tokens: $CONTEXT_LEN
      max_output_tokens: $CONTEXT_LEN
      supports_function_calling: true
      supports_tool_choice: true
      supports_reasoning: true
      supports_vision: true
      supports_response_schema: true
EOF
        n=$(( n + 1 ))
    done
}

write_litellm_config() {
    local bases=() i
    for (( i = 0; i < REPLICAS; i++ )); do bases+=("http://127.0.0.1:$(replica_port "$i")/v1"); done
    mkdir -p "$LITELLM_DIR"
    cp "$SCRIPT_DIR/litellm/ml4_affinity.py" "$LITELLM_DIR/"
    {
        echo "# Generated by serve-ml4.sh $(date -Is) — regenerated on every start."
        echo "model_list:"
        litellm_model_entries "${bases[@]}"
        cat <<EOF
router_settings:
  routing_strategy: $LITELLM_ROUTING
  num_retries: 0
litellm_settings:
  drop_params: false
  request_timeout: 3600
  telemetry: false
$( (( LITELLM_AFFINITY )) && echo '  callbacks: ml4_affinity.affinity     # same conversation -> same replica (prefix cache)')
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
EOF
    } > "$LITELLM_DIR/ml4-litellm.yaml"
}

start_litellm() {   # LiteLLM on 0.0.0.0:PORT in front of the REPLICAS backends
    write_litellm_config
    {
        printf '### serve-ml4.sh litellm  %s  host=%s job=%s\n' "$(date)" "$(hostname -s)" "${SLURM_JOB_ID:-}"
        printf '### image: %s  config: %s\n\n' "$LITELLM_IMAGE" "$LITELLM_DIR/ml4-litellm.yaml"
    } > "$LITELLM_LOG"
    apptainer exec --cleanenv --bind "$LITELLM_DIR":/ml4-litellm \
        --env LITELLM_MASTER_KEY="$LITELLM_MASTER_KEY" --env ML4_BACKEND_KEY="$ML4_API_KEY" \
        --env LITELLM_TELEMETRY=False --env LITELLM_LOG=INFO --env PYTHONNOUSERSITE=1 \
        "$LITELLM_SIF" litellm --config /ml4-litellm/ml4-litellm.yaml --host 0.0.0.0 --port "$PORT" \
        --num_workers "$(( LITELLM_AFFINITY && REPLICAS > 1 ? 1 : LITELLM_WORKERS ))" >>"$LITELLM_LOG" 2>&1 &
    echo "litellm $!" >> "$PID_FILE"
    local lpid=$! start_ts_l
    start_ts_l=$(date +%s)
    until curl -fsS -m 5 "http://127.0.0.1:$PORT/health/liveliness" >/dev/null 2>&1; do
        kill -0 "$lpid" 2>/dev/null || { tail -n 40 "$LITELLM_LOG"; die "LiteLLM exited during startup ($LITELLM_LOG)."; }
        (( $(date +%s) - start_ts_l > 300 )) && die "LiteLLM not up within 300s ($LITELLM_LOG)."
        sleep 3
    done
    log "LiteLLM gateway up on :$PORT -> ${REPLICAS} replica(s), routing $LITELLM_ROUTING$( (( LITELLM_AFFINITY && REPLICAS > 1 )) && echo ' + conversation affinity')."
}

# ── Simple modes (no apptainer) ─────────────────────────────────────────────

case "$MODE" in
    plan)
        API_KEY_ARG="<redacted>"
        resolve_mono_args
        echo "# $(recipe_summary)"
        echo "# arch source: $ARCH_SRC; gpus: ${ALLOC_LIST:-unknown (assumed 8)}; idle: $IDLE_GPUS"
        echo "# weights: $MODEL_PATH ($MODEL_REPO @ ${MODEL_REVISION:0:8}, ~${WEIGHTS_GB} GB)"
        echo "# image: $ENGINE_IMAGE"
        [[ "$ENGINE" == branch ]] && echo "# overlay: $BRANCH_REPO@${BRANCH_SHA:0:12} Python over the image's vllm (prepared in $OVERLAY_DIR)"
        echo "# env:"
        for e in "${R_ENV[@]}"; do echo "  $e"; done
        for (( i = 0; i < REPLICAS; i++ )); do
            build_cmd "$i"
            echo "# replica $i: GPUs $(replica_gpus "$i") -> $(replica_host):$(replica_port "$i")"
            printf ' %q' "${cmd[@]}"; echo
        done
        if [[ "$GATEWAY" == litellm ]]; then
            echo "# litellm: 0.0.0.0:$PORT  routing=$LITELLM_ROUTING affinity=$LITELLM_AFFINITY workers=$LITELLM_WORKERS image=$LITELLM_IMAGE"
        fi
        exit 0 ;;
    litellm-snippet)
        host="${ML4_SNIPPET_HOST:-$(hostname -s 2>/dev/null || echo '<node>')}"
        echo "# Add to your LiteLLM gateway's config.yaml under model_list."
        echo "# Export ML4_BACKEND_KEY on the gateway: the serving key ($API_KEY_FILE)."
        echo "# The node name changes per Slurm job; point api_base at your tunnel if the gateway is off-cluster."
        echo "model_list:"
        litellm_model_entries "http://$host:$PORT/v1"
        exit 0 ;;
    stop) stop_all; log "Stopped."; exit 0 ;;
    status)
        alive="$(pids_alive)"
        if [[ -n "$alive" ]]; then log "Processes:"; sed 's/^/    /' <<<"$alive"; else log "No live processes recorded ($PID_FILE)."; fi
        [[ -n "$alive" ]] && grep -q '^litellm ' <<<"$alive" && ML4_GATEWAY_RUNNING=litellm || ML4_GATEWAY_RUNNING=direct
        key="$(client_key)"
        if curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" -H "Authorization: Bearer $key" >/dev/null 2>&1; then
            log "Endpoint http://$(hostname -s):$PORT/v1 ($ML4_GATEWAY_RUNNING): OK"
        else
            warn "http://127.0.0.1:$PORT/v1/models not answering (not started, still loading, or wrong key)."
        fi
        for f in "$LOG_DIR"/ml4-"$NODE_TAG"-r*.log; do
            [[ -r "$f" ]] || continue
            r="$(basename "$f" .log)"; r="vllm-${r##*-}"
            # only replicas that are running (or every log when nothing is)
            [[ -z "$alive" ]] || grep -q "^$r " <<<"$alive" || continue
            log "$(basename "$f"): $(head -n 2 "$f" | sed -n 's/^### recipe: //p')"
            acc="$(grep -oE 'Mean acceptance length: [0-9.]+' "$f" | tail -n 20 \
                   | awk '{s+=$NF; n++} END {if (n) printf "%.2f (mean of the last %d reports)", s/n, n}')"
            [[ -n "$acc" ]] && log "    DSpark acceptance length: $acc"
            mono="$(grep -v '^###' "$f" | grep -m1 -oE 'Mistral Large 4 specialized decode: [^,]+,[^,]+' || true)"
            [[ -n "$mono" ]] && log "    decode: $mono"
            tps="$(grep -oE 'Avg generation throughput: [0-9.]+ tokens/s' "$f" | tail -n1 || true)"
            [[ -n "$tps" ]] && log "    last $tps"
        done
        exit 0 ;;
    toolcheck|vibecheck)
        ML4_GATEWAY_RUNNING="$GATEWAY"
        [[ -n "$(pids_alive)" ]] && { grep -q '^litellm ' <<<"$(pids_alive)" && ML4_GATEWAY_RUNNING=litellm || ML4_GATEWAY_RUNNING=direct; }
        url="${CHECK_URL:-http://127.0.0.1:$PORT}"
        args=(--base "$url" --key "$(client_key)" --model "$SERVED_MODEL_NAME"
              --effort "${CHECK_EFFORT:-low}")
        if [[ "$MODE" == vibecheck ]]; then
            field="${CHECK_REASONING_FIELD:-}"
            [[ -z "$field" ]] && { [[ "$ML4_GATEWAY_RUNNING" == litellm ]] && field=reasoning_content || field=reasoning; }
            args+=(--vibe --reasoning-field "$field")
        fi
        exec python3 "$SCRIPT_DIR/tools/ml4check.py" "${args[@]}" ;;
    serve|pull|overlay|download|verify-weights|check|gpucheck|flagcheck|loadstat|restart-gateway) ;;
    *) die "Unknown mode '$MODE'. Use: serve | plan | pull | overlay | gpucheck | check | flagcheck | download | verify-weights | toolcheck | vibecheck | litellm-snippet | restart-gateway | status | stop | loadstat" ;;
esac

# ── Weights: verify / download (no GPU needed) ──────────────────────────────

weights_targets() {   # <arg> -> list of fp8/mxfp4
    case "${1:-}" in
        fp8|mxfp4) echo "$1" ;;
        all) echo "fp8 mxfp4" ;;
        "") echo "$WEIGHTS" ;;
        *) die "Expected fp8, mxfp4 or all, got '$1'." ;;
    esac
}
weights_info() {   # <fp8|mxfp4> -> W_PATH W_REPO W_REV W_GB
    case "$1" in
        fp8)   W_PATH="$FP8_PATH";   W_REPO="$FP8_REPO";   W_REV="$FP8_REVISION";   W_GB=$FP8_GB ;;
        mxfp4) W_PATH="$MXFP4_PATH"; W_REPO="$MXFP4_REPO"; W_REV="$MXFP4_REVISION"; W_GB=$MXFP4_GB ;;
    esac
}
verify_weights() {   # <path> [repo rev] -> 0 complete, 1 not
    python3 "$SCRIPT_DIR/tools/verify_weights.py" "$@"
}

if [[ "$MODE" == verify-weights ]]; then
    rc=0
    for w in $(weights_targets "$MODE_ARG"); do
        weights_info "$w"
        log "Verifying $w: $W_PATH ($W_REPO @ ${W_REV:0:8})"
        verify_weights "$W_PATH" "$W_REPO" "$W_REV" ${VERIFY_OFFLINE:+--offline} || rc=1
    done
    exit "$rc"
fi

if [[ "$MODE" == download ]]; then
    if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
        [[ -r "$HF_TOKEN_FILE" ]] || die "HF_TOKEN_FILE '$HF_TOKEN_FILE' is not readable."
        HF_TOKEN="$(<"$HF_TOKEN_FILE")"; export HF_TOKEN
    fi
    for w in $(weights_targets "$MODE_ARG"); do
        weights_info "$w"
        if verify_weights "$W_PATH" "$W_REPO" "$W_REV" --quiet 2>/dev/null; then
            log "$w weights already complete in $W_PATH — nothing to download."; continue
        fi
        mkdir -p "$W_PATH" 2>/dev/null || true
        [[ -w "$W_PATH" ]] || die "$W_PATH is incomplete and not writable by you ($(stat -c '%U:%G %A' "$W_PATH" 2>/dev/null)). Ask its owner, or set ${w^^}_PATH."
        free_gb="$(df -Pk "$W_PATH" | awk 'NR==2 {print int($4/1024/1024)}')"
        (( ${free_gb:-0} >= W_GB )) || warn "Only ${free_gb} GB free under $W_PATH; $W_REPO is ~${W_GB} GB. Check 'rquota'."
        log "Downloading $W_REPO @ ${W_REV:0:8} -> $W_PATH (~${W_GB} GB, resumable) ..."
        (
            umask 022
            export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_TELEMETRY=1
            if command -v hf >/dev/null 2>&1; then
                hf download "$W_REPO" --revision "$W_REV" --local-dir "$W_PATH" --max-workers 16
            else
                [[ -f "$SIF_PATH" ]] || die "No 'hf' CLI on this host and no image at $SIF_PATH ('./serve-ml4.sh pull' first, or 'pip install --user huggingface_hub')."
                apptainer exec --bind "$W_PATH":"$W_PATH" --env HF_TOKEN="${HF_TOKEN:-}" \
                    --env HF_XET_HIGH_PERFORMANCE=1 "$SIF_PATH" \
                    hf download "$W_REPO" --revision "$W_REV" --local-dir "$W_PATH" --max-workers 16
            fi
        ) || die "Download of $w failed. Re-run to resume."
        # Shared store: readable by everyone, like the other models there.
        chmod -R u+rwX,go+rX,go-w "$W_PATH" 2>/dev/null || true
        python3 "$SCRIPT_DIR/tools/verify_weights.py" "$W_PATH" "$W_REPO" "$W_REV" --write-manifest \
            || die "$w downloaded but does not verify. Re-run './serve-ml4.sh download $w'."
        log "$w weights complete in $W_PATH"
    done
    exit 0
fi

# ── Preflight (shared by apptainer modes) ───────────────────────────────────

command -v apptainer >/dev/null 2>&1 \
    || die "apptainer not found — run this INSIDE a compute-node allocation (salloc/sbatch). It is not installed on Bunya login nodes."
command -v curl >/dev/null 2>&1 || die "curl not found on PATH."
mkdir -p "$STATE_DIR" "$LOG_DIR" 2>/dev/null || true
[[ -d "$STATE_DIR" && -w "$STATE_DIR" ]] || die "STATE_DIR '$STATE_DIR' does not exist or is not writable."
chmod 700 "$STATE_DIR" 2>/dev/null || true

: "${APPTAINER_CACHEDIR:=$STATE_DIR/apptainer-cache}"
: "${APPTAINER_TMPDIR:=$STATE_DIR/apptainer-tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" 2>/dev/null || true
export APPTAINER_CACHEDIR APPTAINER_TMPDIR
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR" SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

# Caches vLLM, Triton and torch.compile would otherwise put under ~/.cache on
# /home (quota). One tree per image so a new image never reads stale kernels.
# Per image AND per GPU arch: aiter/FlyDSL modules JIT-built on gfx950 must
# never be loaded on gfx942 (and vice versa) from the shared scratch tree.
CACHE_ROOT="$STATE_DIR/cache-$(digest12 "$ENGINE_IMAGE")-$GFX"
mkdir -p "$CACHE_ROOT"/{vllm,triton,xdg,hf} 2>/dev/null || true
PROBE_TMP="/tmp/ml4-$(id -u)"
PROBE_ENV=(--env "AITER_JIT_DIR=$PROBE_TMP/aiter-jit" --env PYTHONNOUSERSITE=1
           --env "VLLM_CACHE_ROOT=$PROBE_TMP/vllm" --env VLLM_NO_USAGE_STATS=1)

# Apptainer 1.5.0-1.5.2 wrapped mksquashfs in proot, which dies on hosts that
# refuse ptrace; 'build --ignore-proot' is the escape hatch (K3 repo).
pull_one() {   # <sif> <image> <label>
    local sif="$1" image="$2" label="$3" logf rc
    [[ -f "$sif" ]] && { log "$label image present: $sif"; return 0; }
    logf="$(mktemp "$APPTAINER_TMPDIR/ml4-pull.XXXXXX.log")"
    log "Pulling $label image $image -> $sif (one-time) ..."
    if apptainer pull "$sif" "$image" 2>&1 | tee "$logf"; then rc=0; else rc=${PIPESTATUS[0]}; fi
    if (( rc != 0 )); then
        rm -f "$sif"
        if grep -qi 'proot\|ptrace' "$logf"; then
            warn "Apptainer could not use proot here; retrying with build --ignore-proot."
            if PROOT_NO_SECCOMP=1 apptainer build --ignore-proot "$sif" "$image" 2>&1 | tee "$logf"; then rc=0
            else rc=${PIPESTATUS[0]}; rm -f "$sif"; fi
        fi
    fi
    rm -f "$logf"
    (( rc == 0 )) && log "$label image ready: $sif"
    return "$rc"
}

if [[ "$MODE" == pull ]]; then
    pull_one "$SIF_PATH" "$ENGINE_IMAGE" vLLM || die "vLLM image pull failed (network? space in $APPTAINER_CACHEDIR?)."
    if [[ "$GATEWAY" == litellm || "$MODE_ARG" == all || "$MODE_ARG" == litellm ]]; then
        pull_one "$LITELLM_SIF" "$LITELLM_IMAGE" LiteLLM || die "LiteLLM image pull failed."
    else
        log "LiteLLM image not pulled (GATEWAY=direct). './serve-ml4.sh pull all' fetches it too."
    fi
    exit 0
fi
# ── restart-gateway: new LiteLLM in front of the running replicas ─────────
# Changes to LITELLM_ROUTING / LITELLM_AFFINITY / LITELLM_WORKERS without
# reloading the engines (and their weights).
if [[ "$MODE" == restart-gateway ]]; then
    nrep="$(pids_alive | grep -c '^vllm-r' || true)"
    (( nrep > 0 )) || die "No vLLM replicas are running (./serve-ml4.sh status)."
    REPLICAS="$nrep"; GATEWAY=litellm
    for (( i = 0; i < REPLICAS; i++ )); do
        curl -fsS -m 5 "http://127.0.0.1:$(replica_port "$i")/health" >/dev/null 2>&1 \
            || die "Replica $i is not on 127.0.0.1:$(replica_port "$i"): the engines were started with GATEWAY=direct. Restart with GATEWAY=litellm."
    done
    ML4_API_KEY="${ML4_API_KEY:-$(read_key "$API_KEY_FILE")}"
    [[ -n "$ML4_API_KEY" ]] || die "No serving key in $API_KEY_FILE."
    if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
        LITELLM_MASTER_KEY="$(read_key "$LITELLM_KEY_FILE")"
        [[ -n "$LITELLM_MASTER_KEY" ]] || { LITELLM_MASTER_KEY="sk-$(openssl rand -hex 24)"; (umask 077 && printf '%s' "$LITELLM_MASTER_KEY" > "$LITELLM_KEY_FILE"); }
    fi
    [[ -f "$LITELLM_SIF" ]] || pull_one "$LITELLM_SIF" "$LITELLM_IMAGE" LiteLLM || die "LiteLLM image pull failed."
    old="$(pids_alive | awk '$1=="litellm" {print $2}')"
    if [[ -n "$old" ]]; then
        log "Stopping LiteLLM (pid $old) ..."; kill "$old" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$old" 2>/dev/null || break; sleep 1; done
        kill -9 "$old" 2>/dev/null || true
    fi
    grep -v '^litellm ' "$PID_FILE" > "$PID_FILE.tmp" || true; mv "$PID_FILE.tmp" "$PID_FILE"
    for _ in $(seq 1 30); do (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null || break; sleep 1; done
    start_litellm
    disown "$(awk '$1=="litellm" {print $2}' "$PID_FILE")" 2>/dev/null || true
    exit 0
fi

[[ -f "$SIF_PATH" ]] || { log "No vLLM image at $SIF_PATH yet — pulling (one-time)."; pull_one "$SIF_PATH" "$ENGINE_IMAGE" vLLM || die "Pull failed; run './serve-ml4.sh pull' to debug."; }

# ── 'branch' engine: the ML4 branch's Python over the nightly's vllm ────────
# Prepared once per (branch commit, image): a full copy of the image's vllm
# package (compiled .so files included), the branch's vllm/ tree copied on top,
# stale bytecode dropped, and mistral_common >= 1.13 (tekken v15 reasoning
# efforts) installed beside it if the image's is older. Bound over the image's
# vllm at run time, so no image build (and no fakeroot) is needed.
VLLM_SITE=""; OVERLAY_ARGS=()
image_vllm_site() {
    [[ -n "$VLLM_SITE" ]] || VLLM_SITE="$(apptainer exec "${PROBE_ENV[@]}" "$SIF_PATH" python3 -c \
        'import importlib.util,os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))' 2>/dev/null | tail -n1)"
    printf '%s' "$VLLM_SITE"
}
prepare_overlay() {
    [[ "$ENGINE" == branch ]] || return 0
    local site tgz tmp
    site="$(image_vllm_site)"
    [[ "$site" == /* ]] || die "Could not locate vllm inside $SIF_PATH."
    if [[ ! -f "$OVERLAY_DIR/.complete" ]]; then
        log "Preparing the ML4 overlay: $BRANCH_REPO@${BRANCH_SHA:0:12} over $(basename "$SIF_PATH") (one-time) ..."
        rm -rf "${OVERLAY_DIR:?}"; mkdir -p "$OVERLAY_DIR/src" "$STATE_DIR/src"
        tgz="$STATE_DIR/src/vllm-${BRANCH_SHA:0:12}.tar.gz"
        if [[ ! -s "$tgz" ]]; then
            curl -fsSL -m 600 "https://codeload.github.com/$BRANCH_REPO/tar.gz/$BRANCH_SHA" -o "$tgz.part" \
                && mv "$tgz.part" "$tgz" || { rm -f "$tgz.part"; die "Could not fetch $BRANCH_REPO@$BRANCH_SHA (network?)."; }
        fi
        tar -xzf "$tgz" -C "$OVERLAY_DIR/src" --strip-components=1 --wildcards '*/vllm/*' \
            || die "Could not unpack $tgz."
        [[ -d "$OVERLAY_DIR/src/vllm" ]] || die "$tgz has no vllm/ package."
        apptainer exec --bind "$STATE_DIR":"$STATE_DIR" "$SIF_PATH" cp -a "$site" "$OVERLAY_DIR/vllm" \
            || die "Could not copy the image's vllm package."
        chmod -R u+w "$OVERLAY_DIR/vllm"
        cp -a "$OVERLAY_DIR/src/vllm/." "$OVERLAY_DIR/vllm/"
        find "$OVERLAY_DIR/vllm" -name __pycache__ -type d -prune -exec rm -rf {} +
        rm -rf "$OVERLAY_DIR/src"
        local mc
        mc="$(apptainer exec "${PROBE_ENV[@]}" "$SIF_PATH" python3 -c 'import mistral_common as m; print(m.__version__)' 2>/dev/null | tail -n1)"
        if ! python3 -c 'import sys; v=tuple(int(x) for x in sys.argv[1].split(".")[:3]); sys.exit(0 if v >= (1,13,0) else 1)' "${mc:-0}" 2>/dev/null; then
            log "  image mistral_common ${mc:-?} < 1.13.0 — installing 1.13.0 beside the overlay"
            apptainer exec --bind "$STATE_DIR":"$STATE_DIR" --env PYTHONNOUSERSITE=1 "$SIF_PATH" \
                python3 -m pip install --quiet --no-deps --target "$OVERLAY_DIR/site" "mistral_common==1.13.0" \
                || die "pip install mistral_common==1.13.0 failed (network?)."
        fi
        printf '%s\n' "$BRANCH_REPO@$BRANCH_SHA over $ENGINE_IMAGE ($site)" > "$OVERLAY_DIR/.complete"
        log "  overlay ready: $OVERLAY_DIR"
    fi
    OVERLAY_ARGS=(--bind "$OVERLAY_DIR/vllm":"$site")
    [[ -d "$OVERLAY_DIR/site" ]] && OVERLAY_ARGS+=(--env "PYTHONPATH=$OVERLAY_DIR/site")
    # every probe sees the same vllm as the server
    PROBE_ENV+=(--bind "$STATE_DIR":"$STATE_DIR" "${OVERLAY_ARGS[@]}")
}
prepare_overlay
if [[ "$MODE" == overlay ]]; then
    [[ "$ENGINE" == branch ]] || die "ENGINE=$ENGINE uses no overlay (ENGINE=branch does)."
    log "Overlay: $(cat "$OVERLAY_DIR/.complete")"; exit 0
fi

# ── /tmp/aiter_configs: aiter's HARDCODED merged-config path ────────────────
# The host /tmp is bound in and shared between users, so whoever ran aiter
# first owns it and everyone else dies at import (PermissionError ... .lock).
# Bind a per-user, per-job directory over it (K3 repo, 19 Oct 2026).
aiter_config_base="$STATE_DIR/aiter-configs"
AITER_CONFIG_DIR="${AITER_CONFIG_DIR:-$aiter_config_base/job-${SLURM_JOB_ID:-$$}}"
AITER_CONFIG_ARGS=()
aiter_config_setup() {
    if [[ -d "$aiter_config_base" ]]; then
        find "$aiter_config_base" -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    fi
    [[ -e /tmp/aiter_configs ]] || mkdir -m 1777 /tmp/aiter_configs 2>/dev/null || true
    if mkdir -p "$AITER_CONFIG_DIR" 2>/dev/null && [[ -d /tmp/aiter_configs ]] \
       && apptainer exec --bind "$AITER_CONFIG_DIR":/tmp/aiter_configs "$SIF_PATH" \
            sh -c 'touch /tmp/aiter_configs/.w && rm -f /tmp/aiter_configs/.w' >/dev/null 2>&1; then
        AITER_CONFIG_ARGS=(--bind "$AITER_CONFIG_DIR":/tmp/aiter_configs)
    else
        warn "Could not bind $AITER_CONFIG_DIR over /tmp/aiter_configs. If startup dies with
  'Permission denied: /tmp/aiter_configs/*.lock', another user owns it on this node."
    fi
}
aiter_config_setup
PROBE_ENV+=(${AITER_CONFIG_ARGS[@]+"${AITER_CONFIG_ARGS[@]}"})

# ── GPU passthrough ─────────────────────────────────────────────────────────
# '--rocm' binds the node's ROCm libraries ahead of the image's; that broke K3
# when node and image ROCm diverged. 'devices' binds /dev/kfd and /dev/dri only.
# Probe both; cache the verdict per node + image.

ROCM_MODES=(devices rocm)
sif_stamp() { stat -c '%s:%Y' "$1" 2>/dev/null || echo '?'; }
ROCM_MODE_CACHE="$STATE_DIR/.rocm-mode"
rocm_cache_key() { printf '%s|%s' "$(hostname -s 2>/dev/null || echo node)" "$(sif_stamp "$SIF_PATH")"; }
set_rocm_mode_args() {
    case "$1" in
        rocm)    rocm_args=(--rocm) ;;
        devices) rocm_args=(--bind /dev/kfd:/dev/kfd --bind /dev/dri:/dev/dri) ;;
    esac
}
ROCM_VER_CMD='cat /opt/rocm/.info/version 2>/dev/null || ls -d /opt/rocm-* 2>/dev/null | sed -n "1s|.*/opt/rocm-||p"'
# Day-0 images ship ROCm as pip wheels (_rocm_sdk_*), not under /opt/rocm.
sif_rocm_ver() {
    local v
    v="$(apptainer exec "$SIF_PATH" sh -c "$ROCM_VER_CMD" 2>/dev/null | head -n1)"
    [[ -n "$v" ]] || v="$(apptainer exec "$SIF_PATH" python3 -c 'import torch; print("torch", torch.__version__, "HIP", torch.version.hip)' 2>/dev/null | tail -n1)"
    printf '%s' "$v"
}
host_rocm_ver() {
    local v
    v="$(sh -c "$ROCM_VER_CMD" 2>/dev/null | head -n1 || true)"
    [[ -z "$v" ]] && command -v amd-smi >/dev/null 2>&1 \
        && v="$(amd-smi version 2>/dev/null | grep -o 'ROCm version: [0-9.]*' | awk '{print $3}' | head -n1 || true)"
    printf '%s' "$v"
}

make_rocminfo_shim() {   # -> shim_args (host rocminfo output bound over a broken in-image one)
    shim_args=()
    [[ "$ROCMINFO_SHIM" == off ]] && return 0
    local target snap shim
    target="$(apptainer exec "$SIF_PATH" sh -c 'ls -d /opt/rocm*/bin/rocminfo 2>/dev/null | head -n1' 2>/dev/null || true)"
    [[ -n "$target" ]] || return 0
    if [[ "$ROCMINFO_SHIM" != force ]] && apptainer exec --rocm "$SIF_PATH" "$target" >/dev/null 2>&1; then return 0; fi
    command -v rocminfo >/dev/null 2>&1 || { warn "The image's rocminfo fails and the host has none to copy."; return 0; }
    snap="$STATE_DIR/rocminfo-host.txt"
    if ! rocminfo > "$snap" 2>/dev/null || ! grep -q gfx "$snap"; then rm -f "$snap"; return 0; fi
    shim="$STATE_DIR/rocminfo-shim.sh"
    { printf '#!/bin/sh\ncat <<'\''ML4_ROCMINFO_EOF'\''\n'; cat "$snap"; printf 'ML4_ROCMINFO_EOF\n'; } > "$shim"
    chmod +x "$shim"
    shim_args=(--bind "$shim":"$target")
    log "rocminfo shim: binding the host's rocminfo output over $target"
}

GPU_PROBE_PY='
n, gfx, cus, err = -1, "", -1, ""
try:
    import torch
    n = torch.cuda.device_count()
    if n > 0:
        p = torch.cuda.get_device_properties(0)
        gfx = getattr(p, "gcnArchName", "").split(":")[0]
        cus = p.multi_processor_count
except BaseException as e:
    err = "torch: %s: %s" % (type(e).__name__, e)
if not err:
    try:
        import aiter  # noqa: F401
    except BaseException as e:
        err = "aiter: %s: %s" % (type(e).__name__, e)
print("ML4_DEVICES %d" % n); print("ML4_GFX %s" % (gfx or "-")); print("ML4_CUS %d" % cus)
print("ML4_ERR %s" % (" ".join(err.split()) or "-"))
'
gpu_probe() {   # <mode> [apptainer args] -> PROBE_DEVICES/GFX/CUS/ERR
    local mode="$1"; shift
    local out
    set_rocm_mode_args "$mode"
    out="$(apptainer exec "${rocm_args[@]}" "$@" "$SIF_PATH" python3 -c "$GPU_PROBE_PY" 2>&1 || true)"
    PROBE_DEVICES="$(sed -n 's/^ML4_DEVICES //p' <<<"$out" | tail -n1)"
    PROBE_GFX="$(sed -n 's/^ML4_GFX //p' <<<"$out" | tail -n1)"
    PROBE_CUS="$(sed -n 's/^ML4_CUS //p' <<<"$out" | tail -n1)"
    PROBE_ERR="$(sed -n 's/^ML4_ERR //p' <<<"$out" | tail -n1)"
    if [[ -z "$PROBE_DEVICES" ]]; then
        PROBE_DEVICES=-1; PROBE_GFX=-; PROBE_CUS=-1
        PROBE_ERR="$(tail -n 3 <<<"$out" | tr '\n' ' ')"
    fi
    [[ "$PROBE_DEVICES" =~ ^[0-9]+$ && "$PROBE_DEVICES" -gt 0 && "$PROBE_GFX" == gfx* && "$PROBE_ERR" == - ]]
}

if [[ "$MODE" == gpucheck ]]; then
    log "Image: $SIF_PATH"
    printf '\n  node            : %s (%s, %s CUs/GPU, %s GPU(s): %s)\n' "$(hostname -s)" "$GFX" "$(kfd_cus || echo '?')" "$NGPUS" "${ALLOC_LIST:-?}"
    printf '  host ROCm       : %s\n' "$(host_rocm_ver || true)"
    printf '  container ROCm  : %s\n' "$(sif_rocm_ver || true)"
    printf '  amdgpu driver   : %s\n\n' "$(cat /sys/module/amdgpu/version 2>/dev/null || echo '?')"
    make_rocminfo_shim
    ok=""
    for m in "${ROCM_MODES[@]}"; do
        if gpu_probe "$m" ${shim_args[@]+"${shim_args[@]}"} "${PROBE_ENV[@]}"; then
            printf '  %-8s : OK    devices=%s gfx=%s CUs=%s\n' "$m" "$PROBE_DEVICES" "$PROBE_GFX" "$PROBE_CUS"
            [[ -z "$ok" ]] && ok="$m"
        else
            printf '  %-8s : FAIL  devices=%s gfx=%s\n             %s\n' "$m" "$PROBE_DEVICES" "$PROBE_GFX" "$PROBE_ERR"
        fi
    done
    echo
    if [[ -n "$ok" ]]; then
        printf '%s %s\n' "$(rocm_cache_key)" "$ok" > "$ROCM_MODE_CACHE"
        log "Verdict: ROCM_MODE=$ok works on this node."; exit 0
    fi
    rm -f "$ROCM_MODE_CACHE"; warn "Verdict: the container cannot use the GPUs in any mode."; exit 1
fi

# ── Image introspection ─────────────────────────────────────────────────────

IMAGE_HELP=""
image_help() {
    [[ -n "$IMAGE_HELP" ]] || IMAGE_HELP="$(apptainer exec "${PROBE_ENV[@]}" "$SIF_PATH" \
        bash -c 'vllm serve --help=all 2>/dev/null || vllm serve --help 2>&1' 2>/dev/null || true)"
    printf '%s' "$IMAGE_HELP"
}
flags_missing() {   # <argv...> -> every --flag absent from the image's help
    local help a; help="$(image_help)"
    [[ -n "$help" ]] || { echo "__noprobe__"; return 0; }
    for a in "$@"; do
        [[ "$a" == --* ]] || continue
        grep -qE -- "${a%%=*}([ =,]|$)" <<<"$help" || echo "$a"
    done
}

if [[ "$MODE" == check ]]; then
    log "Image: $SIF_PATH ($ENGINE_IMAGE)"
    rc=0
    apptainer exec "${PROBE_ENV[@]}" "$SIF_PATH" python3 - <<'PY' || rc=$?
import sys
try:
    import vllm
    print(f"  vllm {vllm.__version__}")
except Exception as e:
    print(f"  !! vllm import failed: {e}"); sys.exit(3)
try:
    import torch; print(f"  torch {torch.__version__} (HIP {getattr(torch.version, 'hip', None)})")
except Exception as e:
    print(f"  torch: {e}")
for mod in ("aiter", "flydsl", "mistral_common"):
    try:
        m = __import__(mod); print(f"  {mod} {getattr(m, '__version__', 'present')}")
    except Exception as e:
        print(f"  {mod}: MISSING ({type(e).__name__})")
from vllm.model_executor.models.registry import ModelRegistry
archs = set(ModelRegistry.get_supported_archs())
want = ["MistralLarge4ForCausalLM", "PixtralForConditionalGeneration"]
miss = [a for a in want if a not in archs]
for a in want:
    print(f"  {'OK  ' if a in archs else 'MISS'} registry: {a}")
import importlib
for modname in ("vllm.models.mistral_large4.amd.specialized_decode",
                "vllm.models.mistral.mistral_large4.amd.specialized_decode"):
    try:
        importlib.import_module(modname)
        print(f"  OK   AMD specialized decode (TP4 monokernel): {modname}")
        break
    except Exception as e:  # noqa: BLE001
        last = e
else:
    print(f"  MISS AMD specialized decode module ({type(last).__name__}: {last})")
import torch
arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?") if torch.cuda.is_available() else "no GPU visible"
print(f"  device: {arch}")
sys.exit(1 if miss else 0)
PY
    h="$(image_help)"
    for f in --monokernel --speculative-config --safetensors-load-strategy --model-loader-extra-config --additional-config; do
        if grep -q -- "$f" <<<"$h"; then log "  flag OK   $f"; else warn "  flag MISS $f"; fi
    done
    log "  container ROCm: $(sif_rocm_ver || echo '?')   node ROCm: $(host_rocm_ver || echo '?')"
    case "$rc" in 0) log "check passed." ;; *) warn "check FAILED (rc=$rc)." ;; esac
    exit "$rc"
fi

recipe_preflight() {
    local missing
    API_KEY_ARG="${API_KEY_ARG:-x}"
    image_help >/dev/null
    resolve_mono_args
    build_cmd 0
    # shellcheck disable=SC2207
    missing=($(flags_missing "${cmd[@]}"))
    if [[ "${missing[*]:-}" == "__noprobe__" ]]; then
        warn "Could not read 'vllm serve --help' from the image; launching with flags unverified."
    elif (( ${#missing[@]} )); then
        die "This image does not accept: ${missing[*]}
  (from the recipe or EXTRA_ENGINE_ARGS). vLLM would reject them only after the container is up."
    fi
}

if [[ "$MODE" == flagcheck ]]; then
    log "Recipe: $(recipe_summary)"
    recipe_preflight
    log "Specialized decode switch: ${MONO_RESOLVED[*]:-(off)}"
    log "flagcheck passed: the image accepts every flag of this recipe."
    exit 0
fi

if [[ "$MODE" == loadstat ]]; then
    log "Time-to-ready history ($LOADTIMES_FILE):"
    [[ -r "$LOADTIMES_FILE" ]] && tail -15 "$LOADTIMES_FILE" | sed 's/^/    /' || echo "    (nothing yet)"
    for f in "$LOG_DIR"/ml4-"$NODE_TAG"-r*.log; do
        [[ -r "$f" ]] || continue
        log "$(basename "$f"):"
        grep -E 'Loading weights took|Model loading took|Memory profiling takes|KV cache|Maximum concurrency|Graph capturing finished|init engine' "$f" \
            | cut -c1-200 | head -12 | sed 's/^/    /' || true
    done
    exit 0
fi

# ── Serve ───────────────────────────────────────────────────────────────────

[[ "$MODE" == serve ]] || die "internal: unhandled mode $MODE"
[[ -e /dev/kfd && -e /dev/dri ]] || die "/dev/kfd or /dev/dri missing — is this a ROCm GPU node?"
server_running && die "A server is already recorded running: $(pids_alive | tr '\n' ' '). Use 'status' or 'stop' first."
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
port_busy "$PORT" && die "Port $PORT is already in use on this node."
if [[ "$GATEWAY" == litellm ]]; then
    for (( i = 0; i < REPLICAS; i++ )); do port_busy "$(replica_port "$i")" && die "Backend port $(replica_port "$i") is in use (BACKEND_PORT_BASE)."; done
fi
(( GPUS_KNOWN )) || warn "Could not read the allocation's GPU list; assuming 8 GPUs (0-7)."

# Weights present and complete (offline check: index vs files on disk).
verify_weights "$MODEL_PATH" "$MODEL_REPO" "$MODEL_REVISION" --offline --quiet \
    || die "$WEIGHTS weights at $MODEL_PATH are missing or incomplete.
  ./serve-ml4.sh verify-weights $WEIGHTS     # what is missing
  ./serve-ml4.sh download $WEIGHTS           # fetch (~${WEIGHTS_GB} GB, resumable)"
log "Weights: $WEIGHTS — $MODEL_PATH"
host_rocm="$(host_rocm_ver || true)"; sif_rocm="$(sif_rocm_ver || true)"
log "GPUs: $NGPUS x $GFX ($ARCH_SRC); ROCm node ${host_rocm:-?} / container ${sif_rocm:-?}"
(( IDLE_GPUS > 0 )) && warn "$IDLE_GPUS of $NGPUS allocated GPUs will sit idle with this recipe (TP$TP_SIZE x $REPLICAS).
  Allocate --gres=gpu:${GPU_ARCH}:$NEED_GPUS next time, or GATEWAY=litellm to run 2 x TP4 replicas."

# Keys. The serving key always protects vLLM; in litellm mode clients use the
# LiteLLM master key and vLLM listens on 127.0.0.1 only.
gen_key() { openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
if [[ -z "${ML4_API_KEY:-}" ]]; then
    if [[ -s "$API_KEY_FILE" ]]; then ML4_API_KEY="$(<"$API_KEY_FILE")"
    else ML4_API_KEY="$(gen_key)"; (umask 077 && printf '%s' "$ML4_API_KEY" > "$API_KEY_FILE"); log "Generated serving key -> $API_KEY_FILE"; fi
fi
API_KEY_ARG="$ML4_API_KEY"
if [[ "$GATEWAY" == litellm && -z "${LITELLM_MASTER_KEY:-}" ]]; then
    if [[ -s "$LITELLM_KEY_FILE" ]]; then LITELLM_MASTER_KEY="$(<"$LITELLM_KEY_FILE")"
    else LITELLM_MASTER_KEY="sk-$(gen_key)"; (umask 077 && printf '%s' "$LITELLM_MASTER_KEY" > "$LITELLM_KEY_FILE"); log "Generated LiteLLM master key -> $LITELLM_KEY_FILE"; fi
fi
[[ "$GATEWAY" == litellm ]] && { [[ -f "$LITELLM_SIF" ]] || pull_one "$LITELLM_SIF" "$LITELLM_IMAGE" LiteLLM || die "LiteLLM image pull failed."; }

# aiter JIT dir: a writable copy of the image's aiter/jit, bound back at the
# same path. FlyDSL / aiter kernels JIT-compile there during graph capture,
# after the weight load; a read-only .sif kills startup at that point.
AITER_JIT_DIR="${AITER_JIT_DIR:-$CACHE_ROOT/aiter-jit}"
AITER_JIT_TARGET="${AITER_JIT_TARGET:-$(apptainer exec "$SIF_PATH" python3 -c '
import importlib.util, os
s = importlib.util.find_spec("aiter")
o = getattr(s, "origin", None) if s else None
print(os.path.join(os.path.dirname(o), "jit") if o else "")' 2>/dev/null | tail -n1)}"
cache_bind=()
if [[ "$AITER_JIT_TARGET" == /* ]]; then
    if [[ ! -f "$AITER_JIT_DIR/.seeded" ]]; then
        mkdir -p "$AITER_JIT_DIR"
        log "Seeding writable aiter JIT dir from the image (one-off copy) ..."
        apptainer exec --bind "$STATE_DIR":"$STATE_DIR" "$SIF_PATH" cp -a "$AITER_JIT_TARGET/." "$AITER_JIT_DIR/" \
            || die "Failed to copy $AITER_JIT_TARGET into $AITER_JIT_DIR"
        printf '%s\n' "$ENGINE_IMAGE" > "$AITER_JIT_DIR/.seeded"
    fi
    cache_bind=(--bind "$AITER_JIT_DIR":"$AITER_JIT_TARGET")
else
    warn "Could not locate aiter's jit/ in the image; JIT kernels may fail on a read-only image. Set AITER_JIT_TARGET."
fi

# Recipe vs image, before reading a byte of weights.
log "Recipe: $(recipe_summary)"
recipe_preflight
[[ "$SPECULATIVE" == none ]] && warn "SPECULATIVE=none: the DSpark head is a ~1.5-1.9x decode win on Mistral's measurements."
[[ "$GPU_ARCH" == mi300x ]] && warn "MI300X (gfx942) is not in AMD's qualification for this branch (MI350 series). FP8 TP8 is expected to work; see docs/compatibility.md for what was verified."
make_rocminfo_shim

# Page-cache prewarm: GPFS serves many parallel readers far faster than vLLM's
# per-rank sequential reads, and every TP rank (and every replica) then reads
# from RAM. Skipped when the shards are already resident (fincore).
prewarm() {
    local shards=() resident_pct t0 t1 gb
    shopt -s nullglob; shards=("$MODEL_PATH"/consolidated-*.safetensors); shopt -u nullglob
    (( ${#shards[@]} )) || return 0
    if command -v fincore >/dev/null 2>&1; then
        resident_pct="$(fincore -b -n -o RES,SIZE "${shards[@]}" 2>/dev/null | awk '{r+=$1; s+=$2} END {if (s) printf "%d", 100*r/s; else print 0}')"
        if (( ${resident_pct:-0} >= 95 )); then log "Prewarm: ${resident_pct}% of the shards already in page cache — skipping."; return 0; fi
        log "Prewarm: ${resident_pct:-0}% resident; reading ${#shards[@]} shards with $PREWARM_THREADS readers ..."
    else
        log "Prewarm: reading ${#shards[@]} shards with $PREWARM_THREADS readers ..."
    fi
    t0=$(date +%s)
    printf '%s\0' "${shards[@]}" | xargs -0 -P "$PREWARM_THREADS" -I{} dd if={} of=/dev/null bs=16M status=none
    t1=$(date +%s); gb=$WEIGHTS_GB
    log "Prewarm: ~${gb} GB in $(( t1 - t0 ))s ($(awk -v g="$gb" -v s=$(( t1 - t0 )) 'BEGIN{printf "%.1f", s>0?g/s:0}') GB/s)."
}
case "$PREWARM" in
    1) prewarm ;;
    auto) mem_gb=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
          if (( mem_gb > WEIGHTS_GB * 3 / 2 )); then prewarm; else log "Prewarm skipped: ${mem_gb} GB RAM is too small to cache ${WEIGHTS_GB} GB."; fi ;;
esac

# GPU passthrough mode (probed once per node + image).
base_bind=(--bind "$STATE_DIR":"$STATE_DIR" --bind "$MODEL_PATH":"$MODEL_PATH"
    ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"}
    ${cache_bind[@]+"${cache_bind[@]}"} ${AITER_CONFIG_ARGS[@]+"${AITER_CONFIG_ARGS[@]}"}
    ${shim_args[@]+"${shim_args[@]}"})
chosen_mode=""; cache_key="$(rocm_cache_key)"
if [[ "$ROCM_MODE" != auto ]]; then
    gpu_probe "$ROCM_MODE" "${base_bind[@]}" "${PROBE_ENV[@]}" && chosen_mode="$ROCM_MODE" \
        || die "ROCM_MODE=$ROCM_MODE cannot reach the GPUs ($PROBE_ERR). Run gpucheck."
elif [[ "$(cut -d' ' -f1 "$ROCM_MODE_CACHE" 2>/dev/null || true)" == "$cache_key" ]]; then
    chosen_mode="$(cut -d' ' -f2 "$ROCM_MODE_CACHE")"
    log "GPU passthrough: $chosen_mode (remembered for this node + image)"
else
    log "Probing GPU passthrough modes (cached per node + image) ..."
    for m in "${ROCM_MODES[@]}"; do
        if gpu_probe "$m" "${base_bind[@]}" "${PROBE_ENV[@]}"; then
            log "  $m: OK — $PROBE_DEVICES GPU(s), $PROBE_GFX, $PROBE_CUS CUs"; chosen_mode="$m"; break
        fi
        warn "  $m: $PROBE_ERR"
    done
    [[ -n "$chosen_mode" ]] && printf '%s %s\n' "$cache_key" "$chosen_mode" > "$ROCM_MODE_CACHE"
fi
[[ -n "$chosen_mode" ]] || die "No GPU passthrough mode works on this node. Run './serve-ml4.sh gpucheck'."
set_rocm_mode_args "$chosen_mode"

redact() {
    local out=() prev="" a
    for a in "$@"; do
        if [[ "$prev" == --api-key ]]; then out+=("<redacted>")
        elif [[ "$a" == *_KEY=* || "$a" == HF_TOKEN=* ]]; then out+=("${a%%=*}=<redacted>")
        else out+=("$a"); fi
        prev="$a"
    done
    printf '%s' "${out[*]}"
}

: > "$PID_FILE"
cleanup() {
    log "Shutting down ..."
    stop_all >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_ts="$(date +%s)"
for (( i = 0; i < REPLICAS; i++ )); do
    build_cmd "$i"
    gpus="$(replica_gpus "$i")"
    env_args=()
    for e in "${R_ENV[@]}"; do env_args+=(--env "$e"); done
    # One replica = one TP group on its own GPU slice. ROCr enumerates only the
    # slice, so HIP/CUDA indices restart at 0 inside it. Distinct base ports
    # keep two replicas' torch.distributed rendezvous apart.
    env_args+=(--env "ROCR_VISIBLE_DEVICES=$gpus"
               --env "HIP_VISIBLE_DEVICES=$(seq -s, 0 $(( TP_SIZE - 1 )))"
               --env "CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( TP_SIZE - 1 )))"
               # Slurm also exports GPU_DEVICE_ORDINAL=0..7; ROCr aborts when it
               # disagrees with HIP_VISIBLE_DEVICES ("Conflicting visibility").
               --env "GPU_DEVICE_ORDINAL=$(seq -s, 0 $(( TP_SIZE - 1 )))"
               --env "VLLM_PORT=$(( 29500 + 100 * i ))"
               --env "VLLM_CACHE_ROOT=$CACHE_ROOT/vllm"
               --env "TRITON_CACHE_DIR=$CACHE_ROOT/triton"
               --env "XDG_CACHE_HOME=$CACHE_ROOT/xdg"
               --env "HF_HOME=$CACHE_ROOT/hf")
    rlog="$(replica_log "$i")"
    {
        printf '### serve-ml4.sh  %s  host=%s job=%s replica=%s gpus=%s\n' "$(date)" "$(hostname -s)" "${SLURM_JOB_ID:-}" "$i" "$gpus"
        printf '### recipe: %s\n' "$(recipe_summary)"
        printf '### image: %s (%s, passthrough %s)\n' "$ENGINE_IMAGE" "$SIF_PATH" "$chosen_mode"
        [[ "$ENGINE" == branch ]] && printf '### overlay: %s\n' "$(cat "$OVERLAY_DIR/.complete")"
        printf '### env: %s\n' "$(redact "${R_ENV[@]}")"
        printf '### command: %s\n\n' "$(redact "${cmd[@]}")"
    } > "$rlog"
    apptainer exec "${rocm_args[@]}" "${base_bind[@]}" "${env_args[@]}" "$SIF_PATH" "${cmd[@]}" >>"$rlog" 2>&1 &
    echo "vllm-r$i $!" >> "$PID_FILE"
    log "Replica $i: GPUs $gpus -> $(replica_host):$(replica_port "$i")   log: $rlog"
done

wait_healthy() {   # <url> <label> <pid>
    local url="$1" label="$2" pid="$3"
    while true; do
        curl -fsS -m 5 "$url" >/dev/null 2>&1 && return 0
        if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
        (( $(date +%s) - start_ts > READY_TIMEOUT )) && { warn "$label not healthy within ${READY_TIMEOUT}s."; return 1; }
        sleep 10
    done
}

log "Waiting for the engine(s) (timeout ${READY_TIMEOUT}s). Cold start: weights, aiter/FlyDSL JIT, graph capture."
for (( i = 0; i < REPLICAS; i++ )); do
    pid="$(awk -v n="vllm-r$i" '$1==n {print $2}' "$PID_FILE")"
    if ! wait_healthy "http://127.0.0.1:$(replica_port "$i")/health" "replica $i" "$pid"; then
        echo; tail -n 60 "$(replica_log "$i")" || true
        die "Replica $i exited or stalled during startup (log above: $(replica_log "$i")).
     'ValueError ... specialized decode'  ->  MONOKERNEL=0, or see docs/troubleshooting.md
     HIP OOM / KV too small                ->  lower CONTEXT_LEN or MEM_UTIL
     unknown argument                      ->  ./serve-ml4.sh flagcheck"
    fi
    log "Replica $i healthy after $(( $(date +%s) - start_ts ))s."
done
ready_secs=$(( $(date +%s) - start_ts ))

[[ "$GATEWAY" == litellm ]] && start_litellm

printf '%s\t%s\t%s\t%s\tready=%ss\n' "$(date -Is)" "$(hostname -s)" "${SLURM_JOB_ID:-}" "$(recipe_summary)" "$ready_secs" \
    >> "$LOADTIMES_FILE" 2>/dev/null || true

# Snippet for an external gateway (this node's address, the serving key).
if [[ "$GATEWAY" == direct ]]; then
    ML4_SNIPPET_HOST="$(hostname -s)" "$0" litellm-snippet > "$STATE_DIR/litellm-external-snippet.yaml" 2>/dev/null || true
fi

NODE_HOST="$(hostname -s 2>/dev/null || hostname)"
JOBID="${SLURM_JOB_ID:-<jobid>}"
if [[ "$GATEWAY" == litellm ]]; then key_file="$LITELLM_KEY_FILE"; key_name="LiteLLM master key"
else key_file="$API_KEY_FILE"; key_name="serving key"; fi
cat <<EOF

============================================================================
  Mistral Large 4 is up on $NODE_HOST ($GPU_ARCH) after ${ready_secs}s.

  Recipe:      $(recipe_summary)
  Endpoint:    http://$NODE_HOST:$PORT/v1   ($( [[ "$GATEWAY" == litellm ]] && echo "LiteLLM -> vLLM" || echo "vLLM, OpenAI-compatible"))
  Model name:  $SERVED_MODEL_NAME
  API key:     $key_name in $key_file
               export ML4_API_KEY="\$(cat $key_file)"

  Thinking:    "reasoning_effort": "low" | "medium" | "high" (default high).
               Sampling: temperature 1.0, top_p 0.95 (generation_config.json).
  Verify:      ./serve-ml4.sh toolcheck      ./serve-ml4.sh vibecheck
  Laptop:      ssh -N -L $PORT:$NODE_HOST:$PORT \${USER}@bunya1.rcc.uq.edu.au
               ./vibe-setup.sh --via $GATEWAY --host localhost --port $PORT --api-key "\$ML4_API_KEY"
$( [[ "$GATEWAY" == direct ]] && echo "  External LiteLLM gateway snippet: $STATE_DIR/litellm-external-snippet.yaml" )
  Second shell: srun --overlap --jobid $JOBID --pty /bin/bash -l
  Stop:        Ctrl-C, 'scancel $JOBID', or './serve-ml4.sh stop'
============================================================================

EOF

if (( DETACH )); then
    trap - EXIT INT TERM
    while read -r _ p; do disown "$p" 2>/dev/null || true; done < "$PID_FILE"
    log "Detached. Processes keep running until './serve-ml4.sh stop' or the job ends."
    exit 0
fi

log "Attached. Ctrl-C (or scancel) stops everything. Following replica 0's log:"
tail -n 0 -f "$(replica_log 0)" &
TAIL_PID=$!
# Exit (and clean up) as soon as any process dies.
while true; do
    while read -r name p; do
        kill -0 "$p" 2>/dev/null || { kill "$TAIL_PID" 2>/dev/null; die "$name (pid $p) exited — see $LOG_DIR."; }
    done < "$PID_FILE"
    sleep 15
done

exit 0
}
