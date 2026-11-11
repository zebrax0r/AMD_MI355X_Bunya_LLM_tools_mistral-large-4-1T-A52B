#!/usr/bin/env bash
#
# bench-ml4.sh — measure a running Mistral Large 4 endpoint (tok/s, TTFT, TPOT)
# with vLLM's own bench client, run from the same .sif against PORT (vLLM
# directly, or LiteLLM in front of it).
#
#   ./bench-ml4.sh                 sweep BENCH_CONCURRENCY (default "1 2 4 8")
#   ./bench-ml4.sh latency         one stream
#   ./bench-ml4.sh throughput      BENCH_MAX_CONCURRENCY streams (default 16)
#   ./bench-ml4.sh longcontext     100k in / 512 out, one stream, n=4
#
# Tunables: BENCH_INPUT_LEN (1024) BENCH_OUTPUT_LEN (512) BENCH_NUM_PROMPTS (64)
#           BENCH_CONCURRENCY BENCH_MAX_CONCURRENCY BENCH_EFFORT (low)
#           BENCH_EXTRA_ARGS (appended verbatim)
#
# Random prompts flatter speculative decoding: DSpark accepts far more of a
# random continuation than of real agent text. Use ./agentstress.py for the
# agentic shape and compare the "Mean acceptance length" in the server log.
# Results (JSON) go to $STATE_DIR/bench/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-sweep}"
case "$MODE" in sweep|latency|throughput|longcontext) ;; *) echo "use: sweep | latency | throughput | longcontext" >&2; exit 1 ;; esac

ENV_FILE="${ML4_ENV:-$SCRIPT_DIR/ml4.env}"
# shellcheck disable=SC1090
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
STATE_DIR="${STATE_DIR:-/scratch/user/$USER/models/mistral-large-4}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistral-large-4}"
IN="${BENCH_INPUT_LEN:-1024}"; OUT="${BENCH_OUTPUT_LEN:-512}"; N="${BENCH_NUM_PROMPTS:-64}"
CONC="${BENCH_CONCURRENCY:-1 2 4 8}"; MAXC="${BENCH_MAX_CONCURRENCY:-16}"
EFFORT="${BENCH_EFFORT:-low}"

# Which image and tokenizer: whatever the running replica 0 logged.
log0="$STATE_DIR/logs/ml4-$(hostname -s)-r0.log"
[[ -r "$log0" ]] || { echo "No $log0 — is a server running? (./serve-ml4.sh status)" >&2; exit 1; }
SIF="$(sed -n 's/^### image: .* (\(\/[^,]*\.sif\), passthrough.*/\1/p' "$log0" | head -n1)"
if [[ ! -f "$SIF" ]]; then   # renamed since the server started: find it by digest
    dg="$(sed -n 's/^### image: [^ ]*@sha256:\([0-9a-f]\{12\}\).*/\1/p' "$log0" | head -n1)"
    SIF="$(ls -t "$STATE_DIR"/vllm-*-"$dg"*.sif 2>/dev/null | head -n1)"
fi
[[ -f "$SIF" ]] || { echo "Cannot find the server's .sif (see $log0)." >&2; exit 1; }
MODEL_PATH="$(sed -n 's/^### command: vllm serve \([^ ]*\) .*/\1/p' "$log0" | head -n1)"
OVERLAY_BIND=()
if grep -q '^### overlay:' "$log0"; then
    ov="$(ls -d "$STATE_DIR"/overlay-*/vllm 2>/dev/null | head -n1)"
    site="$(apptainer exec "$SIF" python3 -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("vllm").origin))' 2>/dev/null | tail -n1)"
    [[ -n "$ov" && -n "$site" ]] && OVERLAY_BIND=(--bind "$ov":"$site")
fi
if pgrep -u "$(id -u)" -f 'litellm --config' >/dev/null; then KEY="$(cat "$STATE_DIR/ml4-litellm-key")"; else KEY="$(cat "$STATE_DIR/ml4-api-key")"; fi
mkdir -p "$STATE_DIR/bench"

run() {   # <concurrency> <input> <output> <n>
    local c="$1" i="$2" o="$3" n="$4" tag
    tag="$(date +%Y%m%d-%H%M%S)-c${c}-i${i}-o${o}"
    echo "=== concurrency $c, $i in / $o out, n=$n, effort $EFFORT"
    # shellcheck disable=SC2086
    apptainer exec --bind "$STATE_DIR":"$STATE_DIR" --bind "$MODEL_PATH":"$MODEL_PATH" \
        ${OVERLAY_BIND[@]+"${OVERLAY_BIND[@]}"} --env OPENAI_API_KEY="$KEY" --env PYTHONNOUSERSITE=1 \
        --env HF_HUB_OFFLINE=1 "$SIF" \
        vllm bench serve --backend openai-chat --base-url "http://127.0.0.1:$PORT" \
        --endpoint /v1/chat/completions --model "$SERVED_MODEL_NAME" \
        --tokenizer "$MODEL_PATH" --tokenizer-mode mistral \
        --dataset-name random --random-input-len "$i" --random-output-len "$o" \
        --num-prompts "$n" --max-concurrency "$c" --ignore-eos \
        --extra-body "{\"reasoning_effort\": \"$EFFORT\", \"temperature\": 1.0, \"top_p\": 0.95}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$STATE_DIR/bench" --result-filename "$tag.json" \
        ${BENCH_EXTRA_ARGS:-} 2>&1 | grep -vE '^\[aiter\]|^INFO|WARNING'
}

case "$MODE" in
    sweep)       for c in $CONC; do run "$c" "$IN" "$OUT" "$N"; done ;;
    latency)     run 1 "$IN" "$OUT" 16 ;;
    throughput)  run "$MAXC" "$IN" "$OUT" $(( MAXC * 6 )) ;;
    longcontext) run 1 100000 512 4 ;;
esac
echo "Server-side DSpark acceptance (random prompts overstate it):"
grep -oE 'Mean acceptance length: [0-9.]+' "$log0" | tail -n 3 || true
