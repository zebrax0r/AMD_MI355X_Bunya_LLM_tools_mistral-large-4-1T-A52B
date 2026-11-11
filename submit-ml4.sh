#!/usr/bin/env bash
#
# submit-ml4.sh — submit serve-ml4.sbatch for a chosen node class and run shape.
#
#   ./submit-ml4.sh mi355x [options]     bun159-161, 8 x MI355X (288 GB each)
#   ./submit-ml4.sh mi300x [options]     bun144-145, 8 x MI300X (192 GB each)
#
# Options:
#   --gpus N            4 or 8 (mi355x default 8; mi300x always 8 — FP8 is 1.06 TB)
#   --gateway MODE      direct (default) | litellm
#   --agents N          concurrent agent sessions (picks the profile; default 4)
#   --weights W         mxfp4 | fp8 (default: mxfp4 on mi355x, fp8 on mi300x)
#   --port P            client port (default 8000)
#   --time HH:MM:SS     wall time (default 48:00:00)
#   --dry-run           print the sbatch command only
#   anything after --   passed to sbatch verbatim (e.g. -- --account=a_xyz --qos=normal)
#
# The job prints the endpoint banner into logs/ml4-<jobid>.out once healthy.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARCH="${1:-}"; shift || true
case "$ARCH" in mi355x|mi300x) ;; -h|--help|"") sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "First argument must be mi355x or mi300x (got '$ARCH')." >&2; exit 1 ;; esac

GPUS=8; GATEWAY="${GATEWAY:-direct}"; AGENTS="${AGENTS:-4}"; WEIGHTS="${WEIGHTS:-auto}"
PORT="${PORT:-8000}"; TIME="48:00:00"; DRY=0; EXTRA=()
while (( $# )); do
    case "$1" in
        --gpus) GPUS="$2"; shift ;;
        --gateway) GATEWAY="$2"; shift ;;
        --agents) AGENTS="$2"; shift ;;
        --weights) WEIGHTS="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --time) TIME="$2"; shift ;;
        --dry-run) DRY=1 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "Unknown option $1" >&2; exit 1 ;;
    esac
    shift
done
case "$GPUS" in 4|8) ;; *) echo "--gpus must be 4 or 8" >&2; exit 1 ;; esac
[[ "$ARCH" == mi300x && "$GPUS" != 8 ]] && { echo "MI300X needs all 8 GPUs (FP8 weights are 1.06 TB)." >&2; exit 1; }

# Host RAM and cores in proportion to the GPUs (bun159-161: 384 cores / 2 TB;
# bun144-145: 256 cores / 2 TB).
if [[ "$ARCH" == mi355x ]]; then CPUS=$(( GPUS * 12 )); else CPUS=$(( GPUS * 8 )); fi
MEM="$(( GPUS == 8 ? 1500 : 750 ))G"

mkdir -p logs
cmd=(sbatch --job-name="ml4-$ARCH" --gres="gpu:$ARCH:$GPUS" --cpus-per-task="$CPUS" --mem="$MEM"
     --time="$TIME"
     --export="ALL,GPU_ARCH=$ARCH,GATEWAY=$GATEWAY,AGENTS=$AGENTS,WEIGHTS=$WEIGHTS,PORT=$PORT"
     ${EXTRA[@]+"${EXTRA[@]}"} serve-ml4.sbatch)
echo "${cmd[*]}"
(( DRY )) && exit 0
out="$("${cmd[@]}")"; echo "$out"
jid="$(awk '{print $NF}' <<<"$out")"
cat <<EOF

Queued job $jid. Follow it with:
  squeue -j $jid
  tail -f logs/ml4-$jid.out          # the endpoint banner appears when healthy
EOF
