#!/bin/bash --login
#SBATCH --job-name=ml4
#SBATCH --partition=gpu_rocm
#SBATCH --account=a_rcc
#SBATCH --qos=sdf
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1500G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ml4-%j.out
#SBATCH --error=logs/ml4-%j.out
#
# Batch wrapper around serve-ml4.sh. Prefer ./submit-ml4.sh, which sets the GPU
# type/count and exports the run shape for you:
#
#   ./submit-ml4.sh mi355x                       # 8 x MI355X, MXFP4
#   ./submit-ml4.sh mi355x --gpus 4              # 4 x MI355X, MXFP4 TP4 + monokernel
#   ./submit-ml4.sh mi355x --gateway litellm     # 2 x TP4 replicas behind LiteLLM
#   ./submit-ml4.sh mi300x                       # 8 x MI300X, FP8 TP8
#
# Direct use: submit from the repository root; override the GPU type on the
# command line, e.g.  sbatch --gres=gpu:mi300x:8 serve-ml4.sbatch
# Account/QoS: the K3 repos' values. Override with SBATCH_ACCOUNT / SBATCH_QOS
# in your environment, or on the sbatch command line.
#
# Once RUNNING:  grep -A 25 'is up on' logs/ml4-<jobid>.out

set -euo pipefail
# Slurm runs a COPY of this script from its spool dir; SLURM_SUBMIT_DIR is the
# directory sbatch was run from (the repository root).
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[[ -x ./serve-ml4.sh ]] || { echo "Submit from the repository root (serve-ml4.sh not found in $PWD)" >&2; exit 1; }
echo "[ml4-sbatch] Job $SLURM_JOB_ID on $(hostname) at $(date); GPUs ${ROCR_VISIBLE_DEVICES:-?}"
exec ./serve-ml4.sh serve
