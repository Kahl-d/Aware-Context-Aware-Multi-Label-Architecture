#!/bin/bash
#SBATCH --job-name=aware-dapt
#SBATCH --output=logs/dapt_%j.out
#SBATCH --error=logs/dapt_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kkhan@sfsu.edu

# =============================================================================
# AWARE v2 — Domain Adaptive Pre-Training (DAPT)
# Submit from cModels/ directory:
#
#   mkdir -p logs results/dapt
#   sbatch run_dapt.slurm
#
# After DAPT completes, update configs/full.yaml:
#   encoder_path: results/dapt/encoder/
# Then submit run_train.slurm.
# =============================================================================

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source ../../../miniconda/bin/activate llm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p logs results/dapt

cd "$SLURM_SUBMIT_DIR"

# ── Header ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " AWARE v2 — DAPT Run"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $(hostname)"
echo " Started : $(date)"
echo "------------------------------------------------------------"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "============================================================"

# ── DAPT ──────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [$(date +%H:%M:%S)] Starting DAPT..."

python3 scripts/dapt.py \
    --corpus data/dapt_corpus.txt \
    --output_dir results/dapt/ \
    --config configs/full.yaml

echo ""
echo ">>> [$(date +%H:%M:%S)] DAPT complete."
echo ""
echo "============================================================"
echo " DAPT Finished — $(date)"
echo " Encoder saved to: results/dapt/encoder/"
echo " Next step: set encoder_path in configs/full.yaml, then sbatch run_train.slurm"
echo "============================================================"
