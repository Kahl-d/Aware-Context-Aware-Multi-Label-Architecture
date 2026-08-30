#!/bin/bash
#SBATCH --job-name=aware-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=07:55:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kkhan@sfsu.edu

# =============================================================================
# AWARE v2 — Complete Training Pipeline
# SUBMIT from the cModels/ directory:
#
#   mkdir -p logs results/full
#   sbatch run_train.slurm
#
# Pipeline (all automatic, single job):
#   1. DAPT    — domain-adapt DeBERTa-v3-base on ALMA essays (MLM)
#   2. Train   — two-phase fine-tune AWARE model with DAPT encoder
#   3. Eval val  — optimized-threshold evaluation on validation set
#   4. Eval test — MAIN THESIS RESULTS on held-out test set
#   5. Eval train — overfitting diagnostic on training set
# =============================================================================

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source ../../../miniconda/bin/activate llm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"

RESULTS_DIR="results/full"
LOG_FILE="$RESULTS_DIR/training_run.log"

mkdir -p "$RESULTS_DIR" logs

# ── Header ────────────────────────────────────────────────────────────────────
{
echo "============================================================"
echo " AWARE v2 — Complete Training Pipeline"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $(hostname)"
echo " Started : $(date)"
echo "------------------------------------------------------------"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "============================================================"
} | tee "$LOG_FILE"

# ── Run full pipeline ─────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> [$(date +%H:%M:%S)] Starting pipeline..." | tee -a "$LOG_FILE"

python3 scripts/run_pipeline.py \
    --mode full \
    --config configs/full.yaml \
    --data_dir data/ \
    --output_dir "$RESULTS_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# ── Footer ────────────────────────────────────────────────────────────────────
{
echo ""
echo "============================================================"
echo " AWARE v2 — Pipeline Finished"
echo " Ended   : $(date)"
echo " Job ID  : $SLURM_JOB_ID"
echo ""
echo " Results in: $RESULTS_DIR/"
echo "   dapt/encoder/          — domain-adapted DeBERTa"
echo "   best.pt                — best AWARE v2 checkpoint"
echo "   history.json           — per-epoch train/val metrics"
echo "   thresholds.json        — per-theme optimized thresholds"
echo "   evaluation_val.json    — val set results"
echo "   evaluation_test.json   — TEST SET RESULTS (MAIN)"
echo "   evaluation_train.json  — train set (overfitting check)"
echo "   pipeline.log           — detailed pipeline log"
echo "   training_run.log       — this combined SLURM log"
echo "============================================================"
} | tee -a "$LOG_FILE"
