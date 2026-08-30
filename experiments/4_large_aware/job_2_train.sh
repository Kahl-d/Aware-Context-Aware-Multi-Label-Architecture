#!/bin/bash
#SBATCH --job-name=lg-aware
#SBATCH --partition=gpucluster
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -eo pipefail

echo "============================================================"
echo " AWARE — DeBERTa-v3-large (full AWARE pipeline, v4)"
echo " Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "============================================================"

source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_env

cd /Users/923673423/f2-models/Comparison/4_large_aware
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Use FRESH DAPT encoder (must run job_1_dapt.sh first)
DAPT_ENCODER="results/dapt_large/encoder"
if [ ! -d "$DAPT_ENCODER" ]; then
  echo "ERROR: DAPT encoder not found at $DAPT_ENCODER"
  echo "Run job_1_dapt.sh first!"
  exit 1
fi

CONFIG="configs/large_aware.yaml"
OUTPUT="results/final"
mkdir -p "$OUTPUT" logs

echo "Config: $CONFIG"
echo "DAPT encoder: $DAPT_ENCODER (FRESH)"
echo "Output: $OUTPUT"

python scripts/train.py \
  --config "$CONFIG" \
  --data_dir data \
  --output_dir "$OUTPUT" \
  --encoder_path "$DAPT_ENCODER"

for SPLIT in val test train; do
  echo ">>> Evaluating $SPLIT..."
  python scripts/evaluate.py \
    --config "$CONFIG" \
    --data_dir data \
    --results_dir "$OUTPUT" \
    --split "$SPLIT" \
    --encoder_path "$DAPT_ENCODER"
done

echo "============================================================"
echo " DONE — $(date)"
echo "============================================================"
