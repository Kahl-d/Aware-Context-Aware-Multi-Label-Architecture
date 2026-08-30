#!/bin/bash
#SBATCH --job-name=base-aware
#SBATCH --partition=gpucluster
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -eo pipefail

echo "============================================================"
echo " AWARE — DeBERTa-v3-base (full AWARE pipeline)"
echo " Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "============================================================"

source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_env

cd /Users/923673423/f2-models/Comparison/2_base_aware
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

DAPT_ENCODER="results/dapt_base/encoder"
if [ ! -d "$DAPT_ENCODER" ]; then
  echo "ERROR: DAPT encoder not found at $DAPT_ENCODER"
  echo "Run job_1_dapt.sh first!"
  exit 1
fi

CONFIG="configs/base_aware.yaml"
OUTPUT="results/final"
mkdir -p "$OUTPUT" logs

echo "Config: $CONFIG"
echo "DAPT encoder: $DAPT_ENCODER"
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
