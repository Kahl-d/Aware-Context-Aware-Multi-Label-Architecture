#!/bin/bash
#SBATCH --job-name=base-std
#SBATCH --partition=gpucluster
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -eo pipefail

echo "============================================================"
echo " STANDARD BASELINE — DeBERTa-v3-base (no AWARE)"
echo " Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "============================================================"

source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_env

cd /Users/923673423/f2-models/Comparison/1_base_standard
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

CONFIG="configs/base_standard.yaml"
DATA="data/"
OUTPUT="results/final"
mkdir -p "$OUTPUT" logs

python scripts/train.py --config "$CONFIG" --data_dir "$DATA" --output_dir "$OUTPUT"

for SPLIT in val test train; do
  echo ">>> Evaluating $SPLIT..."
  python scripts/evaluate.py \
    --config "$CONFIG" \
    --data_dir "$DATA" \
    --results_dir "$OUTPUT" \
    --split $SPLIT
done

echo "============================================================"
echo " DONE — $(date)"
echo "============================================================"
