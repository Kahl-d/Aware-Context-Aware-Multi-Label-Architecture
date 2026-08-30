#!/bin/bash
#SBATCH --job-name=lg-dapt
#SBATCH --partition=gpucluster
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dapt_%j.out
#SBATCH --error=logs/dapt_%j.err

set -eo pipefail

echo "============================================================"
echo " DAPT — DeBERTa-v3-large MLM pre-training (FRESH)"
echo " Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "============================================================"

source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_env

cd /Users/923673423/f2-models/Comparison/4_large_aware
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

mkdir -p results/dapt_large logs

echo "Config: configs/large_aware.yaml"
echo "Corpus: data/dapt_corpus.txt"
echo "Output: results/dapt_large"

python scripts/dapt.py \
  --config configs/large_aware.yaml \
  --corpus data/dapt_corpus.txt \
  --output_dir results/dapt_large

echo "============================================================"
echo " DAPT DONE — $(date)"
echo "============================================================"
