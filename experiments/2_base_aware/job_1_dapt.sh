#!/bin/bash
#SBATCH --job-name=base-dapt
#SBATCH --partition=gpucluster
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dapt_%j.out
#SBATCH --error=logs/dapt_%j.err

echo "============================================================"
echo " DAPT — DeBERTa-v3-base MLM pre-training"
echo " Job: $SLURM_JOB_ID | $(date)"
echo "============================================================"

source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_env
cd /Users/923673423/f2-models/Comparison/2_base_aware

python scripts/dapt.py \
  --config configs/base_aware.yaml \
  --corpus data/dapt_corpus.txt \
  --output_dir results/dapt_base

echo "DONE — $(date)"
