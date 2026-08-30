#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=gpucluster
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

echo "============================================================"
echo " BASELINES — Majority, Random, TF-IDF models"
echo " Job: $SLURM_JOB_ID | $(date)"
echo "============================================================"

source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_env
cd /Users/923673423/f2-models/Comparison/5_baselines

python scripts/compute_baselines.py

echo "============================================================"
echo " DONE — $(date)"
echo "============================================================"
