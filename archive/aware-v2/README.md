# AWARE v2 — Automated CCW Theme Detection

Essay-aware sentence-level multi-label classification of Community Cultural Wealth (CCW) themes in student reflective essays, using DeBERTa-v3 + BiLSTM.

## Architecture

```
Essay text (all sentences joined)
  → DeBERTa-v3-base encoder (optionally from DAPT)
  → Token embeddings [batch, seq_len, 768]
  → Sentence Mean Pooling (via token-level boundaries)
  → Sentence embeddings [batch, max_sent, 768]
  → BiLSTM Context Encoder (2-layer, bidirectional)
  → Context-aware sentence embeddings [batch, max_sent, 768]
  → Classification Head (Dropout → LayerNorm → Linear)
  → Per-sentence logits [batch, max_sent, 11]
  → Sigmoid + per-theme thresholds → predictions
```

## 11 CCW Themes

Navigational, Attainment, Perseverance, Aspirational, Social, Filial Piety, Spiritual, Familial, Resistance, Community Consciousness, First Gen

## Data

- 86,054 sentences from 12,144 essays (SFSU STEM courses)
- Split by student (alma_id): 80/10/10 train/val/test, zero student leakage
- 69% class_0, 31% themed (11 themes, multi-label)
- Imbalance ratio: 72x (Navigational 9,867 vs First Gen 137)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Local toy test (no GPU needed)
python3 scripts/train.py --config configs/toy.yaml --data_dir data/ --output_dir results/toy/ --toy

# Full training (GPU)
python3 scripts/train.py --config configs/full.yaml --data_dir data/ --output_dir results/full/

# With DAPT first
python3 scripts/dapt.py --corpus data/dapt_corpus.txt --output_dir results/dapt/ --config configs/full.yaml
# Then update configs/full.yaml: encoder_path: results/dapt/encoder/
python3 scripts/train.py --config configs/full.yaml --data_dir data/ --output_dir results/full/

# Evaluate
python3 scripts/evaluate.py --config configs/full.yaml --data_dir data/ --results_dir results/full/ --split test

# Predict single essay
python3 scripts/predict.py --model_dir results/full/ --config configs/full.yaml \
    --text "I came to this country because my family believed in education."

# Batch predict from CSV/Excel
python3 scripts/predict.py --model_dir results/full/ --config configs/full.yaml \
    --input_file essays.csv --output_file annotated.xlsx
```

## HPC (SLURM)

```bash
# 1. DAPT
sbatch run_dapt.slurm

# 2. Training + evaluation
sbatch run_train.slurm
```

## Key Design Decisions (v2 vs v1)

| Change | Why |
|--------|-----|
| DeBERTa-v3-base (86M) instead of large (304M) | Reduce overfitting |
| ASL only (removed SupCon) | SupCon noisy with small batches |
| Linear warmup + cosine decay | Standard for transformers |
| Student-level splits (alma_id) | Prevent data leakage |
| Dropout 0.3, weight_decay 0.01 | Stronger regularization |
| LayerNorm in classification head | Training stability |
| Auto theme weights (sqrt inverse freq) | Data-driven, reproducible |
| All essays for DAPT corpus | Larger domain coverage |

## File Structure

```
cModels/
├── configs/          # YAML configs (toy, full)
├── data/             # Splits created by prepare_data.py
├── scripts/          # All Python code
│   ├── config.py     # Configuration dataclasses
│   ├── model.py      # AWARE model architecture
│   ├── dataset.py    # Data loading + AEDA augmentation
│   ├── losses.py     # Asymmetric Loss
│   ├── metrics.py    # F1, thresholds, bootstrap CI
│   ├── trainer.py    # Two-phase training loop
│   ├── train.py      # Training entry point
│   ├── evaluate.py   # Evaluation + reports
│   ├── predict.py    # Single + batch inference
│   ├── dapt.py       # Domain Adaptive Pre-Training
│   ├── hpo.py        # Optuna hyperparameter search
│   └── run_pipeline.py # Full orchestration
├── results/          # Training outputs
└── *.slurm           # HPC job scripts
```
