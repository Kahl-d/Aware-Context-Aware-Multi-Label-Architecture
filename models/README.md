# Models

The AWARE architecture, its training configuration, and the [model card](MODEL_CARD.md).

## Architecture

```
Essay (all sentences, joined)
  → DeBERTa-v3 encoder (domain-adaptively pre-trained on the essay corpus)
  → token embeddings
  → sentence mean pooling (over token-level sentence boundaries)
  → + sentence position embedding
  → BiLSTM context encoder (2-layer, bidirectional)
  → context-aware sentence embeddings
  → classification head (Dropout → LayerNorm → Linear)
  → per-sentence logits
  → sigmoid + per-theme thresholds → multi-label predictions
```

The essential design choice: **the unit of prediction is the sentence, but the unit of input is
the essay.** A sentence like "so I kept going" means nothing alone; the BiLSTM lets each
sentence be read against the narrative it sits in.

An auxiliary essay-level head provides multi-task supervision during training.

## Training

Three-phase progressive schedule:

| Phase | Epochs | What trains |
|---|---|---|
| 1 | 8 | Frozen encoder; head and BiLSTM only |
| 2 | up to 40 | Progressive unfreezing, full fine-tuning |
| 3 | 5 | Head and BiLSTM retrain |

**Loss:** asymmetric loss (ASL) + class-balanced weights + R-Drop consistency regularization +
essay auxiliary loss.

**After training:** per-theme threshold optimization (worth **+0.084 Macro-F1**) and Platt
calibration.

## Files

| Path | What |
|---|---|
| `aware/` | Architecture and training code — model, dataset, losses, metrics, trainer, DAPT, evaluation |
| `configs/` | Training configuration for the reported model |
| `MODEL_CARD.md` | Intended use, limitations, ethical constraints |

## Weights

**Trained weights are not distributed in this repository.** Contact the author or the ALMA
Project team. Any released checkpoint carries the same use restrictions as the model card.
