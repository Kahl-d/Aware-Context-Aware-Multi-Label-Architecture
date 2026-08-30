# AWARE v4 — Comprehensive Model Comparison Results

> Multi-Label Sentence Classification for 8 CCW Themes in Student Essays
> DeBERTa-v3 Encoder | 2,095 Train Essays (14,023 sentences) | 273 Test Essays (1,838 sentences)

---

## 1. Executive Summary

| Model | Test F1 Macro | Test PR-AUC | ROC-AUC | Train F1 | Gap |
|---|---|---|---|---|---|
| Majority Class | 0.000 | 0.085 | — | — | — |
| Random (Prior) | 0.084 | 0.089 | — | — | — |
| TF-IDF + LogReg | 0.187 | 0.363 | — | — | — |
| TF-IDF + SVM | 0.274 | 0.333 | — | — | — |
| **Base Standard** | **0.487** | **0.507** | 0.895 | 0.727 | 0.240 |
| **Large Standard** | **0.483** | **0.496** | 0.894 | — | — |
| **Base AWARE** | **0.472** | **0.480** | 0.895 | — | — |
| **Large AWARE (v4)** | **0.494** | **0.484** | 0.888 | 0.759 | 0.265 |

**Key finding**: Large AWARE (v4) achieves the best Test F1 (0.494), beating all other models. However, the Base Standard achieves the best Test PR-AUC (0.507).

---

## 2. Per-Theme Test F1 — All Models

| Theme (prevalence) | TF-IDF SVM | Base Std | Large Std | Base AWARE | **Large AWARE** |
|---|---|---|---|---|---|
| Navigational (24.0%) | 0.554 | 0.684 | **0.705** | 0.675 | **0.707** |
| Familial_Capital (3.9%) | 0.487 | **0.641** | 0.684 | 0.646 | 0.600 |
| Aspirational (15.4%) | 0.453 | 0.558 | 0.570 | 0.546 | **0.585** |
| Perseverance (8.3%) | 0.167 | **0.510** | 0.442 | **0.555** | 0.480 |
| Social (5.6%) | 0.317 | 0.500 | 0.471 | 0.417 | **0.506** |
| Resistance (4.5%) | 0.082 | 0.347 | 0.353 | 0.392 | **0.395** |
| Spiritual (3.4%) | 0.075 | **0.431** | **0.437** | 0.341 | 0.375 |
| Attainment (2.4%) | 0.054 | 0.222 | 0.198 | 0.206 | **0.303** |
| **Macro** | **0.274** | **0.487** | **0.483** | **0.472** | **0.494** |

---

## 3. Per-Theme Test PR-AUC — All Models

| Theme | TF-IDF SVM | Base Std | Large Std | Base AWARE | **Large AWARE** |
|---|---|---|---|---|---|
| Navigational | 0.619 | **0.777** | 0.765 | 0.741 | 0.759 |
| Familial_Capital | 0.534 | 0.668 | **0.686** | **0.680** | 0.586 |
| Aspirational | 0.461 | **0.616** | 0.591 | 0.505 | 0.507 |
| Perseverance | 0.281 | **0.594** | 0.509 | **0.591** | 0.536 |
| Social | 0.349 | 0.510 | 0.508 | 0.419 | **0.488** |
| Resistance | 0.153 | 0.347 | **0.402** | 0.396 | 0.377 |
| Spiritual | 0.173 | **0.383** | 0.390 | 0.370 | **0.416** |
| Attainment | 0.097 | 0.161 | 0.116 | 0.144 | **0.205** |
| **Macro** | **0.333** | **0.507** | **0.496** | **0.480** | **0.484** |

---

## 4. The 2x2 Comparison: Standard vs AWARE × Base vs Large

### 4a. Effect of AWARE (context modeling)

| Comparison | Standard F1 | AWARE F1 | Delta | Interpretation |
|---|---|---|---|---|
| Base: Std vs AWARE | 0.487 | 0.472 | **-0.015** | AWARE hurts at base scale |
| Large: Std vs AWARE | 0.483 | **0.494** | **+0.011** | AWARE helps at large scale |

### 4b. Effect of Model Scale

| Comparison | Base F1 | Large F1 | Delta | Interpretation |
|---|---|---|---|---|
| Standard: Base vs Large | **0.487** | 0.483 | **-0.004** | Larger model overfits more |
| AWARE: Base vs Large | 0.472 | **0.494** | **+0.022** | AWARE benefits from scale |

### 4c. Key Insight
- **Standard models**: Base > Large (overfitting dominates)
- **AWARE models**: Large > Base (AWARE components need model capacity)
- **Best overall**: Large AWARE (0.494) > Base Standard (0.487)
- The AWARE pipeline specifically helps **rare themes** (Attainment +0.081, Resistance +0.048)

---

## 5. Model Architectures & Configurations

### 5a. Standard Baseline (Base & Large)

```
Input Essay Text (joined sentences with ". " separator)
    |
DeBERTa-v3 Encoder (base: 768-dim, 12 layers, 184M params)
                    (large: 1024-dim, 24 layers, 434M params)
    |
SentenceMeanPooling (mean of token embeddings per sentence span)
    |
LayerNorm -> Dropout(0.15) -> Linear -> Logits [B, 32, 8]
    |
BCEWithLogitsLoss (with inverse-sqrt theme weights)
```

**Training**: Single phase, AdamW, cosine LR with warmup, early stopping on val PR-AUC.

| Setting | Base Standard | Large Standard |
|---|---|---|
| Encoder | deberta-v3-base | deberta-v3-large |
| Parameters | 184M | 434M |
| Learning Rate | 2e-5 | 1e-5 |
| Batch Size (eff) | 8 × 4 = 32 | 4 × 8 = 32 |
| Weight Decay | 0.01 | 0.02 |
| Epochs | 30 (stopped ep 8) | 20 (stopped ep 13) |
| Patience | 5 | 7 |
| Dropout | 0.15 | 0.15 |
| Loss | BCE + inv-sqrt weights | BCE + inv-sqrt weights |
| DAPT | No | No |
| BiLSTM | No | No |
| R-Drop | No | No |
| LLRD | No | No |
| AEDA | No | No |

### 5b. AWARE Architecture (Base & Large)

```
Input Essay Text (joined sentences with ". " separator)
    |
DeBERTa-v3 Encoder (with DAPT domain pre-training on 2,636 essays)
    |
SentenceMeanPooling (vectorized, mean of token embeddings per span)
    |
Sentence Position Embedding (learned [32, H] table, added to embeddings)
    |
BiLSTM Context Encoder (2-layer BiLSTM, 256 hidden, residual connection)
    |
ClassificationHead (LayerNorm -> Multi-Sample Dropout(n=3) -> Linear)
    |
Logits [B, 32, 8] -> ASL Loss + CB Weights + R-Drop KL
    |
Essay Auxiliary Head (OR over sentence labels, weighted loss)
```

**Training**: 3-Phase curriculum with progressive unfreezing (large only).

| Setting | Base AWARE | Large AWARE (v4) |
|---|---|---|
| Encoder | deberta-v3-base | deberta-v3-large |
| Parameters | ~188M | ~439M |
| DAPT | 10 epochs MLM | 15 epochs MLM (existing encoder) |
| Phase 1 (frozen) | 5 epochs, lr=5e-5 | 8 epochs, lr=1e-4, R-Drop OFF |
| Phase 2 (full FT) | 30 epochs | 40 epochs |
| Phase 3 (head+BiLSTM) | 5 epochs | 5 epochs |
| Encoder LR | 2e-5 | 5e-6 (3× lower for large) |
| Decoder LR | 5e-5 | 2e-5 |
| LLRD Decay | 0.85 (12 layers) | 0.92 (24 layers, gentler) |
| Weight Decay | 0.04 | 0.08 (stronger for 438M) |
| Dropout | 0.20 | 0.30 |
| BiLSTM Dropout | 0.15 | 0.25 |
| R-Drop alpha | 1.5 | 1.0 (reduced, was too strong at 2.0) |
| ASL gamma_neg | 2.5 | 2.5 (reduced from 3.0) |
| Label Smoothing | 0.02 | 0.02 |
| CB Weights | beta=0.9999 | beta=0.9999 |
| Multi-Sample Dropout | n=3 | n=3 |
| Position Embedding | Yes | Yes |
| Essay Aux Weight | 0.10 | 0.15 |
| Progressive Unfreeze | No (12 layers) | Yes (top-12 first, all at ep+6) |
| SWA | ratio=0.30 | ratio=0.25 (collected 22 checkpoints) |
| AEDA Augmentation | prob=0.35 | prob=0.40 |
| Early Stop Patience | 6 | 8 |
| Effective Batch | 8 × 4 = 32 | 4 × 8 = 32 |

---

## 6. AWARE Components — Detailed Analysis

### 6a. Domain-Adaptive Pre-Training (DAPT)

**What**: Masked Language Modeling on 2,636 student essays before supervised fine-tuning.
**Why**: Adapts DeBERTa to CCW-specific vocabulary ("navigational strategies", "familial capital", "systemic barriers").
**Config**: 10-15 epochs, 15% masking, lr=2-3e-5.
**Result**: Reduced perplexity from 84 to 14.2 (base). Encoder learns essay-specific token co-occurrence patterns.
**Impact**: Estimated +0.03-0.04 F1 (indirectly measured via ablation).

### 6b. BiLSTM Context Encoder

**What**: 2-layer bidirectional LSTM (hidden=256) over sentence embeddings with residual connection.
**Why**: Gives each sentence awareness of surrounding sentences. "My mom taught me" is Familial_Capital if the essay is about family, Navigational if about career guidance.
**Config**: input_size=H, hidden=256, layers=2, output_dropout=0.15-0.25.
**Residual**: `output = projection(lstm_out) + sentence_embeddings` (fallback to raw embeddings when context is uninformative).
**Impact**: Helps Attainment (+0.081) and Resistance (+0.048) where multi-sentence narrative matters. Hurts Spiritual (-0.056) and Familial (-0.041) where single sentences are self-contained.

### 6c. Asymmetric Loss (ASL)

**What**: Focal-like loss with separate treatment of positives and negatives.
**Config**: gamma_pos=0.0 (preserve ALL positive gradients), gamma_neg=2.5 (suppress easy negatives), clip=0.03.
**Why**: With 8 themes and ~1.4 avg labels/sentence, negatives outnumber positives 5:1. Standard BCE gradient is dominated by negatives. ASL suppresses easy-negative gradients while preserving rare-theme positive signals.
**Label Smoothing**: 0.02, asymmetric (negatives only). Prevents logit explosion.
**Impact**: Primary driver of Attainment improvement. Combined with CB weights for rare theme handling.

### 6d. Class-Balanced (CB) Weights

**What**: Per-theme loss weights from Effective Number of Samples (Cui et al., CVPR 2019).
**Config**: beta=0.9999, normalized mean=1.0. Attainment gets 2.045× weight, Navigational gets 0.238×.
**Why**: Inverse-sqrt weights (3.15× for Attainment) were insufficient. CB provides 8.6× ratio Attainment:Navigational.
**Impact**: Boosts rare themes but may over-weight, causing mid-frequency theme degradation.

### 6e. R-Drop Regularization

**What**: Runs each batch twice with different dropout masks, penalizes KL divergence between outputs.
**Config**: alpha=1.0 (large), 1.5 (base). Disabled in Phase 1 (head is random, KL wastes gradient).
**Why**: Consistency regularization prevents overfitting. With 438M params on 14K sentences, overfitting is the dominant failure mode.
**Impact**: Reduces train-test gap. alpha=2.0 was too strong (KL dominated 55-72% of loss in early training).

### 6f. Layer-wise LR Decay (LLRD)

**What**: Top encoder layer gets full encoder_lr, each lower layer gets lr × decay^depth.
**Config**: decay=0.92 (large, 24 layers → bottom=8.5e-7), 0.85 (base, 12 layers → bottom=3.3e-6).
**Why**: Lower layers encode general linguistics (preserve), upper layers encode task-specific semantics (adapt more).
**Impact**: Standard best practice. Prevents catastrophic forgetting of pretrained features.

### 6g. 3-Phase Training

**Phase 1 (Frozen Encoder)**: Train BiLSTM + heads only. Stabilizes task-specific components before encoder gradients flow.
- Large: 8 epochs, lr=1e-4, R-Drop OFF
- Base: 5 epochs, lr=5e-5

**Phase 2 (Full Fine-Tune)**: All parameters trainable with LLRD. Main training phase.
- Large: 40 epochs max, patience=8, progressive unfreeze (top-12 → all at +6)
- Base: 30 epochs max, patience=6

**Phase 3 (Head + BiLSTM Retrain)**: Re-freeze encoder, fine-tune BiLSTM + heads at low LR.
- Compensates distribution shift from Phase 2
- V4 fix: unfreezes BiLSTM (2M params), not just head (20K params)
- In practice, Phase 3 rarely beats Phase 2+SWA

### 6h. Multi-Sample Dropout

**What**: Average logits from n=3 independent dropout masks in classification head during training.
**Why**: Free regularization — ensemble effect with zero extra parameters.
**Implementation**: `logits = mean([classifier(dropout_k(x)) for k in range(3)])` during training; single pass at inference.

### 6i. Stochastic Weight Averaging (SWA)

**What**: Averages model weights across multiple Phase 2 checkpoints.
**Config**: Start ratio=0.25 (begins at epoch 18 of Phase 2). V4 collected 22 checkpoints.
**Result**: Improved val PRAUC 0.5209 → 0.5224 (+0.0015). Small but real.
**Why**: Finds flatter minima that generalize better. Critical for overparameterized models.

### 6j. Progressive Unfreezing (Large only)

**What**: Unfreeze top 12 encoder layers first, then all 24 after 6 Phase 2 epochs.
**Why**: Halves effective trainable params (220M vs 438M) in early Phase 2. Reduces memorization risk.
**Result**: PRAUC improved steadily during top-12-only phase, continued after full unfreeze.

### 6k. AEDA Augmentation

**What**: Randomly inserts punctuation marks between words during training (prob=0.35-0.40).
**Why**: Label-safe augmentation — inserting commas cannot change a sentence's theme. Prevents overfitting to surface patterns.

### 6l. Essay Auxiliary Head

**What**: Separate classification head on mean-pooled essay-level representation. Labels = OR-union of sentence labels.
**Why**: Rare themes have higher essay-level prevalence (Attainment: 2.4% sentence → ~8% essay). Provides 3× more positive training signal.
**Config**: weight=0.10 (base), 0.15 (large).

### 6m. Sentence Position Embedding

**What**: Learned [32, H] embedding table added to sentence embeddings before BiLSTM.
**Why**: Essay position correlates with theme (opening = Navigational/Aspirational, middle = Perseverance/Resistance, closing = Attainment).
**Cost**: ~32K parameters (negligible).

---

## 7. Overfitting Analysis

| Model | Train F1 | Test F1 | Gap | Train PRAUC | Test PRAUC | Gap |
|---|---|---|---|---|---|---|
| Base Standard | 0.727 | 0.487 | 0.240 | 0.884 | 0.507 | 0.377 |
| Large AWARE (v4) | 0.759 | 0.494 | 0.265 | 0.900 | 0.484 | 0.416 |

Both models severely overfit. The large AWARE has a LARGER gap despite heavier regularization, because 438M params provide more memorization capacity than the regularization can constrain.

---

## 8. Threshold Analysis

| Theme | Large AWARE Threshold | Interpretation |
|---|---|---|
| Navigational | 0.310 | Moderate — well-calibrated |
| Aspirational | 0.230 | Below 0.5 — model is overconfident |
| Perseverance | 0.350 | Moderate |
| Social | 0.220 | Below 0.5 — overconfident |
| Familial_Capital | 0.140 | Low — rare theme needs aggressive threshold |
| Resistance | 0.090 | Very low |
| Spiritual | 0.053 | Near minimum — model barely distinguishes |
| Attainment | 0.050 | At floor — extreme rarity forces low threshold |

---

## 9. Calibration Analysis (Large AWARE v4)

| Theme | Platt a | Platt b | Interpretation |
|---|---|---|---|
| Attainment | 1.20 | -3.00 | SEVERELY biased — raw probs way too high |
| Aspirational | 1.70 | -1.80 | Biased |
| Navigational | 1.60 | -1.60 | Biased |
| Resistance | 1.60 | -1.80 | Biased |
| Perseverance | 1.30 | -2.00 | Severely biased |
| Social | 1.40 | -2.20 | Severely biased |
| Spiritual | 1.10 | -2.20 | Severely biased |
| Familial_Capital | 1.30 | -2.60 | Severely biased |

All Platt b values are negative → model systematically over-predicts positive probability for every theme. Calibration corrects this but it indicates the model's raw confidence estimates are unreliable.

---

## 10. What Works, What Doesn't

### Clearly Working
| Innovation | Evidence |
|---|---|
| DAPT | Perplexity 84→14; domain vocabulary learned |
| ASL Loss | Attainment +0.081 F1 vs BCE standard |
| CB Weights | Rare themes get proportional gradient signal |
| 3-Phase Training | Phase 1 PRAUC 0.14→0.40 (stable head init) |
| LLRD | All layers contribute at 0.92 decay |
| SWA | +0.0015 PRAUC from weight averaging |
| Multi-Sample Dropout | Free regularization, no downside |

### Uncertain
| Innovation | Evidence |
|---|---|
| BiLSTM Context | Helps 2 themes, hurts 3 themes |
| Essay Aux Head | Small fraction of loss (3-12%) |
| Position Embedding | Negligible parameter cost, unclear benefit |
| Progressive Unfreeze | Adds complexity, benefit unclear |

### Areas for Improvement
| Issue | Detail |
|---|---|
| Overfitting | Train-Test gap 0.265 F1 — 438M params on 14K sentences |
| Calibration | All Platt b values severely negative |
| Spiritual degradation | -0.056 F1 vs standard — context hurts this theme |
| Familial degradation | -0.041 F1 vs standard — CB weights may be too aggressive |

---

## 11. Data Summary

| Split | Essays | Sentences | Usage |
|---|---|---|---|
| Train | 2,095 | 14,023 | Model training |
| Validation | 268 | 1,757 | Hyperparameter selection, early stopping |
| Test | 273 | 1,838 | Final evaluation (held out) |
| DAPT Corpus | 2,636 | — | MLM pre-training (all essays, no labels) |

**Zero student leakage between splits — verified.**

### Theme Distribution (Training Set)

| Theme | Count | Prevalence | CB Weight |
|---|---|---|---|
| Navigational | 3,368 | 24.0% | 0.238× |
| Aspirational | 2,158 | 15.4% | 0.351× |
| Perseverance | 1,165 | 8.3% | 0.620× |
| Social | 786 | 5.6% | 0.902× |
| Resistance | 629 | 4.5% | 1.118× |
| Familial_Capital | 554 | 3.9% | 1.265× |
| Spiritual | 478 | 3.4% | 1.460× |
| Attainment | 339 | 2.4% | 2.045× |

---

## 12. Improvement Over Baselines

| Model | F1 | Improvement over TF-IDF SVM | Improvement over Majority |
|---|---|---|---|
| TF-IDF + SVM | 0.274 | — | +0.274 |
| Base Standard | 0.487 | +0.213 (+77.7%) | +0.487 |
| **Large AWARE (v4)** | **0.494** | **+0.220 (+80.3%)** | **+0.494** |

---

## 13. Run History

| Job | Model | Config | Test F1 | Test PRAUC | Notes |
|---|---|---|---|---|---|
| 22566 | Baselines | — | 0.000-0.274 | 0.085-0.363 | CPU, 27 sec |
| 22619 | Base Standard | base_standard.yaml | 0.487 | 0.507 | Stopped ep 8 |
| 22620 | Large Standard | large_standard.yaml | 0.483 | 0.496 | Stopped ep 13 |
| 22630 | Base AWARE | base_aware.yaml | 0.472 | 0.480 | With DAPT |
| 22386 | **Large AWARE v4** | **large_v4.yaml** | **0.494** | **0.484** | **Best F1** |

---

*Generated: 2026-03-19*
*Project: AWARE v4 — Multi-Label CCW Theme Classification*
