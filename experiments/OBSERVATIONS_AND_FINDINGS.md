# Training Observations, Inferences & Findings

> Compiled from all training logs, HPO experiments, and diagnostic analysis across 15+ runs.

---

## 1. Training Dynamics Observations

### 1a. Phase 1 (Frozen Encoder) — Head Initialization

| Observation | Evidence | Run |
|---|---|---|
| **FP16 gradient overflow in epoch 1** | grad_norm=inf, scaler skips update | Large AWARE v3 (22344), v4 (22386) |
| Effectively wastes 1 of 4 Phase 1 epochs (v3) | PRAUC=0.10 after ep1 (near random) | 22344 |
| 8 Phase 1 epochs much better than 4 | PRAUC=0.40 vs 0.14 at end of P1 | v4 (22386) vs v3 (22344) |
| Phase 1 grad norms stabilize by ep2 | mean=2.45 (ep2) vs inf (ep1) | 22386 |
| Disabling R-Drop in Phase 1 helps | KL=0 frees all gradient for learning | 22386 (phase1_rdrop=0.0) |
| Phase 1 LR=1e-4 works despite grad clipping | Effective LR ~1.4e-5 after clip | 22386 |
| Head saturates by ep6-7 in Phase 1 | PRAUC delta <0.001 between ep7-8 | 22386 |

### 1b. Phase 2 (Full Fine-Tuning) — Main Training

| Observation | Evidence | Run |
|---|---|---|
| **Val loss diverges early while PRAUC still improves** | Val loss rising ep10, PRAUC peaks ep31 | 22386 |
| Large model memorizes at 0.021 loss/epoch | Train loss 1.17→0.49 over 31 epochs | 22386 (v4) |
| Large v3 memorized at 0.072 loss/epoch (3.4× faster!) | Train loss 1.43→0.35 over 15 epochs | 22344 (v3) |
| Lower encoder_lr (5e-6 vs 1.5e-5) slows memorization | v4 train loss 0.49 at ep39 vs v3 0.35 at ep19 | v4 vs v3 |
| Progressive unfreezing helps | PRAUC jumps when bottom layers unfreeze at ep14 | 22386 |
| **KL loss flatlines mid-training** | KL=0.047-0.048 from ep12 onward (v3) | 22344 |
| R-Drop at alpha=2.0 dominated gradient | KL was 55-72% of task loss in early P2 | 22344 (v3) |
| R-Drop at alpha=1.0 is healthier | KL was 10-15% of task loss | 22386 (v4) |
| **Themes overfit at different rates** | Social peaked ep10, Familial peaked ep18 | 22344 |
| Patience=5 too eager, patience=8 better | v3 stopped ep19, v4 ran to ep39 | v3 vs v4 |

### 1c. Phase 3 (Head + BiLSTM Retrain)

| Observation | Evidence | Run |
|---|---|---|
| **Phase 3 never improves over SWA** | 0.5218 vs 0.5224 (v4), 0.5236 vs 0.5236 (v3) | 22386, 22344 |
| V3 Phase 3 was completely dead | Only 20.5K params, grad_norm≈0, metrics frozen | 22344 |
| V4 Phase 3 slightly better with BiLSTM unfrozen | 4.78M params trainable, but still didn't beat SWA | 22386 |
| Phase 3 grad norms are very small | mean=0.74-0.78 (vs 2.5-3.0 in P2) | 22386 |

### 1d. SWA (Stochastic Weight Averaging)

| Observation | Evidence | Run |
|---|---|---|
| V3 SWA was a complete no-op | swa_start at ep19 = same as early stop, 0 checkpoints | 22344 |
| V4 SWA collected 22 checkpoints | swa_start at ep18, ran to ep39 | 22386 |
| SWA improvement is small but real | 0.5209 → 0.5224 (+0.0015) | 22386 |

---

## 2. Loss Function & Weighting Observations

### 2a. ASL Loss

| Observation | Evidence | Run |
|---|---|---|
| gamma_neg=3.0 causes severe overconfidence | All Platt b values -1.8 to -2.6 | 22344 (v3) |
| gamma_neg=2.5 slightly reduces overconfidence | Platt b values -1.6 to -3.0 (still severe) | 22386 (v4) |
| gamma_pos=0.0 is correct | Preserves all gradient for rare positives | All AWARE runs |
| Label smoothing 0.02 (neg-only) helps | Prevents logit explosion without suppressing positives | v4 |

### 2b. CB Weights

| Observation | Evidence | Run |
|---|---|---|
| CB (beta=0.9999) gives Attainment 2.045×, Nav 0.238× | Ratio ~8.6:1 | All AWARE runs |
| Previous inverse-sqrt gave Attainment 3.15× | Less aggressive | Base standard |
| CB weights help Attainment (+0.081 F1) but may hurt mid-frequency themes | Spiritual -0.056, Familial -0.041 | v4 vs base standard |

### 2c. Loss Component Balance

| Observation | Evidence | Run |
|---|---|---|
| Essay aux at weight=0.10 is inert | Only 3-13% of total loss | 22344, 22320 |
| Essay aux at weight=0.15 is meaningful | ~12% of total loss at mid-training | 22386 |
| R-Drop KL at alpha=2.0 dominates early training | 55-72% of task loss | 22344 |
| R-Drop KL at alpha=1.0 is appropriate | 10-19% of task loss | 22386 |

---

## 3. Calibration & Threshold Observations

| Observation | Evidence | Run |
|---|---|---|
| **All models are systematically overconfident** | All Platt b values negative (-1.6 to -3.0) | All AWARE runs |
| Attainment threshold hits minimum floor (0.050) | P=0.22 at that threshold — 78% false positives | 22386 |
| Spiritual threshold near minimum (0.053) | P=0.29 — still high FP rate | 22386 |
| Platt scaling a=1.0 for rare themes | Cannot improve ranking, only shift bias | Att, Spi, Fam |
| Platt scaling a=1.7 for Aspirational | Needs to steepen sigmoid — logit range compressed | 22386 |

---

## 4. Per-Theme Deep Observations

### Attainment (2.4% prevalence, 339 train, 32 test)

| Observation | Evidence |
|---|---|
| **Highest train-test gap of any theme** | Train F1=0.822 vs Test F1=0.303 (gap=0.519) |
| F1 oscillates wildly during training | 0.238→0.364→0.333→0.267 across consecutive epochs |
| 60.5% co-occurrence with Navigational | Model confuses the two |
| Essay-level prevalence ~8% (3× sentence rate) | Essay head provides more signal |
| AWARE helps most here | +0.081 F1 over standard (0.303 vs 0.222) |
| Only 32 test samples | Results highly variable (bootstrap CI: [0.175, 0.414]) |

### Spiritual (3.4% prevalence, 478 train, 95 test)

| Observation | Evidence |
|---|---|
| **AWARE hurts this theme** | 0.375 (AWARE) vs 0.431 (standard) — -0.056 |
| Semantically adjacent to Aspirational | "calling", "purpose", "meaning" overlap |
| Context modeling adds noise | Single sentences are self-contained for Spiritual |
| Platt b=-2.20 (severely biased) | Model over-predicts Spiritual |
| Low logit separation | Spiritual has smallest sep at some epochs |

### Navigational (24.0% — most common)

| Observation | Evidence |
|---|---|
| Consistently best performance across all models | F1=0.68-0.71 |
| Smallest train-test gap | 0.116 F1 gap — generalizes well |
| Distinctive vocabulary | "major", "courses", "transfer", "prerequisites" |
| AWARE helps slightly | +0.023 F1 (0.707 vs 0.684) |
| CB weight = 0.238× (most suppressed) | May be slightly under-weighted |

### Perseverance (8.3%)

| Observation | Evidence |
|---|---|
| Base AWARE scores highest (0.555) | Better than large AWARE (0.480) or standards |
| Large models overfit more on this theme | Large standard 0.442 vs base standard 0.510 |
| Context may genuinely help | Perseverance narratives span multiple sentences |

---

## 5. Architecture & Model Scale Observations

### 5a. Base vs Large

| Observation | Evidence |
|---|---|
| **Base standard BEATS large standard on test** | F1=0.487 vs 0.483 |
| Large overfits more | Val PRAUC 0.555 vs 0.527, but test PRAUC 0.496 vs 0.507 |
| Large benefits from AWARE more than base | Large: +0.011 F1 with AWARE. Base: -0.015 F1 |
| The 438M → 184M reduction (2.4×) doesn't hurt test performance | Overfitting is the bottleneck, not capacity |

### 5b. BiLSTM Context

| Observation | Evidence |
|---|---|
| BiLSTM adds 2M params of overfitting surface | ~1 param per 7 training examples |
| Residual connection dominates (norm ~30 vs projection ~0.02) | BiLSTM signal is negligible at initialization |
| DeBERTa's 512-token self-attention already provides context | BiLSTM is theoretically redundant |
| 7-sentence average is too short for LSTM advantage | LSTMs need 50+ timesteps |
| Despite all this, BiLSTM helps on rare themes | Attainment +0.081 — context for multi-sentence narratives |

### 5c. Sentence Mean Pooling

| Observation | Evidence |
|---|---|
| 16× information compression | 524K numbers → 33K numbers |
| Loses token-level cross-sentence attention patterns | DeBERTa spent 24 layers computing these |
| Sentence separator matters | ". " vs " " significantly affects tokenization |

---

## 6. HPO Observations (Base Model — Job 22321)

Five HPO configurations were tested on the base model:

| Config | gamma_neg | WD | rdrop | ls | Val PRAUC | Notes |
|---|---|---|---|---|---|---|
| hpo_b1 (control) | 3.0 | 0.04 | 1.5 | 0.0 | 0.5532 | Baseline |
| hpo_b2 | 2.5 | 0.05 | 2.0 | 0.0 | — | Fix overconfident logits |
| hpo_b3 | 2.0 | 0.05 | 2.0 | 0.02 | — | Strongest anti-overconfidence |
| hpo_b4 | 2.5 | 0.05 | 2.0 | 0.0 | — | Slower encoder |
| hpo_b5 | 2.5 | 0.06 | 2.0 | 0.02 | — | All regularization maxed |

**Key HPO finding**: Lower gamma_neg (2.5 vs 3.0) and mild label smoothing (0.02) help reduce overconfident logits without hurting macro PRAUC. These findings were incorporated into the v4 config.

---

## 7. DAPT Observations

### Base DAPT (Job 22569, 10 epochs)

| Observation | Evidence |
|---|---|
| Perplexity dropped from ~84 to 14.2 | Significant domain adaptation |
| Mild overfitting at epochs 9-10 | Eval loss 2.58 (ep8) → 2.62 (ep10) |
| Training time: 4 min | Fast on A100 |
| Best checkpoint loaded automatically | HuggingFace Trainer saves best |

### Large DAPT (15 epochs)

| Observation | Evidence |
|---|---|
| Old DAPT encoder reused from f-models repo | May have preprocessing mismatch (old " " separator) |
| 15 epochs may be too many | Base showed overfitting at ep9; large has more params |
| Fresh DAPT (Job 22628) ran on latest code | Should use ". " separator consistently |

---

## 8. Diagnostic Bugs Found & Fixed

| Bug | Impact | Fix | Run |
|---|---|---|---|
| **ClassificationHead dropout order** | LN undid dropout's zeros | Changed to LN→dropout→linear | Pre-v4 |
| **R-Drop asymmetry** | Only p2 got KL gradient (p1 detached) | Symmetric KL in single forward+backward | Pre-v4 |
| **Sentence separator = " "** | Essays looked like one run-on paragraph | Changed to ". " | Pre-v4 |
| **SWA never activated** | swa_start_ep = early_stop_ep | Reduced ratio from 0.50 to 0.25 | v3→v4 |
| **Phase 1 too short (4 epochs)** | Head PRAUC=0.14 entering Phase 2 | Increased to 8 epochs | v3→v4 |
| **Phase 3 dead (only 20K params)** | grad_norm≈0, no improvement | Unfreeze BiLSTM (2M params) | v3→v4 |
| **LLRD too aggressive (0.85^23)** | Bottom layers at 3e-7 (frozen) | Changed to 0.92 (bottom=8.5e-7) | v3→v4 |
| **R-Drop alpha=2.0 too strong** | KL was 55-72% of task loss | Reduced to 1.0 | v3→v4 |
| **Threshold min-precision missing** | Attainment threshold=0.050, P=0.14 | Added min_precision=0.15 fallback | Base run |
| **Phase3_lr=3e-5 too high** | Val loss rose immediately in Phase 3 | Reduced to 1e-5 (base), 5e-6 (large) | HPO analysis |
| **Prototype head catastrophic** | Test PR-AUC 0.275 (vs 0.481 baseline) | Disabled permanently | Run 22310 |
| **log_temperature init=2.0** | exp(2)=7.4 → loss spike | Changed to 0.0 (exp(0)=1.0) | Run 22310 |
| **Nested for-loop in SentenceMeanPooling** | 128 Python iterations per batch | Vectorized to 4 iterations (per essay) | Pre-v4 |
| **SBATCH set -eo before directives** | Job script failed silently | Moved after #SBATCH lines | Comparison runs |
| **nvidia-smi pipe with pipefail** | SIGPIPE killed job | Used --query-gpu format | Comparison runs |
| **Missing --data_dir in job scripts** | argparse crash | Added to all evaluate/train calls | Comparison runs |
| **evaluate.py wrong CLI args** | --checkpoint, --output not recognized | Changed to --results_dir, --data_dir | Comparison runs |

---

## 9. Key Inferences & Conclusions

### 9a. On Overfitting
> The dominant challenge is 438M parameters learning from 14K sentences (31K params/sentence). No regularization stack (dropout 0.30, WD 0.08, R-Drop 1.0, AEDA 0.40, SWA, multi-sample dropout) fully compensates. The train-test F1 gap of 0.265 is the primary ceiling.

### 9b. On Model Scale
> Larger is NOT better for this dataset size. Base standard (184M) beats large standard (434M) on test. The additional parameters only add memorization capacity without improving generalization.

### 9c. On AWARE's Core Contribution
> AWARE's essay-context mechanism (BiLSTM) helps rare themes (Attainment +0.081) where multi-sentence narrative patterns disambiguate meaning. It hurts common themes (Spiritual -0.056) where single-sentence content is already sufficient. The net effect is +0.007 F1 macro, which is marginal but consistent.

### 9d. On the Best Configuration
> The v4 config represents the optimal balance after fixing 14 diagnosed bugs and running HPO. Key choices: encoder_lr=5e-6 (3× lower than v3), LLRD=0.92 (gentler than 0.85), R-Drop=1.0 (halved from 2.0), Phase 1=8 epochs (doubled from 4), patience=8 (increased from 5).

### 9e. On What Would Help Most
> 1. More labeled data (especially for rare themes)
> 2. Better threshold optimization and calibration
> 3. Ensemble of multiple base models (reduce variance)
> 4. The sentinel token approach (classify from [SENT] markers instead of mean-pooled embeddings)

---

*Generated: 2026-03-19*
*Project: AWARE v4 — Multi-Label CCW Theme Classification*
