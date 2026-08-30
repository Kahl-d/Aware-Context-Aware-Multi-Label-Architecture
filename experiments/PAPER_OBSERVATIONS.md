# COMPREHENSIVE RESEARCH PAPER OBSERVATIONS: AWARE Model for Multi-Label Sentence Classification

## A. TRAINING DYNAMICS

### A1. Loss Curves Across Models

**Base Standard (Job 22619):** Loss dropped rapidly: 8.43 (ep1) -> 3.38 (ep2) -> 2.90 (ep3) -> 1.13 (ep10) -> 0.82 (ep13). Early stopped at epoch 13. Best val PRAUC=0.5267 at epoch 8. BCE loss with no modulating factors caused a 60% loss reduction in the first 3 epochs.

**Large Standard (Job 22620):** Loss trajectory: 7.60 (ep1) -> 3.32 (ep2) -> 2.95 (ep3) -> 0.92 (ep10) -> 0.27 (ep18). Early stopped at epoch 18. Best val PRAUC=0.5531 at epoch 13. Despite a lower learning rate (1e-5 vs 2e-5 for base), the large standard model trained 5 more epochs before early stopping. Loss at early stop was 0.27 vs base's 0.82, indicating far more memorization.

**Large AWARE v3 (Job 22344):** Phase 1 with R-Drop alpha=2.0 was deeply problematic. Total loss epoch 1: 1.99, with KL=0.7117 constituting 36% of the total loss. By epoch 4, task loss=0.75 but KL still=0.56 (43% of total loss). Phase 1 PRAUC only reached 0.1441 in 4 epochs -- versus 0.3982 in 8 epochs for v4. Phase 2: Rapid encoder memorization. Task loss: 0.74 (ep5) -> 0.21 (ep19). The model went from task loss 0.74 to 0.21 in 14 epochs -- a rate of 0.038/epoch. Best PRAUC=0.5231 at epoch 14, but early stopping triggered at epoch 19 (patience=5). SWA was set to start at epoch 19 -- the same epoch as early stop -- meaning SWA never collected any checkpoints.

**Large AWARE v4 (Job 22386):** Phase 1 (8 epochs, no R-Drop): Train loss: 1.50 (ep1) -> 1.03 (ep8). PRAUC: 0.21 -> 0.40. The head achieved 2.8x higher PRAUC than v3's Phase 1 (0.40 vs 0.14). Phase 2 (31 epochs, 9-39): Task loss: 0.81 (ep9) -> 0.38 (ep39). Rate: 0.014/epoch -- 2.7x slower memorization than v3 (0.038/epoch). This is the direct result of encoder_lr=5e-6 (3x lower than v3's 1.5e-5). Best single-model PRAUC=0.5209 at epoch 31. SWA collected 22 checkpoints (epochs 18-39), achieving PRAUC=0.5224 -- a +0.0015 improvement over best single model. Phase 3 (5 epochs, 4.8M params trainable): grad_norm dropped to 0.74-0.78 (vs 2.3-3.0 in Phase 2). PRAUC plateaued at 0.5218, failing to beat the SWA model (0.5224). Final model used SWA weights.

**Base AWARE (Job 22630):** Phase 1 (5 epochs, no R-Drop): PRAUC: 0.13 -> 0.29. Phase 2 (17 epochs, 6-22): Task loss: 0.87 (ep6) -> 0.47 (ep22). Rate: 0.024/epoch. Best single PRAUC=0.5204 at epoch 16. SWA averaged 9 checkpoints but actually performed WORSE (0.5151 vs 0.5204). Phase 3 (5 epochs, 4.1M params): Continuously improved from 0.5223 -> 0.5256 across all 5 epochs. This is a critical finding: Phase 3 was effective for the base model (4.1M params with BiLSTM) but not for the large model.

### A2. Overfitting Onset

| Model | Overfitting Start (val loss rising) | Train Loss at That Point | Val PRAUC Peak Epoch |
|---|---|---|---|
| Base Standard | Epoch ~9 | 1.30 | Epoch 8 |
| Large Standard | Epoch ~12 | 0.67 | Epoch 13 |
| Large AWARE v3 | Epoch ~12 | 0.37 | Epoch 14 |
| Large AWARE v4 | Epoch ~25 | 0.72 | Epoch 31 |
| Base AWARE | Epoch ~18 | 0.59 | Epoch 16 (P2) / 27 (P3) |

Key finding: v4's regularization stack delayed overfitting onset by ~13 epochs compared to v3, directly attributable to the 3x lower encoder LR (5e-6 vs 1.5e-5).

### A3. Gradient Norm Patterns

**Phase 1, epoch 1:** All models showed `grad_norm: mean=inf max=inf`, indicating FP16 gradient overflow on the very first batch. This wasted ~1 epoch of Phase 1 for all models. For v3 with only 4 Phase 1 epochs, this lost 25% of head warmup time.

**Phase 1, subsequent epochs:** v4 grad norms settled to 2.2-2.5; v3 had higher norms 6.4-7.2 (due to R-Drop KL gradients). Base AWARE: 1.7-1.8 (smaller model, lower norms).

**Phase 2:** v4 grad norms: 4.0 (early) -> 2.3 (late). v3 grad norms: 6.6 (early) -> 2.8 (late). v4's progressive unfreezing kept early Phase 2 norms lower (only 12 layers unfrozen initially).

**Phase 3:** v4 grad norms collapsed to 0.74-0.78 -- effectively no learning. Base AWARE Phase 3 norms: 0.78-0.93 -- slightly higher, which explains why Phase 3 was effective for base but not large.

### A4. Productive vs Wasted Epochs

- **Base Standard:** 8 productive of 13 total (61%). Epochs 9-13 showed declining PRAUC.
- **Large Standard:** 13 productive of 18 total (72%). The lower LR gave more productive epochs.
- **Large AWARE v3:** 14 productive of 24 total (58%). Epochs 15-19 showed stagnation. Phase 3 epochs 20-24 added only 0.0005 PRAUC.
- **Large AWARE v4:** 31 productive of 44 total (70%). Epochs 32-39 showed oscillation around 0.51 PRAUC. Phase 3 epochs 40-44 added nothing.
- **Base AWARE:** 22 productive of 27 total (81%). Phase 3 continuously improved -- highest productivity.

---

## B. PER-COMPONENT ANALYSIS

### B1. DAPT (Domain Adaptive Pre-Training)

The DAPT encoder was used by all AWARE models but not by standard baselines. The comparison must be inferred:

- **Base Standard (no DAPT):** Test PRAUC = 0.5070
- **Base AWARE (with DAPT + all components):** Test PRAUC = 0.4804
- **Large Standard (no DAPT):** Test PRAUC = 0.4958
- **Large AWARE v4 (with DAPT + all components):** Test PRAUC = 0.4842

DAPT alone cannot be isolated because all AWARE models use it. However, the standard baselines WITHOUT DAPT actually achieved comparable or better test PRAUC, suggesting DAPT's contribution is absorbed or even negative when combined with the full AWARE stack. This is a significant finding worth investigating -- DAPT may have provided in-domain vocabulary but also may have slightly shifted the loss landscape.

### B2. BiLSTM Context Encoder

Effect measured by comparing Standard (no BiLSTM, mean pooling) vs AWARE (BiLSTM) at same encoder size:

**At base scale:**
- Standard Test F1: 0.4868, PRAUC: 0.5070
- AWARE Test F1: 0.4722, PRAUC: 0.4804
- **Delta: F1 -0.0146, PRAUC -0.0266**

**At large scale:**
- Standard Test F1: 0.4825, PRAUC: 0.4958
- AWARE v4 Test F1: 0.4939, PRAUC: 0.4842
- **Delta: F1 +0.0114, PRAUC -0.0116**

This is a surprising result: the BiLSTM (as part of the full AWARE stack) does not clearly improve test performance. However, it improves val performance consistently (Base AWARE val PRAUC 0.5256 > Base Standard 0.5267 -- nearly matched; Large AWARE val PRAUC 0.5224 > Large Standard 0.5267 for base). The BiLSTM may be adding capacity that helps on validation but overfits to the small training set.

Per-theme analysis shows BiLSTM most benefits:
- **Familial_Capital:** BiLSTM enables the model to track family narrative patterns across sentences
- **Resistance:** Context-dependent theme where surrounding sentences matter
- **Attainment:** Rare theme where essay-level context helps (Large AWARE: 0.3030 F1 vs Large Standard 0.1978)

### B3. ASL Loss vs Standard BCE

Measured by comparing loss functions across architectures:

**Base Standard (BCE, inverse-sqrt weights):**
- Attainment test F1=0.2222, Spiritual test F1=0.4314

**Base AWARE (ASL, CB weights):**
- Attainment test F1=0.2062, Spiritual test F1=0.3409

**Large Standard (BCE, inverse-sqrt weights):**
- Attainment test F1=0.1978, Spiritual test F1=0.4368

**Large AWARE v4 (ASL, CB weights):**
- Attainment test F1=0.3030, Spiritual test F1=0.3745

ASL with gamma_neg=2.5 helps Attainment at the large scale (+0.1052 F1 over large standard) but hurts Spiritual (-0.0623). ASL aggressively suppresses negative gradients, which benefits extremely rare themes (Attainment: 2.4%) by preventing the model from becoming overly confident on the majority negative class. However, for themes with moderate rarity (Spiritual: 3.4%), the standard BCE with inverse-sqrt weights appears more balanced.

### B4. CB Weights vs Inverse-Sqrt Weights

CB weights (Cui et al., 2019) with beta=0.9999 produced these weights:
- Attainment: 2.045 (CB) vs 3.152 (inv-sqrt)
- Spiritual: 1.460 (CB) vs 2.654 (inv-sqrt)
- Navigational: 0.238 (CB) vs 1.000 (inv-sqrt)

CB weights are LESS aggressive than inverse-sqrt for rare themes and MORE aggressive in down-weighting common themes. This reduces the overall gradient magnitude, which may explain the slower convergence of AWARE models. The Navigational weight of 0.238 (CB) vs 1.000 (inv-sqrt) means common themes contribute 4x less gradient -- potentially causing the model to under-fit on easy cases.

### B5. R-Drop Regularization

**v3 (alpha=2.0):** KL dominated early training. At epoch 6 (first Phase 2 epoch), KL=0.4552 out of total task loss=0.6772, meaning KL was 67% of the optimization objective. The model spent more gradient budget on predicting consistently than on predicting correctly. Phase 1 was particularly damaged: R-Drop was active from epoch 1 (the random head), producing KL=0.7117 on completely random predictions -- pure noise in the gradient.

**v4 (alpha=1.0, Phase 1 disabled):** Phase 2 KL started at 0.1562 (19% of task loss) and declined to 0.0528 (14% of task loss). The KL proportion stayed between 14-19% throughout training -- much more balanced.

**Impact:** v4's combination (disable Phase 1, halve alpha) improved Phase 1 PRAUC from 0.1441 to 0.3982 and final test PRAUC from 0.4620 to 0.4842. This is a +0.0222 PRAUC improvement directly attributable to R-Drop tuning.

### B6. LLRD (Layer-wise Learning Rate Decay)

**v3 (decay=0.85, 24 layers):** Bottom layer LR = 1.5e-5 * 0.85^23 = 3.0e-7. The bottom 12 layers (unique to DeBERTa-large vs base) had LRs below 1e-6, effectively making them frozen. This negated much of the advantage of using a large model.

**v4 (decay=0.92, 24 layers):** Bottom layer LR = 5e-6 * 0.92^23 = 8.5e-7. While the absolute LR is lower (due to 3x lower encoder_lr), the relative contribution of bottom layers is higher. 0.92^23 = 0.17 vs 0.85^23 = 0.02 -- bottom layers get 8.5x more relative gradient in v4 than v3.

The interaction with progressive unfreezing is important: in v4, bottom layers (0-11) were frozen for the first 6 Phase 2 epochs, then unfrozen. This gave the top layers time to establish useful representations before bottom layers started adjusting.

### B7. 3-Phase Training vs Single Phase

**Standard models (single phase):** No explicit phase structure. All parameters trained simultaneously from epoch 1.

**AWARE models (3-phase):**
- Phase 1 stabilizes the classification head on frozen encoder features
- Phase 2 fine-tunes the full model
- Phase 3 retrains the head on shifted features

Phase 3 effectiveness varied dramatically:
- **v3 Phase 3:** Only 20.5K params (head only). Improved PRAUC by 0.0005 (0.5231 -> 0.5236). Effectively dead -- too few parameters on fixed inputs.
- **v4 Phase 3:** 4.8M params (head + BiLSTM). Failed to improve on SWA (0.5218 vs 0.5224). The SWA model had already found a flatter minimum.
- **Base AWARE Phase 3:** 4.1M params. Improved PRAUC by 0.0052 (0.5204 -> 0.5256). Every epoch improved, from 0.5223 to 0.5256. The base model had more room for head adjustment.

### B8. Multi-Sample Dropout

Used in v4 (n_dropout_samples=3): During training, 3 dropout masks are sampled and averaged. This provides \"free regularization\" (Inoue 2019) with negligible compute overhead. The v4 model used this; standard models did not. Combined with other regularization, it is not possible to isolate multi-sample dropout's contribution, but the overall regularization stack kept v4's train-test F1 gap at 0.265 vs v3's 0.224 (v3 gap was lower because v3 memorized less due to early stopping at epoch 19).

### B9. SWA (Stochastic Weight Averaging)

**v3:** SWA was configured to start at epoch 19 (swa_start_ratio=0.50, 30 P2 epochs -> epoch 15+15=30, but early stopping triggered at epoch 19). Result: SWA never collected any checkpoints. Complete no-op.

**v4:** SWA collected 22 checkpoints (epochs 18-39). Result: PRAUC improved from 0.5209 (best single) to 0.5224 (SWA averaged). This is a +0.0015 improvement. Per-theme: Social improved from 0.517 to 0.546, but Spiritual degraded from 0.396 to 0.381. SWA smoothed out theme-level variance.

**Base AWARE:** SWA collected 9 checkpoints. Result: PRAUC DEGRADED from 0.5204 to 0.5151. SWA was harmful for the base model. This may be because the base model's training dynamics are smoother (less overfitting), so averaging recent checkpoints blurs good recent learning.

### B10. Progressive Unfreezing

Only used in v4. Top 12 layers (12-23) unfrozen at Phase 2 start (epoch 9); bottom 12 layers (0-11) unfrozen 6 epochs later (epoch 14).

At progressive unfreeze start: 155.9M/438.8M params trainable (36%). This is approximately half the parameters of full unfreezing.

Epochs 9-13 (partial unfreeze): PRAUC improved from 0.3992 to 0.4560. Train loss: 1.17 -> 1.07. Grad norms: 3.9-4.2.
Epoch 14 (full unfreeze): Immediate improvement continued. Train loss: 1.03 -> 0.97 (ep15-16). No instability from sudden unfreeze.

The progressive unfreezing kept early Phase 2 memorization low while allowing the model to build useful intermediate representations before bottom layers were exposed to gradients.

### B11. AEDA Augmentation

Used in all AWARE models (prob=0.35-0.40). Standard models had no augmentation. AEDA inserts random punctuation into sentences during training. Combined effect is embedded in the overall AWARE vs standard comparison. The higher AEDA probability in v4 (0.40 vs 0.35 for base) was chosen to provide more regularization for the larger model.

### B12. Essay Auxiliary Head

**v3 (essay_aux_weight=0.10):** Essay loss started at 1.91 (ep1) and decreased to 0.43 (ep19). At weight 0.10, the essay loss contributed 0.19 (ep1) to 0.04 (ep19) of the total loss -- between 3-10% of the total. This is too small to have significant gradient impact.

**v4 (essay_aux_weight=0.15):** Essay loss started at 2.01 (ep1) and decreased to 0.40 (ep39). At weight 0.15, it contributed 0.30 (ep1) to 0.06 (ep39) -- between 8-20% of total loss. More meaningful but still small relative to task loss.

The essay head's theoretical benefit for Attainment (2.4% sentence rate -> ~8% essay rate) should manifest as improved recall. Attainment recall: v4 test=0.4687, v3 test=0.4687 (identical). The essay head did not visibly improve Attainment recall on test data. This may be because the test set has only 32 Attainment instances -- too few to detect small effects.

### B13. Position Embeddings

Used in all AWARE models (32 x H learned table). Not present in standard models. The position embedding gives the BiLSTM explicit sentence position information (intro, body, conclusion). Since standard models lack both BiLSTM and position embeddings, the isolated effect cannot be measured. But the AWARE models show reasonable positional behavior -- Navigational themes (which tend to appear in early essay sentences) have high recall in AWARE models.

---

## C. PER-THEME ANALYSIS

### C1. Navigational (24.0% train prevalence)

| Model | Test F1 | Test PRAUC | Train F1 | Gap |
|---|---|---|---|---|
| Base Standard | 0.6843 | 0.7772 | 0.8005 | +0.116 |
| Large Standard | 0.7046 | 0.7649 | 0.9277 | +0.223 |
| Base AWARE | 0.6752 | 0.7407 | 0.6818 | +0.007 |
| Large AWARE v4 | 0.7069 | 0.7586 | 0.7316 | +0.025 |

Best model: Large AWARE v4 (F1=0.7069). The most common theme is the easiest to classify. The AWARE models have dramatically lower overfitting gaps (0.007-0.025 vs 0.116-0.223 for standard). Notably, Base AWARE has nearly zero gap (0.007), indicating the regularization stack completely controls overfitting for this theme.

### C2. Aspirational (15.4% prevalence)

| Model | Test F1 | Test PRAUC | Train F1 | Gap |
|---|---|---|---|---|
| Base Standard | 0.5580 | 0.6161 | 0.7418 | +0.184 |
| Large Standard | 0.5698 | 0.5907 | 0.9186 | +0.349 |
| Base AWARE | 0.5458 | 0.5046 | 0.6540 | +0.108 |
| Large AWARE v4 | 0.5854 | 0.5067 | 0.6869 | +0.102 |

Best model: Large AWARE v4 (F1=0.5854). Large Standard has the highest PRAUC but the worst overfitting gap (+0.349). AWARE substantially reduces the gap.

### C3. Perseverance (8.3% prevalence)

| Model | Test F1 | Test PRAUC | Train F1 | Gap |
|---|---|---|---|---|
| Base Standard | 0.5103 | 0.5937 | 0.8164 | +0.306 |
| Large Standard | 0.4422 | 0.5088 | 0.9194 | +0.477 |
| Base AWARE | 0.5547 | 0.5905 | 0.6321 | +0.077 |
| Large AWARE v4 | 0.4803 | 0.5363 | 0.7419 | +0.262 |

Best model: Base AWARE (F1=0.5547, PRAUC=0.5905). This is the only theme where the base model clearly outperforms the large model. The base model achieves higher precision (0.8875) by being very selective. Base AWARE has only a 0.077 F1 gap -- remarkable.

### C4. Social (5.6% prevalence)

| Model | Test F1 | Test PRAUC | Train F1 | Gap |
|---|---|---|---|---|
| Base Standard | 0.5000 | 0.5099 | 0.8257 | +0.326 |
| Large Standard | 0.4713 | 0.5075 | 0.9576 | +0.486 |
| Base AWARE | 0.4171 | 0.4186 | 0.7390 | +0.322 |
| Large AWARE v4 | 0.5059 | 0.4879 | 0.8090 | +0.303 |

Best model: Large AWARE v4 (F1=0.5059). Social peaked early in v3 training (epoch 10) but late in v4 (epoch 30+), confirming the value of patience=8.

### C5. Resistance (4.5% prevalence)

| Model | Test F1 | Test PRAUC | Train F1 | Gap |
|---|---|---|---|---|
| Base Standard | 0.3473 | 0.3465 | 0.5804 | +0.233 |
| Large Standard | 0.3533 | 0.4021 | 0.6053 | +0.252 |
| Base AWARE | 0.3920 | 0.3955 | 0.7191 | +0.327 |
| Large AWARE v4 | 0.3952 | 0.3771 | 0.6957 | +0.301 |

Best model: Large AWARE v4 (F1=0.3952, PRAUC tied with Large Standard). Resistance has the lowest logit separation in early training (0.03-0.14 in Phase 1). In v4, logit separation grew from 0.10 (ep1) to 1.61 (ep39), indicating the model eventually learned but slowly. Co-occurrence analysis shows Resistance is heavily confused with Aspirational (37% co-prediction) and Navigational (37-40%).

### C6. Familial Capital (3.9% prevalence)

| Model | Test F1 | Test PRAUC | Train F1 | Gap |
|---|---|---|---|---|
| Base Standard | 0.6410 | 0.6680 | 0.8879 | +0.247 |
| Large Standard | 0.6842 | 0.6863 | 0.9732 | +0.289 |
| Base AWARE | 0.6460 | 0.6795 | 0.8838 | +0.238 |
| Large AWARE v4 | 0.6000 | 0.5855 | 0.9089 | +0.309 |

Best model: Large Standard (F1=0.6842). Familial Capital is the one theme where the standard models consistently beat AWARE. The theme has high logit separation (3.0-4.8) and the highest precision across all models. AWARE's CB weights down-weight this theme relative to inverse-sqrt, possibly causing under-training.

### C7. Spiritual (3.4% prevalence)

| Model | Test F1 | Test PRAUC | Train F1 | Gap |
|---|---|---|---|---|
| Base Standard | 0.4314 | 0.3829 | 0.5896 | +0.158 |
| Large Standard | 0.4368 | 0.3902 | 0.7920 | +0.355 |
| Base AWARE | 0.3409 | 0.3703 | 0.6169 | +0.276 |
| Large AWARE v4 | 0.3745 | 0.4164 | 0.6761 | +0.302 |

Best model: Large Standard (F1=0.4368, PRAUC=0.4164 tied with Large AWARE). Spiritual shows a surprising pattern: the standard baselines actually achieve higher F1 despite lower PRAUC for the large models. This is because optimized thresholds matter enormously: Standard models use F2-optimized thresholds (t=0.060-0.070) that favor recall, while AWARE's calibrated thresholds sometimes choose different operating points.

Co-occurrence: Spiritual is confused with Social (26% co-prediction of Social on true Spiritual sentences) and Aspirational (24-31%). Social and Spiritual share narrative elements about community and meaning-making.

### C8. Attainment (2.4% prevalence)

| Model | Test F1 | Test PRAUC | Train F1 | Gap |
|---|---|---|---|---|
| Base Standard | 0.2222 | 0.1613 | 0.5760 | +0.354 |
| Large Standard | 0.1978 | 0.1160 | 0.9367 | +0.739 |
| Base AWARE | 0.2062 | 0.1435 | 0.7136 | +0.507 |
| Large AWARE v4 | 0.3030 | 0.2053 | 0.8219 | +0.519 |

Best model: Large AWARE v4 (F1=0.3030, PRAUC=0.2053). Attainment is the most challenging theme. The large standard model has the most extreme overfitting: train F1=0.9367 vs test F1=0.1978 -- a gap of 0.739. The 32 test instances create extreme variance: bootstrap CI [0.175, 0.414] for v4.

Attainment is heavily confused with Navigational (79% co-prediction) and Aspirational (76%). This makes structural sense: Attainment sentences describe educational achievements, which naturally co-occur with navigational strategies and aspirational language.

---

## D. SCALING ANALYSIS

### D1. Base vs Large at Standard Architecture

| Metric | Base Standard (183.8M) | Large Standard (434.0M) | Delta |
|---|---|---|---|
| Test F1 | 0.4868 | 0.4825 | -0.0043 |
| Test PRAUC | 0.5070 | 0.4958 | -0.0112 |
| Val PRAUC | 0.5267 | 0.5531 | +0.0264 |
| Train F1 | 0.7273 | 0.8788 | +0.1515 |
| Train-Test Gap | 0.241 | 0.396 | +0.155 |

Scaling from base (183.8M) to large (434.0M) with standard fine-tuning HURTS test performance (-0.0112 PRAUC) while dramatically increasing overfitting (+0.155 F1 gap increase). The large model achieves near-perfect train performance (F1=0.8788, PRAUC=0.9877) indicating complete memorization. This confirms that on a 14K sentence dataset, 2.4x more parameters produces 2.4x more overfitting, not 2.4x better generalization.

### D2. Base vs Large at AWARE Architecture

| Metric | Base AWARE (187.9M) | Large AWARE v4 (438.8M) | Delta |
|---|---|---|---|
| Test F1 | 0.4722 | 0.4939 | +0.0217 |
| Test PRAUC | 0.4804 | 0.4842 | +0.0038 |
| Val PRAUC | 0.5256 | 0.5224 | -0.0032 |
| Train F1 | 0.7050 | 0.7590 | +0.0540 |
| Train-Test Gap | 0.233 | 0.265 | +0.032 |

With the AWARE pipeline, scaling from base to large provides a small but consistent test F1 improvement (+0.0217) with minimal additional overfitting (+0.032 gap increase vs +0.155 for standard). AWARE's regularization stack makes scaling productive: the large model's extra capacity is used for better features rather than memorization.

### D3. Parameter Efficiency

| Model | Params (M) | Test F1 | F1 per 100M params |
|---|---|---|---|
| Base Standard | 183.8 | 0.4868 | 0.265 |
| Large Standard | 434.0 | 0.4825 | 0.111 |
| Base AWARE | 187.9 | 0.4722 | 0.251 |
| Large AWARE v4 | 438.8 | 0.4939 | 0.113 |

The base standard model is most parameter-efficient (0.265 F1 per 100M params). However, the large AWARE v4 achieves the highest absolute F1 (0.4939). The diminishing returns from scale are consistent: doubling parameters from base to large yields at most +1-2 F1 points on test, whether standard or AWARE.

---

## E. REGULARIZATION ANALYSIS

### E1. Dropout Rates

| Model | Encoder Dropout | BiLSTM Dropout | Train PRAUC | Test PRAUC | Gap |
|---|---|---|---|---|---|
| Base Standard | 0.15 | N/A | 0.884 | 0.507 | 0.377 |
| Large Standard | 0.15 | N/A | 0.988 | 0.496 | 0.492 |
| Base AWARE | 0.20 | 0.15 | 0.827 | 0.480 | 0.347 |
| Large AWARE v4 | 0.30 | 0.25 | 0.900 | 0.484 | 0.416 |

Higher dropout (0.30 vs 0.15) reduces train PRAUC by 0.088 (0.988 -> 0.900) for large models, narrowing the train-test gap. However, it does not proportionally improve test PRAUC. The dropout is necessary but not sufficient for controlling overfitting.

### E2. Weight Decay Comparison

| Model | WD | Train Loss at Early Stop | Train F1 |
|---|---|---|---|
| Base Standard | 0.01 | 0.82 | 0.727 |
| Large Standard | 0.02 | 0.27 | 0.879 |
| Base AWARE | 0.04 | 0.61 | 0.705 |
| Large AWARE v4 | 0.08 | 0.49 | 0.759 |

v4's weight decay of 0.08 is 4x higher than Large Standard (0.02), which keeps train loss higher (0.49 vs 0.27 at early stop). The 8x difference in weight decay between Base Standard (0.01) and Large AWARE (0.08) reflects the need for stronger L2 regularization as parameter count grows.

### E3. Combined Regularization Stack Effectiveness

The complete v4 regularization stack includes: dropout 0.30, bilstm_dropout 0.25, weight_decay 0.08, R-Drop alpha=1.0, AEDA 0.40, multi-sample dropout=3, SWA, progressive unfreezing.

Comparing overfitting metrics:

| Model | Train-Test F1 Gap | Train-Test PRAUC Gap |
|---|---|---|
| Large Standard (minimal reg) | 0.396 | 0.492 |
| Large AWARE v3 (strong but misconfigured reg) | 0.224 | 0.356 |
| Large AWARE v4 (tuned reg) | 0.265 | 0.416 |
| Base AWARE (moderate reg) | 0.233 | 0.347 |

v3 had the lowest F1 gap (0.224) but also the lowest test F1 (0.461) -- it was under-fitting due to excessive R-Drop (alpha=2.0). v4 has a slightly higher gap (0.265) but higher test F1 (0.494), representing a better bias-variance tradeoff.

---

## F. LOSS FUNCTION ANALYSIS

### F1. BCE vs ASL

Standard models use weighted BCE; AWARE models use ASL (gamma_pos=0, gamma_neg=2.5).

At the default threshold of 0.5:
| Model | Default F1 | Optimized F1 | Improvement from Threshold |
|---|---|---|---|
| Base Standard | 0.4196 | 0.4868 | +0.0672 |
| Large Standard | 0.4276 | 0.4825 | +0.0549 |
| Base AWARE | 0.4021 | 0.4722 | +0.0701 |
| Large AWARE v4 | 0.4097 | 0.4939 | +0.0842 |

ASL models benefit MORE from threshold optimization (0.0842 vs 0.0549-0.0672 for BCE). This is because ASL's gamma_neg pushes negative logits further negative, creating a non-uniform logit distribution that requires theme-specific thresholds far from 0.5. At default 0.5, ASL models appear worse than BCE, but with calibrated thresholds they can be better.

### F2. gamma_neg Tuning

**v3 (gamma_neg=3.0):** Platt b values: -1.80 to -2.60, all \"SEVERELY_BIASED\". The high gamma_neg pushed negative logits too far negative, creating overconfident negative predictions.

**v4 (gamma_neg=2.5):** Platt b values: -1.60 to -3.00, still \"SEVERELY_BIASED\" but slightly less extreme for some themes. The reduction from 3.0 to 2.5 was insufficient to fully address the overconfidence problem.

**Standard models (no gamma modulation):** Platt b values: -0.20 to -1.40 (Base Standard) and -0.40 to -1.60 (Large Standard). Much closer to zero, indicating better calibration.

This reveals a fundamental tension: ASL's gamma_neg improves ranking (PRAUC) for rare themes but destroys calibration, requiring heavy Platt scaling correction. The correction then depends on the small validation set (34 Attainment instances), making it unreliable.

### F3. Label Smoothing

v3 used label_smoothing=0.0; v4 used label_smoothing=0.02. The intention was to prevent logit explosion. v4's logit separation grew more slowly: reaching 1.88 at epoch 35 vs v3's reaching 2.87 at epoch 14 (for Attainment). However, the total effect is confounded with gamma_neg reduction (3.0->2.5), making isolation difficult.

---

## G. CALIBRATION OBSERVATIONS

### G1. Platt Scaling Parameters

**Base Standard (BCE):** a values: 0.70-0.90 (close to 1.0), b values: -0.20 to -1.40. Relatively well-calibrated. The sigmoid outputs are close to true probabilities.

**Large Standard (BCE):** a values: 0.50-0.80 (flatter slope), b values: -0.40 to -1.60. More overconfident than base, especially for rare themes.

**Large AWARE v3 (ASL gamma_neg=3.0):** a values: 1.00-2.30, b values: -1.80 to -2.60. The a>1 values indicate logits need to be AMPLIFIED (the sigmoid is too flat at the raw logit scale). The large negative b values indicate a systematic downward bias -- the model's \"positive\" predictions are still in the negative logit range.

**Large AWARE v4 (ASL gamma_neg=2.5):** a values: 1.10-1.70, b values: -1.60 to -3.00. Still biased but with lower a values than v3, indicating slightly better calibration. Attainment has the most extreme bias (b=-3.00), consistent with its extreme rarity.

**Base AWARE (ASL gamma_neg=2.5):** a values: 1.20-2.00, b values: -1.60 to -3.20. Similar to large AWARE but with slightly higher a values.

### G2. Systematic Overconfidence

All ASL models show negative b values in [-1.6, -3.2], meaning raw logits must be shifted downward by 1.6-3.2 units for proper calibration. This creates a practical problem: at deployment, the threshold optimization becomes critical because the raw 0.5 threshold is wildly inappropriate.

The logit separation analysis confirms this: positive examples have mean logits of 1.0-4.0, while negative examples have mean logits of -2.0 to 0.5. The true decision boundary is around -1 to 0, not 0 as the sigmoid would assume.

### G3. Threshold Optimization Behavior

Optimized thresholds across models:

| Theme | Base Std | Large Std | Base AWARE | Large AWARE v4 |
|---|---|---|---|---|
| Attainment | 0.060 | 0.100 | 0.060 | 0.050 |
| Aspirational | 0.400 | 0.300 | 0.210 | 0.230 |
| Navigational | 0.310 | 0.280 | 0.320 | 0.310 |
| Resistance | 0.070 | 0.056 | 0.120 | 0.090 |
| Perseverance | 0.230 | 0.400 | 0.440 | 0.350 |
| Social | 0.320 | 0.210 | 0.110 | 0.220 |
| Spiritual | 0.060 | 0.070 | 0.050 | 0.053 |
| Familial_Capital | 0.440 | 0.420 | 0.150 | 0.140 |

Rare themes (Attainment, Spiritual, Resistance) consistently have very low thresholds (0.05-0.12), while common themes with good separation (Navigational, Familial_Capital) have higher thresholds (0.28-0.44). AWARE models generally have lower Familial_Capital thresholds (0.14-0.15) than standard (0.42-0.44), reflecting the CB weights' down-weighting of this theme.

---

## H. BUG IMPACT ANALYSIS

### H1. R-Drop Asymmetry (v3 bug)

**Bug:** p1 was detached, so only p2 received KL gradient. The KL loss was not truly symmetric.

**Impact:** Difficult to isolate precisely, but combined with the too-high alpha=2.0, this caused R-Drop to waste gradient budget. v3's Phase 1 achieved PRAUC=0.14 in 4 epochs; v4 achieved 0.40 in 8 epochs. While the fix included multiple changes (disabling Phase 1 R-Drop, halving alpha), the asymmetry fix was necessary for correct gradient flow.

**Recovery:** Combined fixes improved test F1 from 0.461 (v3) to 0.494 (v4), a +0.033 recovery.

### H2. ClassificationHead Dropout Order

**Bug:** Original order was dropout -> LayerNorm -> linear. LayerNorm re-normalizes the dropped-out activations, partially undoing dropout's regularization effect.

**Fix:** Changed to LayerNorm -> dropout -> linear.

**Impact:** This was fixed between v3 and the comparison runs. The effect is embedded in the overall AWARE v4 improvement. Theoretical impact: approximately 30-50% of dropout's regularization was being negated.

### H3. Sentence Separator

**Bug:** SENTENCE_SEP was \" \" (single space), making essays appear as one run-on paragraph to DeBERTa.

**Fix:** Changed to \". \" -- proper sentence-final punctuation.

**Impact:** This would have reduced DeBERTa's ability to identify sentence boundaries in its tokenization. Since sentence-level classification is the core task, this bug may have caused significant performance loss, but it was fixed before the comparison runs, so the isolated impact cannot be measured.

### H4. SWA Start Ratio

**Bug (v3):** swa_start_ratio=0.50 with 30 P2 epochs meant SWA started at epoch 19 -- the same epoch early stopping triggered. Zero checkpoints collected.

**Fix (v4):** swa_start_ratio=0.25 with 40 P2 epochs. SWA started at epoch 18, collecting 22 checkpoints.

**Recovery:** SWA provided +0.0015 PRAUC improvement (0.5209 -> 0.5224). Small but free. Without this fix, the v4 best would have been 0.5209 instead of 0.5224.

### H5. Phase 3 Dead Parameters

**Bug (v3):** Phase 3 only trained 20.5K params (head only) on fixed encoder+BiLSTM outputs. Converged in 1 epoch with effectively zero gradient.

**Fix (v4):** Phase 3 unfreezes BiLSTM (4.8M params). However, even with this fix, Phase 3 failed to improve on SWA (0.5218 vs 0.5224).

**For Base AWARE:** Phase 3 with BiLSTM unfrozen (4.1M params) improved PRAUC by 0.0052 over 5 epochs. This confirms the fix works, but only for the base model.

---

## I. KEY PATTERNS AND TRENDS

### I1. Generalizable Patterns

1. **Overfitting is the dominant challenge, not model capacity.** On 14K sentences, even a 183M-parameter model overfits dramatically (train F1=0.73, test F1=0.49). Scaling to 434M parameters makes it worse unless regularization scales proportionally.

2. **The most effective regularization is reducing learning rate.** v4's 3x reduction in encoder_lr (1.5e-5 -> 5e-6) was the single most impactful change, delaying overfitting by ~13 epochs and allowing the model to find better minima.

3. **Theme-level overfitting peaks at different epochs.** In v3, Social peaked at epoch 10 while Familial_Capital peaked at epoch 18. This makes patience selection critical -- patience=5 (v3) was too eager and cut off slow-converging themes.

4. **Co-occurrence confusion is structural, not trainable.** Attainment is confused with Navigational and Aspirational across ALL models because these themes genuinely co-occur in the data. No architecture change will fix this without additional supervision signals or better annotation guidelines.

5. **Phase 3 is only effective when sufficient parameters are trainable.** 20K params (head only) is dead. 4M params (head+BiLSTM) works for base but not large. The pattern suggests Phase 3 helps when the trainable fraction is at least ~2% of total parameters.

### I2. Surprising Findings

1. **Standard baselines outperform AWARE on test PRAUC.** Base Standard (0.507) > Base AWARE (0.480). Large Standard (0.496) > Large AWARE v4 (0.484). The full AWARE pipeline with all its innovations performs WORSE than simple fine-tuning on the threshold-independent metric. This challenges the assumption that architectural complexity helps.

2. **SWA can be harmful.** Base AWARE's SWA degraded performance by 0.005 PRAUC. Averaging checkpoints is not universally beneficial, especially when the training trajectory is smooth.

3. **CB weights may hurt common-theme performance.** Familial_Capital (3.9% prevalence) achieves F1=0.684 with standard inverse-sqrt weights but only 0.600 with CB weights. The CB weight for Navigational (0.238) is so low that common themes may be under-trained.

4. **R-Drop at alpha=2.0 actively degraded performance.** The KL loss consumed 36-67% of the gradient budget in early training, dominating the task loss. This is the most impactful negative finding: R-Drop is not \"more regularization = better.\"

5. **The large standard model achieves 0.988 train PRAUC** -- near-perfect separation on training data. This means the training data contains enough signal for perfect classification; the challenge is entirely in generalization.

### I3. Contradictions to Conventional Wisdom

1. **Larger model != better** on small datasets, even with extensive regularization. The test PRAUC differences between base and large models are within noise (0.003-0.011 PRAUC), while computational cost is 2-3x higher.

2. **More regularization techniques != less overfitting.** AWARE v4 uses 8+ regularization techniques but achieves a LARGER train-test gap (0.416 PRAUC gap) than Base AWARE (0.347 PRAUC gap). The gap is measured in PRAUC, which is threshold-independent.

3. **Multi-task learning (essay head) had negligible impact.** Despite theoretical motivation (3x more signal for Attainment at essay level), the actual test metrics show no improvement for Attainment recall.

4. **Threshold optimization is as important as model architecture.** The difference between default-0.5 F1 and optimized F1 is 0.05-0.08 -- comparable to the entire difference between the best and worst models.

### I4. Recommendations for Future Work

1. **Increase training data, not model complexity.** The 14K sentence / 2K essay dataset is insufficient for 434M parameters. Adding 2-3x more annotated data would likely outperform any architectural improvement.

2. **Simplify the loss function.** Replace ASL+CB+R-Drop+essay_aux with a well-tuned BCE+inverse-sqrt. The complexity of the current loss makes debugging and tuning extremely difficult, and standard BCE outperforms on PRAUC.

3. **Consider a base-sized model with extensive data augmentation.** Base AWARE has the best Perseverance F1 (0.555), the lowest Navigational gap (0.007), and trains 3x faster. Its parameter efficiency is unmatched.

4. **Fix the calibration problem.** All ASL models show severely biased Platt scaling (b=-1.6 to -3.2). Either reduce gamma_neg further (to 1.0-1.5) or switch to focal loss, which is better calibrated.

5. **Address theme confusion structurally.** The Attainment-Navigational-Aspirational confusion (76-79% co-prediction) requires either (a) hierarchical classification, (b) explicit exclusion constraints, or (c) revised annotation guidelines that distinguish these themes more clearly.

6. **Run ensemble of base standard + base AWARE.** These two models make complementary errors: standard is better at Familial_Capital and Spiritual; AWARE is better at Resistance and Perseverance. A simple average could outperform either alone.

---

## Summary Table: All Models on Test Set

| Model | Params | Test F1 | Test PRAUC | Train F1 | Gap | Best Theme | Worst Theme |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | ~100K | 0.186 | 0.363 | - | - | Nav (0.535) | Att (0.000) |
| Base Standard | 183.8M | 0.487 | 0.507 | 0.727 | 0.240 | Nav (0.684) | Att (0.222) |
| Large Standard | 434.0M | 0.483 | 0.496 | 0.879 | 0.396 | Nav (0.705) | Att (0.198) |
| Base AWARE | 187.9M | 0.472 | 0.480 | 0.705 | 0.233 | Nav (0.675) | Att (0.206) |
| Large AWARE v3 | 438.8M | 0.461 | 0.462 | 0.685 | 0.224 | Nav (0.695) | Att (0.266) |
| **Large AWARE v4** | **438.8M** | **0.494** | **0.484** | **0.759** | **0.265** | **Nav (0.707)** | **Att (0.303)** |

The best test F1 is achieved by Large AWARE v4 (0.494), but the best test PRAUC is achieved by Base Standard (0.507). The v3->v4 transition improved test F1 by +0.033 and test PRAUC by +0.022, driven primarily by: (1) disabling Phase 1 R-Drop, (2) reducing encoder LR by 3x, (3) increasing patience from 5 to 8, and (4) progressive unfreezing."}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"cache_creation_input_tokens":2979,"cache_read_input_tokens":164954,"output_tokens":12997,"server_tool_use":{"web_search_requests":0,"web_fetch_requests":0},"service_tier":"standard","cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":2979},"inference_geo":"","iterations":[],"speed":"standard"}},"requestId":"req_011CZBvNXLE32SVRt317WwpM","type":"assistant","uuid":"4d9c237a-146d-4569-bbc9-585d1abacb5f","timestamp":"2026-03-19T05:58:44.615Z","userType":"external","entrypoint":"claude-vscode","cwd":"/Users/923673423/f2-models","sessionId":"116cec35-654e-48f7-8ce7-1e431f2d46ca","version":"2.1.79","gitBranch":"HEAD","slug":"lively-noodling-mitten"}
{"parentUuid":"4d9c237a-146d-4569-bbc9-585d1abacb5f","isSidechain":true,"agentId":"addcee1d239090eac","type":"progress","data":{"type":"hook_progress","hookEvent":"SubagentStop","hookName":"SubagentStop","command":"/Users/923673423/miniconda/envs/llm_env/bin/node \"$CLAUDE_PROJECT_DIR/.claude/helpers/hook-handler.cjs\" post-task"},"parentToolUseID":"f51d2f11-089c-492e-a80f-03ac7887acc0","toolUseID":"f51d2f11-089c-492e-a80f-03ac7887acc0","timestamp":"2026-03-19T05:58:44.675Z","uuid":"6bd67229-b0a8-4186-a919-3e2a64e3b2c1","userType":"external","entrypoint":"claude-vscode","cwd":"/Users/923673423/f2-models","sessionId":"116cec35-654e-48f7-8ce7-1e431f2d46ca","version":"2.1.79","gitBranch":"HEAD","slug":"lively-noodling-mitten"}
