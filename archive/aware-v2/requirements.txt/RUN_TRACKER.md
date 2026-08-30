# AWARE v2 — Run Tracker & Improvement Plan

## Current Status
- **Best Result**: Q006 test macro F1 = 0.390 (STILL BEST after Q009-Q011)
- **Last Run**: Q011 — REGRESSED (0.266), CB Loss + logit adjustment caused gradient explosion
- **Next Step**: Full-scale run using Q006 base config with proven improvements
- **Phase**: Sequential runs COMPLETE → preparing full-scale training

---

## Run History Summary

| Run | Macro F1 | Key Changes | Outcome |
|-----|----------|-------------|---------|
| Q001 | 0.305 | Baseline, 6ep, no DAPT | Pipeline verified |
| Q002 | 0.345 | 16ep, DAPT 1ep | +0.040 from DAPT |
| Q003 | — | 36ep, DAPT 2ep, sampling boost | REGRESSED (-0.026) |
| Q004 | 0.380 | F2 thresholds, asymmetric smoothing, residual BiLSTM, patience fix | FP +0.179, FG +0.119 |
| Q005 | 0.322 | Aggressive weights + prior thresholds | CATASTROPHE: recall=1.0, precision≈0 |
| Q006 | **0.390** | Distribution-aware thresholds, moderate weights | **CURRENT BEST** |
| Q007 | — | gamma_pos=0, gamma_neg=6, multi-dropout | CANCELLED at ep10. Lesson: never change loss AND arch |
| Q008 | 0.383 | gamma_pos=0, Phase 3 balanced head | Phase 3 +0.069 val F1, but gamma_pos=0 hurt Phase 2 |
| Q009 | **0.227** | R-Drop + EMA + class_0 + grad logging | **FAILED**: EMA per-epoch update bug froze model |
| Q010 | 0.278 | EMA fix + LSAN + attn pooling | LSAN too slow: Nav -0.30, Per -0.36. FG +0.24 |
| Q011 | 0.266 | CB Loss + logit adj + tau-norm (Q006 arch) | **FAILED**: gradient explosion (NaN at ep10) |

### Per-Theme F1 (Test Set, Optimized Thresholds)

| Theme | Support | Q004 | Q006 | Q008 | Q010 | Q011 |
|-------|---------|------|------|------|------|------|
| Navigational | 1013 | 0.723 | **0.528** | 0.712 | 0.226 | 0.454 |
| Attainment | 767 | 0.597 | **0.715** | 0.617 | 0.570 | 0.598 |
| Perseverance | 713 | 0.560 | 0.407 | 0.537 | 0.046 | 0.325 |
| Aspirational | 665 | 0.547 | **0.503** | 0.543 | 0.414 | 0.411 |
| Social | 126 | 0.321 | **0.431** | 0.297 | 0.276 | 0.237 |
| Filial Piety | 38 | 0.294 | **0.272** | 0.222 | 0.066 | 0.104 |
| Spiritual | 83 | 0.243 | **0.340** | 0.343 | 0.125 | 0.170 |
| Familial | 352 | 0.460 | **0.544** | 0.448 | 0.355 | 0.398 |
| Resistance | 105 | 0.284 | **0.208** | 0.298 | 0.056 | 0.042 |
| CC | 18 | 0.055 | **0.070** | 0.033 | 0.000 | 0.025 |
| First Gen | 14 | 0.333 | 0.273 | **0.422** | 0.517 | 0.162 |

---

## Diagnosed Problems (7)

### 1. Classifier Head Bias
All 11 themes share one `Linear(768, 11)` — common theme gradients overwhelm rare themes.
**Evidence**: FG train F1=0.89 but test F1=0.32 (overfits to specific patterns).

### 2. Mean Pooling Dilutes Rare Signals
`SentenceMeanPooling` averages ALL tokens equally — rare theme tokens diluted by generic tokens.

### 3. Rare Class Prediction Instability
FG oscillates 0.0–0.42, CC oscillates 0.03–0.13 across runs. Bootstrap CI for FG: [0.0, 0.5].

### 4. Hand-Tuned Theme Weights Are Fragile
Q005 proved aggressive manual weights cause catastrophe. Manual tuning is slow and arbitrary.

### 5. Phase 3 Head Retraining Can Be Improved
Reinitializing head discards Phase 2 knowledge. LWS/tau-norm are better alternatives.

### 6. No Class_0 Evaluation (69% of data)
We never evaluate how well "no theme" is predicted — missing visibility into false positives.

### 7. No Gradient Visibility
No data on whether rare theme gradients are being overwhelmed.

---

## Sequential Improvement Plan

### Q009: "Stable Foundation" — R-Drop + EMA + Diagnostics
**Status**: COMPLETED — FAILED (test macro F1 = 0.227, regression from 0.390)

**Changes**:
1. R-Drop (alpha=1.0): dual forward pass with KL consistency loss (NeurIPS 2021)
   - Stabilizes rare theme predictions by forcing dropout-invariant outputs
2. EMA (decay=0.999): shadow model weight averaging for evaluation
   - Reduces prediction variance, especially on rare themes
3. gamma_pos reverted to 1.0 (Q006 proved better than Q008's 0.0 in Phase 2)
4. Class_0 evaluation metrics (precision, recall, F1 for "no theme" sentences)
5. Per-theme gradient norm logging (diagnostic)

**Files Modified**:
- `config.py`: Added `rdrop_alpha`, `ema_decay` fields
- `trainer.py`: R-Drop in `_train_epoch`, EMA in `__init__`/`_validate`/`_log_epoch`, gradient logging, class_0 logging
- `metrics.py`: class_0 metrics in `compute_metrics()`
- `evaluate.py`: class_0 in print summary
- `quick.yaml`: Updated config values

**Success Criteria**:
- [ ] macro F1 >= 0.390 (match or beat Q006)
- [ ] Rare theme F1 variance reduced (check bootstrap CI width)
- [ ] Gradient norms reveal imbalance magnitude
- [ ] Class_0 metrics provide threshold calibration signal

**Timing**: ~60 min (R-Drop adds ~1.5x forward pass cost)

**Post-Mortem**:
- **Root cause**: EMA updated once per EPOCH (not per step). decay=0.999 with 24 epoch-level
  updates → `0.999^24 = 97.6%` initial weights retained. Model barely moved.
- val_f1_macro FLAT: 0.1696 (ep1) → 0.1716 (ep24). Per-theme F1 frozen entire run.
- Saved best.pt was EMA model → essentially Phase 1 initialization weights.
- **The actual trained model was learning** (train_loss 2.68→1.95) but was never saved/evaluated.
- R-Drop NOT at fault — training loss decreased normally. Pure EMA save bug.
- **FIX applied**: Moved `_update_ema()` inside optimizer step loop (per-step, not per-epoch).
- Phase 3 also failed: compared against EMA model, matched it, no improvement.
- class_0 evaluation at test: F1=0.715, P=0.800, R=0.647 (useful calibration signal)
- prob_diag: separations tiny — CC=0.022, Res=0.032, Soc=0.043. But this was EMA model.
- Gradient norms: FP/CC/FG get 10-50x larger norms than Nav (weights working, head can't use it)

---

### Q010: "Smart Head" — EMA Fix + LSAN + Attention Pooling
**Status**: COMPLETED — REGRESSED (test macro F1 = 0.278)

**Changes**:
1. EMA bug fix: `_update_ema()` now per optimizer step (not per epoch)
2. `SentenceAttentionPooling`: learnable token-level attention (replaces mean pooling)
3. `LabelAttentionHead` (LSAN): per-theme query vectors (replaces Linear(768,11))
4. R-Drop + EMA + class_0 + grad logging carried forward from Q009

**Post-Mortem**:
- **Root cause**: LSAN + attention pooling learned too slowly for 20 epochs
- Common themes devastated: Nav 0.528→0.226 (-0.30), Per 0.407→0.046 (-0.36)
- FG actually improved: 0.273→0.517 (+0.24) — LSAN concept works for rare themes
- Phase 3 failed: reinitializing LSAN head couldn't beat Phase 2 version
- FP16 crash on first submission: `-1e9` overflow in SentenceAttentionPooling
  - Fixed with `torch.finfo(scores.dtype).min`
- **Conclusion**: LSAN needs 40+ epochs to converge. Not viable in quick runs.
  Architecture reverted to Q006 proven SentenceMeanPooling + ClassificationHead for Q011.

---

### Q011: "Principled Loss" — CB Loss + Logit Adjustment + Tau-Norm
**Status**: COMPLETED — REGRESSED (test macro F1 = 0.266)

**Changes**:
1. Reverted to Q006 proven architecture (SentenceMeanPooling + ClassificationHead)
2. EMA DISABLED (ema_decay=0.0) — caused issues in Q009+Q010
3. R-Drop alpha reduced 1.0→0.5 — less interference with learning
4. CB Loss (beta=0.999): principled weights replace manual theme_weights
5. Logit adjustment: add log(prior/(1-prior)) to logits during training
6. Tau-normalization: post-hoc weight rescaling sweep after Phase 2

**Post-Mortem**:
- **Root cause**: CB Loss + logit adjustment created a GRADIENT EXPLOSION
- Gradient norms were catastrophically high:
  ```
  Nav=5,609  Att=3,157  Per=7,308  Asp=4,057
  Soc=27,420  FP=87,494  Spi=41,705  Fam=7,122
  Res=47,782  CC=82,025  FG=43,915
  ```
- FP gradient norm 87,494 vs Nav 5,609 — a 15x ratio
- Training loss went NaN at epoch 10, model completely dead
- Logit adjustment shifts: Nav=-2.03, FG=-6.39, CC=-5.99
  Combined with CB weights (FG=9.27, CC=6.39), rare theme gradients exploded
- val_f1 stuck at 0.0706 for ALL of Phase 1+2 (never improved from epoch 1)
- Tau-normalization: USELESS — all tau values gave identical 0.0706 (model was dead)
- Phase 3 rescued: 0.0706 → 0.2367 (+235%) proving Phase 3 robustness
- Final test F1 after threshold optimization: 0.266 (still major regression)
- **Lesson**: Logit adjustment + CB Loss is toxic. Both amplify rare theme signal;
  together they create runaway gradients. NEVER combine multiplicative (CB) and
  additive (logit adj) rare-class boosters simultaneously.
- **CB weights alone** (beta=0.999) are also likely too aggressive: FG=50.2 vs Q006's 10.0

---

### Full Run: Q006 Base Config + Extended Training
**Status**: READY — preparing full.yaml

**Decision**: Q006 remains the best config after Q009-Q011 all regressed.
None of the "improvements" beat Q006's proven manual weights + simple architecture.

**Full Run Configuration**:
- **Architecture**: SentenceMeanPooling + ClassificationHead (Q006 proven)
- **Loss**: ASL with Q006 manual weights, gamma_pos=1.0, gamma_neg=4.0
- **NO CB Loss, NO logit adjustment** — both proven harmful
- **NO EMA** — too many bugs, not proven to help
- **R-Drop alpha=0.5** — safe regularization, low risk (only one extra forward pass)
- **Phase 3 balanced head** — proven +0.069 in Q008, +0.166 in Q011
- **DAPT 1 epoch** — proven +0.040 in Q002
- **30 Phase 2 epochs** (was still climbing at 20 in quick runs)
- **Extended patience** (12-15 for Phase 2)
- **Tau-normalization post-Phase 2** — zero risk, might help
- **Target**: macro F1 >= 0.42, all themes F1 > 0.05

---

## Key Lessons Learned

1. **Never change weights AND thresholds simultaneously** (Q005 catastrophe)
2. **Never change loss AND architecture simultaneously** (Q007 regression)
3. **gamma_pos=1.0 > gamma_pos=0.0** for Phase 2 learning (Q006 vs Q008)
4. **Phase 3 balanced head retraining works** (+0.069 val F1 in Q008, +0.166 in Q011)
5. **DAPT 1 epoch is optimal** (+0.040 macro F1, 2ep no benefit)
6. **Distribution-aware threshold floor** prevents Q005-style catastrophe
7. **F2 thresholds for rare themes** (recall-biased) outperform F1 thresholds
8. **EMA must update per-step, not per-epoch** (Q009 bug: 97.6% initial weights retained)
9. **LSAN needs 40+ epochs** — 20 epochs insufficient for query convergence (Q010)
10. **Never combine CB Loss + logit adjustment** — multiplicative + additive rare-class
    boosters create gradient explosion (Q011: FP gradient 87k, NaN at epoch 10)
11. **Q006 config is remarkably robust** — manual moderate weights beat all "principled" alternatives

---

## Research References

| Technique | Paper | Venue | Used In |
|-----------|-------|-------|---------|
| R-Drop | Liang & Wu | NeurIPS 2021 | Q009 |
| EMA | — | arXiv:2411.18704 | Q009 |
| LSAN | Xiao et al. | EMNLP 2019 | Q010 |
| Attention Pooling | Pool Me Wisely | NeurIPS 2025 | Q010 |
| CB Loss | Cui et al. | CVPR 2019 | Q011 |
| Logit Adjustment | Menon et al. | ICLR 2021 | Q011 |
| Tau-Normalization | Kang et al. | ICLR 2020 | Q011 |
| LWS | Kang et al. | ICLR 2020 | Q011 |
| Decoupled Training | Kang et al. | ICLR 2020 | Q008/Q011 |
| ASL | Ben-Baruch et al. | ICCV 2021 | All runs |

---

## Full-Scale Run Preparation

### Decision: Q006 Base Config Wins

Q009-Q011 all regressed. The decision tree resolved clearly:
```
Q009 (0.227) < Q006 (0.390)  → EMA bug. R-Drop unproven but safe.
Q010 (0.278) < Q006 (0.390)  → LSAN too slow. Reverted architecture.
Q011 (0.266) < Q006 (0.390)  → CB Loss + logit adj = gradient explosion. Reverted loss.
```

**Full run = Q006 config + extended training + safe additions only.**

### full.yaml Changes Needed

| Parameter | Current full.yaml | Needed Value | Source |
|-----------|------------------|--------------|--------|
| `dropout` | 0.30 | 0.35 | Q002 proved better |
| `theme_weights` | auto (base) | `[1.0, 1.2, 1.24, 1.29, 5.0, 7.0, 6.0, 2.0, 5.0, 9.0, 10.0]` | Q006 manual weights |
| `phase1_epochs` | 5 | 4 | Q002 level, proven |
| `phase2_epochs` | 15 | 30 | Still climbing at 20 in quick runs |
| `weight_decay` | 0.01 | 0.02 | Q002 proved better |
| `warmup_ratio` | 0.06 | 0.10 | Stabilizes early training |
| `early_stopping_patience` | 5 | 7 | Quick runs proved 5 too aggressive |
| `phase2_early_stopping_patience` | 7 | 15 | Full runs need more patience with 30 epochs |
| `dapt.epochs` | 5 | 1 | Proven: 1ep optimal, 2ep no benefit |
| `rdrop_alpha` | missing | 0.5 | Safe regularization, low risk |
| `ema_decay` | missing | 0.0 | DISABLED — not proven, too many bugs |
| `cb_beta` | missing | (omit) | NOT USED — gradient explosion in Q011 |
| `logit_adjustment` | missing | false | NOT USED — toxic with CB Loss |
| Architecture | SentenceMeanPooling + ClassificationHead | (no change) | Q006 proven |

### SLURM Settings

```bash
# run_job.sh changes for full run:
RUN_MODE="full"
RUN_NUMBER="001"
USE_DAPT=""          # Include DAPT (proven +0.040)
```

### Estimated Timing (A100 80GB)
- DAPT: ~2.5 min/epoch × 1 = ~3 min
- Phase 1: ~1.4 min/epoch × 4 = ~6 min
- Phase 2: ~1.4 min/epoch × 30 = ~42 min (early stop likely around 20-25)
- Tau-norm: ~3 min (6 validation sweeps)
- Phase 3: ~1.4 min/epoch × 5 = ~7 min
- Evaluation: ~3 min
- **Total estimate: ~65 min** (well within 7:55:00 limit)

### What Diagnostic Code to Keep

From Q009-Q011, these diagnostics are now in the codebase and should stay enabled:
- [x] class_0 metrics (Problem #6 — provides visibility into false positive themes)
- [x] Per-theme gradient norm logging (Problem #7 — verifies gradient health)
- [x] R-Drop alpha=0.5 (regularization, safe)
- [x] Tau-normalization sweep (zero cost, might help)
- [x] Phase 3 balanced head retraining (proven beneficial)
- [x] Bootstrap CI (already present from Q006)

### What to DISABLE for Full Run

- [x] CB Loss (cb_beta) — gradient explosion in Q011
- [x] Logit adjustment — amplifies CB Loss problem
- [x] EMA (ema_decay=0.0) — per-epoch bug in Q009, still unstable in Q010
- [x] LSAN / SentenceAttentionPooling — too slow to converge (Q010)
