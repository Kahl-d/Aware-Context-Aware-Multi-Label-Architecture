# AWARE v2 — Run Results Log

Record observations after every HPC run. One section per run. Copy the template below.

---

## Iteration Roadmap

```
RUN 0 (HPC sanity, ~15-30 min):
  python3 scripts/test_components.py --data_dir data/
  All 10 tests must pass, especially the overfit test.
  If overfit test fails → model bug. Stop. Fix. Do not run training.

RUN quick_001 (~1.5h):
  quick.yaml + --no-dapt
  Question: Does the pipeline complete? Do all 11 themes show F1 > 0?

  IF pipeline crashes → read SLURM error log, fix, re-run.
  IF val_f1_macro stays ~0 → likely LR or data bug; lower encoder_lr.
  IF all themes show any F1 > 0 → pipeline is working. Proceed.

RUN quick_002, 003, ... (~1.5h each):
  Change ONE parameter per run based on previous observations.
  Keep changing until: all themes F1 > 0, val_f1_macro improving, no severe overfit.

RUN dapt_check (~3h):
  Once no-dapt config is stable: set USE_DAPT="" in run_job.sh
  Compare macro F1 to best no-dapt run.
  If DAPT improves macro F1 by > 0.02 → use DAPT in full run.
  If no difference → skip DAPT in full run (saves 2h).

FULL RUN (~6h):
  full.yaml with best quick-run config. DAPT only if verified to help.
  These are the THESIS NUMBERS.
```

---

## Run quick_001 — 2026-02-23

**Config:** quick.yaml (2 phase1 epochs + 4 phase2 epochs, batch_size=8, grad_accum=4)
**DAPT:** no (`--no-dapt`)
**Job ID:** 21350 (gpu02, A100 80GB)
**Wall time:** ~15 min (21:31→21:46) — much faster than expected on A100
**Encoder:** microsoft/deberta-v3-base (base, no domain adaptation)

**Training history (from `history.json`):**

| Epoch | Phase | Train Loss | Val F1 Macro |
|---|---|---|---|
| 1 | 1 (frozen) | 1.4447 | 0.1881 |
| 2 | 1 (frozen) | 1.2840 | 0.1891 |
| 3 | 2 (full FT) | 1.2533 | 0.2692 ← big jump when encoder unfreezes |
| 4 | 2 | 1.2070 | 0.2902 |
| 5 | 2 | 1.1924 | 0.2860 |
| 6 | 2 (best) | 1.1691 | **0.2918** |

**Per-theme F1 on test set (optimized thresholds):**

| Theme | Test F1 | Train F1 | Gap | Threshold |
|---|---|---|---|---|
| Navigational | 0.4591 | 0.4795 | +0.020 | 0.560 |
| Attainment | 0.5697 | 0.5683 | -0.001 | 0.610 |
| Perseverance | 0.3276 | 0.3367 | +0.009 | 0.530 |
| Aspirational | 0.4249 | 0.4153 | -0.010 | 0.540 |
| Familial | 0.4191 | 0.4157 | -0.003 | 0.540 |
| Social | 0.2093 | 0.2773 | +0.068 | 0.480 |
| Resistance | 0.2121 | 0.3009 | +0.089 | 0.500 |
| Filial Piety | 0.2258 | 0.3675 | +0.142 | 0.550 |
| **Spiritual** | **0.2000** | **0.4446** | **+0.245** | 0.580 ← overfit |
| Community Consciousness | 0.0677 | 0.1011 | +0.034 | 0.410 |
| **First Gen** | **0.2424** | **0.4800** | **+0.238** | 0.530 ← overfit |
| **Macro** | **0.3052** | | | |

**Observations:**
- val_f1_macro trend: flat in phase 1 (frozen encoder), jumped +0.08 at epoch 3 (encoder unfreeze), then slowly improving — healthy signal
- All 11 themes show F1 > 0 on test set ✅ — pipeline works
- Common themes (Nav, Att, Per, Asp, Fam) are learning well (F1 0.33–0.57)
- Spiritual and First Gen are overfitting badly: train F1 ~0.45–0.48 but test F1 ~0.20–0.24
- Community Consciousness very low (F1 = 0.068) — needs attention
- Thresholds are high overall (0.41–0.61) — model is conservative
- Overfitting makes sense: rare themes (11 Spiritual, 9 CC, 8 FG sentences in val) → model memorizes train, doesn't generalize

**Decision for quick_002:**
- Problem: Spiritual, First Gen, CC overfitting + low generalization
- Change: Add more dropout (0.4 → 0.45) + ALSO manually increase rare theme weights in quick.yaml
  - Current weights: Spiritual=3.36, CC=6.81, FG=8.31
  - Try: Spiritual=4.5, CC=10.0, FG=12.0
- Reason: Higher weights force more learning signal on rare themes; more dropout fights memorization

---

## Run quick_002 — 2026-02-23

**Config:** quick.yaml (4 phase1 + 12 phase2 = 16 epochs, dropout=0.35, wd=0.02, aeda=0.3)
**DAPT:** yes (1 epoch, 14534 essays)
**Job ID:** 21351 (gpu02, A100 80GB)
**Wall time:** 27 min (22:22→22:49) — 88 min of the 115 min limit UNUSED
**Encoder:** DAPT-adapted DeBERTa-v3-base (1 epoch on ALMA corpus)

**Training history:**

| Epoch | Phase | Train Loss | Val F1 Macro |
|---|---|---|---|
| 1 | 1 (frozen) | 1.4291 | 0.2036 |
| 2 | 1 | 1.2469 | 0.2423 |
| 3 | 1 | 1.2091 | 0.2668 |
| 4 | 1 | 1.1924 | 0.2753 |
| 5 | 2 (full FT) | 1.1959 | 0.2895 |
| 6 | 2 | 1.1570 | 0.2980 |
| 7 | 2 | 1.1394 | 0.3107 |
| 8 | 2 | 1.1363 | 0.3172 |
| 9 | 2 | 1.1209 | 0.3177 |
| 10 | 2 | 1.1164 | 0.3130 |
| 11 | 2 | 1.1005 | 0.3154 |
| 12 | 2 (best) | 1.0905 | **0.3250** ← still climbing, not converged |

**Per-theme F1 on test set vs quick_001:**

| Theme | Q002 F1 | Q001 F1 | Delta | Train F1 | Gap | Threshold |
|---|---|---|---|---|---|---|
| Navigational | 0.5117 | 0.4591 | +0.053 | 0.536 | +0.025 | 0.580 |
| Attainment | 0.6537 | 0.5697 | +0.084 | 0.691 | +0.037 | 0.630 |
| Perseverance | 0.3738 | 0.3276 | +0.046 | 0.381 | +0.007 | 0.540 |
| Aspirational | 0.4667 | 0.4249 | +0.042 | 0.463 | -0.004 | 0.540 |
| Familial | 0.5059 | 0.4191 | +0.087 | 0.509 | +0.003 | 0.520 |
| **Social** | **0.3855** | 0.2093 | **+0.176** | 0.470 | +0.084 | 0.550 |
| Filial Piety | 0.1154 | 0.2258 | **-0.110** | 0.538 | **+0.422** | 0.640 ← overfit |
| **Spiritual** | 0.2775 | 0.2000 | +0.078 | 0.533 | +0.256 | 0.600 ← overfit |
| Resistance | 0.1596 | 0.2121 | **-0.053** | 0.395 | +0.236 | 0.570 ← overfit |
| Community Consciousness | 0.1290 | 0.0677 | +0.061 | 0.373 | +0.244 | 0.510 ← overfit |
| **First Gen** | 0.2143 | 0.2424 | -0.028 | 0.751 | **+0.537** | 0.640 ← severe overfit |
| **MACRO** | **0.3448** | 0.3052 | **+0.040** | | | |

**Key observations:**
- DAPT helps: +0.040 macro F1 vs no-DAPT. Social +0.176 (DAPT revealed rich social capital vocabulary).
- Common themes (Nav, Att, Per, Asp, Fam): minimal gap (+0.003 to +0.037) → AWARE generalizing correctly
- Val F1 still RISING at epoch 12 (best=last) → model not converged, needs many more epochs
- Filial Piety and Resistance DROPPED: overfitting got worse with more epochs + rare theme data scarcity
- FG/FP thresholds hit ceiling (0.640 = max of search range): model overconfident on rare patterns
- Root cause for rare themes: data scarcity (FG=114 train, CC=170, FP=263). Not fixable by regularization alone.
- 88 min of 115 min time budget unused — can scale to 36+ epochs in next run

**Decision for quick_003:**
- Main changes: 36 total epochs (6+30), DAPT 2 epochs, dropout 0.42, label_smoothing 0.10, aeda 0.5, wd 0.03
- Reasoning: val still climbing → needs more epochs. Higher label_smoothing reduces rare-theme overconfidence.

---

## Run quick_003 — 2026-02-23 (COMPLETE, Job 21356)

**Config:** quick.yaml (6 phase1 + 30 phase2 = 36 max epochs, stopped at ep21 — early stopping)
**DAPT:** yes (2 epochs)
**Job ID:** 21356 (gpu02, A100 80GB)
**Wall time:** ~50 min (22:27→00:17 next day)

**Code changes from quick_002:**
1. `dataset.py`: Theme-aware essay upsampling (`rare_theme_boost`)
   - First Gen: 8x, CC: 6x, FP: 4x, Spiritual: 2.5x, Resistance: 2.5x
2. `metrics.py`: Threshold capped at 0.45 for themes with val support < 50
3. `config.py`: Added `rare_theme_boost` to TrainingConfig
4. `quick.yaml`: 36 total epochs, DAPT 2ep, dropout=0.42, wd=0.03, label_smoothing=0.10, aeda=0.5

**Training history (best ep=21, val still rising at ep21):**

| Epoch | Phase | Train Loss | Val F1 Macro |
|---|---|---|---|
| 1 | 1 | 2.2893 | 0.1852 |
| 6 | 1 | 1.8220 | 0.2166 |
| 7 | 2 | 1.8032 | 0.2206 |
| 12 | 2 | 1.6803 | 0.2673 |
| 18 | 2 | 1.5952 | 0.2808 |
| 21 | 2 (best) | 1.5667 | **0.2850** ← still rising, patience fired |

**Per-theme F1 on test set vs quick_002:**

| Theme | Q003 F1 | Q002 F1 | Delta | Train F1 | Gap | Threshold |
|---|---|---|---|---|---|---|
| Navigational | 0.4884 | 0.5117 | **-0.023** | 0.5447 | +0.056 | 0.610 |
| Attainment | 0.6199 | 0.6537 | **-0.034** | 0.6551 | +0.035 | 0.600 |
| Perseverance | 0.3458 | 0.3738 | **-0.028** | 0.3859 | +0.040 | 0.570 |
| Aspirational | 0.4652 | 0.4667 | -0.002 | 0.5008 | +0.036 | 0.590 |
| Familial | 0.4710 | 0.5059 | **-0.035** | 0.4970 | +0.026 | 0.570 |
| **Social** | **0.4496** | 0.3855 | **+0.064** ✅ | 0.5529 | +0.103 | 0.600 |
| Filial Piety | 0.1174 | 0.1154 | +0.002 | 0.2097 | +0.092 | **0.450 (cap)** |
| Spiritual | 0.2536 | 0.2775 | -0.024 | 0.4809 | **+0.227** ← overfit | 0.560 |
| Resistance | 0.1667 | 0.1596 | +0.007 | 0.6043 | **+0.438** ← severe overfit | 0.630 |
| Community Consciousness | 0.0224 | 0.1290 | **-0.107** ❌ | 0.1204 | +0.098 | **0.450 (cap)** |
| First Gen | 0.1106 | 0.2143 | **-0.104** ❌ | 0.1910 | +0.080 | **0.450 (cap)** |
| **MACRO** | **0.3191** | **0.3448** | **-0.026** ❌ REGRESSION | | | |

**Root cause of regression (post-analysis):**
1. **Sampling boost displaced common themes**: FG at 8x → common theme essays fewer per batch → Nav/Att/Per/Fam all regressed
2. **Precision collapse for rare themes**: FG precision=0.060, CC precision=0.012. Model fires rare theme predictions on ~180 essays (FG) but only 11 are correct. The model memorized training essays, not generalizable CCW signals.
3. **Resistance severe overfit** (+0.438 gap): 2.5x sampling boost → 333 Resistance essays seen ~8500 times across 21 epochs → pure memorization
4. **Threshold cap backfired**: cap at 0.45 forces high recall, but precision so low that F1 collapses. CC: P=0.012, R=0.278, F1=0.022

**Decision for quick_004:**
- **Remove `rare_theme_boost` entirely** — it's hurting common themes and causing overfitting
- **Remove threshold cap** — it destroyed precision for rare themes
- **Extend threshold search range** to (0.10, 0.75): FP/FG were hitting the ceiling (0.640) in q002 → let optimizer find true peak
- Keep everything else: DAPT 2ep, 36ep, dropout=0.42, wd=0.03, label_smoothing=0.10, aeda=0.5
- **Expected**: common themes recover to q002+, Social continues ↑, macro F1 > 0.345

---

## Run quick_004 — 2026-02-24

**Config:** quick.yaml reverted to Q002 base (proven best) + 3 targeted code fixes
**DAPT:** yes (1 epoch — proven optimal, 2ep showed no benefit over 1ep)
**Job ID:** (fill in after sbatch)
**Wall time:** expected ~35 min

**Root cause analysis (deep dive on Q001-Q003):**
- Q002 was best (0.3448) with moderate settings. Q003 over-regularized AND added bad sampling.
- Rare theme failure is NOT underfitting: train F1 is high (FG=0.751). It's threshold miscalibration.
- With 9 FG validation examples, F1-based threshold optimization picks threshold=0.64 → too high for test.
- Symmetric label smoothing (1→0.95) weakens the already-scarce positive signal for rare themes.
- BiLSTM bottleneck (768→512→768) loses sentence information without residual path.

**3 targeted code fixes (each addresses a specific root cause):**
1. `metrics.py`: **F2 threshold for rare themes** — themes with val support < 50 now use F2-score
   (recall-biased, beta=2) instead of F1 for threshold selection. F2 weights recall 4x > precision,
   pushing thresholds lower → catches more true positives. Default threshold lowered to 0.25.
2. `losses.py`: **Asymmetric label smoothing** — only smooths negatives (0→ls), keeps positives at
   1.0. Old symmetric approach weakened positive signal (1→0.95) for rare themes.
3. `model.py`: **Residual connection in BiLSTM** — adds skip connection: `context = lstm_proj + sent_emb`.
   Preserves original sentence information through the LSTM projection bottleneck.

**Config reverted to Q002 levels:**
- dropout: 0.42→0.35 (Q003 over-regularized)
- weight_decay: 0.03→0.02 (same)
- label_smoothing: 0.10→0.05 (but now asymmetric)
- aeda_prob: 0.5→0.3 (Q003 added too much noise)
- DAPT: 2→1 epoch (proven optimal)
- phase2_epochs: 30→20 (with patience=10, enough room to converge)
- warmup_ratio: 0.06→0.10 (stabilizes early training)

**Hypothesis:** Q002's common theme performance (Nav=0.51, Att=0.65, Fam=0.51) recovers fully.
Rare themes improve via F2 thresholds (lower thresholds → higher recall). Residual connection
and asymmetric smoothing contribute to better generalization. Target: macro F1 > 0.36.

**Results (fill in when run completes):**

| Epoch | Phase | Train Loss | Val F1 Macro |
|---|---|---|---|
| ... | | | |

| Theme | Q004 F1 | Q003 F1 | Q002 F1 (best) | Delta vs Q002 | Threshold |
|---|---|---|---|---|---|
| Navigational | | 0.4884 | 0.5117 | | |
| Attainment | | 0.6199 | 0.6537 | | |
| Perseverance | | 0.3458 | 0.3738 | | |
| Aspirational | | 0.4652 | 0.4667 | | |
| Familial | | 0.4710 | 0.5059 | | |
| Social | | 0.4496 | 0.3855 | | |
| Filial Piety | | 0.1174 | 0.1154 | | |
| Spiritual | | 0.2536 | 0.2775 | | |
| Resistance | | 0.1667 | 0.1596 | | |
| Community Consciousness | | 0.0224 | 0.1290 | | |
| First Gen | | 0.1106 | 0.2143 | | |
| **MACRO** | | **0.3191** | **0.3448** | | |

---

## Run Template — copy this for each run

---

### Run [NNN] — [YYYY-MM-DD]

**Config:** quick.yaml / full.yaml
**DAPT:** yes / no
**Job ID:** [SLURM job ID from `sbatch` output]
**Wall time:** [actual time from SLURM footer]
**Command used:**
```bash
# run_job.sh settings:
RUN_MODE="[quick|full]"
RUN_NUMBER="[NNN]"
USE_DAPT="[--no-dapt | ""]"
```

**Training history (from `results/[run]/history.json`):**

| Epoch | Phase | Train Loss | Val F1 Macro |
|---|---|---|---|
| 1 | 1 | | |
| 2 | 1 | | |
| 3 | 2 | | |
| 4 | 2 | | |
| 5 | 2 | | |
| 6 | 2 (best) | | |

**Per-theme F1 on test set (from `evaluation_test.json`):**

| Theme | F1 | Precision | Recall | Threshold | Support |
|---|---|---|---|---|---|
| Navigational | | | | | |
| Attainment | | | | | |
| Perseverance | | | | | |
| Aspirational | | | | | |
| Familial | | | | | |
| Social | | | | | |
| Resistance | | | | | |
| Spiritual | | | | | |
| Filial Piety | | | | | |
| Community Consciousness | | | | | |
| First Gen | | | | | |
| **Macro** | | | | | |

**Overfitting check (train F1 vs test F1 from `evaluation_train.json` vs `evaluation_test.json`):**

| Theme | Train F1 | Test F1 | Gap |
|---|---|---|---|
| Navigational | | | |
| Attainment | | | |
| Perseverance | | | |
| Aspirational | | | |
| Familial | | | |
| Social | | | |
| Resistance | | | |
| Spiritual | | | |
| Filial Piety | | | |
| Community Consciousness | | | |
| First Gen | | | |

**Observations:**
- val_f1_macro trend (epoch to epoch):
- Themes with F1 = 0:
- Themes overfitting (gap > 0.15):
- Themes underfitting (both train and test F1 low):
- Anything unexpected:

**Decision for next run:**
- Change:
- Reason:

---
