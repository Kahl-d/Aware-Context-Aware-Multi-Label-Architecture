# Comparison Run Checklist

## Goal
2x2 comparison: Standard vs AWARE × Base vs Large, plus baselines.
Each model runs independently, results checked before submitting next.

---

## Step 1: Baselines
- **Job**: 22566
- **Status**: DONE (27 sec)
- **Results**:

| Model | Macro F1 | Macro PRAUC |
|---|---|---|
| Majority Class | 0.000 | 0.085 |
| Random (Prior) | 0.084 | 0.089 |
| Most Common Label Set | 0.000 | 0.000 |
| TF-IDF + LogReg | 0.187 | 0.363 |
| TF-IDF + RandomForest | 0.098 | 0.352 |
| TF-IDF + SVM | 0.274 | 0.333 |

**Best non-neural baseline**: TF-IDF + SVM (F1=0.274)

---

## Step 2: Base Standard (DeBERTa-v3-base, no AWARE)
- **Job**: 22619
- **Status**: ✅ DONE (early stopped at epoch 13, best at epoch 8)
- **Config**: `1_base_standard/configs/base_standard.yaml`
- **Key settings**: 184M params, BCE loss, single-phase, lr=2e-5, batch=32, 30 epochs

| Metric | Value |
|---|---|
| Test F1 | **0.487** |
| Test PRAUC | **0.507** |
| Val PRAUC | 0.527 |
| Val F1 | 0.535 |
| Train F1 | 0.727 |
| Train PRAUC | 0.884 |
| Train-Test F1 gap | +0.240 |

| Theme | Test F1 | Test PRAUC |
|---|---|---|
| Navigational | 0.684 | 0.777 |
| Familial_Capital | 0.641 | 0.668 |
| Aspirational | 0.558 | 0.616 |
| Perseverance | 0.510 | 0.594 |
| Social | 0.500 | 0.510 |
| Spiritual | 0.431 | 0.383 |
| Resistance | 0.347 | 0.347 |
| Attainment | 0.222 | 0.161 |

**Notes**: Strong baseline! F1=0.487 without any AWARE components. Early stopped at epoch 13 (patience=5). Peaked at epoch 8.

---

## Step 3: Large Standard (DeBERTa-v3-large, no AWARE)
- **Job**: 22620
- **Status**: 🟢 RUNNING on gpu01 (~2 min in)
- **Config**: `3_large_standard/configs/large_standard.yaml`
- **Key settings**: 438M params, BCE loss, single-phase, lr=1e-5, batch=32, 20 epochs
- **Results**: (fill after completion)

| Metric | Value |
|---|---|
| Test F1 | |
| Test PRAUC | |
| Val PRAUC | |
| Train F1 | |
| Train-Test gap | |

---

## Step 4: Base AWARE (DeBERTa-v3-base, full AWARE pipeline)
- **DAPT Job**: 22569 — DONE (4 min, encoder at `2_base_aware/results/dapt_base/encoder/`)
- **Train Job**: (submit after Step 3 checked)
- **Status**: DAPT done, training pending
- **Config**: `2_base_aware/configs/base_aware.yaml`
- **Key settings**: 184M params, ASL loss, 3-phase, BiLSTM, LLRD, R-Drop, AEDA, essay head, CB weights, multi-sample dropout, DAPT
- **Results**: (fill after completion)

| Metric | Value |
|---|---|
| Test F1 | |
| Test PRAUC | |
| Val PRAUC | |
| Train F1 | |
| Train-Test gap | |

---

## Step 5: Large AWARE (DeBERTa-v3-large, full AWARE pipeline)
- **DAPT Job**: (submit after Step 4 checked)
- **Train Job**: (submit after DAPT checked)
- **Status**: PENDING (needs fresh DAPT)
- **Config**: `4_large_aware/configs/large_aware.yaml`
- **Key settings**: 438M params, ASL loss, 3-phase, BiLSTM, LLRD, R-Drop, AEDA, essay head, CB weights, multi-sample dropout, progressive unfreeze, DAPT, SWA
- **Previous proven result**: Test F1=0.494, Test PRAUC=0.484 (job 22386)
- **Results**: (fill after completion)

| Metric | Value |
|---|---|
| Test F1 | |
| Test PRAUC | |
| Val PRAUC | |
| Train F1 | |
| Train-Test gap | |

---

## Final Comparison Table (fill after all complete)

| Model | Params | Test F1 | Test PRAUC | Train-Test Gap | vs Best Baseline |
|---|---|---|---|---|---|
| TF-IDF + SVM | — | 0.274 | 0.333 | — | — |
| Base Standard | 184M | | | | |
| Large Standard | 438M | | | | |
| Base AWARE | 184M | | | | |
| Large AWARE | 438M | | | | |

### Research Questions Answered:
1. **Standard vs AWARE (same encoder)**: How much does the AWARE pipeline add?
   - Base: `Base AWARE F1` - `Base Standard F1` = ?
   - Large: `Large AWARE F1` - `Large Standard F1` = ?
2. **Base vs Large (same approach)**: How much does model scale help?
   - Standard: `Large Standard F1` - `Base Standard F1` = ?
   - AWARE: `Large AWARE F1` - `Base AWARE F1` = ?
3. **All vs baselines**: How much does any neural approach add?
   - Best neural - Best baseline = ?
