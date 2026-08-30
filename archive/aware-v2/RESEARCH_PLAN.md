# AWARE v2 — Research Plan: From 0.390 to Production-Quality F1

## Status: DRAFT — For Discussion Before Implementation

---

## 1. Data Analysis Findings (From master_data.pkl)

### 1.1 Full Corpus Overview
| Metric | Value |
|--------|-------|
| Total essays | 16,148 |
| Total sentences | 112,490 |
| Unique students | 4,179 |
| Themed essays (at least 1 theme) | 9,134 (56.6%) |
| Pure class_0 essays | 7,014 (43.4%) |
| Themed sentences | 26,611 (23.7%) |
| Class_0 sentences | 85,879 (76.3%) |

### 1.2 Theme Distribution
| Theme | Sentences | % of Themed | Essays | Imbalance Ratio |
|-------|-----------|-------------|--------|-----------------|
| Navigational | 9,867 | 37.1% | 4,123 | 1.0x |
| Attainment | 6,876 | 25.8% | 4,092 | 1.4x |
| Perseverance | 6,439 | 24.2% | 2,977 | 1.5x |
| Aspirational | 5,993 | 22.5% | 3,357 | 1.6x |
| Familial | 3,083 | 11.6% | 1,843 | 3.2x |
| Social | 1,205 | 4.5% | 743 | 8.2x |
| Resistance | 968 | 3.6% | 424 | 10.2x |
| Spiritual | 873 | 3.3% | 388 | 11.3x |
| Filial Piety | 326 | 1.2% | 167 | 30.3x |
| Community Consciousness | 225 | 0.8% | 106 | 43.9x |
| First Gen | 137 | 0.5% | 79 | 72.0x |

### 1.3 Critical Co-Occurrence Patterns
Top theme pairs by co-occurrence count:
| Pair | Co-occurrences | Implication |
|------|---------------|-------------|
| Aspirational + Attainment | 2,467 | Very tightly coupled |
| Attainment + Navigational | 1,941 | Common pair |
| Aspirational + Navigational | 1,779 | Common pair |
| Perseverance + Resistance | 699 | Resistance almost never alone |
| Familial + Filial Piety | 125 | FP is a refinement of Familial |
| First Gen + Familial | 96 | FG is deeply tied to Familial |

### 1.4 Rare Theme Dependency (This Is Key)
| Rare Theme | % Multi-label | Top Co-occurring | P(co-occur) |
|------------|---------------|-------------------|-------------|
| First Gen | 96.4% | Familial | 70.1% |
| Resistance | 90.2% | Perseverance | 72.2% |
| Community Consciousness | 72.9% | Spiritual | 30.7% |
| Filial Piety | 78.5% | Familial | 38.3% |
| Spiritual | 58.2% | Social | 35.7% |

**Key insight:** Rare themes are NOT independent labels. They are refinements that almost always appear WITH common themes. First Gen with only 5 single-label instances out of 137 means the model needs to learn "what makes a Familial sentence ALSO First Gen."

### 1.5 Class_0 Dominance Problem
| Metric | Value |
|--------|-------|
| class_0 as % of ALL sentences | 76.3% |
| class_0 as % within themed essays | 59.9% |
| class_0 / First Gen ratio | 626.9x |
| Essays that are 100% class_0 | 7,014 (43.4%) |
| Essays with >= 80% class_0 | 9,456 (58.6%) |
| Essays with 0% class_0 | 752 (4.7%) |

### 1.6 Multi-Label Distribution
| Labels per sentence | Count | % of themed |
|--------------------|-------|-------------|
| 1 label | 19,801 | 74.4% |
| 2 labels | 4,788 | 18.0% |
| 3 labels | 1,623 | 6.1% |
| 4 labels | 278 | 1.0% |
| 5+ labels | 121 | 0.5% |

---

## 2. Critical Issues Found in Current Pipeline

### Issue 1: TRAINING-TIME EVALUATION USES WRONG THRESHOLDS
**File:** `scripts/trainer.py` line 726
**Problem:** During training, `_validate()` calls `flatten_masked_preds_labels()` with DEFAULT threshold=0.5 for ALL themes. But rare theme probabilities may never reach 0.5 — they hover at 0.1-0.3 for positives.

**Impact:**
- val_f1_macro during training is ARTIFICIALLY LOW for rare themes
- Early stopping decisions are based on this misleading metric
- The "best model" is selected by a metric that doesn't reflect actual performance
- Threshold optimization only happens AFTER training ends (line 317)
- The model that's best at 0.5 threshold is NOT necessarily best with optimized thresholds

**This is a chicken-and-egg problem:** We need the model to optimize thresholds, but we need thresholds to properly evaluate the model during training.

### Issue 2: CLASS_0 GRADIENT DOMINANCE
**File:** `scripts/dataset.py` line 155-160, `scripts/losses.py` line 107
**Problem:** class_0 sentences have ALL-ZERO label vectors. In the loss function, each class_0 sentence generates 11 NEGATIVE loss signals (one per theme). With 69% class_0 in training data, the model receives ~69% pure negative signal.

**Impact:**
- The model learns "predict nothing" as the safest strategy
- For First Gen (0.2% of data), saying "no" is correct 99.84% of the time
- Even with loss weighting, the sheer volume of negative examples overwhelms
- This is a DATA problem, not a loss function problem

### Issue 3: SHARED CLASSIFIER GRADIENT COMPETITION
**File:** `scripts/model.py` line 148-160
**Problem:** All 11 themes share one `Linear(768, 11)` layer. Common theme gradients (Nav: ~10K examples) dominate weight updates. Rare theme gradients (FG: 137) barely influence the shared weights.

**Impact:**
- The shared classifier's weight space is shaped primarily by common themes
- Rare themes don't get enough gradient signal to learn distinct patterns
- Q011 showed gradient norm for FP was 87,494 vs Nav at 5,609 (15x) when forced — but this was with CB Loss which caused explosion

### Issue 4: THRESHOLD INSTABILITY FOR RARE THEMES
**File:** `scripts/metrics.py` line 130-224
**Problem:** With ~14 First Gen validation examples, threshold optimization is extremely noisy. A single prediction flip changes the optimal threshold dramatically.

**Impact:**
- Threshold varies wildly between runs (FG: 0.0-0.517 across Q004-Q010)
- Bootstrap CI for First Gen: [0.0, 0.5] — completely unreliable
- The "distribution-aware floor" helps prevent catastrophe but doesn't solve instability

### Issue 5: MISSING EVALUATION METRICS
**Problem:** Only per-theme F1 and macro/micro F1 are computed. Missing several important multi-label metrics.

**Missing metrics:**
- **PR-AUC per theme**: Threshold-independent, better for imbalanced data than F1
- **Exact Match Ratio**: How often ALL themes are correct for a sentence
- **Hamming Loss**: Fraction of incorrect labels
- **Label Ranking Loss**: How well the model ranks true labels above false ones
- **Weighted F1**: F1 weighted by support (complements macro F1)

### Issue 6: NO MONITORING OF WHAT CLASS_0 DOES TO LEARNING
**Problem:** While class_0 F1 is now computed (added in Q009), there's no analysis of:
- How many themed sentences get predicted as "no theme" (false negatives)
- How the model's class_0 predictions change during training
- Whether the model is essentially a "no theme" classifier that occasionally detects common themes

---

## 3. Research-Backed Solutions

### 3.1 Data Strategy: Separate class_0 from Theme Classification

**Research backing:** Multi-label classification fundamentals — the "no label" class should NOT be treated as 11 simultaneous negatives.

**Proposed approach:**
1. Start from master_data.pkl (ALL 16,148 essays)
2. Extract ALL sentences with their annotations
3. **Do NOT include class_0 as a training signal in the multi-label loss**
4. Instead, use a **two-stage approach** OR **exclude class_0 and let it emerge naturally**:
   - When all 11 theme probabilities are below their thresholds → classify as class_0
   - This prevents class_0 from dominating gradients

**Sampling strategy options to test:**
- **Option A:** Only use themed sentences for training (26,611 sentences). Class_0 emerges as "model predicts nothing above threshold."
  - Risk: Model never sees "normal" text and over-predicts themes
- **Option B:** Include class_0 at reduced ratio (~30-40% instead of 69%). Strategically prefer class_0 from themed essays (contextually richer).
  - Better: Model learns what "no theme" looks like without being overwhelmed
- **Option C:** Two-stage model. Stage 1: binary "has theme / no theme" → Stage 2: multi-label "which themes?"
  - Best: Cleanly separates two different tasks

**Recommendation:** Start with Option B (reduced class_0, ~40%) as baseline, then test Option C (two-stage) as the architectural innovation.

### 3.2 Architecture: Theme-Group Multi-Head with Conditioning

**Research backing:**
- Distribution-Balanced Loss (Wu et al., ECCV 2020) — per-group handling of different frequency classes
- GCN for label correlation (Chen et al., CVPR 2019) — model label dependencies
- Cascade classifiers — condition rare predictions on common predictions

**Current architecture:**
```
DeBERTa → mean pool → BiLSTM → Linear(768, 11)  [SHARED]
```

**Proposed architecture:**
```
DeBERTa → mean pool → BiLSTM
  |
  +--→ Common Head: Linear(768, 4) [Nav, Att, Per, Asp]
  |      Standard training, these have enough data
  |
  +--→ Medium Head: Linear(768, 3) [Fam, Soc, Res]
  |      Upweighted training
  |
  +--→ Rare Head: Linear(768 + 7, 4) [Spi, FP, CC, FG]
         INPUT: context_embedding CONCAT [common_logits, medium_logits]
         Rare themes see what common themes the model already detected
         Matches data structure: FG is 70% conditional on Familial
```

**Why conditioning works for this data:**
- First Gen: P(Familial | First Gen) = 0.701. If the model knows "this is Familial," it has strong signal for "might also be First Gen"
- Resistance: P(Perseverance | Resistance) = 0.722. Knowing Perseverance is present helps predict Resistance
- This is NOT circular — common themes have enough data to be reliably predicted first

**Alternative: Ablate with simpler approach first**
- Per-theme binary heads: 11 × Linear(768, 1). Each theme gets independent gradient flow.
- This is simpler than multi-head groups and might work well enough.

### 3.3 Evaluation: Fix During-Training Metrics

**Research backing:**
- PR-AUC is the standard for imbalanced classification (Davis & Goadrich, 2006)
- Macro F1 at fixed threshold is misleading for long-tail distributions

**Proposed fixes:**

1. **During training validation, use PR-AUC instead of F1 at 0.5 threshold:**
   - PR-AUC doesn't depend on threshold selection
   - Better model selection for rare themes
   - Can still report F1 as secondary metric

2. **For best model selection, use macro-averaged PR-AUC:**
   - Average PR-AUC across all 11 themes
   - This prevents threshold-dependent artifacts

3. **Keep threshold optimization as a POST-TRAINING step:**
   - This is correct — thresholds should be optimized on val set after training
   - But the model SELECTION during training should NOT depend on thresholds

4. **Add more evaluation metrics:**
   - Per-theme PR-AUC (threshold-independent)
   - Hamming loss (overall label quality)
   - Exact match ratio (strictest multi-label metric)
   - Per-theme confusion: when FG is missed, what WAS predicted?

### 3.4 Loss Function: Keep ASL But Fix class_0 Handling

**Research backing:** ASL (Ridnik et al., ICCV 2021) is proven and already in our codebase. The issue isn't the loss function — it's what data feeds into it.

**Key changes:**
- **Do NOT change ASL parameters** — Q006 proved gamma_pos=1.0, gamma_neg=4.0 works
- **Reduce class_0 data** feeding into loss (Section 3.1)
- **Consider per-head loss computation** with separate weights per head group
- **DO NOT use CB Loss** — Q011 proved it causes gradient explosion with logit adjustment

### 3.5 DAPT: Keep But Ablate Properly

**Research backing:** Gururangan et al. (ACL 2020) — DAPT proven effective for domain adaptation.

**Current state:** Q002 showed DAPT 1 epoch = +0.040 macro F1. But this was confounded with other changes.

**Required:** Clean ablation — run identical config WITH and WITHOUT DAPT. Must be a new experiment, not relying on Q001 vs Q002 comparison (which had different epochs).

---

## 4. Proposed Experiment Plan

### Phase 0: Fix Evaluation (Before Any Training)
1. Add PR-AUC per theme to metrics.py
2. Change `_validate()` in trainer.py to use macro PR-AUC for model selection
3. Add Hamming loss, exact match ratio to compute_metrics()
4. Verify metric computations with hand-computed toy examples
5. **No model changes yet — just fix how we measure**

### Phase 1: Data Experiments (New Dataset from master_data.pkl)
Create `build_dataset.py` with configurable class_0 ratio:
1. **Dataset A:** Current approach (69% class_0) — baseline
2. **Dataset B:** Reduced class_0 (40%) — strategic sampling from themed essays
3. **Dataset C:** Reduced class_0 (50%) — middle ground
4. **Dataset D:** Themed sentences only (0% class_0) — extreme test

Run baseline model (current Q006 arch) on each dataset. Measure:
- Which class_0 ratio gives best macro F1?
- Which gives best rare theme F1?
- Which gives best PR-AUC?

### Phase 2: Architecture Experiments (On Best Dataset)
Using whichever dataset won in Phase 1:
1. **Arch A:** Current shared head — Linear(768, 11) — baseline
2. **Arch B:** Per-theme heads — 11 × Linear(768, 1) — independent gradient flow
3. **Arch C:** Theme-group heads — common/medium/rare with conditioning
4. **Arch D:** No BiLSTM — sentence-only classification — proves context matters

### Phase 3: Ablation Experiments
1. With vs. without DAPT (on best data + best arch)
2. With vs. without BiLSTM context
3. With vs. without AEDA augmentation
4. With vs. without per-theme loss weights

### Phase 4: Full-Scale Training
Best configuration from Phases 1-3, trained for 30+ epochs on HPC.

---

## 5. Key Questions to Discuss with Khalid

1. **class_0 ratio:** Should we test multiple ratios (40%, 50%, 60%) or go straight to one? Option B (reduced class_0) seems safest; Option C (two-stage) is more innovative but riskier.

2. **Architecture complexity:** Theme-group multi-head with conditioning is the most innovative but also most complex. Per-theme binary heads (11 × Linear(768,1)) is simpler and might work well. Which to try first?

3. **Evaluation change priority:** Should we fix evaluation FIRST before any other changes? I think yes — we need correct measurement before we can improve.

4. **Timeline reality check:** With April deadline, can we do all 4 phases? Minimum viable: Phase 0 (evaluation fix) + one data experiment + one architecture experiment + full run = ~4-5 HPC runs.

5. **Thesis narrative:** Even if F1 doesn't dramatically improve, the systematic approach (data analysis → evaluation fix → architecture comparison → ablation) is itself a strong thesis contribution. Do you agree?

---

## 6. References

- Ridnik et al. (2021). "Asymmetric Loss for Multi-Label Classification." ICCV. [Current loss function]
- Wu et al. (2020). "Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets." ECCV.
- Chen et al. (2019). "Multi-Label Image Recognition with Graph Convolutional Networks." CVPR.
- Gururangan et al. (2020). "Don't Stop Pretraining." ACL. [DAPT]
- Davis & Goadrich (2006). "The Relationship Between PR and ROC Curves." ICML. [PR-AUC for imbalanced]
- Menon et al. (2021). "Long-tail Learning via Logit Adjustment." ICLR. [Tried in Q011, failed]
- Cui et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples." CVPR. [Tried in Q011, failed]
- He et al. (2023). "DeBERTaV3." ICLR. [Our encoder]
- Kang et al. (2020). "Decoupling Representation and Classifier for Long-Tailed Recognition." ICLR. [Phase 3 concept]

---

## 7. Files That Will Be Modified (When Approved)

**Evaluation fixes (Phase 0):**
- `scripts/metrics.py` — Add PR-AUC, Hamming loss, exact match
- `scripts/trainer.py` — Change model selection to use PR-AUC

**Data (Phase 1):**
- `scripts/build_dataset.py` — NEW: configurable dataset builder from master_data.pkl

**Architecture (Phase 2):**
- `scripts/model.py` — Add ablation flags, multi-head options
- `scripts/config.py` — Add architecture config fields
- `scripts/losses.py` — Per-head loss computation

**Infrastructure:**
- `scripts/test_components.py` — Tests for all new components
