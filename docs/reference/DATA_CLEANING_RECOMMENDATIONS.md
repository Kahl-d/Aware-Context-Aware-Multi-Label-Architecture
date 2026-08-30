# ALMA — Final Data Preparation Recommendations for Model Training

## Goal

Create a DeBERTa model that can **identify CCW themes in new, unseen student essays**.
Every recommendation below is evaluated against this single goal. The question for each
decision is: *"Will this help the model generalize to new data?"*

---

## RECOMMENDATION 1: Merge Familial + Filial_Piety into "Familial_Capital"

### Verdict: STRONGLY RECOMMEND

### Evidence
| Metric | Value | What It Means |
|--------|-------|---------------|
| Centroid similarity (all data) | 0.987 | Nearly identical in embedding space |
| Centroid similarity (single-label) | 0.947 | Still very high even with purest samples |
| Label correlation | +0.23 | Highest inter-theme co-occurrence |
| Filial_Piety single-label count | 53 | Too few for reliable training |
| Filial_Piety multi-label rate | 77.9% | Rarely appears alone |
| Filial_Piety LOO-KNN mismatch | 64% | Majority misclassified by neighbors |
| Combined count after merge | 880 (640+240) | Reasonable training size |

### Why
- At 0.987 cosine similarity, these two centroids are essentially the same point in
  384-dimensional space. No model — not DeBERTa, not GPT-4 — can reliably distinguish
  two classes whose centroids differ by only 0.013 in cosine distance.
- Filial_Piety is a cultural subset of Familial capital (family as obligation/duty vs.
  family as general support). In Yosso's original framework, Filial Piety is not a
  separate construct — it was added as an extension. Merging it back aligns with theory.
- The merged class "Familial_Capital" (880 samples) has a healthy training size and retains
  both concepts. The model doesn't lose information — it just stops trying to distinguish
  the indistinguishable.

### How
```python
df['Familial'] = ((df['Familial'] == 1) | (df['Filial_Piety'] == 1)).astype(int)
df = df.drop(columns=['Filial_Piety'])
# Rename to 'Familial_Capital' for clarity in thesis
df = df.rename(columns={'Familial': 'Familial_Capital'})
```

### Impact on model
- Reduces 11 theme heads to 10
- Eliminates the single most confusable pair
- Increases Familial training samples by 37.5%
- No loss of theoretical coverage (Filial Piety is a subset of Familial)

---

## RECOMMENDATION 2: Drop First_Gen as a Separate Theme

### Verdict: STRONGLY RECOMMEND

### Evidence
| Metric | Value | What It Means |
|--------|-------|---------------|
| Total samples | 33 | Extremely rare (0.2%) |
| Single-label samples | 4 | Essentially never appears alone |
| Multi-label rate | 87.9% | Almost always co-occurs with other themes |
| Centroid shift (single vs all) | 0.4262 | MASSIVE — centroid is unreliable |
| LOO-KNN mismatch | 88% | Neighbors almost never agree it's First_Gen |
| Imbalance ratio | 269:1 vs Class_0 | Extreme |

### Why
- With only 33 samples (4 single-label), the model cannot learn what First_Gen "looks like."
  Even with data augmentation, 33 samples cannot define a reliable decision boundary in
  384-dimensional space.
- The 0.4262 centroid shift proves the centroid computed from all 33 samples is completely
  different from the centroid of the 4 single-label samples. The class has no stable
  representation.
- 88% LOO-KNN mismatch means that in the embedding space, First_Gen points look like
  other classes to their neighbors. There is no spatial signal for this class.
- Theoretically, First-Generation status is a demographic attribute, not a linguistic
  theme. Unlike Aspirational ("I want to succeed") or Resistance ("I overcame barriers"),
  First-Gen doesn't have a distinctive language pattern. Students don't consistently
  use specific words to express being first-generation.

### How
- **Option A (Recommended)**: Absorb First_Gen labels into their co-occurring themes.
  Since 87.9% are multi-label, most sentences already have other valid labels. The 4
  single-label First_Gen sentences can be re-examined manually:
  ```python
  df = df.drop(columns=['First_Gen'])
  # The 4 single-label First_Gen sentences become Class_0
  # (they had no other theme, and First_Gen is removed)
  ```
- **Option B**: Merge into Resistance (both involve overcoming systemic barriers), but
  this conflates distinct constructs.

### Impact on model
- Reduces theme heads from 10 (after Filial_Piety merge) to 9
- Eliminates the most extreme imbalance (269:1)
- Prevents the model from wasting capacity on an unlearnable class
- Loses 4 sentences to Class_0 reclassification (negligible)

---

## RECOMMENDATION 3: Keep Community_Consciousness Despite Small Size

### Verdict: KEEP (with augmentation)

### Evidence
| Metric | Value | Why Keep |
|--------|-------|---------|
| Total samples | 119 | Small but 3.6x larger than First_Gen |
| Single-label samples | 19 | Barely enough for a representation |
| d' (separability) | 1.88 | Actually HIGH — it IS separable |
| CC ↔ Spiritual similarity | 0.930 | High but not extreme (vs 0.987 for Fam/FP) |
| Centroid shift | 0.2289 | Moderate — centroid is somewhat stable |
| LOO-KNN mismatch | 75% | High but driven by small size, not confusion |

### Why Keep
- Unlike First_Gen, CC has a d' of 1.88 (easy separability when measured on single-label
  sentences). The semantic signal exists — there just aren't enough training samples.
- 119 samples is sufficient with aggressive augmentation (back-translation, synonym
  replacement can bring it to 400-500).
- Theoretically, Community Consciousness is a distinct CCW construct — it describes
  awareness of collective community issues, which is different from Social (interpersonal
  relationships) or Spiritual (inner strength/faith).

### What To Do
- Apply 3-5x data augmentation specifically for CC
- Use Focal Loss with high alpha for CC
- Monitor per-class F1 during training — if CC F1 < 0.3 after tuning, reconsider merging
  into Social (the next-closest theme at 0.907 similarity)
- Set a low decision threshold for CC (optimize on validation set)

---

## RECOMMENDATION 4: Handle Resistance with Extra Care

### Verdict: KEEP but acknowledge difficulty

### Evidence
| Metric | Value | Concern |
|--------|-------|---------|
| d' (separability) | 0.39 | HARD — lowest of all themes |
| Class_0 ↔ Resistance similarity | 0.933 | Hardest theme/Class_0 boundary |
| Social ↔ Resistance similarity | 0.932 | Also confused with Social |
| LOO-KNN mismatch | 52% | Coin flip accuracy |
| Boundary removal rate | 35.6% | Highest removal rate of any class |
| No cluster dominance | — | Never dominates any hierarchical cluster |

### Why It's Hard
- Resistance language ("I proved them wrong," "despite challenges") overlaps heavily with
  Perseverance ("I overcame difficulties") and generic Class_0 language.
- Theoretically, Resistance requires identifying a *systemic* barrier being challenged.
  The word "challenge" appears in Resistance, Perseverance, AND Class_0 sentences.
- The model must learn the CONTEXT of challenge language, not just the words themselves.
  This is exactly what DeBERTa's contextual attention should capture.

### What To Do
- **Do NOT merge or drop** — Resistance is theoretically distinct and important to CCW
- Apply 2x augmentation (830 → ~1,600 samples)
- Use Focal Loss with gamma=3 (higher than default 2) for Resistance
- Optimize threshold aggressively on validation set (likely need threshold < 0.3)
- In thesis: report Resistance F1 separately and acknowledge this is the hardest theme
- Consider multi-task learning: train a separate binary Resistance classifier as auxiliary

---

## RECOMMENDATION 5: Do NOT Remove the 1,790 Boundary Candidates (Yet)

### Verdict: WAIT — let the model decide

### Evidence
The 1,790 boundary candidates include:
- 255 Class_0 points (2.9% of Class_0)
- 1,535 themed points (especially Resistance at 35.6%, Spiritual at 29.4%)

### Why Wait
- Removing 1,790 points (9.9%) ON TOP of the 1,705 already removed in Round 1 means
  we'd have removed 3,495 sentences (17.7% of original data). This is aggressive.
- More importantly, the "boundary candidates" are identified using FROZEN embeddings
  (all-MiniLM-L6-v2). DeBERTa fine-tuning will learn its OWN embedding space that may
  separate these points correctly.
- Removing Resistance points at 35.6% rate would leave only 473 Resistance sentences —
  dangerously small for a class that's already the hardest to learn.
- The points that "look like Class_0" in MiniLM space may look like their correct theme
  in DeBERTa fine-tuned space.

### What To Do Instead
1. **Train first on all 18,019 sentences** (after merges/drops above)
2. **After Phase 1 training**, compute DeBERTa embeddings from the [CLS] token
3. **Re-run boundary analysis** using DeBERTa embeddings — identify which points are STILL
   problematic in the fine-tuned space
4. **Remove the stubborn points** (those misclassified even by the fine-tuned model) and
   retrain
5. This iterative approach removes only what the actual model finds confusing, not what
   a generic embedding model finds confusing

---

## RECOMMENDATION 6: Final Theme Taxonomy (9 Themes + Class_0)

### After applying Recommendations 1-4:

| # | Theme | Samples | % | Status |
|---|-------|---------|---|--------|
| 0 | Class_0 | 8,874 | 49.2% | Keep (4 ex-First_Gen added) |
| 1 | Navigational | 4,322 | 24.0% | Keep as-is |
| 2 | Aspirational | 2,756 | 15.3% | Keep as-is |
| 3 | Perseverance | 1,516 | 8.4% | Keep as-is |
| 4 | Social | 978 | 5.4% | Keep as-is |
| 5 | Familial_Capital | 880 | 4.9% | MERGED (Familial + Filial_Piety) |
| 6 | Resistance | 830 | 4.6% | Keep (needs extra care) |
| 7 | Spiritual | 710 | 3.9% | Keep as-is |
| 8 | Attainment | 452 | 2.5% | Keep as-is |
| 9 | Community_Consciousness | 119 | 0.7% | Keep (with augmentation) |
| — | ~~First_Gen~~ | ~~33~~ | — | DROPPED |
| — | ~~Filial_Piety~~ | ~~240~~ | — | MERGED into Familial_Capital |

**Total: 10 classes (9 themes + Class_0)**
**Dataset: 18,019 sentences** (unchanged — we're only relabeling, not removing)

### New Imbalance Profile
| Theme | Imbalance Ratio (vs Class_0) |
|-------|------------------------------|
| Navigational | 2.1:1 |
| Aspirational | 3.2:1 |
| Perseverance | 5.9:1 |
| Social | 9.1:1 |
| Familial_Capital | 10.1:1 (was 13.9:1) |
| Resistance | 10.7:1 |
| Spiritual | 12.5:1 |
| Attainment | 19.6:1 |
| Community_Consciousness | 74.5:1 |

Maximum imbalance drops from **269:1** (First_Gen) to **74.5:1** (CC).

---

## RECOMMENDATION 7: Data Augmentation Strategy

### Augment only the rarest classes to reach minimum viable training size:

| Theme | Current | Target | Augmentation | Method |
|-------|---------|--------|-------------|--------|
| Community_Consciousness | 119 | ~500 | 4x | Back-translation + synonym replacement |
| Attainment | 452 | ~700 | 1.5x | Back-translation |
| Spiritual | 710 | ~800 | 1.1x | Light synonym replacement |
| Resistance | 830 | ~1,200 | 1.4x | Back-translation (careful with context) |
| Familial_Capital | 880 | ~1,000 | 1.1x | Light synonym replacement |
| All others | — | — | No augmentation | Large enough already |

### Augmentation methods (in priority order):
1. **Back-translation** (English → Spanish → English, English → French → English):
   Most reliable, preserves meaning while varying surface form
2. **Synonym replacement** (WordNet): Replace 1-2 content words per sentence
3. **Random word insertion**: Add contextually relevant words
4. **Do NOT use**: Random deletion (too risky for short sentences), paraphrasing LLMs
   (may change theme meaning)

### Important:
- Augment ONLY training set (never validation/test)
- Verify augmented sentences still express the theme (spot-check 10% manually)
- For Resistance: be especially careful — paraphrasing may lose the "systemic barrier"
  signal that distinguishes Resistance from Perseverance

---

## RECOMMENDATION 8: Training Configuration

### Model
- **DeBERTa-v3-base** (86M params, 12 layers, 768 hidden)
- Multi-label classification head: Linear(768, 9) for 9 themes
- Class_0 derived: if all 9 theme outputs < threshold → Class_0

### Loss Function
- **Focal Loss** (recommended over weighted BCE)
  - gamma = 2.0 (default), increase to 3.0 for Resistance and CC
  - alpha per theme = inverse class frequency
  - Better handles extreme imbalance by down-weighting easy examples

### Data Splits
- **Split by essay_id** (NOT sentence_id) — CRITICAL for preventing data leakage
- Use **iterative stratification** (skmultilearn) for multi-label splits
- Ratio: 70% train / 15% validation / 15% test
- Verify each split has representation of all 9 themes

### Threshold Optimization
- Do NOT use fixed 0.5 threshold
- After training, optimize per-theme threshold on validation set
- Use F1-maximizing threshold per class (grid search 0.1 to 0.9)
- Expected optimal thresholds:
  - Navigational, Aspirational: ~0.4-0.5 (high-signal themes)
  - Resistance, CC: ~0.2-0.3 (need lower threshold due to difficulty)

### Evaluation Metrics
- **Primary**: Macro-F1 across 9 themes (treats all themes equally)
- **Secondary**: Micro-F1, per-class F1, per-class AUROC
- **Report**: Full classification report, per-theme confusion matrix, threshold curves

---

## RECOMMENDATION 9: What NOT to Do

### Do NOT merge Aspirational + Attainment
Despite high centroid similarity (0.951), these are theoretically distinct:
- Aspirational = hoping/dreaming about the future
- Attainment = concrete achievement/completion of goals
Attainment has 452 samples (viable) and d'=1.55 (easily separable in single-label).
The model should learn this distinction.

### Do NOT merge Social + Resistance
Despite high centroid similarity (0.932):
- Social = community/peer relationships and support
- Resistance = challenging systemic barriers
These are core CCW constructs that would lose theoretical meaning if merged.

### Do NOT merge CC + Spiritual
Despite centroid similarity of 0.930:
- CC = awareness of community-level issues
- Spiritual = inner strength, faith, cultural practices
These address fundamentally different forms of cultural wealth.

### Do NOT remove Class_0 sentences aggressively
The 255 boundary Class_0 candidates represent only 2.9% of Class_0. Removing them
would marginally help but risks removing sentences that discuss real challenges in a
way that ISN'T themed (the model needs to learn this "not a theme" signal).

### Do NOT downsample Class_0
Class_0 at 49.2% is already near-balanced with themed sentences (50.8%).
Downsampling would lose valuable "what themes DON'T look like" training signal.

---

## RECOMMENDATION 10: Expected Model Performance Ranges

Based on the data analysis, realistic per-class F1 expectations:

| Theme | Expected F1 Range | Confidence | Reasoning |
|-------|-------------------|-----------|-----------|
| Navigational | 0.75-0.85 | High | Largest theme, d'=1.07, 21% KNN error |
| Aspirational | 0.70-0.80 | High | Large, d'=1.06 |
| Perseverance | 0.65-0.75 | Medium | Good size, d'=0.99 |
| Social | 0.55-0.70 | Medium | Moderate size, scattered |
| Familial_Capital | 0.60-0.75 | Medium | Merged gives good size, d'=1.56 single-label |
| Spiritual | 0.55-0.70 | Medium | Moderate size, d'=1.10 |
| Attainment | 0.50-0.65 | Medium | Small but separable (d'=1.55) |
| Resistance | 0.35-0.50 | Low | Hardest theme, d'=0.39, 52% KNN error |
| CC | 0.30-0.50 | Low | Very small, needs augmentation |
| **Macro-F1** | **0.55-0.65** | Medium | Dragged down by Resistance and CC |

These are BEFORE any advanced techniques (curriculum learning, multi-task, etc.).
With threshold optimization and augmentation, each could improve by 5-10 points.

---

## IMPLEMENTATION ORDER

### Step 1: Data Preparation (this session)
1. Apply Familial + Filial_Piety merge → Familial_Capital
2. Drop First_Gen column (4 single-label sentences become Class_0)
3. Verify final dataset: 18,019 sentences, 9 themes + Class_0
4. Save as `ALMA_final_training_dataset.csv`

### Step 2: Train/Val/Test Split
1. Split by essay_id using iterative stratification
2. Verify theme representation in all splits
3. Save split indices for reproducibility

### Step 3: Data Augmentation (training set only)
1. Augment CC (4x), Attainment (1.5x), Resistance (1.4x)
2. Light augmentation for Spiritual, Familial_Capital
3. Verify augmented samples manually (10% spot-check)

### Step 4: Phase 1 Training (Baseline)
1. DeBERTa-v3-base, Focal Loss, macro-F1 metric
2. Train 10 epochs with early stopping (patience=3)
3. Evaluate: per-class F1, threshold optimization

### Step 5: Phase 2 (Iterative Refinement)
1. Extract DeBERTa embeddings from trained model
2. Re-run boundary analysis with DeBERTa embeddings
3. Remove stubborn misclassified points
4. Retrain on cleaned data

### Step 6: Phase 3 (Advanced Techniques if Needed)
1. Curriculum learning (easy themes first, hard themes later)
2. Multi-task auxiliary classifiers for Resistance
3. Consider DeBERTa-v3-large if base plateaus

---

## DECISION SUMMARY

| Decision | Action | Confidence |
|----------|--------|-----------|
| Familial + Filial_Piety | **MERGE** | 95% — evidence is overwhelming |
| First_Gen | **DROP** | 95% — unlearnable with 33 samples |
| Community_Consciousness | **KEEP** + augment | 75% — borderline, monitor F1 |
| Resistance | **KEEP** + extra care | 90% — theoretically essential |
| Aspirational + Attainment | **DO NOT merge** | 85% — distinct despite similarity |
| Social + Resistance | **DO NOT merge** | 90% — core CCW constructs |
| CC + Spiritual | **DO NOT merge** | 80% — fundamentally different |
| Remove 1,790 boundary pts | **WAIT** — train first, then re-evaluate | 80% |
| Downsample Class_0 | **DO NOT** — already balanced | 95% |
| Final taxonomy | **9 themes + Class_0** (10 classes) | 90% |

---
*Based on analysis of 36 plots across 3 analysis rounds, covering UMAP/PCA embeddings,
hierarchical and K-Means clustering, LOO-KNN classification, centroid similarity,
label correlation, separability metrics, boundary violations, and local density analysis.*
*Date: March 2026*
