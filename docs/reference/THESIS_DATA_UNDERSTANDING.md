# ALMA Data Understanding Report — Complete Analysis for Thesis Writing

## 1. Dataset Overview

The ALMA dataset contains **18,019 sentences** extracted from student reflective essays
written in Physics and Astronomy labs at San Francisco State University (Fall 2018 - Spring 2025).
Each sentence has been human-annotated for the presence of **11 Community Cultural Wealth (CCW)
themes** based on Yosso's (2005) theoretical framework, extended with additional constructs.

Sentences with no theme annotations are classified as **Class_0** (no CCW theme detected).
This is a **multi-label classification** task — a single sentence can express multiple themes
simultaneously.

### 1.1 Data Processing Pipeline

The dataset underwent two rounds of processing:

**Round 1 (Master Dataset Construction):**
- 4,079 essays assembled from 8 annotated CSV files + 12 unannotated prompt folders
- Sentence-level segmentation produced 19,724 sentences
- Binary theme labels derived from text excerpts (positive) vs 0/NaN (negative)
- Mojibake characters cleaned; duplicates removed

**Round 2 (Embedding-Based Quality Cleaning):**
- Removed 1,705 sentences (8.6%) using three rules:
  - R1: 1,337 Class_0 sentences in confusion zones (7+ of 10 nearest neighbors are themed)
  - R2: 238 single-label theme outliers closer to Class_0 centroid than own theme centroid
  - R3: 130 Class_0 sentences with very high similarity to rare theme centroids (P90 strict)
- Result: 18,019 sentences with improved Class_0:Themed balance (0.97:1)


## 2. Class Distribution

| Theme | Count | % of Dataset | Imbalance Ratio |
|-------|-------|-------------|-----------------|
| Class_0 | 8,870 | 49.2% | 1.0x (baseline) |
| Navigational | 4,322 | 24.0% | 2.1x |
| Aspirational | 2,756 | 15.3% | 3.2x |
| Perseverance | 1,516 | 8.4% | 5.9x |
| Social | 978 | 5.4% | 9.1x |
| Resistance | 830 | 4.6% | 10.7x |
| Spiritual | 710 | 3.9% | 12.5x |
| Familial | 640 | 3.6% | 13.9x |
| Attainment | 452 | 2.5% | 19.6x |
| Filial_Piety | 240 | 1.3% | 36.9x |
| Community_Consciousness | 119 | 0.7% | 74.5x |
| First_Gen | 33 | 0.2% | 268.8x |

Key observations:
- Class distribution spans 3 orders of magnitude (8,870 to 33)
- Top 3 themes (Navigational, Aspirational, Perseverance) account for 74.6% of all theme labels
- Bottom 3 themes (Filial_Piety, CC, First_Gen) account for only 3.4%
- Class_0:Themed ratio is 0.97:1 (near-balanced at the binary level)


## 3. Multi-Label Structure

| Label Type | Count | Percentage |
|------------|-------|-----------|
| Class_0 (0 themes) | 8,870 | 49.2% |
| Single-label (1 theme) | 6,390 | 35.5% |
| Multi-label (2+ themes) | 2,759 | 15.3% |
| Maximum themes per sentence | 7 | — |

### 3.1 Multi-Label Rates by Theme

Themes differ dramatically in how often they appear alone vs. with other themes:

| Theme | % Multi-Label | Single-Label Count |
|-------|--------------|-------------------|
| First_Gen | 87.9% | 4 |
| Community_Consciousness | 84.0% | 19 |
| Filial_Piety | 77.9% | 53 |
| Attainment | 77.9% | 100 |
| Resistance | 68.9% | 258 |
| Familial | 64.4% | 228 |
| Aspirational | 62.0% | 1,046 |
| Social | 51.6% | 473 |
| Navigational | 42.5% | 2,484 |
| Spiritual | 27.6% | 514 |
| Perseverance | 20.1% | 1,211 |

**Implications:**
- First_Gen essentially never appears alone (only 4 single-label sentences). It always
  co-occurs with other themes, meaning it functions more as a modifier than an independent theme.
- Community_Consciousness and Filial_Piety are similar — they rarely stand alone.
- Perseverance and Spiritual are the most "independent" themes — they mostly appear alone.

### 3.2 Theme Co-occurrence Patterns (Jaccard Similarity)

All theme-pair Jaccard similarities are low (<0.20), meaning themes rarely co-occur
in the same sentence. The most common co-occurrence is Navigational + Aspirational (Jaccard ~0.19).
This is important: multi-label classification is needed, but most sentences have only 1-2 themes.

### 3.3 Most Common Label Combinations

1. Navigational only (1,684 sentences)
2. Perseverance only (1,211)
3. Aspirational only (1,046)
4. Spiritual only (514)
5. Social only (473)
6. Navigational + Resistance (213)
7. Aspirational + Navigational (143)
8. Familial only (228)


## 4. Embedding Space Analysis (all-MiniLM-L6-v2, 384 dimensions)

### 4.1 Dimensionality and Structure

- **PCA**: 50 components capture only ~65% of variance. The data is truly high-dimensional —
  there is no low-dimensional subspace that separates the classes.
- **UMAP 2D**: Reveals a single large connected manifold with no clean cluster boundaries.
  Themed points and Class_0 points are deeply interleaved throughout the space.
- **Optimal K (silhouette)**: K=2. The data naturally forms only TWO groups, not 12.
  This means the 12-class taxonomy is a human construct imposed on a continuous semantic space,
  not something the embedding model can discover from text alone.

### 4.2 Centroid Similarity Analysis

Cosine similarity between class centroids reveals how semantically distinct each theme is
from others in the embedding space:

**Most Similar (Hardest to Separate):**
| Pair | Cosine Similarity | Interpretation |
|------|------------------|----------------|
| Familial ↔ Filial_Piety | 0.987 | Nearly identical embeddings |
| Aspirational ↔ Attainment | 0.951 | Very high overlap |
| Aspirational ↔ Navigational | 0.948 | Very high overlap |
| Aspirational ↔ Resistance | 0.940 | High overlap |
| Class_0 ↔ Resistance | 0.933 | Hardest Class_0/theme boundary |
| Social ↔ Resistance | 0.932 | High overlap |
| CC ↔ Spiritual | 0.930 | High overlap |

**Most Different (Easiest to Separate):**
| Pair | Cosine Similarity | Interpretation |
|------|------------------|----------------|
| First_Gen ↔ Navigational | 0.691 | Most distinct pair |
| First_Gen ↔ Class_0 | 0.695 | First_Gen is furthest from Class_0 |
| First_Gen ↔ Perseverance | 0.654 | Very different semantics |
| Familial ↔ Class_0 | 0.750 | Family language distinct from generic |

### 4.3 Single-Label Centroid Analysis

When we compute centroids using only single-label sentences (purest class representations),
separation improves:

| Pair | All-Data Similarity | Single-Label Similarity | Improvement |
|------|-------------------|------------------------|-------------|
| Familial ↔ Filial_Piety | 0.987 | 0.947 | +0.040 |
| Aspirational ↔ Navigational | 0.948 | 0.901 | +0.047 |
| Aspirational ↔ Resistance | 0.940 | 0.877 | +0.063 |
| Class_0 ↔ Resistance | 0.933 | 0.928 | +0.005 |

Multi-label sentences pull centroids toward each other, making classes appear more similar
than they truly are. Single-label centroids are the "ground truth" class representations.

### 4.4 Centroid Shift from Multi-Label

How much does the centroid move when we use only single-label sentences?

| Theme | Euclidean Shift | Interpretation |
|-------|----------------|----------------|
| First_Gen | 0.4262 | MASSIVE shift — centroid is completely different |
| Community_Consciousness | 0.2289 | Large shift |
| Filial_Piety | 0.1614 | Moderate shift |
| Attainment | 0.1272 | Moderate shift |
| Familial | 0.1051 | Moderate shift |
| Resistance | 0.0921 | Moderate shift |
| Social | 0.0777 | Small shift |
| Aspirational | 0.0746 | Small shift |
| Navigational | 0.0644 | Small shift |
| Perseverance | 0.0495 | Minimal shift |
| Spiritual | 0.0309 | Minimal shift |
| Class_0 | 0.0000 | No shift (all are single-label by definition) |

Themes with high multi-label rates (First_Gen, CC, Filial_Piety) show the largest centroid
shifts, confirming that multi-label contamination distorts their representations.


## 5. Separability Analysis (d-prime)

d' (d-prime) measures how separable each class is from all other classes, computed on
single-label sentences using cosine similarity to own centroid vs nearest other centroid:

| Theme | d' | Difficulty | Single-Label N |
|-------|-----|-----------|---------------|
| First_Gen | 6.01 | Easy* | 4 |
| Community_Consciousness | 1.88 | Easy* | 19 |
| Familial | 1.62 | Easy | 228 |
| Filial_Piety | 1.56 | Easy | 53 |
| Attainment | 1.55 | Easy | 100 |
| Spiritual | 1.10 | Easy | 514 |
| Navigational | 1.07 | Easy | 2,484 |
| Aspirational | 1.06 | Easy | 1,046 |
| Social | 0.99 | Moderate | 473 |
| Perseverance | 0.99 | Moderate | 1,211 |
| Resistance | 0.39 | Hard | 258 |
| Class_0 | 0.28 | Very Hard | 8,870 |

*First_Gen and CC d' values are inflated by tiny sample size (4 and 19 respectively)

**Key findings:**
- Most themes have d' > 1.0, meaning they ARE separable with fine-tuning
- Resistance (d'=0.39) is genuinely hard — its language overlaps with both Class_0 and
  other themes. The model will struggle most with this theme.
- Class_0 (d'=0.28) is the hardest — it's not a coherent semantic class but a catch-all
  for "everything else," so it doesn't cluster in embedding space.


## 6. Leave-One-Out KNN Analysis

Using leave-one-out k=15 nearest neighbors (excluding self) as a proxy for how well a
classifier could separate the classes from embeddings alone:

| Theme | LOO-KNN Mismatch % | Interpretation |
|-------|-------------------|----------------|
| First_Gen | 88% | Nearly all points misclassified |
| Community_Consciousness | 75% | Three-quarters misclassified |
| Filial_Piety | 64% | Majority misclassified |
| Spiritual | 56% | Majority misclassified |
| Resistance | 52% | Coin flip |
| Familial | 50% | Coin flip |
| Attainment | 44% | Slightly better than chance |
| Social | 43% | Slightly better than chance |
| Perseverance | 35% | Moderate accuracy |
| Aspirational | 33% | Moderate accuracy |
| Navigational | 21% | Good accuracy |
| Class_0 | 20% | Good accuracy (large size helps) |

**Overall**: 38.7% of all points are misclassified by KNN on embeddings alone.
This is why fine-tuning DeBERTa is essential — frozen embeddings cannot separate these classes.

### 6.1 Confusion Patterns

When Class_0 is misclassified by KNN, it's most often predicted as:
1. Perseverance (~400 instances)
2. Navigational (~350)
3. Aspirational (~250)

When themed points are misclassified:
- 60.1% predicted as Class_0 (theme language looks generic)
- 39.9% predicted as another theme (inter-theme confusion)


## 7. Local Neighborhood Density

Average % of k=20 nearest neighbors that share the same primary label:

| Theme | Avg Same-Class Density |
|-------|----------------------|
| Class_0 | 56.8% |
| Navigational | 30.5% |
| Aspirational | 19.8% |
| Perseverance | 17.4% |
| Social | 10.2% |
| Resistance | 9.1% |
| Spiritual | 8.8% |
| Familial | 8.5% |
| Attainment | 7.8% |
| Filial_Piety | 7.0% |
| Community_Consciousness | 7.6% |
| First_Gen | 5.8% |

**Interpretation**: Only Class_0, Navigational, and Aspirational have enough density to form
local clusters. All other themes are scattered — their points are isolated among points
from other classes. This is the core challenge: rare themes don't cluster spatially,
so the model must learn semantic features rather than spatial proximity.


## 8. Clustering Analysis

### 8.1 K-Means (Optimal K=2)

The silhouette score peaks at K=2, meaning the data naturally splits into just two groups.
Both K=2 clusters are ~50% Class_0 each with similar theme distributions. The K=2 split
appears to reflect topic/prompt differences rather than Class_0-vs-themed separation.

### 8.2 Hierarchical Clustering (K=12, Ward)

When forced to K=12 clusters to match the class count:

**Theme-dominated clusters (3 of 12):**
- One cluster is Perseverance-heavy (~35-40%, only ~25% Class_0)
- One cluster is Aspirational-concentrated with some Perseverance
- One cluster is Navigational-dominated (~32%) with ~37% Class_0

**Mixed clusters (9 of 12):**
- All other clusters have 40-70% Class_0 contamination
- No cluster is dominated by Familial, Social, Resistance, Attainment, Spiritual,
  Filial_Piety, CC, or First_Gen

**Label distribution across clusters:**
- Navigational: 65% captured by 3 clusters (decent concentration)
- Perseverance: 50% captured by 2 clusters (moderate)
- Aspirational: Spread across 4 clusters (scattered)
- Resistance: No single cluster captures >20% (worst concentration)
- Class_0: Spread across all clusters (everywhere)


## 9. Theme-vs-Class_0 Overlap (KDE Density Analysis)

For each theme, the density overlap with Class_0 in UMAP space:

- **Navigational**: Has its own distinct region (bottom-right) but also overlaps Class_0 in center
- **Aspirational**: Occupies upper-center region, moderate Class_0 overlap
- **Perseverance**: Distinct bottom-left cluster, least Class_0 overlap
- **Resistance**: Almost completely overlapping with Class_0 — nearly indistinguishable
- **Social**: Moderate overlap, scattered
- **Spiritual**: Has a small distinct pocket but mostly mixed
- **Familial**: Scattered throughout Class_0 regions
- **Filial_Piety**: Very scattered, no distinct region
- **Community_Consciousness**: Too few points to form any visible cluster
- **First_Gen**: Too few points, completely scattered


## 10. Boundary Violation Analysis

### 10.1 Class_0 Sitting on Themed Data Points

Among 8,870 Class_0 sentences:
- Average cosine similarity to nearest themed neighbor: 0.567
- P95 threshold: 0.76 (444 Class_0 points with sim > 0.76 to nearest themed point)
- These 444 Class_0 points sit directly inside themed regions, primarily overlapping with:
  - Perseverance (~200 points)
  - Navigational (~90 points)
  - Spiritual (~60 points)

### 10.2 Conservative Removal Candidates

Using multi-signal criteria (LOO-KNN + local density + embedding similarity):
- **1,790 total removal candidates** (9.9% of dataset)
  - R1: Class_0 on theme zones (P95 sim + <30% density): 62
  - R2: Class_0 LOO mismatch + isolated: 207
  - R3: Class_0 deep in theme zone (<10% density): 48
  - R4: Themed points predicted as Class_0 (conf>0.4, <15% density): 1,202
  - R5: Themed points confused with other themes (conf>0.5, <10% density): 333


## 11. Label Correlation Matrix

Phi coefficient correlations between theme labels:
- **Navigational ↔ Class_0**: -0.55 (strong negative — Navigational is the strongest indicator of NOT being Class_0)
- **Familial ↔ Filial_Piety**: +0.23 (strongest positive inter-theme correlation — they co-occur)
- **Navigational ↔ Aspirational**: +0.19 (tend to co-occur)
- **CC ↔ Spiritual**: +0.12 (slight positive)
- Most inter-theme correlations near 0 (themes are largely independent)


## 12. Summary of Key Data Characteristics

1. **Extreme class imbalance**: 269:1 ratio between largest (Class_0) and smallest (First_Gen)
2. **High centroid similarity**: Most class pairs have cosine sim > 0.85 — the semantic
   space is fundamentally overlapping. DeBERTa fine-tuning must learn to separate what
   pre-trained embeddings cannot.
3. **Multi-label complexity**: 15.3% of sentences have 2+ themes. Some themes (First_Gen,
   CC, Filial_Piety) almost never appear alone — they function as co-occurring modifiers.
4. **Heterogeneous separability**: Themes range from easily separable (Familial, d'=1.62)
   to nearly inseparable from Class_0 (Resistance, d'=0.39).
5. **Class_0 is not a true class**: It's a catch-all for "no theme detected," which means
   it has no coherent semantic identity. It's defined by absence, not presence.
6. **Natural clustering = 2 groups**: The embedding space doesn't support 12 distinct
   clusters. Only Perseverance, Navigational, and Aspirational form any spatial concentration.
   All other themes are too rare and scattered to cluster.
7. **38.7% LOO-KNN error rate**: Pre-trained embeddings alone classify only 61.3% of
   points correctly. Fine-tuning is essential to move accuracy above this baseline.

---
*This document was generated from comprehensive analysis of 36 plots across 3 analysis rounds
(comprehensive, single-label, boundary) on the ALMA processed dataset (18,019 sentences).*
*Embedding model: all-MiniLM-L6-v2 (384 dimensions)*
*Date: March 2026*
