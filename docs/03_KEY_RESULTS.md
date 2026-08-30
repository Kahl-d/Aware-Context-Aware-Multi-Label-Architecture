# Key Results, Claims & Numbers for Dashboard

## Headline Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Dataset size | 17,622 sentences | From 2,636 annotated essays |
| CCW themes | 8 | After consolidation from 11 |
| Best Macro-F1 | **0.494** [0.464, 0.520] | DeBERTa-v3-large v4 |
| Improvement over baseline | **+30.8%** | vs TF-IDF (0.378) |
| Single-theme F1 range | **0.568-0.896** | With PERFECT precision |
| Threshold optimization gain | **+0.084 F1** | Bigger than 5x parameter scaling |
| Parameters | 438M (large), 86M (base) | DeBERTa-v3 backbone |
| Unannotated essays | 1,443 | Available for model inference |

## Per-Theme Performance (Large v4, Test Set)

| Theme | F1 | Precision | Recall | PR-AUC | ROC-AUC | Support | Threshold |
|-------|-----|-----------|--------|--------|---------|---------|-----------|
| Navigational | **0.707** | 0.669 | 0.749 | 0.759 | 0.908 | 430 | 0.31 |
| Familial_Capital | **0.600** | 0.533 | 0.686 | 0.586 | 0.941 | 70 | 0.14 |
| Aspirational | **0.585** | 0.549 | 0.627 | 0.507 | — | 276 | 0.23 |
| Social | **0.506** | 0.538 | 0.478 | 0.488 | — | 90 | 0.22 |
| Perseverance | **0.480** | 0.570 | 0.415 | 0.536 | — | 176 | 0.35 |
| Resistance | **0.395** | 0.303 | 0.570 | 0.377 | — | 86 | 0.09 |
| Spiritual | **0.375** | 0.291 | 0.526 | 0.416 | — | 95 | 0.053 |
| Attainment | **0.303** | 0.224 | 0.469 | 0.205 | — | 32 | 0.05 |

## Model Comparison Table

| Model | Params | Macro-F1 | 95% CI | PR-AUC | ROC-AUC | Hamming | Exact Match |
|-------|--------|----------|--------|--------|---------|---------|-------------|
| Majority class | 0 | 0.000 | — | 0.085 | — | — | — |
| Random (prior) | 0 | 0.084 | — | 0.085 | — | — | — |
| Oracle constant | 0 | 0.151 | — | 0.085 | — | — | — |
| TF-IDF + LogReg | ~10K | 0.378 | — | 0.350 | — | — | — |
| **AWARE Base** | 86M | 0.474 | [0.445, 0.497] | 0.473 | 0.893 | 0.089 | 0.502 |
| AWARE Large v3 | 438M | 0.461 | [0.431, 0.486] | 0.462 | 0.901 | 0.086 | 0.522 |
| **AWARE Large v4** | **438M** | **0.494** | **[0.464, 0.520]** | **0.484** | **0.888** | **0.082** | **0.537** |

## THE Key Finding: Multi-Label Bottleneck

**Single-theme F1 (sentences with exactly 1 theme):**

| Theme | Single-Theme F1 | Overall F1 | Gap |
|-------|-----------------|-----------|-----|
| Navigational | **0.896** | 0.707 | -0.189 |
| Aspirational | **0.866** | 0.589 | -0.277 |
| Attainment | **0.727** | 0.303 | -0.424 |

**Claim:** "The model detects themes with perfect precision when they appear alone. The 0.494 macro-F1 reflects multi-label disentanglement difficulty, not detection failure."

## Most Quotable Findings (for Paper page)

1. **"Single-theme F1 = 0.568-0.896 with perfect precision. The 0.494 macro-F1 reflects multi-label disentanglement, not detection failure."**

2. **"Threshold optimization: +0.084 F1 (17% relative). Scaling 5x parameters: +0.020 F1 (4% relative). Calibration > Parameters."**

3. **"Context helps Resistance +118%, Aspirational only +6%. The difference: Resistance requires narrative framing; Aspirational has distinctive vocabulary."**

4. **"Embedding analysis predicted per-theme performance BEFORE training: Navigational (d'=1.07) → F1=0.707; Attainment (77.3% multi-label) → F1=0.303."**

5. **"438M parameters memorize 339 Attainment examples (train PR-AUC=0.959) but can't generalize (test=0.205). Future improvements come from DATA, not architecture."**

6. **"7 documented failure modes when scaling transformers on small datasets. Each has a targeted fix. Zero architectural changes, +0.033 F1."**

## The 7 Failure Modes (v3 → v4)

| # | Failure | v3 Setting | v4 Fix | Impact |
|---|---------|-----------|--------|--------|
| 1 | Encoder memorized too fast | LR=1.5e-5 | LR=5.0e-6 | Slowed memorization 3x |
| 2 | SWA never activated | Start at 50%, terminated at ep15 | Start at 25% | 0→22 checkpoints |
| 3 | LLRD froze bottom layers | decay=0.85 (24 layers → 0.02) | decay=0.92 | Bottom LR: 3e-7→8.5e-7 |
| 4 | R-Drop dominated loss | α=2.0 (55-72% of loss) | α=1.0 (~11%) | Task loss restored |
| 5 | Phase 1 too short | 4 epochs (PR-AUC=0.144) | 8 epochs (PR-AUC=0.398) | Stable head init |
| 6 | Early stopping too eager | patience=5 | patience=8 | Allow late-peaking themes |
| 7 | Phase 3 ineffective | 20.5K params only | Same (acknowledged) | Minimal impact |

**Result:** v3 F1=0.461 → v4 F1=0.494 (+0.033), zero architectural changes.

## Overfitting Analysis (Train-Test Gap)

| Theme | Train F1 | Test F1 | Gap | Train PR-AUC | Test PR-AUC | Gap |
|-------|---------|---------|-----|-------------|-------------|-----|
| Navigational | 0.732 | 0.707 | **0.025** | 0.802 | 0.759 | 0.043 |
| Aspirational | 0.687 | 0.585 | 0.102 | 0.758 | 0.507 | 0.251 |
| Perseverance | 0.742 | 0.480 | 0.262 | 0.859 | 0.536 | 0.323 |
| Social | 0.809 | 0.506 | 0.303 | 0.904 | 0.488 | 0.416 |
| Resistance | 0.696 | 0.395 | 0.301 | 0.956 | 0.377 | 0.579 |
| Spiritual | 0.676 | 0.375 | 0.301 | 0.992 | 0.416 | 0.576 |
| Familial_Capital | 0.909 | 0.600 | 0.309 | 0.969 | 0.586 | 0.383 |
| **Attainment** | **0.822** | **0.303** | **0.519** | 0.959 | 0.205 | **0.754** |
| **Macro** | 0.759 | 0.494 | **0.265** | 0.900 | 0.484 | 0.416 |

## Dataset Statistics for Dashboard

| Statistic | Value |
|-----------|-------|
| Total annotated essays | 2,636 (V4) / 2,710 (master) |
| Total sentences (V4) | 17,622 |
| Unannotated essays | 1,388 |
| Class_0 (no theme) | 8,815 (50.0%) |
| Single-label | 6,288 (35.7%) |
| Multi-label (2+) | 2,519 (14.3%) |
| Max themes per sentence | 6 |
| Max imbalance ratio | 21.8:1 (Attainment vs Class_0) |
| Courses | 26 unique |
| Semesters | Fall 2018 through Spring 2025 |
| Prompts | 5 reflective writing prompts |
| Annotators/coders | 6+ (AB, PL, KW, AM, NE, KC) |

## Available Plots (74 total)

### Models_light/plots_comparison/ (26 files)
- 01_class_distribution.png
- 02_label_correlation.png
- 03_multi_label_analysis.png
- 04_train_test_frequency.png
- 05_per_theme_f1_comparison.png
- 06_macro_f1_prauc_comparison.png
- 07_improvement_waterfall.png
- 08_overfitting_analysis.png
- 09_threshold_calibration.png
- 10_confusion_cooccurrence.png
- 11_umap_test.png, 11_umap_train.png
- 12_pca_test.png, 12_pca_train.png
- 13_centroid_similarity_test.png, 13_centroid_similarity_train.png
- 14_separability_comparison.png
- aware_architecture_diagram.png
- plot_dapt_loss.png
- plot_density_finetuned.png, plot_density_pretrained.png, plot_density_pretrained_vs_finetuned.png
- plot_umap_finetuned.png, plot_umap_pretrained.png, plot_umap_pretrained_vs_finetuned.png
- plot_training_curves.png
- plot_v3_vs_v4_comparison.png
- embeddings_train.npz, embeddings_test.npz

### Data/Data_Processing_v2/plots/ (18 files)
- UMAP, PCA, clustering, density analysis visualizations

### Data/Data_Processing_v2/plots_boundary/ (9 files)
- Boundary violation analysis visualizations

### Data/Data_Processing_v2/plots_single_label/ (12 files)
- Single-label subset analysis

### Data/Final_Data/ (8 files)
- V1→V4 UMAP/density comparison plots

## Diagrams to Create for AWARE Paper Page

1. **AWARE Architecture** — DeBERTa → Pooling → Position Embedding → BiLSTM → Head (interactive SVG)
2. **Data Pipeline Flow** — Raw → V1 → V2 → V3 → V4 (interactive nodes with counts)
3. **3-Phase Training Timeline** — Phase 1 (frozen) → Phase 2 (progressive unfreeze + SWA) → Phase 3 (head retrain)
4. **Loss Composition** — ASL 77% + R-Drop 11% + Essay Aux 12% (animated pie/bar)
5. **v3 vs v4 Fix Comparison** — Side-by-side config diffs with metric impact
6. **Threshold Optimization** — Before/after waterfall chart
7. **Embedding Space Before/After** — Pre-trained vs fine-tuned UMAP (side-by-side)
