# ALMA Final Datasets — Detailed Distribution Report

## Overview

| Metric | V1 (Original) | V2 (Merged+Cleaned) | V3 (Boundary Cleaned) | V4 (No CC) |
|--------|---------------|---------------------|----------------------|------------|
| **File** | v1_original_processed.csv | v2_merged_cleaned.csv | v3_boundary_cleaned.csv | v4_no_cc.csv |
| **Total sentences** | 18,019 | 17,859 | 16,395 | 17,622 |
| **Unique essays** | 2,698 | 2,675 | 2,665 | 2,636 |
| **Theme columns** | 11 | 9 | 9 | 8 |
| **Class_0:Themed ratio** | 0.97:1 | 0.98:1 | 1.10:1 | 1.00:1 |


## V1 — Original Processed Dataset (18,019 rows, 11 themes)

### Label Type Breakdown
| Type | Count | % |
|------|-------|---|
| Class_0 (0 themes) | 8,870 | 49.2% |
| Single-label (1 theme) | 6,390 | 35.5% |
| Multi-label (2 themes) | 2,186 | 12.1% |
| Multi-label (3 themes) | 494 | 2.7% |
| Multi-label (4+ themes) | 79 | 0.4% |
| Max themes/sentence | 7 | — |

### Per-Theme Distribution
| Theme | Total | % of Data | Single-Label | Multi-Label | % Multi | Imbalance vs Class_0 |
|-------|-------|-----------|-------------|-------------|---------|---------------------|
| Class_0 | 8,870 | 49.2% | 8,870 | 0 | 0.0% | 1.0x |
| Navigational | 4,322 | 24.0% | 2,484 | 1,838 | 42.5% | 2.1x |
| Aspirational | 2,756 | 15.3% | 1,046 | 1,710 | 62.0% | 3.2x |
| Perseverance | 1,516 | 8.4% | 1,211 | 305 | 20.1% | 5.9x |
| Social | 978 | 5.4% | 473 | 505 | 51.6% | 9.1x |
| Resistance | 830 | 4.6% | 258 | 572 | 68.9% | 10.7x |
| Spiritual | 710 | 3.9% | 514 | 196 | 27.6% | 12.5x |
| Familial | 640 | 3.6% | 228 | 412 | 64.4% | 13.9x |
| Attainment | 452 | 2.5% | 100 | 352 | 77.9% | 19.6x |
| Filial_Piety | 240 | 1.3% | 53 | 187 | 77.9% | 37.0x |
| Community_Consciousness | 119 | 0.7% | 19 | 100 | 84.0% | 74.5x |
| First_Gen | 33 | 0.2% | 4 | 29 | 87.9% | 268.8x |

Average themes per themed sentence: 1.38

### Top 15 Label Combinations
| Rank | Combination | Count | % |
|------|------------|-------|---|
| 1 | Class_0 | 8,870 | 49.2% |
| 2 | Navigational | 2,484 | 13.8% |
| 3 | Perseverance | 1,211 | 6.7% |
| 4 | Aspirational | 1,046 | 5.8% |
| 5 | Aspirational + Navigational | 825 | 4.6% |
| 6 | Spiritual | 514 | 2.9% |
| 7 | Social | 473 | 2.6% |
| 8 | Resistance | 258 | 1.4% |
| 9 | Familial | 228 | 1.3% |
| 10 | Navigational + Resistance | 213 | 1.2% |
| 11 | Attainment + Navigational | 143 | 0.8% |
| 12 | Aspirational + Social | 127 | 0.7% |
| 13 | Navigational + Social | 106 | 0.6% |
| 14 | Attainment | 100 | 0.6% |
| 15 | Familial + Navigational | 88 | 0.5% |

---

## V2 — Merged + Cleaned Dataset (17,859 rows, 9 themes)

**Changes from V1:**
- Familial (640) + Filial_Piety (240) merged into Familial_Capital (748)
  - 98 sentences had BOTH labels, so 748 not 880
- First_Gen dropped: 23 complete essays removed (160 sentences)

### Label Type Breakdown
| Type | Count | % |
|------|-------|---|
| Class_0 (0 themes) | 8,848 | 49.5% |
| Single-label (1 theme) | 6,370 | 35.7% |
| Multi-label (2 themes) | 2,111 | 11.8% |
| Multi-label (3 themes) | 463 | 2.6% |
| Multi-label (4+ themes) | 67 | 0.4% |
| Max themes/sentence | 6 | — |

### Per-Theme Distribution
| Theme | Total | % of Data | Single-Label | Multi-Label | % Multi | Imbalance vs Class_0 |
|-------|-------|-----------|-------------|-------------|---------|---------------------|
| Class_0 | 8,848 | 49.5% | 8,848 | 0 | 0.0% | 1.0x |
| Navigational | 4,274 | 23.9% | 2,460 | 1,814 | 42.4% | 2.1x |
| Aspirational | 2,708 | 15.2% | 1,037 | 1,671 | 61.7% | 3.3x |
| Perseverance | 1,504 | 8.4% | 1,205 | 299 | 19.9% | 5.9x |
| Social | 974 | 5.5% | 472 | 502 | 51.5% | 9.1x |
| Resistance | 826 | 4.6% | 256 | 570 | 69.0% | 10.7x |
| Familial_Capital | 748 | 4.2% | 317 | 431 | 57.6% | 11.8x |
| Spiritual | 697 | 3.9% | 509 | 188 | 27.0% | 12.7x |
| Attainment | 428 | 2.4% | 96 | 332 | 77.6% | 20.7x |
| Community_Consciousness | 111 | 0.6% | 18 | 93 | 83.8% | 79.7x |

Average themes per themed sentence: 1.36

### Top 15 Label Combinations
| Rank | Combination | Count | % |
|------|------------|-------|---|
| 1 | Class_0 | 8,848 | 49.5% |
| 2 | Navigational | 2,460 | 13.8% |
| 3 | Perseverance | 1,205 | 6.7% |
| 4 | Aspirational | 1,037 | 5.8% |
| 5 | Aspirational + Navigational | 818 | 4.6% |
| 6 | Spiritual | 509 | 2.9% |
| 7 | Social | 472 | 2.6% |
| 8 | Familial_Capital | 317 | 1.8% |
| 9 | Resistance | 256 | 1.4% |
| 10 | Navigational + Resistance | 213 | 1.2% |
| 11 | Attainment + Navigational | 141 | 0.8% |
| 12 | Aspirational + Social | 124 | 0.7% |
| 13 | Aspirational + Familial_Capital | 117 | 0.7% |
| 14 | Navigational + Social | 106 | 0.6% |
| 15 | Familial_Capital + Navigational | 98 | 0.5% |

---

## V3 — Boundary Cleaned Dataset (16,395 rows, 9 themes)

**Changes from V2:**
- 1,464 boundary violation sentences removed:
  - 255 Class_0 sentences sitting on top of themed data points
  - 880 themed sentences strongly predicted as Class_0 by neighbors (conf > 0.5)
  - 329 themed sentences strongly confused with a different theme

### Label Type Breakdown
| Type | Count | % |
|------|-------|---|
| Class_0 (0 themes) | 8,593 | 52.4% |
| Single-label (1 theme) | 5,638 | 34.4% |
| Multi-label (2 themes) | 1,750 | 10.7% |
| Multi-label (3 themes) | 365 | 2.2% |
| Multi-label (4+ themes) | 49 | 0.3% |
| Max themes/sentence | 6 | — |

### Per-Theme Distribution
| Theme | Total | % of Data | Single-Label | Multi-Label | % Multi | Imbalance vs Class_0 |
|-------|-------|-----------|-------------|-------------|---------|---------------------|
| Class_0 | 8,593 | 52.4% | 8,593 | 0 | 0.0% | 1.0x |
| Navigational | 3,787 | 23.1% | 2,284 | 1,503 | 39.7% | 2.3x |
| Aspirational | 2,414 | 14.7% | 960 | 1,454 | 60.2% | 3.6x |
| Perseverance | 1,247 | 7.6% | 1,051 | 196 | 15.7% | 6.9x |
| Social | 788 | 4.8% | 384 | 404 | 51.3% | 10.9x |
| Familial_Capital | 634 | 3.9% | 279 | 355 | 56.0% | 13.6x |
| Resistance | 619 | 3.8% | 189 | 430 | 69.5% | 13.9x |
| Spiritual | 518 | 3.2% | 390 | 128 | 24.7% | 16.6x |
| Attainment | 354 | 2.2% | 87 | 267 | 75.4% | 24.3x |
| Community_Consciousness | 81 | 0.5% | 14 | 67 | 82.7% | 106.1x |

Average themes per themed sentence: 1.34

### Top 15 Label Combinations
| Rank | Combination | Count | % |
|------|------------|-------|---|
| 1 | Class_0 | 8,593 | 52.4% |
| 2 | Navigational | 2,284 | 13.9% |
| 3 | Perseverance | 1,051 | 6.4% |
| 4 | Aspirational | 960 | 5.9% |
| 5 | Aspirational + Navigational | 765 | 4.7% |
| 6 | Spiritual | 390 | 2.4% |
| 7 | Social | 384 | 2.3% |
| 8 | Familial_Capital | 279 | 1.7% |
| 9 | Resistance | 189 | 1.2% |
| 10 | Navigational + Resistance | 144 | 0.9% |
| 11 | Attainment + Navigational | 115 | 0.7% |
| 12 | Aspirational + Familial_Capital | 110 | 0.7% |
| 13 | Aspirational + Social | 102 | 0.6% |
| 14 | Attainment | 87 | 0.5% |
| 15 | Navigational + Social | 84 | 0.5% |

---

## V4 — No Community Consciousness Dataset (17,622 rows, 8 themes)

**Changes from V2:**
- Community_Consciousness dropped: 39 complete essays removed (237 sentences, 1.3% loss)
- Collateral theme losses from dropped essays: Spiritual -7.5%, Attainment -5.4%, Familial_Capital -4.8%

### Label Type Breakdown
| Type | Count | % |
|------|-------|---|
| Class_0 (0 themes) | 8,815 | 50.0% |
| Single-label (1 theme) | 6,288 | 35.7% |
| Multi-label (2 themes) | 2,030 | 11.5% |
| Multi-label (3 themes) | 429 | 2.4% |
| Multi-label (4+ themes) | 60 | 0.3% |
| Max themes/sentence | 6 | — |

### Per-Theme Distribution
| Theme | Total | % of Data | Single-Label | Multi-Label | % Multi | Imbalance vs Class_0 |
|-------|-------|-----------|-------------|-------------|---------|---------------------|
| Class_0 | 8,815 | 50.0% | 8,815 | 0 | 0.0% | 1.0x |
| Navigational | 4,245 | 24.1% | 2,449 | 1,796 | 42.3% | 2.1x |
| Aspirational | 2,658 | 15.1% | 1,029 | 1,629 | 61.3% | 3.3x |
| Perseverance | 1,475 | 8.4% | 1,197 | 278 | 18.8% | 6.0x |
| Social | 960 | 5.4% | 470 | 490 | 51.0% | 9.2x |
| Resistance | 793 | 4.5% | 245 | 548 | 69.1% | 11.1x |
| Familial_Capital | 712 | 4.0% | 305 | 407 | 57.2% | 12.4x |
| Spiritual | 645 | 3.7% | 501 | 144 | 22.3% | 13.7x |
| Attainment | 405 | 2.3% | 92 | 313 | 77.3% | 21.8x |

Average themes per themed sentence: 1.35

### Top 15 Label Combinations
| Rank | Combination | Count | % |
|------|------------|-------|---|
| 1 | Class_0 | 8,815 | 50.0% |
| 2 | Navigational | 2,449 | 13.9% |
| 3 | Perseverance | 1,197 | 6.8% |
| 4 | Aspirational | 1,029 | 5.8% |
| 5 | Aspirational + Navigational | 812 | 4.6% |
| 6 | Spiritual | 501 | 2.8% |
| 7 | Social | 470 | 2.7% |
| 8 | Familial_Capital | 305 | 1.7% |
| 9 | Resistance | 245 | 1.4% |
| 10 | Navigational + Resistance | 213 | 1.2% |
| 11 | Attainment + Navigational | 139 | 0.8% |
| 12 | Aspirational + Social | 121 | 0.7% |
| 13 | Aspirational + Familial_Capital | 114 | 0.6% |
| 14 | Navigational + Social | 106 | 0.6% |
| 15 | Familial_Capital + Navigational | 98 | 0.6% |

---

## V1 → V2 → V3 / V4 Theme-by-Theme Comparison

| Theme | V1 | V2 | V3 | V4 | V1→V2 Loss | V2→V3 Loss | V2→V4 Loss |
|-------|-----|-----|-----|-----|-----------|-----------| -----------|
| Class_0 | 8,870 | 8,848 | 8,593 | 8,815 | -22 (0.2%) | -255 (2.9%) | -33 (0.4%) |
| Navigational | 4,322 | 4,274 | 3,787 | 4,245 | -48 (1.1%) | -487 (11.4%) | -29 (0.7%) |
| Aspirational | 2,756 | 2,708 | 2,414 | 2,658 | -48 (1.7%) | -294 (10.9%) | -50 (1.8%) |
| Perseverance | 1,516 | 1,504 | 1,247 | 1,475 | -12 (0.8%) | -257 (17.1%) | -29 (1.9%) |
| Social | 978 | 974 | 788 | 960 | -4 (0.4%) | -186 (19.1%) | -14 (1.4%) |
| Resistance | 830 | 826 | 619 | 793 | -4 (0.5%) | -207 (25.1%) | -33 (4.0%) |
| Familial | 640 | — | — | — | merged | — | — |
| Filial_Piety | 240 | — | — | — | merged | — | — |
| Familial_Capital | — | 748 | 634 | 712 | (new) | -114 (15.2%) | -36 (4.8%) |
| Spiritual | 710 | 697 | 518 | 645 | -13 (1.8%) | -179 (25.7%) | -52 (7.5%) |
| Attainment | 452 | 428 | 354 | 405 | -24 (5.3%) | -74 (17.3%) | -23 (5.4%) |
| Community_Consciousness | 119 | 111 | 81 | DROPPED | -8 (6.7%) | -30 (27.0%) | dropped |
| First_Gen | 33 | DROPPED | DROPPED | DROPPED | dropped | — | — |

---

## Key Observations

1. **V2 is conservative**: Only 160 sentences lost (0.9%). The merge adds no data loss —
   just relabeling. The only loss is from dropping First_Gen complete essays.

2. **V3 is aggressive**: 1,464 additional sentences removed (8.2% of V2).
   Hardest hit: CC (-27%), Spiritual (-26%), Resistance (-25%). Class_0 shifts to majority (52.4%).

3. **V4 is minimal-loss**: Only 237 sentences lost (1.3% of V2). Drops CC entirely but
   collateral damage to other themes is small (max 7.5% for Spiritual).

4. **V4 achieves perfect balance**: Class_0:Themed = 1.00:1 (50.0% / 50.0%).
   Best balanced of all 4 datasets.

5. **V4 eliminates worst imbalance**: Max imbalance drops from 79.7x (CC in V2) to
   21.8x (Attainment in V4) — a 3.7x improvement in worst-case ratio.

6. **Multi-label stays stable**: V1 15.3% → V2 14.8% → V4 14.3% → V3 13.2%.
   V4 preserves multi-label structure better than V3.

7. **Familial_Capital merge effect**: V1 had Familial=640 + Filial_Piety=240 = 880 total labels,
   but 98 sentences had BOTH, so Familial_Capital = 748 unique sentences.

8. **V3 vs V4 trade-off**: V3 has cleaner boundaries but fewer samples and more imbalance.
   V4 has more samples, better balance, but retains boundary noise. V4 is recommended
   as the primary training dataset; V3 as an alternative for comparison.

---
*Generated from prepare_final_datasets.py*
*Source: ALMA_processed_master_dataset.csv (18,019 sentences from student reflective essays)*
*Date: March 2026*
