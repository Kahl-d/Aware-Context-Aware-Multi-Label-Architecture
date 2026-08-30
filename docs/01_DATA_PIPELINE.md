# Data Pipeline — Raw to Dashboard JSON

## Data Journey Overview

```
Raw XLSX (13,563 files across 7 semesters)
    ↓ Stage 1: Assembly & Normalization
2,710 annotated essays + 1,388 unannotated
    ↓ Stage 2: Sentence Segmentation + Label Propagation
19,724 sentences (11 themes) — ALMA_sentence_level_dataset.csv
    ↓ Stage 3: Embedding-Based Semantic Cleaning (1,705 removed)
18,019 sentences — V1
    ↓ Stage 4: Theme Consolidation
V2: 17,859 (merge Familial+Filial_Piety, drop First_Gen)
V3: 16,395 (aggressive boundary cleaning)
V4: 17,622 (drop Community_Consciousness) — FINAL TRAINING
```

## Source Files for Dashboard

| File | Path | Records | Purpose |
|------|------|---------|---------|
| V4 (final) | `Data/Final_Data/v4_no_cc.csv` | 17,622 | Final training dataset (8 themes) |
| V1 | `Data/Final_Data/v1_original_processed.csv` | 18,019 | After cleaning (11 themes) |
| V2 | `Data/Final_Data/v2_merged_cleaned.csv` | 17,859 | After merge + First_Gen drop |
| V3 | `Data/Final_Data/v3_boundary_cleaned.csv` | 16,395 | Aggressive boundary cleaning |
| Master sentences | `Data/ALMA_Master_Dataset/ALMA_sentence_level_dataset.csv` | 19,724 | Pre-cleaning master |
| Master essays | `Data/ALMA_Master_Dataset/annotated/ALMA_all_annotated_combined.csv` | 2,710 | Essay-level with excerpts |
| Unannotated | `Data/ALMA_Master_Dataset/unannotated/*/unannotated_*.csv` | ~1,388 | No labels, 12 prompt folders |
| Split stats | `Models_inference/data/splits_stats.json` | — | Train/val/test info |
| Split pkl | `Models_inference/data/{train,val,test}_data.pkl` | — | Actual split assignments |
| V3 removal log | `Data/Final_Data/v3_removal_log.csv` | 1,465 | Why sentences were dropped in V3 |

## V4 CSV Columns (19 total)

```
essay_id, sentence_id, sentence, sentence_length, alma_id, course,
semester, year, prompt, source_file, coder,
Attainment, Aspirational, Navigational, Resistance, Perseverance,
Social, Spiritual, Familial_Capital, Class_0
```

## Data Tagging Strategy for Dashboard

Every sentence in the dashboard gets these computed tags:

| Tag | Type | Values | How to Compute |
|-----|------|--------|----------------|
| `annotated` | boolean | true/false | Has human labels (not null) |
| `dataset_versions` | string[] | ["v1","v2","v3","v4"] | Match by (essay_id, sentence_id) across all CSVs |
| `used_for_training` | boolean | true/false | Present in V4 dataset |
| `split` | string/null | "train"/"val"/"test"/null | From pkl files (essay_id → split) |
| `dropped_reason` | string/null | "semantic_cleaning"/"boundary"/"theme_consolidation"/null | Compare master vs V1, V1 vs V4 |

## Theme Evolution Across Versions

| Version | Themes | Changes |
|---------|--------|---------|
| Master (19,724) | 11: Attainment, Aspirational, Navigational, Resistance, Perseverance, Filial_Piety, Familial, Community_Consciousness, Social, Spiritual, First_Gen | Original full set |
| V1 (18,019) | 11 | Same themes, 1,705 ambiguous sentences removed |
| V2 (17,859) | 9 | Merged Familial+Filial_Piety → Familial_Capital; Dropped First_Gen (33 sentences, 87.9% LOO error) |
| V3 (16,395) | 9 | Additional 1,464 boundary violations removed |
| V4 (17,622) | 8 | Dropped Community_Consciousness from V2 (119 sentences, 84% multi-label) |

## Data Preparation Scripts Needed

### Script 1: `prepare_sentences.py`
- Load V4 as primary dataset (17,622 rows) — tag as `used_for_training: true`
- Load split pkl files → map essay_id to train/val/test
- Load V1 (18,019) → identify V1-only sentences → tag `dropped_reason: "theme_consolidation"`
- Load V2, V3 → compute `dataset_versions` array per sentence
- Load master (19,724) → identify master-only sentences → tag `dropped_reason: "semantic_cleaning"`
- For V1-only sentences: map 11 themes → 8 theme schema where possible
- **Output:** `sentences.json` (~15MB, ~20K records)

### Script 2: `prepare_unannotated.py`
- Read 12 CSV folders under `Data/ALMA_Master_Dataset/unannotated/`
- Segment each essay into sentences (same regex as model)
- Create records with all null labels, `annotated: false`
- **Output:** merged into `sentences.json` + `unannotated_essays.json`

### Script 3: `prepare_embeddings.py`
- Load `Models_light/plots_comparison/embeddings_{train,test}.npz`
- Extract or compute UMAP 2D + PCA 2D coordinates
- **Output:** `embeddings_umap.json`, `embeddings_pca.json`

### Script 4: `prepare_results.py`
- Merge evaluation_test.json from large v4, base, baselines
- **Output:** `evaluation_results.json`

### Script 5: `prepare_training_history.py`
- Merge history.json from base, large v3, large v4
- **Output:** `training_history.json`

### Script 6: `validate_data.py`
- Verify: 17,622 V4 sentences, 2,636 essays, 1,388 unannotated essays
- Verify: no essay_id in >1 split
- Verify: theme counts match thesis numbers exactly
- Verify: unannotated data NOT in any training split

## Unannotated Data Detail

Location: `Data/ALMA_Master_Dataset/unannotated/`

| Prompt Folder | Essay Count |
|---------------|-------------|
| what_are_my_goals | 297 |
| how_have_the_values_of_my_community_or_my_family | 232 |
| ai_in_stem | 188 |
| why_do_i_want_to_go_into_the_stem_field | 178 |
| when_life_gets_challenging | 124 |
| what_strategies_can_i_put_in_place_to_keep_me_focused | 103 |
| contribute_to_stem | 73 |
| who_can_i_go_to_when_times_are_tough | 67 |
| have_you_made_a_connection_with_someone_in_the_class | 51 |
| select_a_concept_that_was_covered_in_class | 28 |
| personal_values | 1 |
| **Total** | **~1,388** |

Columns: alma_id, user_id, course, semester, year, prompt, essay, essay_length, source_file

## Split Statistics

- **Train:** 2,095 essays / 14,023 sentences / 1,708 students
- **Val:** 268 essays / 1,757 sentences / 210 students
- **Test:** 273 essays / 1,842 sentences / 220 students
- **No data leakage:** Students don't appear in multiple splits
