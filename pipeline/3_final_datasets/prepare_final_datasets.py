"""
ALMA — Prepare 3 Final Datasets for Model Training

Dataset v1: Original processed data (18,019 sentences, 11 themes)
Dataset v2: Familial+Filial_Piety merged, First_Gen essays dropped (9 themes)
Dataset v3: v2 + boundary point removals (9 themes, cleaned boundaries)

All outputs saved to Final_Data/ folder.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Data_Processing_v2')
SOURCE_FILE = os.path.join(DATA_DIR, 'ALMA_processed_master_dataset.csv')
BOUNDARY_FILE = os.path.join(DATA_DIR, 'v2_boundary_removable.csv')

ALL_THEMES_V1 = ['Aspirational', 'Familial', 'Social', 'Navigational', 'Resistance',
                 'Attainment', 'First_Gen', 'Perseverance', 'Filial_Piety',
                 'Community_Consciousness', 'Spiritual']

ALL_THEMES_V2 = ['Aspirational', 'Familial_Capital', 'Social', 'Navigational', 'Resistance',
                 'Attainment', 'Perseverance', 'Community_Consciousness', 'Spiritual']


def print_distribution(df, themes, label):
    """Print class distribution for a dataset."""
    print(f"\n  --- {label} Distribution ---")
    print(f"  Total sentences: {len(df)}")

    if 'Class_0' not in df.columns:
        df = df.copy()
        df['Class_0'] = (df[themes].sum(axis=1) == 0).astype(int)

    c0 = df['Class_0'].sum()
    themed = len(df) - c0
    print(f"  Class_0: {c0} ({c0/len(df)*100:.1f}%)")
    print(f"  Themed:  {themed} ({themed/len(df)*100:.1f}%)")
    print(f"  Ratio:   {c0/themed:.2f}:1")
    print()

    for t in themes:
        n = (df[t] == 1).sum()
        print(f"    {t:30s}: {n:5d} ({n/len(df)*100:.1f}%)")

    # Multi-label stats
    theme_sums = df[themes].sum(axis=1)
    multi = (theme_sums >= 2).sum()
    single = (theme_sums == 1).sum()
    print(f"\n  Single-label: {single}, Multi-label: {multi}, Class_0: {c0}")

    # Unique essays
    if 'essay_id' in df.columns:
        print(f"  Unique essays: {df['essay_id'].nunique()}")


def create_v1(df):
    """Dataset v1: Original processed data — just copy."""
    print("=" * 70)
    print("DATASET V1: Original Processed Data")
    print("=" * 70)

    v1 = df.copy()
    outpath = os.path.join(BASE_DIR, 'v1_original_processed.csv')
    v1.to_csv(outpath, index=False)

    print(f"  Saved: {outpath}")
    print(f"  Rows: {len(v1)}")
    print(f"  Themes: {len(ALL_THEMES_V1)} ({', '.join(ALL_THEMES_V1)})")
    print_distribution(v1, ALL_THEMES_V1, "V1")

    return v1


def create_v2(df):
    """Dataset v2: Merge Familial+Filial_Piety, drop First_Gen complete essays."""
    print("\n" + "=" * 70)
    print("DATASET V2: Merged + First_Gen Essays Dropped")
    print("=" * 70)

    v2 = df.copy()

    # Step 1: Merge Familial + Filial_Piety → Familial_Capital
    print("\n  Step 1: Merging Familial + Filial_Piety → Familial_Capital")
    fam_before = (v2['Familial'] == 1).sum()
    fp_before = (v2['Filial_Piety'] == 1).sum()
    both = ((v2['Familial'] == 1) & (v2['Filial_Piety'] == 1)).sum()

    v2['Familial_Capital'] = ((v2['Familial'] == 1) | (v2['Filial_Piety'] == 1)).astype(int)
    v2 = v2.drop(columns=['Familial', 'Filial_Piety'])

    fc_after = (v2['Familial_Capital'] == 1).sum()
    print(f"    Familial: {fam_before}, Filial_Piety: {fp_before}, Both: {both}")
    print(f"    Familial_Capital: {fc_after}")

    # Step 2: Drop First_Gen — remove COMPLETE ESSAYS that have any First_Gen label
    print("\n  Step 2: Dropping First_Gen (complete essays)")
    fg_sentences = v2[v2['First_Gen'] == 1]
    fg_essay_ids = fg_sentences['essay_id'].unique()
    fg_all_sentences = v2[v2['essay_id'].isin(fg_essay_ids)]

    print(f"    First_Gen labeled sentences: {len(fg_sentences)}")
    print(f"    First_Gen essays: {len(fg_essay_ids)}")
    print(f"    Total sentences in those essays: {len(fg_all_sentences)}")

    # What we lose
    print(f"    Losing from other themes:")
    for t in ALL_THEMES_V2:
        lost = (fg_all_sentences[t] == 1).sum()
        total = (v2[t] == 1).sum()
        if lost > 0:
            print(f"      {t:30s}: {lost:4d} lost ({lost/total*100:.1f}% of {total})")

    c0_lost = (fg_all_sentences[ALL_THEMES_V2].sum(axis=1) == 0).sum()
    print(f"      {'Class_0':30s}: {c0_lost:4d} lost")

    # Drop the essays
    v2 = v2[~v2['essay_id'].isin(fg_essay_ids)].copy()

    # Drop the First_Gen column
    v2 = v2.drop(columns=['First_Gen'])

    # Reset index
    v2 = v2.reset_index(drop=True)

    # Recompute Class_0
    v2['Class_0'] = (v2[ALL_THEMES_V2].sum(axis=1) == 0).astype(int)

    outpath = os.path.join(BASE_DIR, 'v2_merged_cleaned.csv')
    v2.to_csv(outpath, index=False)

    print(f"\n  Saved: {outpath}")
    print(f"  Rows: {len(v2)} (dropped {len(df) - len(v2)} sentences)")
    print(f"  Themes: {len(ALL_THEMES_V2)} ({', '.join(ALL_THEMES_V2)})")
    print_distribution(v2, ALL_THEMES_V2, "V2")

    return v2


def create_v3(df, v2):
    """Dataset v3: v2 + boundary point removals (finalized)."""
    print("\n" + "=" * 70)
    print("DATASET V3: V2 + Boundary Removals (Finalized)")
    print("=" * 70)

    # Load boundary analysis (indices refer to original 18,019-row dataset)
    boundary = pd.read_csv(BOUNDARY_FILE, index_col=0)

    # Which original indices are still in v2?
    # v2 was reset, but we can match by essay_id + sentence_id
    # The boundary CSV has original indices, the original df has same indices
    original_indices_in_v2 = set(df.index) - set(
        df[df['essay_id'].isin(df[df['First_Gen'] == 1]['essay_id'].unique())].index
    )
    boundary_still_valid = boundary[boundary.index.isin(original_indices_in_v2)]

    print(f"\n  Boundary candidates from analysis: {len(boundary)}")
    print(f"  Already removed by v2 (First_Gen essays): {len(boundary) - len(boundary_still_valid)}")
    print(f"  Remaining candidates: {len(boundary_still_valid)}")

    # --- FINALIZED REMOVAL STRATEGY ---
    print("\n  === Finalized Boundary Removal Strategy ===")

    # Group 1: Class_0 in theme zones (R1+R2+R3) — REMOVE ALL
    # These are Class_0 points sitting on top of themed data
    class0_reasons = ['class0_on_theme_P95', 'class0_loo_mismatch_isolated', 'class0_deep_in_theme_zone']
    g1 = boundary_still_valid[boundary_still_valid['removal_reason'].isin(class0_reasons)]
    print(f"\n  Group 1 — Class_0 in theme zones: {len(g1)} (REMOVE ALL)")
    print(f"    class0_on_theme_P95:          {(g1['removal_reason']=='class0_on_theme_P95').sum()}")
    print(f"    class0_loo_mismatch_isolated:  {(g1['removal_reason']=='class0_loo_mismatch_isolated').sum()}")
    print(f"    class0_deep_in_theme_zone:     {(g1['removal_reason']=='class0_deep_in_theme_zone').sum()}")

    # Group 2: Theme predicted as Class_0 — TIGHTEN to conf > 0.5
    # Original was conf > 0.4. At 0.5+, neighbors STRONGLY say it's Class_0
    g2_all = boundary_still_valid[boundary_still_valid['removal_reason'] == 'theme_predicted_class0']
    if 'loo_confidence' in g2_all.columns:
        g2 = g2_all[g2_all['loo_confidence'] > 0.5]
    else:
        g2 = g2_all  # fallback
    print(f"\n  Group 2 — Theme→Class_0 (conf>0.5): {len(g2)} of {len(g2_all)} (TIGHTENED)")

    # Group 3: Theme confused with other theme — REMOVE ALL
    # These have strong neighbor disagreement about which theme
    g3 = boundary_still_valid[boundary_still_valid['removal_reason'] == 'theme_strong_confusion']
    print(f"\n  Group 3 — Theme confusion: {len(g3)} (REMOVE ALL)")

    # Combine
    all_remove_idx = set(g1.index) | set(g2.index) | set(g3.index)
    print(f"\n  Total boundary removals: {len(all_remove_idx)}")

    # Now map original indices to v2 rows
    # v2 was created by dropping First_Gen essays from df, then resetting index
    # We need to find which v2 rows correspond to which original indices
    fg_essay_ids = df[df['First_Gen'] == 1]['essay_id'].unique()
    df_after_drop = df[~df['essay_id'].isin(fg_essay_ids)].copy()
    # df_after_drop has original indices (before reset_index)
    # all_remove_idx are original indices
    # We need to find which of these are in df_after_drop
    valid_remove_idx = all_remove_idx & set(df_after_drop.index)

    # Create v3 by removing these rows from df_after_drop, then resetting
    v3_pre = df_after_drop.drop(index=valid_remove_idx)

    # Apply same transformations as v2
    v3 = v3_pre.copy()
    v3['Familial_Capital'] = ((v3['Familial'] == 1) | (v3['Filial_Piety'] == 1)).astype(int)
    v3 = v3.drop(columns=['Familial', 'Filial_Piety', 'First_Gen'])
    v3['Class_0'] = (v3[ALL_THEMES_V2].sum(axis=1) == 0).astype(int)
    v3 = v3.reset_index(drop=True)

    # Per-class impact
    print(f"\n  Per-class removals:")
    for t in ALL_THEMES_V2:
        v2_count = (v2[t] == 1).sum()
        v3_count = (v3[t] == 1).sum()
        lost = v2_count - v3_count
        pct = lost / v2_count * 100 if v2_count > 0 else 0
        print(f"    {t:30s}: {v2_count:5d} → {v3_count:5d} (lost {lost}, {pct:.1f}%)")

    v2_c0 = v2['Class_0'].sum()
    v3_c0 = v3['Class_0'].sum()
    print(f"    {'Class_0':30s}: {v2_c0:5d} → {v3_c0:5d} (lost {v2_c0-v3_c0}, {(v2_c0-v3_c0)/v2_c0*100:.1f}%)")

    outpath = os.path.join(BASE_DIR, 'v3_boundary_cleaned.csv')
    v3.to_csv(outpath, index=False)

    print(f"\n  Saved: {outpath}")
    print(f"  Rows: {len(v3)} (dropped {len(v2) - len(v3)} from v2, {len(df) - len(v3)} from original)")
    print(f"  Themes: {len(ALL_THEMES_V2)} ({', '.join(ALL_THEMES_V2)})")
    print_distribution(v3, ALL_THEMES_V2, "V3")

    # Save removal log for v3
    removal_log = boundary_still_valid.loc[boundary_still_valid.index.isin(valid_remove_idx)].copy()
    log_path = os.path.join(BASE_DIR, 'v3_removal_log.csv')
    removal_log.to_csv(log_path)
    print(f"\n  Removal log saved: {log_path} ({len(removal_log)} entries)")

    return v3


def print_comparison(v1, v2, v3):
    """Print side-by-side comparison of all 3 datasets."""
    print("\n" + "=" * 70)
    print("COMPARISON: All 3 Datasets")
    print("=" * 70)

    print(f"\n  {'':30s} {'V1':>10s} {'V2':>10s} {'V3':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Total sentences':30s} {len(v1):10,d} {len(v2):10,d} {len(v3):10,d}")

    # Class_0
    for ds, themes, name in [(v1, ALL_THEMES_V1, 'V1'),
                              (v2, ALL_THEMES_V2, 'V2'),
                              (v3, ALL_THEMES_V2, 'V3')]:
        if 'Class_0' not in ds.columns:
            ds['Class_0'] = (ds[themes].sum(axis=1) == 0).astype(int)

    v1_c0 = v1['Class_0'].sum() if 'Class_0' in v1.columns else (v1[ALL_THEMES_V1].sum(axis=1)==0).sum()
    v2_c0 = v2['Class_0'].sum()
    v3_c0 = v3['Class_0'].sum()
    print(f"  {'Class_0':30s} {v1_c0:10,d} {v2_c0:10,d} {v3_c0:10,d}")
    print(f"  {'Unique essays':30s} {v1['essay_id'].nunique():10,d} {v2['essay_id'].nunique():10,d} {v3['essay_id'].nunique():10,d}")
    print(f"  {'Theme count':30s} {'11':>10s} {'9':>10s} {'9':>10s}")

    # Per theme comparison
    print(f"\n  Per-theme counts:")
    print(f"  {'Theme':30s} {'V1':>10s} {'V2':>10s} {'V3':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    # Common themes
    for t in ['Aspirational', 'Social', 'Navigational', 'Resistance',
              'Attainment', 'Perseverance', 'Community_Consciousness', 'Spiritual']:
        v1_n = (v1[t] == 1).sum()
        v2_n = (v2[t] == 1).sum()
        v3_n = (v3[t] == 1).sum()
        print(f"  {t:30s} {v1_n:10,d} {v2_n:10,d} {v3_n:10,d}")

    # Familial / Filial_Piety / Familial_Capital
    v1_fam = (v1['Familial'] == 1).sum()
    v1_fp = (v1['Filial_Piety'] == 1).sum()
    v2_fc = (v2['Familial_Capital'] == 1).sum()
    v3_fc = (v3['Familial_Capital'] == 1).sum()
    print(f"  {'Familial (v1)':30s} {v1_fam:10,d} {'—':>10s} {'—':>10s}")
    print(f"  {'Filial_Piety (v1)':30s} {v1_fp:10,d} {'—':>10s} {'—':>10s}")
    print(f"  {'Familial_Capital (v2/v3)':30s} {'—':>10s} {v2_fc:10,d} {v3_fc:10,d}")

    # First_Gen
    v1_fg = (v1['First_Gen'] == 1).sum()
    print(f"  {'First_Gen (v1 only)':30s} {v1_fg:10,d} {'DROPPED':>10s} {'DROPPED':>10s}")

    # Balance
    print(f"\n  Class_0:Themed ratio:")
    for name, ds, themes in [('V1', v1, ALL_THEMES_V1), ('V2', v2, ALL_THEMES_V2), ('V3', v3, ALL_THEMES_V2)]:
        c0 = (ds[themes].sum(axis=1) == 0).sum()
        themed = len(ds) - c0
        print(f"    {name}: {c0/themed:.3f}:1 ({c0} Class_0 vs {themed} themed)")


def main():
    print("Loading source data...")
    df = pd.read_csv(SOURCE_FILE)
    print(f"  Source: {len(df)} sentences, {df['essay_id'].nunique()} essays")

    v1 = create_v1(df)
    v2 = create_v2(df)
    v3 = create_v3(df, v2)
    print_comparison(v1, v2, v3)

    print("\n" + "=" * 70)
    print("ALL 3 DATASETS CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"  v1_original_processed.csv  — {len(v1):,} rows, 11 themes")
    print(f"  v2_merged_cleaned.csv      — {len(v2):,} rows,  9 themes")
    print(f"  v3_boundary_cleaned.csv    — {len(v3):,} rows,  9 themes")
    print(f"\n  Location: {BASE_DIR}")


if __name__ == '__main__':
    main()
