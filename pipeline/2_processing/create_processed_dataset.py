"""
Create Processed Master Dataset — Clean noisy data points for DeBERTa training.

Removal rules:
  R1: Class_0 in KNN confusion zones (7+ of 10 neighbors themed) → 1,337 sentences
  R2: Single-label theme outliers closer to Class_0 centroid → 238 sentences
  R3: Class_0 with very high similarity to theme centroids (P90) → 130 sentences

Total removal: 1,705 sentences (8.6%) — no overlaps between rules.

Input:
  - ALMA_sentence_level_dataset.csv (19,724 sentences — NOT modified)
  - flagged_datapoints.csv (4,310 flags)
  - sentence_embeddings.npy (19724 x 384)

Output:
  - ALMA_processed_master_dataset.csv (18,019 sentences)
  - removal_log.csv (1,705 removed rows with reasons)
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Configuration
# ============================================================
BASE_DIR = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/Data_Processing"
INPUT_FILE = os.path.join(BASE_DIR, "ALMA_sentence_level_dataset.csv")
FLAGS_FILE = os.path.join(BASE_DIR, "flagged_datapoints.csv")
EMBED_FILE = os.path.join(BASE_DIR, "sentence_embeddings.npy")
OUTPUT_FILE = os.path.join(BASE_DIR, "ALMA_processed_master_dataset.csv")
LOG_FILE = os.path.join(BASE_DIR, "removal_log.csv")

THEMES = [
    'Attainment', 'First_Gen', 'Aspirational', 'Navigational', 'Resistance',
    'Perseverance', 'Filial_Piety', 'Familial', 'Community_Consciousness',
    'Social', 'Spiritual'
]


# ============================================================
# Load data
# ============================================================
def load_data():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)
    flags = pd.read_csv(FLAGS_FILE)
    embeddings = np.load(EMBED_FILE)

    print(f"  Dataset:    {len(df)} sentences, {df['essay_id'].nunique()} essays")
    print(f"  Flags:      {len(flags)} total, {flags['datapoint_index'].nunique()} unique")
    print(f"  Embeddings: {embeddings.shape}")

    assert len(df) == len(embeddings), "Dataset and embeddings size mismatch!"
    return df, flags, embeddings


# ============================================================
# Compute centroids
# ============================================================
def compute_centroids(df, embeddings):
    print("\n" + "=" * 70)
    print("COMPUTING CENTROIDS")
    print("=" * 70)

    centroids = {}
    for theme in THEMES:
        mask = df[theme] == 1
        centroids[theme] = embeddings[mask].mean(axis=0)
        print(f"  {theme:<30} centroid from {mask.sum():>5} sentences")

    mask0 = df[THEMES].sum(axis=1) == 0
    centroids['Class_0'] = embeddings[mask0].mean(axis=0)
    print(f"  {'Class_0':<30} centroid from {mask0.sum():>5} sentences")

    return centroids


# ============================================================
# R1: Class_0 in confusion zones
# ============================================================
def removal_1_confusion_zone(df, flags):
    """Remove Class_0 sentences with 7+ of 10 nearest neighbors being themed."""
    print("\n" + "=" * 70)
    print("REMOVAL 1: CLASS_0 IN CONFUSION ZONES")
    print("=" * 70)

    cz = flags[flags['flag_type'] == 'Class0_in_confusion_zone']
    r1_indices = set(cz['datapoint_index'].values)

    # Verify all are truly Class_0
    n_labels = df.iloc[list(r1_indices)][THEMES].sum(axis=1)
    assert (n_labels == 0).all(), "Some confusion zone sentences are not Class_0!"

    print(f"  Flagged: {len(r1_indices)} Class_0 sentences")
    print(f"  Neighbor themes they're confused with:")
    for theme, group in cz.groupby('target_theme'):
        print(f"    {theme:<30} {len(group):>5}")

    return r1_indices


# ============================================================
# R2: Single-label theme outliers closer to Class_0
# ============================================================
def removal_2_theme_outliers(df, flags, embeddings, centroids):
    """Remove single-label theme outliers closer to Class_0 centroid than own theme."""
    print("\n" + "=" * 70)
    print("REMOVAL 2: SINGLE-LABEL THEME OUTLIERS -> CLASS_0")
    print("=" * 70)

    to_flags = flags[flags['flag_type'] == 'theme_outlier']
    to_indices = to_flags['datapoint_index'].unique()

    n_labels = df.iloc[to_indices][THEMES].sum(axis=1)
    single_label_indices = to_indices[n_labels.values == 1]
    multi_label_indices = to_indices[n_labels.values > 1]

    print(f"  Total theme outlier sentences: {len(to_indices)}")
    print(f"  Multi-label (KEPT):            {len(multi_label_indices)}")
    print(f"  Single-label (checking):       {len(single_label_indices)}")

    class0_centroid = centroids['Class_0'].reshape(1, -1)

    r2_indices = set()
    r2_kept = 0
    r2_per_theme_removed = {}
    r2_per_theme_kept = {}

    for idx in single_label_indices:
        row = df.iloc[idx]
        active = [t for t in THEMES if row[t] == 1]
        if len(active) != 1:
            continue
        theme = active[0]

        emb = embeddings[idx].reshape(1, -1)
        sim_to_theme = cosine_similarity(emb, centroids[theme].reshape(1, -1))[0, 0]
        sim_to_class0 = cosine_similarity(emb, class0_centroid)[0, 0]

        if sim_to_class0 > sim_to_theme:
            r2_indices.add(idx)
            r2_per_theme_removed[theme] = r2_per_theme_removed.get(theme, 0) + 1
        else:
            r2_kept += 1
            r2_per_theme_kept[theme] = r2_per_theme_kept.get(theme, 0) + 1

    print(f"\n  Removed (closer to Class_0): {len(r2_indices)}")
    print(f"  Kept (closer to own theme):  {r2_kept}")
    print(f"\n  Per-theme breakdown:")
    all_themes = sorted(set(list(r2_per_theme_removed.keys()) + list(r2_per_theme_kept.keys())))
    for theme in all_themes:
        removed = r2_per_theme_removed.get(theme, 0)
        kept = r2_per_theme_kept.get(theme, 0)
        print(f"    {theme:<30} removed={removed:>3}, kept={kept:>3}")

    return r2_indices


# ============================================================
# R3: Near-theme Class_0 (P90 strict)
# ============================================================
def removal_3_near_theme_strict(df, flags, r1_indices):
    """Remove Class_0 with very high similarity to theme centroids (P90 threshold)."""
    print("\n" + "=" * 70)
    print("REMOVAL 3: NEAR-THEME CLASS_0 (P90 STRICT)")
    print("=" * 70)

    nt = flags[flags['flag_type'] == 'Class0_near_theme']
    nt_remaining = nt[~nt['datapoint_index'].isin(r1_indices)]

    print(f"  Total near_theme flags:          {len(nt)}")
    print(f"  Already in R1 (excluded):        {len(nt) - len(nt_remaining)}")
    print(f"  Remaining to evaluate:           {len(nt_remaining)} flags, "
          f"{nt_remaining['datapoint_index'].nunique()} unique")

    threshold = nt_remaining['similarity'].quantile(0.90)
    high_sim = nt_remaining[nt_remaining['similarity'] >= threshold]
    r3_indices = set(high_sim['datapoint_index'].values)

    print(f"\n  P90 threshold:                   {threshold:.4f}")
    print(f"  Removed:                         {len(r3_indices)} unique sentences")
    print(f"\n  Per-theme:")
    for theme, group in high_sim.groupby('target_theme'):
        print(f"    {theme:<30} {len(group):>5} flags "
              f"(sim: {group['similarity'].min():.4f} - {group['similarity'].max():.4f})")

    return r3_indices, threshold


# ============================================================
# Apply removals and save
# ============================================================
def apply_removals(df, r1, r2, r3):
    print("\n" + "=" * 70)
    print("APPLYING REMOVALS")
    print("=" * 70)

    all_removed = r1 | r2 | r3

    # Verify no overlap
    assert len(r1 & r2) == 0, "R1 and R2 should not overlap"
    assert len(r1 & r3) == 0, "R1 and R3 should not overlap"
    assert len(r2 & r3) == 0, "R2 and R3 should not overlap"

    print(f"  R1 (confusion_zone Class_0):     {len(r1):>6}")
    print(f"  R2 (single-label outliers->C0):  {len(r2):>6}")
    print(f"  R3 (near_theme P90 Class_0):     {len(r3):>6}")
    print(f"  {'_' * 42}")
    print(f"  TOTAL REMOVED:                   {len(all_removed):>6} ({len(all_removed)/len(df)*100:.1f}%)")

    # Create removal log
    log_rows = []
    for idx in sorted(all_removed):
        row = df.iloc[idx]
        reasons = []
        if idx in r1:
            reasons.append('R1_confusion_zone')
        if idx in r2:
            reasons.append('R2_theme_outlier_to_class0')
        if idx in r3:
            reasons.append('R3_near_theme_p90')

        active_themes = [t for t in THEMES if row[t] == 1]

        log_rows.append({
            'original_index': idx,
            'essay_id': int(row['essay_id']),
            'sentence_id': int(row['sentence_id']),
            'sentence': row['sentence'][:200],
            'removal_reason': '|'.join(reasons),
            'original_labels': ','.join(active_themes) if active_themes else 'Class_0'
        })

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_FILE, index=False)
    print(f"\n  Removal log saved: {LOG_FILE}")

    # Drop removed rows
    df_processed = df.drop(index=list(all_removed)).reset_index(drop=True)

    return df_processed, log_df


# ============================================================
# Distribution comparison
# ============================================================
def print_distribution(df_before, df_after):
    print("\n" + "=" * 70)
    print("DISTRIBUTION COMPARISON: BEFORE vs AFTER")
    print("=" * 70)

    b_c0 = (df_before[THEMES].sum(axis=1) == 0).sum()
    a_c0 = (df_after[THEMES].sum(axis=1) == 0).sum()
    b_themed = (df_before[THEMES].sum(axis=1) > 0).sum()
    a_themed = (df_after[THEMES].sum(axis=1) > 0).sum()

    print(f"\n  {'Category':<30} {'Before':>8} {'%':>6}  {'After':>8} {'%':>6}  {'Change':>8}  {'Ratio B':>8} {'Ratio A':>8}")
    print(f"  {'=' * 96}")

    # Class_0
    print(f"  {'Class_0':<30} {b_c0:>8} {b_c0/len(df_before)*100:>5.1f}%  "
          f"{a_c0:>8} {a_c0/len(df_after)*100:>5.1f}%  {a_c0-b_c0:>+8}  "
          f"{'--':>8} {'--':>8}")
    print(f"  {'-' * 96}")

    # Per theme
    for theme in THEMES:
        bp = (df_before[theme] == 1).sum()
        ap = (df_after[theme] == 1).sum()
        br = (len(df_before) - bp) / max(bp, 1)
        ar = (len(df_after) - ap) / max(ap, 1)
        improved = " ^" if ar < br else ""
        print(f"  {theme:<30} {bp:>8} {bp/len(df_before)*100:>5.1f}%  "
              f"{ap:>8} {ap/len(df_after)*100:>5.1f}%  {ap-bp:>+8}  "
              f"{br:>6.0f}:1 {ar:>6.0f}:1{improved}")

    print(f"  {'-' * 96}")
    print(f"  {'TOTAL THEMED':<30} {b_themed:>8} {b_themed/len(df_before)*100:>5.1f}%  "
          f"{a_themed:>8} {a_themed/len(df_after)*100:>5.1f}%  {a_themed-b_themed:>+8}")
    print(f"  {'TOTAL SENTENCES':<30} {len(df_before):>8} {'100%':>6}  "
          f"{len(df_after):>8} {'100%':>6}  {len(df_after)-len(df_before):>+8}")

    print(f"\n  Class_0:Themed ratio:")
    print(f"    Before: {b_c0/b_themed:.2f}:1")
    print(f"    After:  {a_c0/a_themed:.2f}:1")

    # Multi-label
    print(f"\n  Multi-label distribution:")
    for label, data in [('Before', df_before), ('After', df_after)]:
        n_labels = data[THEMES].sum(axis=1)
        counts = n_labels.value_counts().sort_index()
        parts = [f"{int(n)}L:{c}" for n, c in counts.items()]
        print(f"    {label}: {', '.join(parts)}")


# ============================================================
# Main
# ============================================================
def main():
    print()
    print("+" + "=" * 68 + "+")
    print("|  ALMA Processed Master Dataset Creator                            |")
    print("|  Cleaning noisy data points for DeBERTa training                  |")
    print("+" + "=" * 68 + "+")

    df, flags, embeddings = load_data()
    centroids = compute_centroids(df, embeddings)

    r1 = removal_1_confusion_zone(df, flags)
    r2 = removal_2_theme_outliers(df, flags, embeddings, centroids)
    r3, r3_threshold = removal_3_near_theme_strict(df, flags, r1)

    df_processed, removal_log = apply_removals(df, r1, r2, r3)

    # Save processed dataset
    df_processed.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Processed dataset saved: {OUTPUT_FILE}")
    print(f"  Shape: {df_processed.shape}")

    # Distribution comparison
    print_distribution(df, df_processed)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    c0_before = (df[THEMES].sum(axis=1) == 0).sum()
    c0_after = (df_processed[THEMES].sum(axis=1) == 0).sum()
    print(f"  Original:  {len(df):>6} sentences ({c0_before} Class_0)")
    print(f"  Processed: {len(df_processed):>6} sentences ({c0_after} Class_0)")
    print(f"  Removed:   {len(df)-len(df_processed):>6} sentences ({(len(df)-len(df_processed))/len(df)*100:.1f}%)")
    print(f"    R1 confusion_zone: {len(r1)} Class_0")
    print(f"    R2 theme_outliers: {len(r2)} themed (single-label, closer to C0)")
    print(f"    R3 near_theme P90: {len(r3)} Class_0 (sim > {r3_threshold:.4f})")
    print(f"\n  Output: {OUTPUT_FILE}")
    print(f"  Log:    {LOG_FILE}")


if __name__ == '__main__':
    main()
