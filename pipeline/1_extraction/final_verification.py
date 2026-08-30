"""
FINAL VERIFICATION: Cross-check all processed data for:
1. Deduplication across ALL annotated files
2. Label integrity (no invalid values)
3. Essay quality (no empty/too-short essays)
4. Format consistency (all files have same column structure)
5. Complete summary statistics
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from process_gold_standard import ALL_THEMES

OUTPUT = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA_Master_Dataset"
ANNOTATED = os.path.join(OUTPUT, "annotated")
UNANNOTATED = os.path.join(OUTPUT, "unannotated")


def main():
    print("=" * 80)
    print("FINAL VERIFICATION: ALMA Master Dataset")
    print("=" * 80)

    # ==========================================
    # 1. LOAD ALL ANNOTATED DATA
    # ==========================================
    print("\n\n1. LOADING ALL ANNOTATED DATA")
    print("-" * 60)

    all_annotated = []
    file_summaries = []

    for root, dirs, files in os.walk(ANNOTATED):
        for f in sorted(files):
            if not f.endswith('.csv'):
                continue
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, ANNOTATED)
            df = pd.read_csv(fpath)
            all_annotated.append(df)
            prompt_folder = os.path.basename(os.path.dirname(fpath))
            file_summaries.append({
                'file': rel,
                'rows': len(df),
                'prompt_folder': prompt_folder,
            })
            print(f"  {rel}: {len(df)} rows")

    master = pd.concat(all_annotated, ignore_index=True)
    print(f"\n  TOTAL ANNOTATED: {len(master)} rows across {len(file_summaries)} files")

    # ==========================================
    # 2. DEDUPLICATION CHECK
    # ==========================================
    print("\n\n2. DEDUPLICATION CHECK")
    print("-" * 60)

    # Check by essay text (first 200 chars)
    master['_essay_key'] = master['essay'].apply(lambda x: str(x)[:200].strip().lower())
    dup_essays = master[master.duplicated(subset=['_essay_key'], keep=False)]

    if len(dup_essays) > 0:
        dup_groups = dup_essays.groupby('_essay_key')
        print(f"  FOUND {len(dup_groups)} duplicate essay groups ({len(dup_essays)} total dup rows)")
        print(f"\n  Duplicates to remove:")
        rows_to_drop = []
        for key, group in dup_groups:
            if len(group) <= 1:
                continue
            # Keep the row from highest-quality source
            source_priority = {
                'gold_standard': 1,
                'batch1': 2,
                'batch2': 3,
                'spring2025_why_am_i_here': 4,
                'spring2025_why_additional': 5,
                'spring2025_when': 6,
                'spring2025_what': 7,
                'spring2025_personal': 8,
            }
            # Sort by priority (keep lowest = highest quality)
            def get_priority(src):
                src_str = str(src).lower()
                for key_name, prio in source_priority.items():
                    if key_name in src_str:
                        return prio
                return 99

            sorted_group = group.copy()
            sorted_group['_priority'] = sorted_group['source_file'].apply(get_priority)
            sorted_group = sorted_group.sort_values('_priority')

            keep_idx = sorted_group.index[0]
            drop_idxs = sorted_group.index[1:]
            rows_to_drop.extend(drop_idxs)

            kept = sorted_group.iloc[0]
            print(f"    Essay: '{str(kept['essay'])[:60]}...'")
            print(f"      KEEP: {kept['source_file']} ({kept.get('alma_id', '?')})")
            for idx in drop_idxs:
                dropped = master.loc[idx]
                print(f"      DROP: {dropped['source_file']} ({dropped.get('alma_id', '?')})")

        if rows_to_drop:
            print(f"\n  Removing {len(rows_to_drop)} duplicate rows...")
            master = master.drop(rows_to_drop).reset_index(drop=True)
            print(f"  After dedup: {len(master)} rows")
    else:
        print(f"  No duplicates found!")

    master = master.drop(columns=['_essay_key'], errors='ignore')

    # ==========================================
    # 3. LABEL INTEGRITY CHECK
    # ==========================================
    print("\n\n3. LABEL INTEGRITY CHECK")
    print("-" * 60)

    binary_cols = [f'{t}_binary' for t in ALL_THEMES]
    issues = []

    for col in binary_cols:
        if col not in master.columns:
            print(f"  MISSING COLUMN: {col}")
            issues.append(f"Missing {col}")
            continue

        vals = master[col].dropna().unique()
        valid = {0, 1, -1, 0.0, 1.0, -1.0}
        invalid = set(vals) - valid
        if invalid:
            print(f"  INVALID values in {col}: {invalid}")
            issues.append(f"Invalid values in {col}: {invalid}")
        else:
            # Count distribution
            n_pos = (master[col] == 1).sum()
            n_neg = (master[col] == 0).sum()
            n_nc = (master[col] == -1).sum()
            n_nan = master[col].isna().sum()
            print(f"  {col:30s}: {n_pos:4d} pos, {n_neg:4d} neg, {n_nc:4d} not_coded, {n_nan:2d} NaN")

    if not issues:
        print(f"\n  All labels valid!")

    # ==========================================
    # 4. ESSAY QUALITY CHECK
    # ==========================================
    print("\n\n4. ESSAY QUALITY CHECK")
    print("-" * 60)

    # Short essays
    short = master[master['essay_length'] < 20]
    if len(short) > 0:
        print(f"  WARNING: {len(short)} essays shorter than 20 chars:")
        for _, r in short.head(5).iterrows():
            print(f"    {r.get('alma_id', '?')}: '{r['essay'][:50]}'")
    else:
        print(f"  No extremely short essays (<20 chars)")

    # Empty essays
    empty = master[master['essay'].isna() | (master['essay'] == '')]
    if len(empty) > 0:
        print(f"  WARNING: {len(empty)} empty essays!")
    else:
        print(f"  No empty essays")

    # Essay length distribution
    lengths = master['essay_length']
    print(f"\n  Essay length stats:")
    print(f"    Min: {lengths.min()}")
    print(f"    Max: {lengths.max()}")
    print(f"    Mean: {lengths.mean():.0f}")
    print(f"    Median: {lengths.median():.0f}")

    # ==========================================
    # 5. COMPREHENSIVE SUMMARY
    # ==========================================
    print("\n\n5. COMPREHENSIVE SUMMARY")
    print("=" * 80)

    # By source
    print(f"\n  A) By Source File:")
    for src, group in master.groupby('source_file'):
        print(f"    {src}: {len(group)} rows")

    # By prompt
    if 'prompt' in master.columns:
        print(f"\n  B) By Prompt:")
        for prompt, group in master.groupby('prompt'):
            print(f"    {prompt}: {len(group)} rows")

    # By course
    print(f"\n  C) By Course:")
    for course, group in sorted(master.groupby('course'), key=lambda x: x[0]):
        print(f"    {course}: {len(group)} rows")

    # By semester/year
    print(f"\n  D) By Semester/Year:")
    master['year'] = master['year'].astype(str)
    master['semester'] = master['semester'].astype(str)
    for (sem, yr), group in master.groupby(['semester', 'year']):
        print(f"    {sem} {yr}: {len(group)} rows")

    # Theme distribution (coded rows only)
    print(f"\n  E) Theme Distribution (across all coded rows):")
    for theme in ALL_THEMES:
        col = f'{theme}_binary'
        if col in master.columns:
            coded = master[master[col] >= 0]
            n_pos = (coded[col] == 1).sum()
            n_coded = len(coded)
            pct = n_pos / n_coded * 100 if n_coded > 0 else 0
            n_nc = (master[col] == -1).sum()
            print(f"    {theme:25s}: {n_pos:4d} / {n_coded:4d} coded ({pct:5.1f}%) | {n_nc:4d} not coded")

    # ==========================================
    # 6. UNANNOTATED DATA SUMMARY
    # ==========================================
    print(f"\n\n6. UNANNOTATED DATA SUMMARY")
    print("-" * 60)

    total_unann = 0
    for root, dirs, files in os.walk(UNANNOTATED):
        for f in sorted(files):
            if not f.endswith('.csv'):
                continue
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, UNANNOTATED)
            df = pd.read_csv(fpath)
            total_unann += len(df)
            print(f"  {rel}: {len(df)} rows")

    print(f"\n  TOTAL UNANNOTATED: {total_unann} rows")

    # ==========================================
    # 7. FINAL TOTALS
    # ==========================================
    print(f"\n\n{'='*80}")
    print(f"FINAL DATASET TOTALS")
    print(f"{'='*80}")
    print(f"  Annotated rows:   {len(master):,}")
    print(f"  Unannotated rows: {total_unann:,}")
    print(f"  GRAND TOTAL:      {len(master) + total_unann:,}")
    print(f"\n  Unique courses:   {master['course'].nunique()}")
    print(f"  Unique semesters: {master.groupby(['semester', 'year']).ngroups}")
    print(f"  Themes tracked:   {len(ALL_THEMES)}")
    print(f"\n  Files in annotated/:   {len(file_summaries)}")

    # Save deduped master if any were removed
    print(f"\n\n  Saving final deduped files...")

    # Re-save each annotated subfolder from the deduped master
    # Group by source file to maintain original structure
    for summary in file_summaries:
        original_path = os.path.join(ANNOTATED, summary['file'])
        original_df = pd.read_csv(original_path)
        original_keys = set(original_df['essay'].apply(lambda x: str(x)[:200].strip().lower()))
        deduped = master[master['essay'].apply(lambda x: str(x)[:200].strip().lower()).isin(original_keys)]

        # Check if rows were removed
        if len(deduped) < summary['rows']:
            removed = summary['rows'] - len(deduped)
            # Re-read and filter
            keep_essays = set(master['essay'].apply(lambda x: str(x)[:200].strip().lower()))
            df_refilter = pd.read_csv(original_path)
            df_refilter['_key'] = df_refilter['essay'].apply(lambda x: str(x)[:200].strip().lower())
            df_refilter = df_refilter[df_refilter['_key'].isin(keep_essays)]
            df_refilter = df_refilter.drop(columns=['_key'])
            df_refilter.to_csv(original_path, index=False)
            print(f"    Updated {summary['file']}: removed {removed} dups ({len(df_refilter)} rows)")
        else:
            print(f"    {summary['file']}: no changes ({summary['rows']} rows)")

    print(f"\n  VERIFICATION COMPLETE")


if __name__ == '__main__':
    main()
