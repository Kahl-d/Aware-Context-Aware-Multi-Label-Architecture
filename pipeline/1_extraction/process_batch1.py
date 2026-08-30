"""
Process TIER 2: Batch 1 compiled data
Only adds rows NOT already in Gold Standard (dedup by essay text)
"""
import pandas as pd
import numpy as np
import os
import re
import json
import sys

# Add scripts dir to path for shared functions
sys.path.insert(0, os.path.dirname(__file__))
from process_gold_standard import (
    parse_alma_id, binarize_label, extract_excerpts, clean_text, ALL_THEMES
)

BASE = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA Project-selected"
OUTPUT = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA_Master_Dataset"
ANNOTATED_WHY = os.path.join(OUTPUT, "annotated", "why_am_i_here")

# Batch 1 column to standard theme mapping
BATCH1_THEME_MAP = {
    'Attainment': 'Attainment',
    'First Gen': 'First_Gen',
    'Aspirational': 'Aspirational',
    'Navigational': 'Navigational',
    'Resistance': 'Resistance',
    'Perseverance': 'Perseverance',
    'Filial Piety': 'Filial_Piety',
    'Familial': 'Familial',
    'Community Consciousness': 'Community_Consciousness',
    'Social': 'Social',
    'Spiritual': 'Spiritual',
}


def main():
    # Load existing Gold Standard for dedup
    gold_path = os.path.join(ANNOTATED_WHY, "gold_standard_fall2019.csv")
    if os.path.exists(gold_path):
        gold = pd.read_csv(gold_path)
        existing_essays = set()
        for _, r in gold.iterrows():
            existing_essays.add(str(r['essay'])[:200].strip().lower())
        print(f"Loaded {len(gold)} Gold Standard rows for dedup")
    else:
        existing_essays = set()
        print("WARNING: No Gold Standard file found, no dedup will happen")

    # Read Batch 1
    path = os.path.join(BASE, "ALMA 2024/Data/gian_reina_to_be_reconciled/batch1_compiled.xlsx")
    print(f"\n{'='*70}")
    print(f"PROCESSING: batch1_compiled.xlsx")
    print(f"{'='*70}")

    df = pd.read_excel(path, sheet_name='Sheet1')
    df = df.dropna(how='all')
    print(f"Total rows: {len(df)}")

    # Process each row
    new_rows = []
    skipped_dup = 0
    skipped_short = 0

    for idx, row in df.iterrows():
        alma_id = row.get('Alma ID')
        if pd.isna(alma_id):
            continue

        alma_id = str(alma_id).strip()
        essay = str(row.get('Essay: Why I am here?', '')).strip()

        if pd.isna(row.get('Essay: Why I am here?')) or len(essay) < 10:
            skipped_short += 1
            continue

        # Clean essay
        essay = clean_text(essay)

        # Dedup by essay text
        essay_key = essay[:200].strip().lower()
        if essay_key in existing_essays:
            skipped_dup += 1
            continue

        existing_essays.add(essay_key)

        # Parse metadata
        semester, year, course = parse_alma_id(alma_id)

        # Build record
        record = {
            'alma_id': alma_id,
            'course': course,
            'semester': semester,
            'year': year,
            'prompt': 'Why am I here?',
            'essay': essay,
            'essay_length': len(essay),
            'source_file': 'batch1_compiled.xlsx',
            'source_sheet': 'Sheet1',
        }

        # Process each theme
        for batch_col, std_name in BATCH1_THEME_MAP.items():
            val = row.get(batch_col)
            record[f'{std_name}_binary'] = binarize_label(val)
            excerpts = extract_excerpts(val)
            record[f'{std_name}_excerpts'] = ' /%/ '.join(excerpts) if excerpts else ''

        # Ensure all themes present
        for theme in ALL_THEMES:
            if f'{theme}_binary' not in record:
                record[f'{theme}_binary'] = 0
                record[f'{theme}_excerpts'] = ''

        new_rows.append(record)

    print(f"\nResults:")
    print(f"  New rows added: {len(new_rows)}")
    print(f"  Skipped (duplicate of Gold): {skipped_dup}")
    print(f"  Skipped (short/empty essay): {skipped_short}")

    if new_rows:
        # Create DataFrame
        meta_cols = ['alma_id', 'course', 'semester', 'year', 'prompt', 'essay', 'essay_length', 'source_file', 'source_sheet']
        binary_cols = [f'{t}_binary' for t in ALL_THEMES]
        excerpt_cols = [f'{t}_excerpts' for t in ALL_THEMES]
        all_cols = meta_cols + binary_cols + excerpt_cols

        new_df = pd.DataFrame(new_rows)[all_cols]
        new_df = new_df.sort_values('alma_id').reset_index(drop=True)

        # Save
        output_path = os.path.join(ANNOTATED_WHY, "batch1_new_rows.csv")
        new_df.to_csv(output_path, index=False)

        print(f"\nOutput: {output_path}")

        # Summary
        print(f"\nBy course:")
        for course, group in new_df.groupby('course'):
            print(f"  {course}: {len(group)} rows")

        print(f"\nTheme distribution (new rows only):")
        for theme in ALL_THEMES:
            n_pos = new_df[f'{theme}_binary'].sum()
            pct = n_pos / len(new_df) * 100
            print(f"  {theme:25s}: {n_pos:3d} / {len(new_df)} ({pct:5.1f}%)")

        # Verify: read and spot-check some labels
        print(f"\n=== SPOT CHECK: 5 random rows ===")
        sample = new_df.sample(min(5, len(new_df)), random_state=42)
        for _, r in sample.iterrows():
            active_themes = [t for t in ALL_THEMES if r[f'{t}_binary'] == 1]
            print(f"\n{r['alma_id']} | {r['course']} | {r['semester']} {r['year']}")
            print(f"  Essay ({r['essay_length']} chars): {r['essay'][:120]}...")
            print(f"  Themes: {active_themes}")
            for t in active_themes[:2]:
                excerpt = str(r[f'{t}_excerpts'])[:100]
                print(f"    {t}: \"{excerpt}...\"")


if __name__ == '__main__':
    main()
