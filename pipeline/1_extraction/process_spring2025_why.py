"""
Process TIER 3: Spring 2025 "Why am I here?" merged coded files
These have 5 themes only (perseverance, community, Strong Spiritual, Resistance, Weak Spiritual)
Other themes are NOT coded (not negative, just not evaluated)
"""
import pandas as pd
import numpy as np
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from process_gold_standard import clean_text, binarize_label, extract_excerpts, ALL_THEMES

BASE = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA Project-selected"
OUTPUT = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA_Master_Dataset"
ANNOTATED_WHY = os.path.join(OUTPUT, "annotated", "why_am_i_here")

# Spring 2025 theme column variants → standard name
SPRING25_THEME_MAP = {
    'perseverance': 'Perseverance',
    'community': 'Community_Consciousness',  # Maps to Community + Social
    'strong spiritual': 'Strong_Spiritual',  # Keep as sub-label
    'resistance': 'Resistance',
    'resistance capital': 'Resistance',  # Variant name in PHYS112
    'weak spiritual': 'Weak_Spiritual',  # Keep as sub-label
}

# Files to process with their header row (0-indexed)
FILES = [
    {
        'name': 'PHYS102-Spring2025-Why am I here - KW  AM NE .xlsx',
        'header_row': 0,
        'course': 'PHYS102',
    },
    {
        'name': 'PHYS112-Spring2025-Why am I here-KW AM NE.xlsx',
        'header_row': 1,  # Headers on row 2!
        'course': 'PHYS112',
    },
    {
        'name': 'PHYS122-Spring2025-Why am I here - KW AM NE .xlsx',
        'header_row': 0,
        'course': 'PHYS122',
    },
    {
        'name': 'PHYS232-Spring2025-Why am I here - KW AM NE  (GROUP).xlsx',
        'header_row': 0,
        'course': 'PHYS232',
    },
]


def main():
    # Load existing data for dedup
    existing_essays = set()
    for f in ['gold_standard_fall2019.csv', 'batch1_new_rows.csv']:
        fpath = os.path.join(ANNOTATED_WHY, f)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            for _, r in df.iterrows():
                existing_essays.add(str(r['essay'])[:200].strip().lower())
            print(f"Loaded {len(df)} rows from {f} for dedup")
    print(f"Total existing essays for dedup: {len(existing_essays)}")

    all_rows = []
    skipped_dup = 0
    skipped_empty = 0
    skipped_unannotated = 0

    for file_info in FILES:
        fname = file_info['name']
        path = os.path.join(BASE, 'ALMA 2025/Khalid-Allyson-Phuong-Kirsten-Aline-Nada/Final4Khalid', fname)

        print(f"\n{'='*70}")
        print(f"FILE: {fname}")
        print(f"{'='*70}")

        # Read with correct header row
        df = pd.read_excel(path, header=file_info['header_row'])
        df = df.dropna(how='all')

        # Clean column names
        clean_cols = {}
        for c in df.columns:
            if c is not None:
                cleaned = str(c).strip().replace('\xa0', ' ').strip()
                clean_cols[c] = cleaned
        df = df.rename(columns=clean_cols)

        print(f"Columns: {list(df.columns)}")
        print(f"Total rows: {len(df)}")

        # Identify columns
        id_col = 'user_id'
        essay_col = 'submission'

        if id_col not in df.columns or essay_col not in df.columns:
            print(f"  ERROR: Missing required columns ({id_col}, {essay_col})")
            continue

        data = df[df[id_col].notna()]
        print(f"Data rows: {len(data)}")

        # Map theme columns
        theme_map = {}
        for c in df.columns:
            c_clean = str(c).strip().lower()
            if c_clean in SPRING25_THEME_MAP:
                theme_map[c] = SPRING25_THEME_MAP[c_clean]

        print(f"Theme columns mapped:")
        for raw, std in theme_map.items():
            print(f"  {raw} → {std}")

        # Theme distribution
        print(f"\nTheme distribution:")
        for raw_col, std_name in theme_map.items():
            vals = data[raw_col]
            n_zero = ((vals == 0) | (vals == 0.0)).sum()
            n_nan = vals.isna().sum()
            n_text = len(vals) - n_zero - n_nan
            pct = n_text / len(data) * 100 if len(data) > 0 else 0
            print(f"  {std_name:25s}: {n_text:3d} pos ({pct:.1f}%), {n_zero:3d} zero, {n_nan:3d} NaN")

        # Process rows
        for idx, row in data.iterrows():
            user_id = row[id_col]
            essay = row.get(essay_col, '')
            if pd.isna(essay) or len(str(essay).strip()) < 10:
                skipped_empty += 1
                continue

            essay = clean_text(str(essay))

            # Check if ALL theme columns are NaN (unannotated row)
            all_nan = all(pd.isna(row.get(c)) for c in theme_map.keys())
            if all_nan:
                skipped_unannotated += 1
                continue

            # Dedup
            essay_key = essay[:200].strip().lower()
            if essay_key in existing_essays:
                skipped_dup += 1
                continue
            existing_essays.add(essay_key)

            # Extract course from course_name column
            course_name = row.get('course_name', '')
            course = file_info['course']

            record = {
                'alma_id': f"S25.{course}.{int(user_id) if not pd.isna(user_id) else 'UNK'}",
                'user_id': str(int(user_id)) if not pd.isna(user_id) else '',
                'course': course,
                'semester': 'Spring',
                'year': '2025',
                'prompt': 'Why am I here?',
                'essay': essay,
                'essay_length': len(essay),
                'source_file': fname,
                'source_sheet': df.columns.name if hasattr(df.columns, 'name') else 'Sheet1',
            }

            # Process coded themes
            for raw_col, std_name in theme_map.items():
                val = row.get(raw_col)
                record[f'{std_name}_binary'] = binarize_label(val)
                excerpts = extract_excerpts(val)
                record[f'{std_name}_excerpts'] = ' /%/ '.join(excerpts) if excerpts else ''

            # Combine Strong + Weak Spiritual into single Spiritual label
            strong = record.get('Strong_Spiritual_binary', 0)
            weak = record.get('Weak_Spiritual_binary', 0)
            record['Spiritual_binary'] = 1 if (strong == 1 or weak == 1) else 0
            strong_exc = record.get('Strong_Spiritual_excerpts', '')
            weak_exc = record.get('Weak_Spiritual_excerpts', '')
            all_exc = [e for e in [strong_exc, weak_exc] if e]
            record['Spiritual_excerpts'] = ' /%/ '.join(all_exc) if all_exc else ''

            # Mark NOT CODED themes with -1 (distinguish from 0 = coded as absent)
            not_coded_themes = ['Attainment', 'First_Gen', 'Aspirational', 'Navigational',
                               'Filial_Piety', 'Familial', 'Social']
            for theme in not_coded_themes:
                record[f'{theme}_binary'] = -1  # Not coded
                record[f'{theme}_excerpts'] = 'NOT_CODED'

            all_rows.append(record)

    # Create output
    print(f"\n{'='*70}")
    print(f"SPRING 2025 'WHY AM I HERE?' PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"New rows: {len(all_rows)}")
    print(f"Skipped (empty): {skipped_empty}")
    print(f"Skipped (duplicate): {skipped_dup}")
    print(f"Skipped (unannotated): {skipped_unannotated}")

    if all_rows:
        df_out = pd.DataFrame(all_rows)

        # Define output columns
        meta_cols = ['alma_id', 'user_id', 'course', 'semester', 'year', 'prompt',
                     'essay', 'essay_length', 'source_file', 'source_sheet']

        # Standard 11-theme binary + excerpt columns
        binary_cols = [f'{t}_binary' for t in ALL_THEMES]
        excerpt_cols = [f'{t}_excerpts' for t in ALL_THEMES]
        # Also keep the sub-labels
        extra_cols = ['Strong_Spiritual_binary', 'Strong_Spiritual_excerpts',
                      'Weak_Spiritual_binary', 'Weak_Spiritual_excerpts']

        all_cols = meta_cols + binary_cols + excerpt_cols + extra_cols
        for col in all_cols:
            if col not in df_out.columns:
                df_out[col] = 0 if '_binary' in col else ''

        df_out = df_out[all_cols]
        df_out = df_out.sort_values('alma_id').reset_index(drop=True)

        output_path = os.path.join(ANNOTATED_WHY, "spring2025_why_am_i_here.csv")
        df_out.to_csv(output_path, index=False)
        print(f"Output: {output_path}")

        # Summary
        print(f"\nBy course:")
        for c, g in df_out.groupby('course'):
            print(f"  {c}: {len(g)} rows")

        print(f"\nCoded theme distribution:")
        coded_themes = ['Perseverance', 'Community_Consciousness', 'Resistance', 'Spiritual']
        for t in coded_themes:
            n = (df_out[f'{t}_binary'] == 1).sum()
            total = (df_out[f'{t}_binary'] >= 0).sum()
            pct = n / total * 100 if total > 0 else 0
            print(f"  {t:25s}: {n:3d} / {total} coded ({pct:.1f}%)")

        print(f"\nNot coded themes: {', '.join(not_coded_themes)}")
        print("These were not evaluated in Spring 2025 coding → marked as -1")

        # Spot check
        print(f"\n=== SPOT CHECK ===")
        for _, r in df_out.head(3).iterrows():
            active = [t for t in ALL_THEMES if r[f'{t}_binary'] == 1]
            print(f"\n{r['alma_id']} | {r['course']}")
            print(f"  Essay ({r['essay_length']} chars): {r['essay'][:120]}...")
            print(f"  Active themes: {active}")


if __name__ == '__main__':
    # Handle unresolvable column names
    not_coded_themes = ['Attainment', 'First_Gen', 'Aspirational', 'Navigational',
                       'Filial_Piety', 'Familial', 'Social']
    main()
