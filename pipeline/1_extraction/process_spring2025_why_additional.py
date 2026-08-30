"""
Process ADDITIONAL Spring 2025 "Why am I here?" courses: PHYS222 and ASTR116
These have NO merged file - only individual coder files exist.
Using AB coder (most complete: has strong/weak spiritual split) as primary.
PL coder used for additional coverage where AB is missing.
"""
import pandas as pd
import numpy as np
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from process_gold_standard import clean_text, binarize_label, extract_excerpts, ALL_THEMES

BASE = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA Project-selected"
SPRING25_DIR = os.path.join(BASE, "ALMA 2025/Khalid-Allyson-Phuong-Kirsten-Aline-Nada/Spring2025-whyamIhere")
OUTPUT = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA_Master_Dataset"
ANNOTATED_WHY = os.path.join(OUTPUT, "annotated", "why_am_i_here")

# Theme mapping for these files (note: AB files have strong/weak spiritual split)
THEME_MAP_AB = {
    'perseverance': 'Perseverance',
    'community': 'Community_Consciousness',
    'strong spiritual': 'Strong_Spiritual',
    'weak spiritual': 'Weak_Spiritual',
    'weak spirtual': 'Weak_Spiritual',  # typo in ASTR116
    'resistance': 'Resistance',
    'resistance\xa0': 'Resistance',  # NBSP variant
}

THEME_MAP_PL = {
    'perseverance': 'Perseverance',
    'community': 'Community_Consciousness',
    'spiritual': 'Spiritual',
    'resistance': 'Resistance',
    'resistance\xa0': 'Resistance',
}

# Files to process (course, AB file, PL file)
FILES = [
    {
        'course': 'PHYS222',
        'ab_file': 'PHYS222-Spring2025-Why am I here - AB.xlsx',
        'pl_file': 'PHYS222-Spring2025-Why am I here- PL .xlsx',
    },
    {
        'course': 'ASTR116',
        'ab_file': 'ASTR116-Spring2025-Why am I here (AB).xlsx',
        'pl_file': 'ASTR116-Spring2025-Why am I here-PL.xlsx',
    },
]


def process_coder_file(filepath, course, theme_map, coder_name):
    """Process a single coder file."""
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return []

    df = df.dropna(how='all')

    # Clean column names
    clean_cols = {}
    for c in df.columns:
        if c is not None:
            cleaned = str(c).strip().replace('\xa0', ' ').strip()
            clean_cols[c] = cleaned
    df = df.rename(columns=clean_cols)

    # Remove unnamed columns
    df = df[[c for c in df.columns if 'Unnamed' not in str(c)]]

    # Find columns
    id_col = next((c for c in df.columns if 'user_id' in str(c).lower()), None)
    essay_col = next((c for c in df.columns if 'submission' in str(c).lower()), None)

    if not id_col or not essay_col:
        print(f"  ERROR: Missing columns. Found: {list(df.columns)}")
        return []

    data = df[df[id_col].notna()]

    # Map theme columns
    mapped_themes = {}
    for c in df.columns:
        c_clean = str(c).strip().lower()
        if c_clean in theme_map:
            mapped_themes[c] = theme_map[c_clean]

    print(f"  Coder {coder_name}: {len(data)} rows, themes: {list(mapped_themes.values())}")

    rows = []
    for idx, row in data.iterrows():
        user_id = row[id_col]
        essay = row.get(essay_col, '')
        if pd.isna(essay) or len(str(essay).strip()) < 10:
            continue

        essay = clean_text(str(essay))

        # Check if all themes are NaN (unannotated)
        all_nan = all(pd.isna(row.get(c)) for c in mapped_themes.keys())
        if all_nan:
            continue

        record = {
            'alma_id': f"S25.{course}.{int(user_id) if not pd.isna(user_id) else 'UNK'}",
            'user_id': str(int(user_id)) if not pd.isna(user_id) else '',
            'course': course,
            'semester': 'Spring',
            'year': '2025',
            'prompt': 'Why am I here?',
            'essay': essay,
            'essay_length': len(essay),
            'source_file': os.path.basename(filepath),
            'source_sheet': 'Sheet1',
            'coder': coder_name,
        }

        # Process coded themes
        for raw_col, std_name in mapped_themes.items():
            val = row.get(raw_col)
            record[f'{std_name}_binary'] = binarize_label(val)
            excerpts = extract_excerpts(val)
            record[f'{std_name}_excerpts'] = ' /%/ '.join(excerpts) if excerpts else ''

        # Combine Strong + Weak Spiritual into single Spiritual (for AB files)
        strong = record.get('Strong_Spiritual_binary', None)
        weak = record.get('Weak_Spiritual_binary', None)
        if strong is not None or weak is not None:
            record['Spiritual_binary'] = 1 if (strong == 1 or weak == 1) else 0
            strong_exc = record.get('Strong_Spiritual_excerpts', '')
            weak_exc = record.get('Weak_Spiritual_excerpts', '')
            all_exc = [e for e in [strong_exc, weak_exc] if e]
            record['Spiritual_excerpts'] = ' /%/ '.join(all_exc) if all_exc else ''

        # NOT CODED themes
        for theme in ['Attainment', 'First_Gen', 'Aspirational', 'Navigational',
                      'Filial_Piety', 'Familial', 'Social']:
            record[f'{theme}_binary'] = -1
            record[f'{theme}_excerpts'] = 'NOT_CODED'

        rows.append(record)

    return rows


def main():
    # Load existing data for dedup
    existing_essays = set()
    for f in os.listdir(ANNOTATED_WHY):
        if f.endswith('.csv'):
            fpath = os.path.join(ANNOTATED_WHY, f)
            df = pd.read_csv(fpath)
            for _, r in df.iterrows():
                existing_essays.add(str(r['essay'])[:200].strip().lower())
            print(f"Loaded {len(df)} rows from {f} for dedup")
    print(f"Total existing essays for dedup: {len(existing_essays)}")

    all_new_rows = []

    for file_info in FILES:
        course = file_info['course']
        print(f"\n{'='*60}")
        print(f"PROCESSING: {course} Spring 2025 'Why am I here?'")
        print(f"{'='*60}")

        # Process AB file (primary - has strong/weak spiritual split)
        ab_path = os.path.join(SPRING25_DIR, file_info['ab_file'])
        if os.path.exists(ab_path):
            ab_rows = process_coder_file(ab_path, course, THEME_MAP_AB, 'AB')
        else:
            ab_rows = []
            print(f"  AB file not found: {file_info['ab_file']}")

        # Process PL file (supplementary)
        pl_path = os.path.join(SPRING25_DIR, file_info['pl_file'])
        if os.path.exists(pl_path):
            pl_rows = process_coder_file(pl_path, course, THEME_MAP_PL, 'PL')
        else:
            pl_rows = []
            print(f"  PL file not found: {file_info['pl_file']}")

        # Use AB as primary, add PL rows for essays not in AB
        ab_essays = set()
        added_ab = 0
        for r in ab_rows:
            essay_key = r['essay'][:200].strip().lower()
            if essay_key not in existing_essays:
                all_new_rows.append(r)
                existing_essays.add(essay_key)
                ab_essays.add(essay_key)
                added_ab += 1

        added_pl = 0
        for r in pl_rows:
            essay_key = r['essay'][:200].strip().lower()
            if essay_key not in existing_essays and essay_key not in ab_essays:
                all_new_rows.append(r)
                existing_essays.add(essay_key)
                added_pl += 1

        print(f"  Added from AB: {added_ab}, Added from PL (unique): {added_pl}")

    # Save
    print(f"\n{'='*60}")
    print(f"ADDITIONAL SPRING 2025 PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"New rows: {len(all_new_rows)}")

    if all_new_rows:
        df_out = pd.DataFrame(all_new_rows)

        meta_cols = ['alma_id', 'user_id', 'course', 'semester', 'year', 'prompt',
                     'essay', 'essay_length', 'source_file', 'source_sheet', 'coder']
        binary_cols = [f'{t}_binary' for t in ALL_THEMES]
        excerpt_cols = [f'{t}_excerpts' for t in ALL_THEMES]
        extra_cols = ['Strong_Spiritual_binary', 'Strong_Spiritual_excerpts',
                      'Weak_Spiritual_binary', 'Weak_Spiritual_excerpts']

        all_cols = meta_cols + binary_cols + excerpt_cols + extra_cols
        for col in all_cols:
            if col not in df_out.columns:
                df_out[col] = 0 if '_binary' in col else ''

        df_out = df_out[all_cols].sort_values('alma_id').reset_index(drop=True)

        output_path = os.path.join(ANNOTATED_WHY, "spring2025_why_additional.csv")
        df_out.to_csv(output_path, index=False)
        print(f"Output: {output_path}")

        print(f"\nBy course:")
        for c, g in df_out.groupby('course'):
            print(f"  {c}: {len(g)} rows")

        print(f"\nBy coder:")
        for c, g in df_out.groupby('coder'):
            print(f"  {c}: {len(g)} rows")

        print(f"\nCoded theme distribution:")
        for t in ['Perseverance', 'Community_Consciousness', 'Resistance', 'Spiritual']:
            n_pos = (df_out[f'{t}_binary'] == 1).sum()
            total = (df_out[f'{t}_binary'] >= 0).sum()
            pct = n_pos / total * 100 if total > 0 else 0
            print(f"  {t:25s}: {n_pos:3d} / {total} coded ({pct:.1f}%)")


if __name__ == '__main__':
    main()
