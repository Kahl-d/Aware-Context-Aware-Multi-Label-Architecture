"""
Process TIER 5: Spring 2025 Other Prompts (merged files)
Prompts: "when life gets challenging", "what are my goals", "personal values"
Same 5-theme coding as Spring 2025 "Why am I here?"
"""
import pandas as pd
import numpy as np
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from process_gold_standard import clean_text, binarize_label, extract_excerpts, ALL_THEMES

BASE = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA Project-selected"
KHALID_DIR = os.path.join(BASE, "ALMA 2025/Khalid-Allyson-Phuong-Kirsten-Aline-Nada")
OUTPUT = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA_Master_Dataset"

# Prompt to output folder mapping
PROMPT_FOLDER_MAP = {
    'What do I do when life gets challenging?': 'when_life_gets_challenging',
    'What are my goals and how can this class help me achieve these goals?': 'what_are_my_goals',
    'What are my goals? How is this class helping me achieve these goals?': 'what_are_my_goals',
    'How do your personal values intersect with the values of the scientific community (as you perceive them)?': 'personal_values',
}

SPRING25_THEME_MAP = {
    'perseverance': 'Perseverance',
    'community': 'Community_Consciousness',
    'strong spiritual': 'Strong_Spiritual',
    'weak spiritual': 'Weak_Spiritual',
    'resistance': 'Resistance',
    'resistance capital': 'Resistance',
}

# Merged files to process
MERGED_FILES = [
    ('Spring2025-OtherPrompts/KW AM NE - ASTR116-S25-when life gets challenging.xlsx', 'ASTR116', 0),
    ('Spring2025-OtherPrompts/KW NE AM - PHYS102-S25-when life gets challenging.xlsx', 'PHYS102', 0),
    ('Spring2025-OtherPrompts/KW NE AM - PHYS122-S25-What are my goals.xlsx', 'PHYS122', 0),
    ('Spring2025-OtherPrompts/KW NE AM - PHYS232-S25-personal values.xlsx', 'PHYS232', 0),
]

# Individual coder files for courses NOT in merged (check these for additional data)
# These are single-coder annotations - lower quality but only source for these courses
INDIVIDUAL_FILES = {
    'when_life_gets_challenging': [
        # PHYS112 - no merged file exists
        ('Spring2025-OtherPrompts/PHYS112-S25-when life gets challenging-AB.xlsx', 'PHYS112', 0, 'AB'),
        ('Spring2025-OtherPrompts/PHYS112-S25-when life gets challenging-PL.xlsx', 'PHYS112', 0, 'PL'),
        # PHYS222 - no merged file exists
        ('Spring2025-OtherPrompts/KW-PHYS222-S25-when life gets challenging.xlsx', 'PHYS222', 0, 'KW'),
    ],
    'what_are_my_goals': [
        # PHYS102 - no merged
        ('Spring2025-OtherPrompts/KW-PHYS102-S25-What are my goals.xlsx', 'PHYS102', 0, 'KW'),
        # PHYS242 - no merged
        ('Spring2025-OtherPrompts/KW-PHYS242-S25-What are my goals.xlsx', 'PHYS242', 0, 'KW'),
    ],
    'personal_values': [
        # PHYS242 - no merged
        ('Spring2025-OtherPrompts/PHYS242-S25-personal values-AB.xlsx', 'PHYS242', 0, 'AB'),
        ('Spring2025-OtherPrompts/PHYS242-S25-personal values-PL.xlsx', 'PHYS242', 0, 'PL'),
    ],
}


def process_file(filepath, course, header_row, coder='merged'):
    """Process a single Spring 2025 other-prompts file."""
    try:
        df = pd.read_excel(filepath, header=header_row)
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return [], 'unknown'

    df = df.dropna(how='all')
    clean_cols = {}
    for c in df.columns:
        if c is not None:
            clean_cols[c] = str(c).strip().replace('\xa0', ' ').strip()
    df = df.rename(columns=clean_cols)

    # Remove unnamed columns
    df = df[[c for c in df.columns if 'Unnamed' not in str(c)]]

    # Find columns
    id_col = next((c for c in df.columns if 'user_id' in str(c).lower()), None)
    essay_col = next((c for c in df.columns if 'submission' in str(c).lower()), None)
    prompt_col = next((c for c in df.columns if 'prompt' in str(c).lower()), None)

    if not id_col or not essay_col:
        print(f"  ERROR: Missing columns. Found: {list(df.columns)}")
        return [], 'unknown'

    data = df[df[id_col].notna()]

    # Get prompt
    prompt = 'Unknown'
    if prompt_col and len(data) > 0:
        prompt = str(data[prompt_col].dropna().iloc[0]).strip() if len(data[prompt_col].dropna()) > 0 else 'Unknown'

    # Determine output folder
    folder = 'other_prompts'
    for key_prompt, folder_name in PROMPT_FOLDER_MAP.items():
        if key_prompt.lower()[:30] in prompt.lower()[:30]:
            folder = folder_name
            break

    # Map theme columns
    theme_map = {}
    for c in df.columns:
        c_clean = str(c).strip().lower()
        if c_clean in SPRING25_THEME_MAP:
            theme_map[c] = SPRING25_THEME_MAP[c_clean]

    rows = []
    for idx, row in data.iterrows():
        user_id = row[id_col]
        essay = row.get(essay_col, '')
        if pd.isna(essay) or len(str(essay).strip()) < 10:
            continue

        essay = clean_text(str(essay))

        # Check if all themes are NaN (unannotated)
        all_nan = all(pd.isna(row.get(c)) for c in theme_map.keys())
        if all_nan:
            continue

        record = {
            'alma_id': f"S25.{course}.{int(user_id) if not pd.isna(user_id) else 'UNK'}",
            'user_id': str(int(user_id)) if not pd.isna(user_id) else '',
            'course': course,
            'semester': 'Spring',
            'year': '2025',
            'prompt': prompt,
            'essay': essay,
            'essay_length': len(essay),
            'source_file': os.path.basename(filepath),
            'source_sheet': 'Sheet1',
            'coder': coder,
        }

        # Process coded themes
        for raw_col, std_name in theme_map.items():
            val = row.get(raw_col)
            record[f'{std_name}_binary'] = binarize_label(val)
            excerpts = extract_excerpts(val)
            record[f'{std_name}_excerpts'] = ' /%/ '.join(excerpts) if excerpts else ''

        # Combine Spiritual
        strong = record.get('Strong_Spiritual_binary', 0)
        weak = record.get('Weak_Spiritual_binary', 0)
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

    return rows, folder


def main():
    # Collect all essays for dedup
    existing_essays = set()

    # Process merged files (highest quality)
    results_by_folder = {}

    print("=== PROCESSING MERGED FILES ===")
    for fpath, course, header_row in MERGED_FILES:
        full_path = os.path.join(KHALID_DIR, fpath)
        fname = os.path.basename(fpath)
        print(f"\n{'='*60}")
        print(f"FILE: {fname}")
        print(f"{'='*60}")

        rows, folder = process_file(full_path, course, header_row)
        print(f"  Rows: {len(rows)}, Folder: {folder}")

        if folder not in results_by_folder:
            results_by_folder[folder] = []

        # Dedup
        added = 0
        for r in rows:
            essay_key = r['essay'][:200].strip().lower()
            if essay_key not in existing_essays:
                results_by_folder[folder].append(r)
                existing_essays.add(essay_key)
                added += 1

        print(f"  Added: {added}, Skipped (dup): {len(rows) - added}")

        # Theme distribution
        if rows:
            print(f"  Theme distribution:")
            for t in ['Perseverance', 'Community_Consciousness', 'Resistance', 'Spiritual']:
                n = sum(1 for r in rows if r.get(f'{t}_binary', 0) == 1)
                pct = n / len(rows) * 100
                print(f"    {t:25s}: {n:3d} / {len(rows)} ({pct:.1f}%)")

    # Process individual coder files for courses NOT in merged
    print("\n\n=== PROCESSING INDIVIDUAL CODER FILES (additional courses) ===")
    for prompt_folder, files in INDIVIDUAL_FILES.items():
        for fpath, course, header_row, coder in files:
            full_path = os.path.join(KHALID_DIR, fpath)
            fname = os.path.basename(fpath)

            if not os.path.exists(full_path):
                print(f"  SKIP: {fname} (not found)")
                continue

            print(f"\n  FILE: {fname} (coder: {coder})")
            rows, folder = process_file(full_path, course, header_row, coder=coder)

            if prompt_folder not in results_by_folder:
                results_by_folder[prompt_folder] = []

            added = 0
            for r in rows:
                essay_key = r['essay'][:200].strip().lower()
                if essay_key not in existing_essays:
                    results_by_folder[prompt_folder].append(r)
                    existing_essays.add(essay_key)
                    added += 1

            print(f"    Rows: {len(rows)}, Added: {added}, Dup: {len(rows) - added}")

    # Save results by folder
    print(f"\n\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    for folder, rows in results_by_folder.items():
        if not rows:
            continue

        output_dir = os.path.join(OUTPUT, "annotated", folder)
        os.makedirs(output_dir, exist_ok=True)

        df_out = pd.DataFrame(rows)

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

        output_path = os.path.join(output_dir, f"spring2025_{folder}.csv")
        df_out.to_csv(output_path, index=False)

        print(f"\n{folder}: {len(df_out)} rows → {output_path}")
        print(f"  By course: {dict(df_out.groupby('course').size())}")
        print(f"  By coder: {dict(df_out.groupby('coder').size())}")


if __name__ == '__main__':
    main()
