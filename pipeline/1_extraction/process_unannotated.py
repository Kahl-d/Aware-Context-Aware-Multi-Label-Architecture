"""
Process unannotated Spring 2025 raw CSVs.
These are student essays that have NOT been coded for CCW themes.
Organized by prompt type into unannotated/ subfolders.
Dedup against all annotated data to avoid duplicates.
"""
import pandas as pd
import numpy as np
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from process_gold_standard import clean_text

BASE = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA Project-selected"
RAW_DIR = os.path.join(BASE, "New Data Spring 2025")
OUTPUT = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA_Master_Dataset"

# Map prompt text to output folder
PROMPT_FOLDER_MAP = {
    'why am i here': 'why_am_i_here',
    'what do i do when life gets challenging': 'when_life_gets_challenging',
    'what are my goals': 'what_are_my_goals',
    'how do your personal values intersect': 'personal_values',
    'how can i contribute to stem': 'contribute_to_stem',
    'what is the role of ai in stem': 'ai_in_stem',
}

# Map course folder names to standardized course codes
COURSE_MAP = {
    'ASTR 116': 'ASTR116',
    'PHYS 102': 'PHYS102',
    'PHYS 112': 'PHYS112',
    'PHYS 122': 'PHYS122',
    'PHYS 222': 'PHYS222',
    'PHYS 232': 'PHYS232',
    'PHYS 242': 'PHYS242',
}


def get_prompt_folder(prompt_text):
    """Map prompt text to output folder name."""
    prompt_lower = prompt_text.strip().lower().rstrip('?')
    for key, folder in PROMPT_FOLDER_MAP.items():
        if key in prompt_lower:
            return folder
    # Fallback: create slug from prompt
    slug = re.sub(r'[^a-z0-9]+', '_', prompt_lower)[:50].strip('_')
    return slug


def main():
    # Load ALL annotated essays for dedup
    existing_essays = set()
    annotated_dir = os.path.join(OUTPUT, "annotated")
    for root, dirs, files in os.walk(annotated_dir):
        for f in files:
            if f.endswith('.csv'):
                fpath = os.path.join(root, f)
                try:
                    df = pd.read_csv(fpath)
                    for _, r in df.iterrows():
                        existing_essays.add(str(r['essay'])[:200].strip().lower())
                except:
                    pass
    print(f"Loaded {len(existing_essays)} annotated essays for dedup")

    # Process each course folder
    results = {}
    total_new = 0
    total_dup = 0
    total_empty = 0

    course_dirs = [d for d in sorted(os.listdir(RAW_DIR))
                   if os.path.isdir(os.path.join(RAW_DIR, d)) and not d.startswith('.')]

    for course_dir in course_dirs:
        course_path = os.path.join(RAW_DIR, course_dir)

        # Extract course code
        course = 'Unknown'
        for key, code in COURSE_MAP.items():
            if key in course_dir:
                course = code
                break

        csv_files = [f for f in sorted(os.listdir(course_path))
                     if f.endswith('.csv') and not f.startswith('.')]

        print(f"\n{'='*60}")
        print(f"COURSE: {course} ({course_dir})")
        print(f"{'='*60}")

        for csv_file in csv_files:
            csv_path = os.path.join(course_path, csv_file)
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"  ERROR reading {csv_file}: {e}")
                continue

            print(f"\n  File: {csv_file} ({len(df)} rows)")

            # Get prompt
            prompt = 'Unknown'
            if 'prompt' in df.columns and len(df) > 0:
                prompt = str(df['prompt'].dropna().iloc[0]).strip() if len(df['prompt'].dropna()) > 0 else 'Unknown'

            folder = get_prompt_folder(prompt)
            print(f"  Prompt: {prompt}")
            print(f"  Folder: {folder}")

            if folder not in results:
                results[folder] = []

            # Process each row
            added = 0
            for idx, row in df.iterrows():
                user_id = row.get('user_id', '')
                essay = row.get('submission', '')

                if pd.isna(essay) or len(str(essay).strip()) < 10:
                    total_empty += 1
                    continue

                essay = clean_text(str(essay))
                if len(essay) < 10:
                    total_empty += 1
                    continue

                # Dedup
                essay_key = essay[:200].strip().lower()
                if essay_key in existing_essays:
                    total_dup += 1
                    continue
                existing_essays.add(essay_key)

                record = {
                    'alma_id': f"S25.{course}.{int(user_id) if not pd.isna(user_id) else 'UNK'}",
                    'user_id': str(int(user_id)) if not pd.isna(user_id) else '',
                    'course': course,
                    'semester': 'Spring',
                    'year': '2025',
                    'prompt': prompt,
                    'essay': essay,
                    'essay_length': len(essay),
                    'source_file': csv_file,
                }

                results[folder].append(record)
                added += 1

            print(f"  Added: {added}")
            total_new += added

    # Save results by folder
    print(f"\n\n{'='*60}")
    print(f"UNANNOTATED DATA PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total new: {total_new}")
    print(f"Total duplicates (in annotated): {total_dup}")
    print(f"Total empty/short: {total_empty}")

    for folder, rows in sorted(results.items()):
        if not rows:
            continue

        output_dir = os.path.join(OUTPUT, "unannotated", folder)
        os.makedirs(output_dir, exist_ok=True)

        df_out = pd.DataFrame(rows)
        cols = ['alma_id', 'user_id', 'course', 'semester', 'year', 'prompt',
                'essay', 'essay_length', 'source_file']
        df_out = df_out[cols].sort_values('alma_id').reset_index(drop=True)

        output_path = os.path.join(output_dir, f"unannotated_{folder}.csv")
        df_out.to_csv(output_path, index=False)

        print(f"\n{folder}: {len(df_out)} rows -> {output_path}")
        print(f"  By course: {dict(df_out.groupby('course').size())}")


if __name__ == '__main__':
    main()
