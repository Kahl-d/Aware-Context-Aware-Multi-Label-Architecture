"""
Process TIER 4: Batch 2 reconciled data (merged_reconciled_annotations_complete.xlsx)
Spring 2020 + Fall 2020 data, 5 themes only (Aspirational, Familial, Social, Navigational, Resistance)

CORRECTED ANALYSIS: This data is actually well-distributed (NOT 99% positive).
Previous analysis error was counting '0' as positive text.
Real distribution: Aspirational 67%, Navigational 76%, Social 30%, Resistance 20%, Familial 18%
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

# Batch 2 uses 5 core CCW themes only
BATCH2_THEMES = ['Aspirational', 'Familial', 'Social', 'Navigational', 'Resistance']


def parse_batch2_essay_id(essay_id):
    """Parse Batch 2 Essay ID format: S20.PHYS0112.04.003.376 or F20.ASTR0116.02.012.1242"""
    if pd.isna(essay_id):
        return '', '', ''

    essay_id = str(essay_id).strip()
    parts = essay_id.split('.')
    if len(parts) < 3:
        return '', '', ''

    # Semester + Year
    sem_code = parts[0]
    semester_map = {'S20': ('Spring', '2020'), 'F20': ('Fall', '2020')}
    semester, year = semester_map.get(sem_code, ('Unknown', 'Unknown'))

    # Course - strip leading zeros (PHYS0112 -> PHYS112, SCI0115 -> SCI115, SCI0333 -> SCI333)
    course_raw = parts[1] if len(parts) >= 2 else ''
    course = re.sub(r'([A-Z]+)0+(\d+)', r'\1\2', course_raw)

    return semester, year, course


def main():
    # Load existing data for dedup
    existing_essays = set()
    for f in ['gold_standard_fall2019.csv', 'batch1_new_rows.csv', 'spring2025_why_am_i_here.csv']:
        fpath = os.path.join(ANNOTATED_WHY, f)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            for _, r in df.iterrows():
                existing_essays.add(str(r['essay'])[:200].strip().lower())
            print(f"Loaded {len(df)} rows from {f} for dedup")
    print(f"Total existing essays for dedup: {len(existing_essays)}")

    # Read Batch 2
    path = os.path.join(BASE, "ALMA 2024/Data/gian_reina_to_be_reconciled/merged_reconciled_annotations_complete.xlsx")
    print(f"\n{'='*70}")
    print(f"PROCESSING: merged_reconciled_annotations_complete.xlsx (Batch 2)")
    print(f"{'='*70}")

    df = pd.read_excel(path)
    df = df.dropna(how='all')
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    # Columns: ['Alma ID', 'Essay ID', 'Annotated Essays', 'Aspirational', 'Familial', 'Social', 'Navigational', 'Resistance']

    # Column mapping
    alma_id_col = 'Alma ID'
    essay_id_col = 'Essay ID'
    essay_col = 'Annotated Essays'

    all_rows = []
    skipped_dup = 0
    skipped_empty = 0
    skipped_unannotated = 0
    skipped_pdf = 0
    skipped_html = 0

    for idx, row in df.iterrows():
        alma_id_num = row.get(alma_id_col)
        essay_id = row.get(essay_id_col)
        essay = row.get(essay_col, '')

        if pd.isna(essay) or len(str(essay).strip()) < 10:
            skipped_empty += 1
            continue

        essay = str(essay).strip()

        # Skip entries that are just "(PDF File)" or similar non-text
        if essay.lower().startswith('(pdf') or essay.lower() == 'pdf file':
            skipped_pdf += 1
            continue

        # Remove HTML <mark> tags from annotated essays
        essay = re.sub(r'</?mark[^>]*>', '', essay)
        essay = re.sub(r'</?[a-z][^>]*>', '', essay)  # Remove other HTML tags

        # Clean text
        essay = clean_text(essay)

        if len(essay) < 10:
            skipped_empty += 1
            continue

        # Check if ALL theme columns are 0 or NaN (unannotated row)
        all_negative = True
        for t in BATCH2_THEMES:
            val = row.get(t)
            if binarize_label(val) == 1:
                all_negative = False
                break

        if all_negative:
            # Check if it has all NaN (completely unannotated) vs all 0 (coded as having none)
            all_nan = all(pd.isna(row.get(t)) for t in BATCH2_THEMES)
            if all_nan:
                skipped_unannotated += 1
                continue
            # all 0 = coded as having no themes present → keep this, it's a valid negative example

        # Dedup by essay text
        essay_key = essay[:200].strip().lower()
        if essay_key in existing_essays:
            skipped_dup += 1
            continue
        existing_essays.add(essay_key)

        # Parse metadata from Essay ID
        semester, year, course = parse_batch2_essay_id(essay_id)

        # Build alma_id string
        if not pd.isna(essay_id):
            alma_id_str = str(essay_id).strip()
        else:
            alma_id_str = f"B2.{alma_id_num}" if not pd.isna(alma_id_num) else f"B2.UNK.{idx}"

        record = {
            'alma_id': alma_id_str,
            'user_id': str(int(alma_id_num)) if not pd.isna(alma_id_num) else '',
            'course': course,
            'semester': semester,
            'year': year,
            'prompt': 'Why am I here?',
            'essay': essay,
            'essay_length': len(essay),
            'source_file': 'merged_reconciled_annotations_complete.xlsx',
            'source_sheet': 'Sheet1',
            'coder': 'reconciled',
        }

        # Process coded themes (5 themes)
        for theme in BATCH2_THEMES:
            val = row.get(theme)
            record[f'{theme}_binary'] = binarize_label(val)
            excerpts = extract_excerpts(val)
            record[f'{theme}_excerpts'] = ' /%/ '.join(excerpts) if excerpts else ''

        # Mark NOT CODED themes with -1 (the other 6 themes not in Batch 2)
        not_coded_themes = ['Attainment', 'First_Gen', 'Perseverance',
                           'Filial_Piety', 'Community_Consciousness', 'Spiritual']
        for theme in not_coded_themes:
            record[f'{theme}_binary'] = -1
            record[f'{theme}_excerpts'] = 'NOT_CODED'

        all_rows.append(record)

    # Results
    print(f"\n{'='*70}")
    print(f"BATCH 2 PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"New rows: {len(all_rows)}")
    print(f"Skipped (empty/short): {skipped_empty}")
    print(f"Skipped (duplicate): {skipped_dup}")
    print(f"Skipped (unannotated): {skipped_unannotated}")
    print(f"Skipped (PDF file): {skipped_pdf}")

    if all_rows:
        df_out = pd.DataFrame(all_rows)

        meta_cols = ['alma_id', 'user_id', 'course', 'semester', 'year', 'prompt',
                     'essay', 'essay_length', 'source_file', 'source_sheet', 'coder']
        binary_cols = [f'{t}_binary' for t in ALL_THEMES]
        excerpt_cols = [f'{t}_excerpts' for t in ALL_THEMES]
        all_cols = meta_cols + binary_cols + excerpt_cols

        for col in all_cols:
            if col not in df_out.columns:
                df_out[col] = 0 if '_binary' in col else ''

        df_out = df_out[all_cols].sort_values('alma_id').reset_index(drop=True)

        output_path = os.path.join(ANNOTATED_WHY, "batch2_reconciled.csv")
        df_out.to_csv(output_path, index=False)
        print(f"\nOutput: {output_path}")

        # Summary
        print(f"\nBy course:")
        for c, g in df_out.groupby('course'):
            print(f"  {c}: {len(g)} rows")

        print(f"\nBy semester/year:")
        for (s, y), g in df_out.groupby(['semester', 'year']):
            print(f"  {s} {y}: {len(g)} rows")

        print(f"\nCoded theme distribution:")
        for t in BATCH2_THEMES:
            n_pos = (df_out[f'{t}_binary'] == 1).sum()
            n_neg = (df_out[f'{t}_binary'] == 0).sum()
            total = n_pos + n_neg
            pct = n_pos / total * 100 if total > 0 else 0
            print(f"  {t:15s}: {n_pos:4d} pos / {total:4d} total ({pct:.1f}%)")

        print(f"\nRows by # positive themes:")
        pos_counts = df_out[[f'{t}_binary' for t in BATCH2_THEMES]].apply(
            lambda row: (row == 1).sum(), axis=1
        )
        for n in range(6):
            cnt = (pos_counts == n).sum()
            print(f"  {n} themes: {cnt} rows ({cnt/len(df_out)*100:.1f}%)")

        print(f"\nNot coded themes: {', '.join(not_coded_themes)}")
        print("These 6 themes were not evaluated in Batch 2 -> marked as -1")

        # Spot check
        print(f"\n=== SPOT CHECK ===")
        for _, r in df_out.sample(min(5, len(df_out)), random_state=42).iterrows():
            active = [t for t in BATCH2_THEMES if r[f'{t}_binary'] == 1]
            print(f"\n{r['alma_id']} | {r['course']} | {r['semester']} {r['year']}")
            print(f"  Essay ({r['essay_length']} chars): {r['essay'][:120]}...")
            print(f"  Active themes: {active}")


if __name__ == '__main__':
    main()
