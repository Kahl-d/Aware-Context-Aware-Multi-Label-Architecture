"""
RECOVERY: Add 22 truly missing essays found during comprehensive audit.

Sources:
1. validated_data/all_capitals_published_quotes_IRR.xlsx
   - Fall 2019 PHYS122: 20 essays (skipped because gold standard file was mislabeled)
   - S18 SI: 1 essay
2. validated_data/all_capitals_unpublished_quotes_IRR.xlsx
   - Fall 2019 PHYS102: 1 essay
3. batch_2_merge_reconciliations_wout_nav.xlsx
   - 2 essays with annotations not in reconciled file

These essays have human annotations (11 themes for validated, 5 for batch2).
Text comparison used aggressive normalization (strip all non-alnum) to confirm genuinely missing.
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


def normalize_for_dedup(text):
    """Aggressive normalization for dedup comparison."""
    t = str(text).lower()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t[:200]


def load_master_essays():
    """Load all master dataset essays with aggressive normalization."""
    master = set()
    annotated_dir = os.path.join(OUTPUT, "annotated")
    for root, dirs, files in os.walk(annotated_dir):
        for f in files:
            if f.endswith('.csv'):
                fp = os.path.join(root, f)
                df = pd.read_csv(fp)
                for _, r in df.iterrows():
                    key = normalize_for_dedup(r['essay'])
                    master.add(key)
    return master


# Theme column mapping for validated_data files (11 themes)
VALIDATED_THEME_MAP = {
    'Attainment FINAL': 'Attainment',
    'Attainment - FINAL': 'Attainment',
    'Attainment final': 'Attainment',
    'Attainment': 'Attainment',
    'First Gen FINAL': 'First_Gen',
    'First Gen - FInal': 'First_Gen',
    'First Generation': 'First_Gen',
    'Aspirational FINAL': 'Aspirational',
    'Aspirational - FINAL': 'Aspirational',
    'Aspirational ': 'Aspirational',
    'Aspirational': 'Aspirational',
    'Navigational FINAL': 'Navigational',
    'Navigational - FINAL': 'Navigational',
    'Naviagational ': 'Navigational',  # typo in source
    'Navigational': 'Navigational',
    'Resistance FINAL': 'Resistance',
    'Resistance - FINAL': 'Resistance',
    'Resistance': 'Resistance',
    'Perserverance FINAL': 'Perseverance',  # typo in source
    'Perseverance FINAL': 'Perseverance',
    'Perseverance': 'Perseverance',
    'Filial Piety FINAL': 'Filial_Piety',
    'Filial Piety': 'Filial_Piety',
    'Familial FINAL': 'Familial',
    'Familial': 'Familial',
    'Community Consciousness FINAL ': 'Community_Consciousness',
    'Community Consciousness FINAL': 'Community_Consciousness',
    'Community Consciousness': 'Community_Consciousness',
    'Social FINAL': 'Social',
    'Social': 'Social',
    'Spiritual FINAL': 'Spiritual',
    'Spiritual': 'Spiritual',
}

# Semester/course mapping for validated data sheets
SHEET_META = {
    'S18 SI': ('Spring', '2018', 'SI-PHYS111'),
    'F18 SI': ('Fall', '2018', 'SI-PHYS111'),
    'S19 SI': ('Spring', '2019', 'SI-PHYS111'),
    'F19 SI': ('Fall', '2019', 'SI-PHYS111'),
    'Fall 2018 - PHYS 112 Lab': ('Fall', '2018', 'PHYS112'),
    'Spring 2019 - PHYS 112 Lab': ('Spring', '2019', 'PHYS112'),
    'Fall 2019 - PHYS 112 Lab': ('Fall', '2019', 'PHYS112'),
    'Fall 2019 - PHYS122': ('Fall', '2019', 'PHYS122'),
    'Fall 2019 - PHYS222': ('Fall', '2019', 'PHYS222'),
    'Fall 2019 - PHYS232': ('Fall', '2019', 'PHYS232'),
    'Fall 2019 - ASTR116': ('Fall', '2019', 'ASTR116'),
    'Pilot': ('Pilot', 'Pre-2018', 'Pilot'),
    'Spring 2018 SI 333': ('Spring', '2018', 'SI-PHYS333'),
    'Spring 2018 - SI': ('Spring', '2018', 'SI-PHYS111'),
    'Fall 2019 - PHYS102': ('Fall', '2019', 'PHYS102'),
    'Fall 2019 - PHYS242': ('Fall', '2019', 'PHYS242'),
}


def main():
    existing_essays = load_master_essays()
    print(f"Master dataset: {len(existing_essays)} unique normalized essays")

    # ========================================
    # Part 1: Recover from validated_data
    # ========================================
    print(f"\n{'='*60}")
    print(f"PART 1: RECOVER FROM VALIDATED_DATA")
    print(f"{'='*60}")

    published = os.path.join(BASE, "ALMA 2024/Data/validated_data/batch_1_validated/all_capitals/all_capitals_published_quotes_IRR.xlsx")
    unpublished = os.path.join(BASE, "ALMA 2024/Data/validated_data/batch_1_validated/all_capitals/all_capitals_unpublished_quotes_IRR.xlsx")

    new_rows = []

    for label, fpath in [("PUBLISHED", published), ("UNPUBLISHED", unpublished)]:
        xl = pd.ExcelFile(fpath)
        for sheet in xl.sheet_names:
            df = pd.read_excel(fpath, sheet_name=sheet)
            df = df.dropna(how='all')

            # Get metadata for this sheet
            semester, year, course = SHEET_META.get(sheet, ('Unknown', 'Unknown', 'Unknown'))

            # Find ID column
            id_col = None
            for c in df.columns:
                if 'alma' in str(c).lower() and 'id' in str(c).lower():
                    id_col = c
                    break
                if 'id' in str(c).lower() and 'code' in str(c).lower():
                    id_col = c
                    break
            if not id_col:
                id_col = df.columns[0]

            # Find essay column
            essay_col = None
            for c in df.columns:
                if 'essay' in str(c).lower() or 'why' in str(c).lower():
                    essay_col = c
                    break
            if not essay_col:
                continue

            # Map theme columns
            mapped_themes = {}
            for c in df.columns:
                c_str = str(c).strip()
                if c_str in VALIDATED_THEME_MAP:
                    mapped_themes[c] = VALIDATED_THEME_MAP[c_str]

            sheet_added = 0
            for _, r in df.iterrows():
                essay = str(r.get(essay_col, ''))
                if len(essay.strip()) < 10:
                    continue

                key = normalize_for_dedup(essay)
                if key in existing_essays:
                    continue

                # This essay is truly missing!
                existing_essays.add(key)

                essay_clean = clean_text(essay)
                alma_id = str(r.get(id_col, 'UNK')).strip()

                record = {
                    'alma_id': alma_id,
                    'user_id': '',
                    'course': course,
                    'semester': semester,
                    'year': year,
                    'prompt': 'Why am I here?',
                    'essay': essay_clean,
                    'essay_length': len(essay_clean),
                    'source_file': os.path.basename(fpath),
                    'source_sheet': sheet,
                    'coder': 'validated_IRR',
                }

                # Process themes
                for raw_col, std_name in mapped_themes.items():
                    val = r.get(raw_col)
                    record[f'{std_name}_binary'] = binarize_label(val)
                    excerpts = extract_excerpts(val)
                    record[f'{std_name}_excerpts'] = ' /%/ '.join(excerpts) if excerpts else ''

                # Fill missing themes with -1
                for theme in ALL_THEMES:
                    if f'{theme}_binary' not in record:
                        record[f'{theme}_binary'] = -1
                        record[f'{theme}_excerpts'] = 'NOT_CODED'

                new_rows.append(record)
                sheet_added += 1

            if sheet_added > 0:
                print(f"  {label}/{sheet}: {sheet_added} new essays recovered")

    print(f"\nTotal recovered from validated_data: {len(new_rows)}")

    # ========================================
    # Part 2: Recover from batch_2_merge
    # ========================================
    print(f"\n{'='*60}")
    print(f"PART 2: RECOVER FROM BATCH_2_MERGE")
    print(f"{'='*60}")

    merge_path = os.path.join(BASE, "ALMA 2024/Data/gian_reina_to_be_reconciled/batch_2_merge_reconciliations_wout_nav.xlsx")
    recon_path = os.path.join(BASE, "ALMA 2024/Data/gian_reina_to_be_reconciled/merged_reconciled_annotations_complete.xlsx")

    df_merge = pd.read_excel(merge_path)
    df_merge = df_merge.dropna(how='all')
    df_recon = pd.read_excel(recon_path)
    df_recon = df_recon.dropna(how='all')

    recon_ids = set(df_recon['Essay ID'].dropna().astype(str))

    BATCH2_THEMES = ['Aspirational', 'Familial', 'Social', 'Navigational', 'Resistance']

    merge_rows = []
    for eid in sorted(set(df_merge['Essay ID'].dropna().astype(str)) - recon_ids):
        rows = df_merge[df_merge['Essay ID'] == eid]
        essay = str(rows.iloc[0].get('Annotated Essays', ''))

        if pd.isna(essay) or len(str(essay).strip()) < 10 or essay == 'nan':
            continue

        # Clean
        essay_clean = re.sub(r'</?mark[^>]*>', '', essay)
        essay_clean = re.sub(r'</?[a-z][^>]*>', '', essay_clean)
        essay_clean = clean_text(essay_clean)

        if len(essay_clean) < 10:
            continue

        key = normalize_for_dedup(essay_clean)
        if key in existing_essays:
            continue
        existing_essays.add(key)

        # Parse metadata
        parts = eid.split('.')
        sem_map = {'S20': ('Spring', '2020'), 'F20': ('Fall', '2020')}
        semester, year = sem_map.get(parts[0], ('Unknown', 'Unknown'))
        course = re.sub(r'([A-Z]+)0+(\d+)', r'\1\2', parts[1]) if len(parts) >= 2 else 'Unknown'

        alma_id_num = rows.iloc[0].get('Alma ID')

        record = {
            'alma_id': eid,
            'user_id': str(int(alma_id_num)) if not pd.isna(alma_id_num) else '',
            'course': course,
            'semester': semester,
            'year': year,
            'prompt': 'Why am I here?',
            'essay': essay_clean,
            'essay_length': len(essay_clean),
            'source_file': 'batch_2_merge_reconciliations_wout_nav.xlsx',
            'source_sheet': 'Sheet1',
            'coder': 'pre-reconciliation',
        }

        # Use majority vote across coders for theme labels
        for theme in BATCH2_THEMES:
            vals = rows[theme].dropna()
            pos = sum(1 for v in vals if binarize_label(v) == 1)
            neg = sum(1 for v in vals if binarize_label(v) == 0)
            record[f'{theme}_binary'] = 1 if pos > neg else 0

            # Collect excerpts from positive coders
            excerpts = []
            for v in vals:
                excs = extract_excerpts(v)
                excerpts.extend(excs)
            record[f'{theme}_excerpts'] = ' /%/ '.join(excerpts) if excerpts else ''

        # Not coded themes
        for theme in ['Attainment', 'First_Gen', 'Perseverance', 'Filial_Piety',
                       'Community_Consciousness', 'Spiritual']:
            record[f'{theme}_binary'] = -1
            record[f'{theme}_excerpts'] = 'NOT_CODED'

        merge_rows.append(record)
        print(f"  Recovered: {eid} ({course}, {semester} {year})")

    print(f"\nTotal recovered from batch_2_merge: {len(merge_rows)}")

    # ========================================
    # Part 3: Save recovered essays
    # ========================================
    all_recovered = new_rows + merge_rows
    print(f"\n{'='*60}")
    print(f"TOTAL RECOVERED: {len(all_recovered)} essays")
    print(f"{'='*60}")

    if not all_recovered:
        print("No essays to recover!")
        return

    # Add validated_data essays to gold_standard file
    if new_rows:
        gs_path = os.path.join(ANNOTATED_WHY, "gold_standard_fall2019.csv")
        gs_df = pd.read_csv(gs_path)
        print(f"\nExisting gold_standard: {len(gs_df)} rows")

        new_df = pd.DataFrame(new_rows)

        # Ensure same columns
        for col in gs_df.columns:
            if col not in new_df.columns:
                if '_binary' in col:
                    new_df[col] = -1
                elif '_excerpts' in col:
                    new_df[col] = 'NOT_CODED'
                else:
                    new_df[col] = ''

        # Use only gold standard columns
        new_df = new_df[gs_df.columns]

        combined = pd.concat([gs_df, new_df], ignore_index=True)
        combined.to_csv(gs_path, index=False)
        print(f"Updated gold_standard: {len(combined)} rows (+{len(new_rows)} recovered)")

        # Show distribution of recovered
        print(f"\nRecovered validated essays by source:")
        for _, r in new_df.iterrows():
            active = [t for t in ALL_THEMES if r.get(f'{t}_binary') == 1]
            print(f"  {r['alma_id']} | {r['course']} | {r['semester']} {r['year']} | themes: {active}")

    # Add batch_2_merge essays to batch2 file
    if merge_rows:
        b2_path = os.path.join(ANNOTATED_WHY, "batch2_reconciled.csv")
        b2_df = pd.read_csv(b2_path)
        print(f"\nExisting batch2: {len(b2_df)} rows")

        merge_df = pd.DataFrame(merge_rows)

        for col in b2_df.columns:
            if col not in merge_df.columns:
                if '_binary' in col:
                    merge_df[col] = -1
                elif '_excerpts' in col:
                    merge_df[col] = 'NOT_CODED'
                else:
                    merge_df[col] = ''

        merge_df = merge_df[b2_df.columns]

        combined = pd.concat([b2_df, merge_df], ignore_index=True)
        combined.to_csv(b2_path, index=False)
        print(f"Updated batch2: {len(combined)} rows (+{len(merge_rows)} recovered)")

    print(f"\n{'='*60}")
    print(f"RECOVERY COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
