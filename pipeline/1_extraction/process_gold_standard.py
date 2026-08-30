"""
Process TIER 1: Fall 2019 Gold Standard files (Final4Khalid)
Each file is processed individually, analyzed, and added to master dataset.
"""
import pandas as pd
import numpy as np
import os
import re
import json
from pathlib import Path

# Paths
BASE = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA Project-selected"
OUTPUT = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA_Master_Dataset"
ANNOTATED_WHY = os.path.join(OUTPUT, "annotated", "why_am_i_here")

# Standard theme names (canonical)
THEME_MAP = {
    # Attainment variants
    'attainment final': 'Attainment',
    'attainment - final': 'Attainment',
    'attainment': 'Attainment',
    # First Gen variants
    'first gen final': 'First_Gen',
    'first gen - final': 'First_Gen',
    'first generation': 'First_Gen',
    # Aspirational variants
    'aspirational final': 'Aspirational',
    'aspirational - final': 'Aspirational',
    'aspirational': 'Aspirational',
    # Navigational variants
    'navigational final': 'Navigational',
    'navigational - final': 'Navigational',
    'naviagational': 'Navigational',
    # Resistance variants
    'resistance final': 'Resistance',
    'resistance - final': 'Resistance',
    'resistance': 'Resistance',
    # Perseverance variants (note: source data has typo "perserverance")
    'perserverance final': 'Perseverance',
    'perserverant  - final': 'Perseverance',
    'perserverant': 'Perseverance',
    'perserverant - final': 'Perseverance',
    'perseverance final': 'Perseverance',
    'perseverance': 'Perseverance',
    # Filial Piety variants
    'filial piety final': 'Filial_Piety',
    'filial piety - final': 'Filial_Piety',
    'filial piety': 'Filial_Piety',
    # Familial variants
    'familial final': 'Familial',
    'familial - final': 'Familial',
    'familial': 'Familial',
    # Community Consciousness variants
    'community consciousness final': 'Community_Consciousness',
    'community consciousness - final': 'Community_Consciousness',
    'community consciousness': 'Community_Consciousness',
    # Social variants
    'social final': 'Social',
    'social - final': 'Social',
    'social': 'Social',
    # Spiritual variants
    'spiritual final': 'Spiritual',
    'spiritual - final': 'Spiritual',
    'spiritual': 'Spiritual',
}

# All 11 standard theme names in order
ALL_THEMES = [
    'Attainment', 'First_Gen', 'Aspirational', 'Navigational', 'Resistance',
    'Perseverance', 'Filial_Piety', 'Familial', 'Community_Consciousness',
    'Social', 'Spiritual'
]


def normalize_column_name(col):
    """Map raw column name to standard theme name."""
    if col is None:
        return None
    clean = str(col).strip().lower()
    # Remove extra whitespace
    clean = re.sub(r'\s+', ' ', clean)
    return THEME_MAP.get(clean, None)


def parse_alma_id(alma_id):
    """Extract metadata from Alma ID format: F19.PHYS222.01.01 or F18.PHYS.112.01.01"""
    alma_id = str(alma_id).strip()

    semester_map = {
        'F': 'Fall', 'S': 'Spring', 'W': 'Winter',
        'f': 'Fall', 's': 'Spring', 'w': 'Winter'
    }

    semester = ''
    year = ''
    course = ''

    # Handle pilot study IDs (no dots)
    if 'pilot' in alma_id.lower():
        return 'Pilot', 'Pre-2018', 'Pilot'

    parts = alma_id.split('.')
    if len(parts) < 2:
        return semester, year, course

    # Handle "SCI 111.01.01" format (no semester prefix, space in name)
    first = parts[0].strip()
    if first.upper().startswith('SCI'):
        semester = 'Unknown'
        year = 'Unknown'
        # Extract course number from "SCI 111" or "SCI111"
        match = re.search(r'(\d+)', first)
        if match:
            course = f"SCI{match.group(1)}"
        else:
            course = first.upper().replace(' ', '')
        return semester, year, course

    # First part: semester+year (e.g., F19, S18)
    if first[0:1] in semester_map and len(first) >= 3:
        semester = semester_map[first[0]]
        yr = first[1:]
        if yr.isdigit():
            year = f"20{yr}" if int(yr) < 50 else f"19{yr}"

    # Course detection - handle both formats:
    # F19.PHYS222.01.01 (course as one token) vs F18.PHYS.112.01.01 (split)
    # Also: S18.W2.333.01.01 (SI course) and F20.PHYS0102.02.000.001
    # Also: F18.SI.111.04.01 (SI course variant)
    second = parts[1].upper() if len(parts) >= 2 else ''
    third = parts[2] if len(parts) >= 3 else ''

    # Check if second part is W2 or SI (SI courses)
    if second == 'W2' and third:
        course = f"SI-PHYS{third}"
    elif second == 'SI' and third:
        course = f"SI-PHYS{third}"
    elif re.match(r'^[A-Z]+\d+$', second):
        # Combined format: PHYS222, ASTR116, PHYS0102, SCI111
        course = second.replace('PHYS0', 'PHYS').replace('ASTR0', 'ASTR')
    elif re.match(r'^[A-Z]+$', second) and third.isdigit():
        # Split format: PHYS.112 → PHYS112
        course = f"{second}{third}"
        course = course.replace('PHYS0', 'PHYS').replace('ASTR0', 'ASTR')
    else:
        course = second

    return semester, year, course


def binarize_label(val):
    """Convert annotation value to binary: 1 if positive text, 0 if negative."""
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return 0 if val == 0 or val == 0.0 else 1
    # It's a string (text excerpt = positive)
    val_str = str(val).strip()
    if val_str == '' or val_str == '0' or val_str == '0.0' or val_str == 'nan':
        return 0
    return 1


def extract_excerpts(val):
    """Extract text excerpts from annotation, handling /%/ delimiter."""
    if pd.isna(val):
        return []
    if isinstance(val, (int, float)):
        if val == 0 or val == 0.0:
            return []
        return []

    val_str = str(val).strip()
    if val_str == '' or val_str == '0' or val_str == '0.0':
        return []

    # Split by /%/ delimiter
    excerpts = val_str.split('/%/')
    # Clean each excerpt
    cleaned = []
    for e in excerpts:
        e = e.strip()
        if e and e != '0' and e != '0.0':
            # Remove weird characters but preserve essential punctuation
            e = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', e)
            if len(e) > 5:  # Skip very short fragments
                cleaned.append(e)
    return cleaned


def clean_text(text):
    """Clean text: fix encoding issues, normalize unicode, remove artifacts."""
    if not text or pd.isna(text):
        return ''
    text = str(text).strip()
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Fix common mojibake (Windows-1252 → UTF-8 double-encoding)
    mojibake = {
        '\u2019': "'", '\u2018': "'",  # smart single quotes
        '\u201c': '"', '\u201d': '"',  # smart double quotes
        '\u2014': '--', '\u2013': '-',  # em/en dash
        '\u2026': '...',  # ellipsis
        '\xa0': ' ',  # non-breaking space
        '†': '',  # dagger from anonymization
        # Mojibake patterns (UTF-8 bytes read as Windows-1252)
        '\u00e2\u0080\u0099': "'",  # right single quote
        '\u00e2\u0080\u009c': '"',  # left double quote
        '\u00e2\u0080\u009d': '"',  # right double quote
        '\u00e2\u0080\u0094': '--',  # em dash
        '\u00e2\u0080\u0093': '-',  # en dash
        '\u00e2\u0080\u00a6': '...',  # ellipsis
        '\u00c3\u00ad': 'i',  # i with acute
        '\u00c3\u00a8': 'e',  # e with grave
        '\u00c3\u00ae': 'i',  # i with circumflex
        '\u00c3\u00b1': 'n',  # n with tilde
    }
    for bad, good in mojibake.items():
        text = text.replace(bad, good)
    # Additional mojibake: ‚Äô → ', ‚Äú → ", etc.
    text = text.replace('\u201a\u00c4\u00f4', "'")  # ‚Äô → '
    text = text.replace('\u201a\u00c4\u00fa', '"')  # ‚Äú → "
    text = text.replace('\u201a\u00c4\u00f9', '"')  # ‚Äù → "
    text = text.replace('\u201a\u00c4\u00ee', '--')  # ‚Äî → --
    text = text.replace('\u201a\u00c4\u00ec', '-')  # ‚Äì → -
    # Pattern-based cleanup for remaining mojibake
    text = re.sub(r'‚Äô', "'", text)
    text = re.sub(r'‚Äú', '"', text)
    text = re.sub(r'‚Äù', '"', text)
    text = re.sub(r'‚Äî', '--', text)
    text = re.sub(r'‚Äì', '-', text)
    text = re.sub(r'‚Ä†', '', text)
    text = re.sub(r'‚Ä¶', '...', text)  # ellipsis
    text = re.sub(r'‚Äò', "'", text)  # left single quote variant
    text = re.sub(r'‚Äë', "'", text)  # another quote variant
    text = re.sub(r'‚Ä¢', '-', text)  # dash variant
    text = re.sub(r'‚Ä[^\s]{0,2}', '', text)  # catch-all for remaining ‚Ä patterns
    text = re.sub(r'√≠', 'i', text)
    text = re.sub(r'√¨', 'i', text)
    text = re.sub(r'√Æ', 'i', text)
    text = re.sub(r'√±', 'n', text)
    text = re.sub(r'√©', 'e', text)
    text = re.sub(r'√°', 'o', text)
    text = re.sub(r'√¥', 'o', text)
    text = re.sub(r'√¼', 'u', text)
    # Clean up any remaining unusual characters
    text = re.sub(r'[√]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_sheet(df, source_file, sheet_name, id_col, essay_col, theme_cols_map):
    """Process a single sheet into standardized format."""
    rows = []

    for idx, row in df.iterrows():
        alma_id = row.get(id_col)
        if pd.isna(alma_id):
            continue

        alma_id = str(alma_id).strip()
        essay = row.get(essay_col, '')
        if pd.isna(essay):
            essay = ''
        essay = str(essay).strip()

        if not essay or len(essay) < 10:
            continue

        # Clean essay text
        essay = clean_text(essay)

        # Parse metadata
        semester, year, course = parse_alma_id(alma_id)

        # Build row
        record = {
            'alma_id': alma_id,
            'course': course,
            'semester': semester,
            'year': year,
            'prompt': 'Why am I here?',
            'essay': essay,
            'essay_length': len(essay),
            'source_file': source_file,
            'source_sheet': sheet_name,
        }

        # Process each theme
        for raw_col, standard_name in theme_cols_map.items():
            val = row.get(raw_col)
            record[f'{standard_name}_binary'] = binarize_label(val)
            excerpts = extract_excerpts(val)
            record[f'{standard_name}_excerpts'] = ' /%/ '.join(excerpts) if excerpts else ''

        # Add missing themes as 0
        for theme in ALL_THEMES:
            if f'{theme}_binary' not in record:
                record[f'{theme}_binary'] = 0
                record[f'{theme}_excerpts'] = ''

        rows.append(record)

    return rows


def analyze_file(filepath, file_label):
    """Deeply analyze a single Excel file."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {file_label}")
    print(f"Path: {filepath}")
    print(f"{'='*70}")

    wb_sheets = pd.ExcelFile(filepath).sheet_names
    print(f"Sheets: {wb_sheets}")

    all_rows = []
    analysis = {
        'file': file_label,
        'sheets': [],
        'total_rows': 0,
        'issues': [],
    }

    for sheet_name in wb_sheets:
        print(f"\n--- Sheet: {sheet_name} ---")
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        df = df.dropna(how='all')

        # Remove unnamed columns
        real_cols = [c for c in df.columns if c is not None and 'Unnamed' not in str(c)]
        df = df[real_cols]

        # Find ID column
        id_col = None
        for c in df.columns:
            c_lower = str(c).strip().lower()
            if c_lower in ('alma id', 'id code', 'alma_id'):
                id_col = c
                break

        if id_col is None:
            print(f"  WARNING: No ID column found! Columns: {list(df.columns)}")
            analysis['issues'].append(f"Sheet '{sheet_name}': No ID column")
            continue

        # Find essay column
        essay_col = None
        for c in df.columns:
            c_lower = str(c).strip().lower()
            if 'essay' in c_lower:
                essay_col = c
                break

        if essay_col is None:
            print(f"  WARNING: No essay column found! Columns: {list(df.columns)}")
            analysis['issues'].append(f"Sheet '{sheet_name}': No essay column")
            continue

        # Map theme columns
        theme_cols_map = {}
        unmapped_cols = []
        skip_cols = {id_col, essay_col}

        # Handle duplicate column names (e.g., two "Alma ID" columns)
        seen_ids = 0
        for c in df.columns:
            if c in skip_cols:
                continue
            standard = normalize_column_name(c)
            if standard:
                theme_cols_map[c] = standard
            elif c not in skip_cols:
                c_lower = str(c).strip().lower()
                if c_lower in ('alma id', 'id code'):
                    seen_ids += 1
                    if seen_ids > 0:
                        continue  # Skip duplicate ID columns
                # Check if it's Linguistic (which we skip)
                if 'linguistic' in c_lower or 'pluriversal' in c_lower:
                    print(f"  SKIPPING column: {c} (Linguistic/Pluriversal - not in taxonomy)")
                    continue
                unmapped_cols.append(c)

        if unmapped_cols:
            print(f"  UNMAPPED columns: {unmapped_cols}")
            analysis['issues'].append(f"Sheet '{sheet_name}': Unmapped columns {unmapped_cols}")

        # Filter to rows with IDs
        data = df[df[id_col].notna()].copy()
        n_rows = len(data)

        print(f"  Rows: {n_rows}")
        print(f"  ID column: {id_col}")
        print(f"  Essay column: {essay_col}")
        print(f"  Theme columns mapped: {len(theme_cols_map)}")
        for raw, std in theme_cols_map.items():
            print(f"    {raw} → {std}")

        # Check for duplicates
        dupes = data[id_col].duplicated()
        if dupes.any():
            dup_ids = data[dupes][id_col].tolist()
            print(f"  DUPLICATE IDs: {dup_ids}")
            analysis['issues'].append(f"Sheet '{sheet_name}': Duplicate IDs: {dup_ids}")

        # Theme distribution
        print(f"\n  Theme distribution:")
        for raw_col, std_name in theme_cols_map.items():
            vals = data[raw_col]
            n_pos = sum(binarize_label(v) for v in vals)
            n_neg = len(vals) - n_pos
            pct = n_pos / len(vals) * 100 if len(vals) > 0 else 0
            print(f"    {std_name:25s}: {n_pos:3d} pos ({pct:5.1f}%), {n_neg:3d} neg")

        # Process rows
        sheet_rows = process_sheet(data, file_label, sheet_name, id_col, essay_col, theme_cols_map)
        all_rows.extend(sheet_rows)

        analysis['sheets'].append({
            'name': sheet_name,
            'rows': n_rows,
            'themes_mapped': len(theme_cols_map),
        })
        analysis['total_rows'] += n_rows

    print(f"\nTotal processed rows: {len(all_rows)}")
    return all_rows, analysis


def main():
    # Output directory
    os.makedirs(ANNOTATED_WHY, exist_ok=True)

    # Track all processed data for dedup
    # IMPORTANT: Alma IDs can be same across courses (same student, different essay)
    # So dedup by (alma_id + essay_hash) not just alma_id
    master_rows = []
    seen_essay_hashes = set()  # hash of (alma_id, essay_text_first_100_chars)
    analyses = []

    # === FILE 1: ASTR116 ===
    rows, analysis = analyze_file(
        os.path.join(BASE, "ALMA 2025/Khalid-Allyson-Phuong-Kirsten-Aline-Nada/Final4Khalid/Alma-ASTR116-F19-forKhalid.xlsx"),
        "Alma-ASTR116-F19-forKhalid.xlsx"
    )
    for r in rows:
        # Dedup by essay content (not just ID - same student can have different essays in different courses)
        essay_hash = hash((r['alma_id'], r['essay'][:200].strip().lower()))
        if essay_hash not in seen_essay_hashes:
            master_rows.append(r)
            seen_essay_hashes.add(essay_hash)
        else:
            print(f"  DEDUP: Skipping {r['alma_id']} - duplicate essay")
    analyses.append(analysis)

    # === FILE 2: PHYS112 (3 sheets: Fall 2018, Spring 2019, Fall 2019) ===
    rows, analysis = analyze_file(
        os.path.join(BASE, "ALMA 2025/Khalid-Allyson-Phuong-Kirsten-Aline-Nada/Final4Khalid/Alma-PHYS112-S18-F19-forKhalid.xlsx"),
        "Alma-PHYS112-S18-F19-forKhalid.xlsx"
    )
    for r in rows:
        # Dedup by essay content (not just ID - same student can have different essays in different courses)
        essay_hash = hash((r['alma_id'], r['essay'][:200].strip().lower()))
        if essay_hash not in seen_essay_hashes:
            master_rows.append(r)
            seen_essay_hashes.add(essay_hash)
        else:
            print(f"  DEDUP: Skipping {r['alma_id']} - duplicate essay")
    analyses.append(analysis)

    # === FILE 3: PHYS122 (SKIP - confirmed duplicate of PHYS112 Fall 2019) ===
    print(f"\n{'='*70}")
    print("SKIPPING: Alma-PHYS122-F19-forKhalid.xlsx")
    print("REASON: Contains PHYS112 data (IDs: F19.PHYS112.*), duplicate of File 2 Fall 2019 sheet")
    print(f"{'='*70}")
    analyses.append({
        'file': 'Alma-PHYS122-F19-forKhalid.xlsx',
        'sheets': [],
        'total_rows': 0,
        'issues': ['SKIPPED: Mislabeled file - contains PHYS112 data, duplicate of File 2'],
    })

    # === FILE 4: PHYS222 ===
    rows, analysis = analyze_file(
        os.path.join(BASE, "ALMA 2025/Khalid-Allyson-Phuong-Kirsten-Aline-Nada/Final4Khalid/Alma-PHYS222-F19-forKhalid.xlsx"),
        "Alma-PHYS222-F19-forKhalid.xlsx"
    )
    for r in rows:
        # Dedup by essay content (not just ID - same student can have different essays in different courses)
        essay_hash = hash((r['alma_id'], r['essay'][:200].strip().lower()))
        if essay_hash not in seen_essay_hashes:
            master_rows.append(r)
            seen_essay_hashes.add(essay_hash)
        else:
            print(f"  DEDUP: Skipping {r['alma_id']} - duplicate essay")
    analyses.append(analysis)

    # === FILE 5: PHYS232 ===
    rows, analysis = analyze_file(
        os.path.join(BASE, "ALMA 2025/Khalid-Allyson-Phuong-Kirsten-Aline-Nada/Final4Khalid/Alma-PHYS232-Fall2019-forKhalid.xlsx"),
        "Alma-PHYS232-Fall2019-forKhalid.xlsx"
    )
    for r in rows:
        # Dedup by essay content (not just ID - same student can have different essays in different courses)
        essay_hash = hash((r['alma_id'], r['essay'][:200].strip().lower()))
        if essay_hash not in seen_essay_hashes:
            master_rows.append(r)
            seen_essay_hashes.add(essay_hash)
        else:
            print(f"  DEDUP: Skipping {r['alma_id']} - duplicate essay")
    analyses.append(analysis)

    # === FILE 6: Unpublished Quotes (5 sheets, various semesters) ===
    rows, analysis = analyze_file(
        os.path.join(BASE, "ALMA 2025/Khalid-Allyson-Phuong-Kirsten-Aline-Nada/Final4Khalid/Alma-unpublished-quotes-forKhalid.xlsx"),
        "Alma-unpublished-quotes-forKhalid.xlsx"
    )
    for r in rows:
        # Dedup by essay content (not just ID - same student can have different essays in different courses)
        essay_hash = hash((r['alma_id'], r['essay'][:200].strip().lower()))
        if essay_hash not in seen_essay_hashes:
            master_rows.append(r)
            seen_essay_hashes.add(essay_hash)
        else:
            print(f"  DEDUP: Skipping {r['alma_id']} - duplicate essay")
    analyses.append(analysis)

    # === CREATE MASTER DATAFRAME ===
    if master_rows:
        # Define column order
        meta_cols = ['alma_id', 'course', 'semester', 'year', 'prompt', 'essay', 'essay_length', 'source_file', 'source_sheet']
        binary_cols = [f'{t}_binary' for t in ALL_THEMES]
        excerpt_cols = [f'{t}_excerpts' for t in ALL_THEMES]
        all_cols = meta_cols + binary_cols + excerpt_cols

        master_df = pd.DataFrame(master_rows)
        # Ensure all columns exist
        for col in all_cols:
            if col not in master_df.columns:
                master_df[col] = 0 if '_binary' in col else ''

        master_df = master_df[all_cols]

        # Sort by alma_id
        master_df = master_df.sort_values('alma_id').reset_index(drop=True)

        # Save
        output_path = os.path.join(ANNOTATED_WHY, "gold_standard_fall2019.csv")
        master_df.to_csv(output_path, index=False)

        print(f"\n{'='*70}")
        print("GOLD STANDARD PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total rows: {len(master_df)}")
        print(f"Unique Alma IDs: {master_df['alma_id'].nunique()}")
        print(f"Output: {output_path}")

        # Summary by course
        print(f"\nBy course:")
        for course, group in master_df.groupby('course'):
            print(f"  {course}: {len(group)} rows")

        print(f"\nBy semester/year:")
        for (sem, yr), group in master_df.groupby(['semester', 'year']):
            print(f"  {sem} {yr}: {len(group)} rows")

        # Theme distribution across entire dataset
        print(f"\nOverall theme distribution:")
        for theme in ALL_THEMES:
            n_pos = master_df[f'{theme}_binary'].sum()
            pct = n_pos / len(master_df) * 100
            print(f"  {theme:25s}: {n_pos:3d} / {len(master_df)} ({pct:5.1f}%)")

        # Check for any essays that seem too short or problematic
        short_essays = master_df[master_df['essay_length'] < 50]
        if len(short_essays) > 0:
            print(f"\nWARNING: {len(short_essays)} essays shorter than 50 chars:")
            for _, row in short_essays.iterrows():
                print(f"  {row['alma_id']}: '{row['essay'][:80]}'")

    # Save analysis report
    report_path = os.path.join(OUTPUT, "plans", "gold_standard_analysis.json")
    with open(report_path, 'w') as f:
        json.dump(analyses, f, indent=2)
    print(f"\nAnalysis report: {report_path}")


if __name__ == '__main__':
    main()
