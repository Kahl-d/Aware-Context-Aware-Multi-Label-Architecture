"""
Create sentence-level dataset from essay-level annotated data.

Output columns:
  essay_id, sentence_id, sentence, sentence_length,
  alma_id, course, semester, year, prompt, source_file, coder,
  Attainment, First_Gen, Aspirational, Navigational, Resistance,
  Perseverance, Filial_Piety, Familial, Community_Consciousness,
  Social, Spiritual

Label logic:
  - If essay theme_binary == 1: match sentence against excerpts → 1 if match, 0 if no match
  - If essay theme_binary == 0: all sentences → 0
  - If essay theme_binary == -1: all sentences → 0 (not coded = treated as negative)

Excerpt handling (/%/ delimiter):
  - Excerpts are text passages highlighted by annotators as evidence for a theme
  - Multiple excerpts per theme separated by ' /%/ '
  - Each excerpt is matched independently against each sentence
  - A sentence can match excerpts from MULTIPLE themes simultaneously (multi-label)
  - Matching uses: substring containment + word overlap (60% threshold)
  - Encoding normalization handles unicode artifacts (â€™ → ', etc.)
"""
import pandas as pd
import numpy as np
import re
import os
import unicodedata

# ============================================================
# Configuration
# ============================================================
INPUT = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA_Master_Dataset/annotated/ALMA_all_annotated_combined.csv"
OUTPUT_DIR = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/ALMA_Master_Dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ALMA_sentence_level_dataset.csv")

THEMES = [
    'Attainment', 'First_Gen', 'Aspirational', 'Navigational', 'Resistance',
    'Perseverance', 'Filial_Piety', 'Familial', 'Community_Consciousness',
    'Social', 'Spiritual'
]

META_COLS = ['alma_id', 'course', 'semester', 'year', 'prompt', 'source_file', 'coder']


# ============================================================
# Text normalization for matching
# ============================================================
def normalize_text(text):
    """Normalize text for fuzzy matching: lowercase, strip all non-alnum, collapse whitespace.

    Critical: must handle unicode vs ASCII apostrophes consistently.
    e.g., i've (unicode \u2019) and I've (ASCII ') must both → 'ive'
    """
    if not isinstance(text, str):
        return ''
    # Normalize unicode (handle encoding artifacts like â€™, Ã©, etc.)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    # REMOVE (not replace) non-alnum except spaces — prevents i've → 'i ve' vs 'ive' mismatch
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# Sentence splitting
# ============================================================
def split_into_sentences(text):
    """Split essay text into sentences using regex.

    Handles:
    - Standard sentence endings (. ? !)
    - Avoids splitting on abbreviations (Mr., Dr., etc.)
    - Keeps sentences with minimum length
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return []

    # Clean up the text first
    text = text.strip()

    # Split on sentence-ending punctuation followed by space+uppercase or end
    # This regex looks for: .!? followed by space(s) and optional quote, then uppercase letter or end
    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z"\'])|(?<=[.!?])\s*$',
        text
    )

    # If regex didn't split well (single long block), try simpler split
    if len(sentences) <= 1 and len(text) > 200:
        sentences = re.split(r'(?<=[.!?])\s+', text)

    # Clean and filter
    result = []
    for s in sentences:
        s = s.strip()
        if len(s) >= 10:  # Minimum sentence length
            result.append(s)

    # If still no sentences (text has no punctuation), treat whole text as one sentence
    if not result and len(text.strip()) >= 10:
        result = [text.strip()]

    return result


# ============================================================
# Excerpt parsing and matching
# ============================================================
def parse_excerpts(excerpt_text):
    """Parse excerpt column: split by /%/ delimiter, return list of normalized excerpts."""
    if not isinstance(excerpt_text, str):
        return []
    if excerpt_text.strip() in ('', 'NOT_CODED', 'nan', '0'):
        return []

    parts = excerpt_text.split('/%/')
    excerpts = []
    for p in parts:
        p = p.strip()
        if len(p) >= 10 and p not in ('NOT_CODED', '0', 'nan'):
            excerpts.append(p)
    return excerpts


def sentence_matches_excerpts(sentence_norm, excerpts_norm, threshold=0.5):
    """Check if a normalized sentence matches any excerpt.

    A sentence matches if:
    1. The sentence is a substring of an excerpt (sentence within excerpt), OR
    2. An excerpt is a substring of the sentence (excerpt within sentence), OR
    3. >=50% of sentence words found in an excerpt (handles multi-sentence excerpts)
    4. >=50% of excerpt words found in the sentence (handles short excerpts)

    Each theme's excerpts (split by /%/) are checked independently.
    Multiple themes can match the same sentence (multi-label).
    Returns True/False.
    """
    if not excerpts_norm:
        return False

    sent_words = set(sentence_norm.split())
    if not sent_words:
        return False

    for exc_norm in excerpts_norm:
        # Substring match (either direction)
        if sentence_norm in exc_norm or exc_norm in sentence_norm:
            return True

        exc_words = set(exc_norm.split())
        if not exc_words:
            continue

        overlap = sent_words & exc_words

        # Check coverage from both directions
        sent_coverage = len(overlap) / len(sent_words)  # how much of sentence is in excerpt
        exc_coverage = len(overlap) / len(exc_words)     # how much of excerpt is in sentence

        # Match if either direction has sufficient overlap
        if sent_coverage >= threshold or exc_coverage >= threshold:
            return True

    return False


# ============================================================
# Main processing
# ============================================================
def main():
    print("Loading combined annotated data...")
    df = pd.read_csv(INPUT)
    print(f"  {len(df)} essays loaded")

    all_rows = []
    essay_count = 0
    total_sentences = 0
    skipped = 0

    for idx, row in df.iterrows():
        essay_text = str(row.get('essay', ''))
        if len(essay_text.strip()) < 10:
            skipped += 1
            continue

        # Split into sentences
        sentences = split_into_sentences(essay_text)
        if not sentences:
            skipped += 1
            continue

        essay_count += 1

        # Pre-compute: for each theme, get binary label + normalized excerpts
        theme_data = {}
        for theme in THEMES:
            binary = row.get(f'{theme}_binary', -1)
            try:
                binary = int(float(binary))
            except (ValueError, TypeError):
                binary = -1

            excerpts_raw = parse_excerpts(row.get(f'{theme}_excerpts'))
            excerpts_norm = [normalize_text(e) for e in excerpts_raw if normalize_text(e)]

            theme_data[theme] = {
                'binary': binary,
                'excerpts_norm': excerpts_norm,
            }

        # Process each sentence
        for sent_idx, sentence in enumerate(sentences):
            sent_norm = normalize_text(sentence)
            if not sent_norm:
                continue

            record = {
                'essay_id': essay_count,
                'sentence_id': sent_idx + 1,
                'sentence': sentence,
                'sentence_length': len(sentence),
            }

            # Add metadata
            for col in META_COLS:
                record[col] = row.get(col, '')

            # Label each theme independently (multi-label: each theme checked separately)
            for theme in THEMES:
                td = theme_data[theme]
                if td['binary'] == -1:
                    record[theme] = 0   # Not coded → treat as 0
                elif td['binary'] == 0:
                    record[theme] = 0   # Essay negative → all sentences negative
                elif td['binary'] == 1:
                    # Essay positive → check if this sentence matches excerpts
                    if td['excerpts_norm']:
                        match = sentence_matches_excerpts(sent_norm, td['excerpts_norm'])
                        record[theme] = 1 if match else 0
                    else:
                        # Positive but no excerpts → mark all sentences as 1
                        # (annotator marked positive without highlighting specific text)
                        record[theme] = 1
                else:
                    record[theme] = 0

            all_rows.append(record)
            total_sentences += 1

        if essay_count % 500 == 0:
            print(f"  Processed {essay_count} essays → {total_sentences} sentences so far...")

    print(f"\nProcessing complete:")
    print(f"  Essays processed: {essay_count}")
    print(f"  Essays skipped (too short): {skipped}")
    print(f"  Total sentences: {total_sentences}")

    # Create DataFrame
    columns = ['essay_id', 'sentence_id', 'sentence', 'sentence_length'] + META_COLS + THEMES
    result_df = pd.DataFrame(all_rows, columns=columns)

    # Save
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Shape: {result_df.shape}")

    # ============================================================
    # Summary statistics
    # ============================================================
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")

    print(f"\nTotal rows (sentences): {len(result_df)}")
    print(f"Total unique essays: {result_df['essay_id'].nunique()}")
    print(f"Sentences per essay: mean={result_df.groupby('essay_id').size().mean():.1f}, "
          f"median={result_df.groupby('essay_id').size().median():.0f}, "
          f"max={result_df.groupby('essay_id').size().max()}")

    print(f"\nSentence length: mean={result_df['sentence_length'].mean():.0f} chars, "
          f"median={result_df['sentence_length'].median():.0f}, "
          f"min={result_df['sentence_length'].min()}, max={result_df['sentence_length'].max()}")

    print(f"\n--- Theme Distribution (sentence-level, binary 0/1) ---")
    print(f"{'Theme':<30} {'Pos (1)':>8} {'Neg (0)':>8} {'Total':>8} {'%Pos':>8}")
    print(f"{'-'*62}")
    for theme in THEMES:
        pos = (result_df[theme] == 1).sum()
        neg = (result_df[theme] == 0).sum()
        total = pos + neg
        pct = (pos / total * 100) if total > 0 else 0
        print(f"{theme:<30} {pos:>8} {neg:>8} {total:>8} {pct:>7.1f}%")

    # Multi-label analysis
    print(f"\n--- Multi-label Distribution ---")
    pos_counts = (result_df[THEMES] == 1).sum(axis=1)
    for n in range(12):
        count = (pos_counts == n).sum()
        if count > 0:
            print(f"  {n} themes positive: {count} sentences ({count/len(result_df)*100:.1f}%)")

    # Distribution by prompt
    print(f"\n--- Sentences by Prompt ---")
    for prompt, group in result_df.groupby('prompt'):
        print(f"  {prompt}: {len(group)} sentences from {group['essay_id'].nunique()} essays")

    # Distribution by semester
    print(f"\n--- Sentences by Semester/Year ---")
    for (sem, yr), group in result_df.groupby(['semester', 'year']):
        print(f"  {sem} {yr}: {len(group)} sentences")


if __name__ == '__main__':
    main()
