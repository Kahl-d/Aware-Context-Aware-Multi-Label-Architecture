"""
build_dataset.py — Two-step data pipeline for AWARE v2.

Step 1: master_data.pkl → model_data.pkl (essay selection + formatting)
Step 2: model_data.pkl → train/val/test splits (student-level stratification)

Replaces prepare_data.py with a clean pipeline that starts from master_data.pkl
and gives full control over class_0 sampling strategy.

Usage:
    # Step 1: Build model_data.pkl from master_data.pkl
    python build_dataset.py build_model_data \
        --master_path ../../Data_for_modeling/master_data.pkl \
        --output_path ../data/model_data.pkl \
        --n_pure_c0 1000

    # Step 2: Build train/val/test splits
    python build_dataset.py build_splits \
        --model_data_path ../data/model_data.pkl \
        --master_path ../../Data_for_modeling/master_data.pkl \
        --output_dir ../data/

    # Both steps at once
    python build_dataset.py build_all \
        --master_path ../../Data_for_modeling/master_data.pkl \
        --output_dir ../data/ \
        --n_pure_c0 1000
"""

import argparse
import json
import logging
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Canonical theme order (same as config.py)
THEMES = [
    "Navigational", "Attainment", "Perseverance", "Aspirational",
    "Social", "Filial Piety", "Spiritual", "Familial",
    "Resistance", "Community Consciousness", "First Gen",
]
THEME_TO_IDX = {t: i for i, t in enumerate(THEMES)}
NUM_THEMES = len(THEMES)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_master_data(path: Path) -> dict:
    """Load master_data.pkl (essay_id → EssayEntry)."""
    # Add processing_workspace to path for EssayEntry class
    alma_pipeline_dir = path.parent.parent / "Data" / "processing_workspace"
    if alma_pipeline_dir.exists():
        sys.path.insert(0, str(alma_pipeline_dir))
    alt_dir = path.parent / "processing_workspace"
    if alt_dir.exists():
        sys.path.insert(0, str(alt_dir))
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded master_data: %d essays from %s", len(data), path)
    return data


# ---------------------------------------------------------------------------
# Step 1: build_model_data
# ---------------------------------------------------------------------------

def classify_essays(master_data: dict) -> tuple:
    """Classify essays as themed (≥1 non-class_0 annotation) or pure_c0.

    Returns (themed_ids, pure_c0_ids) as sorted lists.
    """
    themed = []
    pure_c0 = []
    for eid, entry in master_data.items():
        has_theme = False
        for themes in entry.annotations.values():
            if themes - {"class_0"}:
                has_theme = True
                break
        if has_theme:
            themed.append(eid)
        else:
            pure_c0.append(eid)
    return sorted(themed), sorted(pure_c0)


def convert_entry_to_essay_format(entry) -> dict:
    """Convert an EssayEntry to the essay-level format used by AWAREDataset.

    Returns:
        {"sentences": [str, ...], "annotations": [set, ...]}
    """
    sentences = list(entry.sentences)
    annotations = []

    # Build lookup: lowered sentence text → set of themes
    ann_lookup = {}
    for sent_text, themes in entry.annotations.items():
        clean = themes - {"class_0"}
        ann_lookup[sent_text.lower().strip()] = clean

    for sent in sentences:
        key = sent.lower().strip()
        annotations.append(ann_lookup.get(key, set()))

    return {"sentences": sentences, "annotations": annotations}


def build_model_data(
    master_path: str,
    output_path: str,
    n_pure_c0: int = 1000,
    seed: int = 42,
):
    """Step 1: master_data.pkl → model_data.pkl.

    Includes ALL themed essays + n_pure_c0 randomly sampled pure class_0 essays.
    """
    master_data = load_master_data(Path(master_path))

    themed_ids, pure_c0_ids = classify_essays(master_data)
    logger.info("Themed essays: %d, Pure c0 essays: %d", len(themed_ids), len(pure_c0_ids))

    # Sample pure c0 essays
    rng = np.random.RandomState(seed)
    n_sample = min(n_pure_c0, len(pure_c0_ids))
    sampled_c0_ids = sorted(rng.choice(pure_c0_ids, size=n_sample, replace=False).tolist())
    logger.info("Sampled %d pure c0 essays (from %d available)", n_sample, len(pure_c0_ids))

    # Combine
    selected_ids = sorted(set(themed_ids + sampled_c0_ids))

    # Convert to essay-level format
    essays = {}
    for eid in selected_ids:
        essays[eid] = convert_entry_to_essay_format(master_data[eid])

    # Compute stats
    stats = _compute_model_data_stats(essays, themed_ids, sampled_c0_ids)

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "essays": essays,
        "essay_ids": selected_ids,
        "metadata": {
            "n_themed_essays": len(themed_ids),
            "n_pure_c0_essays": n_sample,
            "n_pure_c0_available": len(pure_c0_ids),
            "seed": seed,
        },
    }
    with open(output, "wb") as f:
        pickle.dump(model_data, f)
    logger.info("Saved model_data: %d essays → %s", len(selected_ids), output)

    # Save stats alongside
    stats_path = output.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved stats → %s", stats_path)

    _print_model_data_summary(stats)
    return model_data


def _compute_model_data_stats(essays: dict, themed_ids: list, sampled_c0_ids: list) -> dict:
    """Compute comprehensive statistics for model_data."""
    total_sents = 0
    themed_sents = 0
    c0_sents = 0
    theme_counts = Counter()
    multi_label_counts = Counter()

    for eid, essay in essays.items():
        for ann in essay["annotations"]:
            total_sents += 1
            if ann:
                themed_sents += 1
                multi_label_counts[len(ann)] += 1
                for theme in ann:
                    theme_counts[theme] += 1
            else:
                c0_sents += 1

    c0_ratio = c0_sents / total_sents if total_sents > 0 else 0

    return {
        "total_essays": len(essays),
        "themed_essays": len(themed_ids),
        "pure_c0_essays": len(sampled_c0_ids),
        "total_sentences": total_sents,
        "themed_sentences": themed_sents,
        "class_0_sentences": c0_sents,
        "class_0_ratio": round(c0_ratio, 4),
        "theme_counts": {t: theme_counts.get(t, 0) for t in THEMES},
        "multi_label_distribution": dict(sorted(multi_label_counts.items())),
    }


def _print_model_data_summary(stats: dict):
    """Print model_data summary to stdout."""
    print("\n" + "=" * 70)
    print("MODEL DATA SUMMARY")
    print("=" * 70)
    print(f"Total essays:     {stats['total_essays']:>8}")
    print(f"  Themed:         {stats['themed_essays']:>8}")
    print(f"  Pure class_0:   {stats['pure_c0_essays']:>8}")
    print(f"Total sentences:  {stats['total_sentences']:>8}")
    print(f"  Themed:         {stats['themed_sentences']:>8}")
    print(f"  Class_0:        {stats['class_0_sentences']:>8} ({stats['class_0_ratio']:.1%})")
    print(f"\nPer-theme counts:")
    for theme in THEMES:
        count = stats['theme_counts'].get(theme, 0)
        print(f"  {theme:<25} {count:>6}")
    print(f"\nMulti-label distribution:")
    for n_labels, count in sorted(stats['multi_label_distribution'].items()):
        print(f"  {n_labels} label(s): {count:>6}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 2: build_splits
# ---------------------------------------------------------------------------

def get_essay_theme_vector(essay: dict) -> np.ndarray:
    """Binary vector [11] indicating which themes appear in this essay."""
    vec = np.zeros(NUM_THEMES, dtype=np.float32)
    for ann in essay["annotations"]:
        for theme in ann:
            if theme in THEME_TO_IDX:
                vec[THEME_TO_IDX[theme]] = 1.0
    return vec


def stratified_student_split(
    essays: dict,
    essay_to_alma: dict,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple:
    """Split essays by student (alma_id), stratified by theme presence.

    Uses iterative stratification: rarest themes assigned first to guarantee
    proportional representation across all splits.

    Returns (train_ids, val_ids, test_ids).
    """
    rng = np.random.RandomState(seed)

    # Group essays by student
    student_essays = defaultdict(list)
    for eid, alma_id in essay_to_alma.items():
        student_essays[alma_id].append(eid)

    students = sorted(student_essays.keys())
    logger.info("Splitting %d students into train/val/test", len(students))

    # Per-student theme vector (union across all their essays)
    student_vectors = {}
    for sid in students:
        vec = np.zeros(NUM_THEMES, dtype=np.float32)
        for eid in student_essays[sid]:
            vec = np.maximum(vec, get_essay_theme_vector(essays[eid]))
        student_vectors[sid] = vec

    # Theme rarity order
    theme_counts = np.zeros(NUM_THEMES)
    for sid in students:
        theme_counts += student_vectors[sid]
    theme_order = np.argsort(theme_counts)  # rarest first

    train_students, val_students, test_students = _manual_stratified_split(
        students, student_vectors, theme_order,
        train_ratio, val_ratio, test_ratio, rng,
    )

    # Convert student splits → essay splits
    train_ids = [eid for s in train_students for eid in student_essays[s]]
    val_ids = [eid for s in val_students for eid in student_essays[s]]
    test_ids = [eid for s in test_students for eid in student_essays[s]]

    logger.info(
        "Split: train=%d essays (%d students), val=%d (%d), test=%d (%d)",
        len(train_ids), len(train_students),
        len(val_ids), len(val_students),
        len(test_ids), len(test_students),
    )
    return train_ids, val_ids, test_ids


def _manual_stratified_split(
    students, student_vectors, theme_order,
    train_ratio, val_ratio, test_ratio, rng,
):
    """Iterative stratification: process themes rarest first, assign proportionally."""
    train_set = set()
    val_set = set()
    test_set = set()
    assigned = set()

    for theme_idx in theme_order:
        theme_students = [s for s in students if student_vectors[s][theme_idx] > 0 and s not in assigned]
        if not theme_students:
            continue
        rng.shuffle(theme_students)
        n = len(theme_students)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_batch = theme_students[:n_train]
        val_batch = theme_students[n_train:n_train + n_val]
        test_batch = theme_students[n_train + n_val:]
        if not test_batch and len(train_batch) > 2:
            test_batch = [train_batch.pop()]
        if not val_batch and len(train_batch) > 2:
            val_batch = [train_batch.pop()]

        train_set.update(train_batch)
        val_set.update(val_batch)
        test_set.update(test_batch)
        assigned.update(train_batch + val_batch + test_batch)

    # Assign remaining students
    remaining = [s for s in students if s not in assigned]
    rng.shuffle(remaining)
    n = len(remaining)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_set.update(remaining[:n_train])
    val_set.update(remaining[n_train:n_train + n_val])
    test_set.update(remaining[n_train + n_val:])

    # Resolve overlaps (keep first assignment)
    for s in train_set & val_set:
        val_set.discard(s)
    for s in train_set & test_set:
        test_set.discard(s)
    for s in val_set & test_set:
        test_set.discard(s)

    return list(train_set), list(val_set), list(test_set)


def verify_splits(essays, train_ids, val_ids, test_ids, essay_to_alma) -> bool:
    """Verify split quality: no leakage, all themes present, no overlap."""
    issues = []

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    if train_set & val_set:
        issues.append(f"Train/val overlap: {len(train_set & val_set)} essays")
    if train_set & test_set:
        issues.append(f"Train/test overlap: {len(train_set & test_set)} essays")
    if val_set & test_set:
        issues.append(f"Val/test overlap: {len(val_set & test_set)} essays")

    train_students = {essay_to_alma[e] for e in train_ids}
    val_students = {essay_to_alma[e] for e in val_ids}
    test_students = {essay_to_alma[e] for e in test_ids}
    if train_students & val_students:
        issues.append(f"Student leakage train/val: {len(train_students & val_students)}")
    if train_students & test_students:
        issues.append(f"Student leakage train/test: {len(train_students & test_students)}")
    if val_students & test_students:
        issues.append(f"Student leakage val/test: {len(val_students & test_students)}")

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        themes_present = set()
        for eid in ids:
            for ann in essays[eid]["annotations"]:
                themes_present.update(ann)
        missing = set(THEMES) - themes_present
        if missing:
            issues.append(f"{name} missing themes: {missing}")

    total = len(train_ids) + len(val_ids) + len(test_ids)
    if total != len(essays):
        issues.append(f"Essay count mismatch: {total} vs {len(essays)}")

    if issues:
        for issue in issues:
            logger.error("SPLIT ISSUE: %s", issue)
        return False
    logger.info("All split verification checks PASSED")
    return True


def compute_theme_counts(essays: dict, essay_ids: list) -> dict:
    """Count sentences per theme in a split."""
    counts = Counter()
    for eid in essay_ids:
        for ann in essays[eid]["annotations"]:
            for theme in ann:
                counts[theme] += 1
        counts["class_0"] += sum(1 for a in essays[eid]["annotations"] if not a)
    return dict(counts)


def compute_theme_weights(essays: dict, essay_ids: list) -> list:
    """Inverse sqrt frequency weights. Returns list of 11 floats."""
    counts = np.zeros(NUM_THEMES)
    for eid in essay_ids:
        for ann in essays[eid]["annotations"]:
            for theme in ann:
                if theme in THEME_TO_IDX:
                    counts[THEME_TO_IDX[theme]] += 1

    max_count = counts.max()
    weights = []
    for i in range(NUM_THEMES):
        w = math.sqrt(max_count / counts[i]) if counts[i] > 0 else 1.0
        weights.append(round(w, 4))
    return weights


def compute_essay_weights(essays: dict, essay_ids: list) -> dict:
    """Per-essay sampling weight. Rare-theme essays get higher weight."""
    theme_essay_counts = Counter()
    themed_ids = []
    for eid in essay_ids:
        essay_themes = set()
        for ann in essays[eid]["annotations"]:
            essay_themes.update(ann)
        if essay_themes:
            themed_ids.append(eid)
            for t in essay_themes:
                theme_essay_counts[t] += 1

    total_themed = len(themed_ids)
    weights = {}
    for eid in essay_ids:
        essay_themes = set()
        for ann in essays[eid]["annotations"]:
            essay_themes.update(ann)
        if not essay_themes:
            weights[eid] = 0.5  # pure class_0: down-weighted
        else:
            w = max(
                math.sqrt(total_themed / max(theme_essay_counts.get(t, 1), 1))
                for t in essay_themes
            )
            weights[eid] = round(w, 4)
    return weights


def create_toy_dataset(essays: dict, essay_ids: list, n: int = 100, seed: int = 42) -> list:
    """Create toy dataset of ~n essays, ensuring all 11 themes are represented."""
    rng = np.random.RandomState(seed)

    theme_essays = defaultdict(list)
    for eid in essay_ids:
        for ann in essays[eid]["annotations"]:
            for theme in ann:
                if theme in THEME_TO_IDX:
                    theme_essays[theme].append(eid)

    selected = set()
    for theme in sorted(THEMES, key=lambda t: len(theme_essays.get(t, []))):
        candidates = [e for e in theme_essays.get(theme, []) if e not in selected]
        if candidates:
            picks = rng.choice(candidates, size=min(3, len(candidates)), replace=False)
            selected.update(picks)

    remaining = [e for e in essay_ids if e not in selected]
    rng.shuffle(remaining)
    needed = max(0, n - len(selected))
    selected.update(remaining[:needed])

    return sorted(selected)


def build_split_data(essays: dict, essay_ids: list, essay_weights: dict) -> dict:
    """Build the output format for a split (compatible with AWAREDataset)."""
    return {
        "essays": {eid: essays[eid] for eid in essay_ids},
        "essay_ids": list(essay_ids),
        "weights": {eid: essay_weights.get(eid, 1.0) for eid in essay_ids},
    }


def build_dapt_corpus(master_data: dict) -> str:
    """Build DAPT corpus: all essay text from master_data."""
    texts = []
    for eid, entry in master_data.items():
        if hasattr(entry, "essay_text") and entry.essay_text:
            texts.append(entry.essay_text.strip())
        elif hasattr(entry, "sentences") and entry.sentences:
            texts.append(" ".join(entry.sentences))
    logger.info("DAPT corpus: %d essays", len(texts))
    return "\n\n".join(texts)


def build_splits(
    model_data_path: str,
    master_path: str,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    toy_size: int = 100,
    skip_dapt: bool = False,
):
    """Step 2: model_data.pkl → train/val/test_data.pkl."""
    # Load model_data
    with open(model_data_path, "rb") as f:
        model_data = pickle.load(f)
    essays = model_data["essays"]
    essay_ids = model_data["essay_ids"]
    logger.info("Loaded model_data: %d essays", len(essay_ids))

    # Load master_data for alma_id mapping
    master_data = load_master_data(Path(master_path))

    # Build essay → alma_id mapping
    essay_to_alma = {}
    missing = 0
    for eid in essay_ids:
        if eid in master_data:
            essay_to_alma[eid] = master_data[eid].metadata.alma_id
        else:
            essay_to_alma[eid] = f"synthetic_{eid}"
            missing += 1
    if missing:
        logger.warning("%d essays not in master_data (synthetic alma_ids)", missing)
    n_students = len(set(essay_to_alma.values()))
    logger.info("Mapped %d essays to %d students", len(essay_to_alma), n_students)

    # Stratified split
    train_ids, val_ids, test_ids = stratified_student_split(
        essays, essay_to_alma,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        seed=seed,
    )

    # Verify
    ok = verify_splits(essays, train_ids, val_ids, test_ids, essay_to_alma)
    if not ok:
        logger.error("Split verification FAILED")
        sys.exit(1)

    # Compute weights
    train_weights = compute_essay_weights(essays, train_ids)
    all_weights = {**train_weights}
    for eid in val_ids + test_ids:
        all_weights[eid] = 1.0

    theme_weights = compute_theme_weights(essays, train_ids)
    logger.info("Theme weights (train): %s", dict(zip(THEMES, theme_weights)))

    # Save splits
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_data = build_split_data(essays, ids, all_weights)
        path = out / f"{name}_data.pkl"
        with open(path, "wb") as f:
            pickle.dump(split_data, f)
        logger.info("Saved %s: %d essays → %s", name, len(ids), path)

    # Toy dataset
    toy_ids = create_toy_dataset(essays, train_ids, n=toy_size, seed=seed)
    toy_val_ids = create_toy_dataset(essays, val_ids, n=min(30, len(val_ids)), seed=seed + 1)
    toy_data = {
        "train": build_split_data(essays, toy_ids, all_weights),
        "val": build_split_data(essays, toy_val_ids, all_weights),
    }
    with open(out / "toy_data.pkl", "wb") as f:
        pickle.dump(toy_data, f)
    logger.info("Saved toy dataset: train=%d, val=%d", len(toy_ids), len(toy_val_ids))

    # DAPT corpus
    if not skip_dapt:
        corpus = build_dapt_corpus(master_data)
        dapt_path = out / "dapt_corpus.txt"
        with open(dapt_path, "w") as f:
            f.write(corpus)
        logger.info("Saved DAPT corpus: %.1f MB → %s", len(corpus) / 1e6, dapt_path)

    # Stats
    stats = {
        "seed": seed,
        "total_essays": len(essays),
        "total_sentences": sum(len(e["sentences"]) for e in essays.values()),
        "total_students": n_students,
        "theme_order": THEMES,
        "theme_weights": dict(zip(THEMES, theme_weights)),
        "splits": {},
    }
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        n_sents = sum(len(essays[e]["sentences"]) for e in ids)
        n_studs = len(set(essay_to_alma[e] for e in ids))
        tc = compute_theme_counts(essays, ids)
        stats["splits"][name] = {
            "essays": len(ids),
            "sentences": n_sents,
            "students": n_studs,
            "theme_counts": {t: tc.get(t, 0) for t in THEMES},
            "class_0_sentences": tc.get("class_0", 0),
        }
    stats["leakage_check"] = {
        "train_val": len(set(essay_to_alma[e] for e in train_ids) & set(essay_to_alma[e] for e in val_ids)),
        "train_test": len(set(essay_to_alma[e] for e in train_ids) & set(essay_to_alma[e] for e in test_ids)),
        "val_test": len(set(essay_to_alma[e] for e in val_ids) & set(essay_to_alma[e] for e in test_ids)),
    }

    stats_path = out / "splits_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved split stats → %s", stats_path)

    # Generate dataset report
    _generate_dataset_report(stats, model_data.get("metadata", {}), out)

    # Print summary
    _print_splits_summary(stats, theme_weights)


def _generate_dataset_report(stats: dict, metadata: dict, output_dir: Path):
    """Generate DATASET_REPORT.md with all decisions and statistics."""
    lines = [
        "# AWARE v2 Dataset Report",
        "",
        "## Pipeline",
        f"- Source: master_data.pkl ({metadata.get('n_themed_essays', '?')} themed + "
        f"{metadata.get('n_pure_c0_essays', '?')} sampled pure c0 from "
        f"{metadata.get('n_pure_c0_available', '?')} available)",
        f"- Seed: {stats['seed']}",
        f"- Total essays: {stats['total_essays']}",
        f"- Total sentences: {stats['total_sentences']}",
        f"- Total students: {stats['total_students']}",
        "",
        "## Design Decisions",
        "",
        "### Why include all themed essays?",
        "Every themed essay contains annotated sentences that provide training signal.",
        "Removing any themed essay reduces the already-scarce positive examples for rare themes.",
        "",
        "### Why reduce pure class_0 essays?",
        f"The full corpus has {metadata.get('n_pure_c0_available', 7014)} pure class_0 essays.",
        "Including all of them inflates the class_0 ratio to ~69%, causing:",
        "- Gradient dominance: each c0 sentence generates 11 negative signals",
        "- For rare themes like First Gen (137 sentences), the model sees 625x more negatives",
        f"By sampling {metadata.get('n_pure_c0_essays', '?')} pure c0 essays, we reduce the",
        "class_0 ratio while maintaining enough true-negative signal for real-world deployment.",
        "",
        "### Why keep some pure class_0 essays?",
        "The model must handle real-world input where many essays have no themes.",
        "Pure c0 essays provide true-negative examples. Without them, the model",
        "would only see class_0 sentences in the context of themed essays,",
        "potentially biasing it toward over-prediction.",
        "",
        "## Split Statistics",
        "",
    ]

    for name in ["train", "val", "test"]:
        s = stats["splits"][name]
        total_sents = s["sentences"]
        c0 = s["class_0_sentences"]
        c0_pct = c0 / total_sents * 100 if total_sents > 0 else 0
        lines.append(f"### {name.upper()}")
        lines.append(f"- Essays: {s['essays']}")
        lines.append(f"- Sentences: {total_sents}")
        lines.append(f"- Students: {s['students']}")
        lines.append(f"- Class_0: {c0} ({c0_pct:.1f}%)")
        lines.append("")
        lines.append(f"| Theme | Count |")
        lines.append(f"|-------|-------|")
        for theme in THEMES:
            lines.append(f"| {theme} | {s['theme_counts'].get(theme, 0)} |")
        lines.append("")

    lines.append("## Leakage Check")
    lc = stats["leakage_check"]
    lines.append(f"- Train/Val student overlap: {lc['train_val']}")
    lines.append(f"- Train/Test student overlap: {lc['train_test']}")
    lines.append(f"- Val/Test student overlap: {lc['val_test']}")
    lines.append("")

    lines.append("## Theme Weights (inverse sqrt frequency, from training set)")
    lines.append("")
    lines.append("| Theme | Weight |")
    lines.append("|-------|--------|")
    for theme in THEMES:
        w = stats["theme_weights"].get(theme, 1.0)
        lines.append(f"| {theme} | {w} |")

    report_path = output_dir / "DATASET_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved dataset report → %s", report_path)


def _print_splits_summary(stats: dict, theme_weights: list):
    """Print splits summary."""
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    for name in ["train", "val", "test"]:
        s = stats["splits"][name]
        print(f"\n{name.upper():>5}: {s['essays']:>6} essays, {s['sentences']:>6} sents, {s['students']:>5} students")
        for theme in THEMES:
            print(f"        {theme:<25} {s['theme_counts'].get(theme, 0):>6}")
        print(f"        {'class_0':<25} {s['class_0_sentences']:>6}")
    print(f"\nTheme weights: {dict(zip(THEMES, theme_weights))}")
    print(f"Student leakage: {stats['leakage_check']}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AWARE v2 Data Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step")

    # Step 1
    p1 = subparsers.add_parser("build_model_data", help="Step 1: master_data → model_data")
    p1.add_argument("--master_path", type=str, required=True)
    p1.add_argument("--output_path", type=str, required=True)
    p1.add_argument("--n_pure_c0", type=int, default=1000)
    p1.add_argument("--seed", type=int, default=42)

    # Step 2
    p2 = subparsers.add_parser("build_splits", help="Step 2: model_data → train/val/test")
    p2.add_argument("--model_data_path", type=str, required=True)
    p2.add_argument("--master_path", type=str, required=True)
    p2.add_argument("--output_dir", type=str, required=True)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--train_ratio", type=float, default=0.80)
    p2.add_argument("--val_ratio", type=float, default=0.10)
    p2.add_argument("--test_ratio", type=float, default=0.10)
    p2.add_argument("--toy_size", type=int, default=100)
    p2.add_argument("--skip_dapt", action="store_true")

    # Both steps
    pa = subparsers.add_parser("build_all", help="Run both steps")
    pa.add_argument("--master_path", type=str, required=True)
    pa.add_argument("--output_dir", type=str, required=True)
    pa.add_argument("--n_pure_c0", type=int, default=1000)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--train_ratio", type=float, default=0.80)
    pa.add_argument("--val_ratio", type=float, default=0.10)
    pa.add_argument("--test_ratio", type=float, default=0.10)
    pa.add_argument("--toy_size", type=int, default=100)
    pa.add_argument("--skip_dapt", action="store_true")

    args = parser.parse_args()

    if args.command == "build_model_data":
        build_model_data(args.master_path, args.output_path, args.n_pure_c0, args.seed)

    elif args.command == "build_splits":
        build_splits(
            args.model_data_path, args.master_path, args.output_dir,
            seed=args.seed, train_ratio=args.train_ratio,
            val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            toy_size=args.toy_size, skip_dapt=args.skip_dapt,
        )

    elif args.command == "build_all":
        out_dir = Path(args.output_dir)
        model_data_path = out_dir / "model_data.pkl"
        build_model_data(args.master_path, str(model_data_path), args.n_pure_c0, args.seed)
        build_splits(
            str(model_data_path), args.master_path, args.output_dir,
            seed=args.seed, train_ratio=args.train_ratio,
            val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            toy_size=args.toy_size, skip_dapt=args.skip_dapt,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
