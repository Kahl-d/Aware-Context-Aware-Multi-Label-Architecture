"""
prepare_data.py — Convert model_data.pkl → essay-level format + stratified splits.

Splits by alma_id (student) to prevent leakage. Ensures all 11 themes
appear in every split. Computes per-essay sampling weights.

Usage:
    python scripts/prepare_data.py --data_dir ../Data_for_modeling --output_dir data/
    python scripts/prepare_data.py --data_dir ../Data_for_modeling --output_dir data/ --toy_only
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

# Canonical theme order (same as old pipeline)
THEMES = [
    "Navigational", "Attainment", "Perseverance", "Aspirational",
    "Social", "Filial Piety", "Spiritual", "Familial",
    "Resistance", "Community Consciousness", "First Gen",
]
THEME_TO_IDX = {t: i for i, t in enumerate(THEMES)}
NUM_THEMES = len(THEMES)


def load_model_data(path: Path) -> list:
    """Load flat sentence-level model_data.pkl."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded model_data: %d sentences", len(data))
    return data


def load_master_data(path: Path) -> dict:
    """Load master_data.pkl (essay_id → EssayEntry with metadata.alma_id)."""
    # master_data.pkl uses alma_pipeline custom classes — add to path if needed
    alma_pipeline_dir = path.parent.parent / "Data" / "processing_workspace"
    if alma_pipeline_dir.exists():
        sys.path.insert(0, str(alma_pipeline_dir))
    # Also try relative to data_dir
    alt_dir = path.parent.parent / "processing_workspace"
    if alt_dir.exists():
        sys.path.insert(0, str(alt_dir))
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded master_data: %d essays", len(data))
    return data


def group_sentences_to_essays(model_data: list) -> dict:
    """
    Group flat sentence records into essay-level format.

    Returns:
        {essay_id: {"sentences": [str, ...], "annotations": [set, ...]}}
    """
    essays = defaultdict(list)
    for row in model_data:
        essays[row["essay_id"]].append(row)

    result = {}
    for eid, rows in essays.items():
        rows.sort(key=lambda r: r["sentence_order"])
        sentences = []
        annotations = []
        for r in rows:
            sentences.append(r["sentence"])
            labels = set(r["labels"]) - {"class_0"}
            annotations.append(labels)
        result[eid] = {"sentences": sentences, "annotations": annotations}

    logger.info("Grouped %d essays from %d sentences", len(result), len(model_data))
    return result


def build_alma_id_mapping(essays: dict, master_data: dict) -> dict:
    """
    Map essay_id → alma_id (student identifier) from master_data.

    For essays not in master_data, assign a unique synthetic alma_id.
    """
    essay_to_alma = {}
    missing = 0
    for eid in essays:
        if eid in master_data:
            essay_to_alma[eid] = master_data[eid].metadata.alma_id
        else:
            essay_to_alma[eid] = f"synthetic_{eid}"
            missing += 1
    if missing:
        logger.warning("%d essays not found in master_data (assigned synthetic alma_ids)", missing)
    n_students = len(set(essay_to_alma.values()))
    logger.info("Mapped %d essays to %d unique students (alma_ids)", len(essay_to_alma), n_students)
    return essay_to_alma


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
    """
    Split essays by student (alma_id), stratified by theme presence.

    Uses iterative stratification: group essays by student, compute per-student
    theme vector (union of all their essays), then stratify students.

    Returns (train_ids, val_ids, test_ids) — lists of essay_ids.
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

    # Iterative stratification (simplified but effective):
    # Sort themes by rarity. For each theme (rarest first), ensure proportional
    # representation in all splits.
    theme_counts = np.zeros(NUM_THEMES)
    for sid in students:
        theme_counts += student_vectors[sid]
    theme_order = np.argsort(theme_counts)  # rarest first

    # Try iterstrat if available, otherwise fall back to manual stratification
    try:
        from iterstratification import MultilabelStratifiedShuffleSplit
        _use_iterstrat = True
    except ImportError:
        try:
            from skmultilearn.model_selection import IterativeStratification
            _use_iterstrat = False  # use skmultilearn
        except ImportError:
            _use_iterstrat = None

    if _use_iterstrat is True:
        # Use iterstrat
        X = np.arange(len(students)).reshape(-1, 1)
        y = np.array([student_vectors[s] for s in students])

        # First split: train vs (val+test)
        splitter1 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed,
        )
        train_idx, temp_idx = next(splitter1.split(X, y))

        # Second split: val vs test (from the temp set)
        relative_test = test_ratio / (val_ratio + test_ratio)
        X_temp = X[temp_idx]
        y_temp = y[temp_idx]
        splitter2 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=relative_test, random_state=seed,
        )
        val_sub_idx, test_sub_idx = next(splitter2.split(X_temp, y_temp))
        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        train_students = [students[i] for i in train_idx]
        val_students = [students[i] for i in val_idx]
        test_students = [students[i] for i in test_idx]
    else:
        # Manual stratified split (works without extra deps)
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
    """
    Manual iterative stratification when iterstrat is not available.

    For each theme (rarest first), greedily assign students who have that theme
    proportionally across splits. Then assign remaining students randomly.
    """
    train_set = set()
    val_set = set()
    test_set = set()
    assigned = set()

    # Process themes rarest first
    for theme_idx in theme_order:
        theme_students = [s for s in students if student_vectors[s][theme_idx] > 0 and s not in assigned]
        if not theme_students:
            continue
        rng.shuffle(theme_students)
        n = len(theme_students)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Ensure at least 1 student with this theme in each split
        train_batch = theme_students[:n_train]
        val_batch = theme_students[n_train:n_train + n_val]
        test_batch = theme_students[n_train + n_val:]
        # If test is empty, take from train
        if not test_batch and len(train_batch) > 2:
            test_batch = [train_batch.pop()]
        if not val_batch and len(train_batch) > 2:
            val_batch = [train_batch.pop()]

        train_set.update(train_batch)
        val_set.update(val_batch)
        test_set.update(test_batch)
        assigned.update(train_batch + val_batch + test_batch)

    # Assign remaining students (those with no themed essays or already missed)
    remaining = [s for s in students if s not in assigned]
    rng.shuffle(remaining)
    n = len(remaining)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_set.update(remaining[:n_train])
    val_set.update(remaining[n_train:n_train + n_val])
    test_set.update(remaining[n_train + n_val:])

    # Remove any duplicates (students assigned via rare theme then also via another)
    # Priority: keep in the split they were first assigned to
    all_assigned = train_set | val_set | test_set
    # Check for overlap (shouldn't happen, but be safe)
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set
    for s in overlap_tv:
        val_set.discard(s)
    for s in overlap_tt:
        test_set.discard(s)
    for s in overlap_vt:
        test_set.discard(s)

    return list(train_set), list(val_set), list(test_set)


def verify_splits(essays, train_ids, val_ids, test_ids, essay_to_alma):
    """Verify split quality: no leakage, all themes present, no overlap."""
    issues = []

    # 1. No essay overlap
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    if train_set & val_set:
        issues.append(f"Train/val overlap: {len(train_set & val_set)} essays")
    if train_set & test_set:
        issues.append(f"Train/test overlap: {len(train_set & test_set)} essays")
    if val_set & test_set:
        issues.append(f"Val/test overlap: {len(val_set & test_set)} essays")

    # 2. No student leakage
    train_students = {essay_to_alma[e] for e in train_ids}
    val_students = {essay_to_alma[e] for e in val_ids}
    test_students = {essay_to_alma[e] for e in test_ids}
    leak_tv = train_students & val_students
    leak_tt = train_students & test_students
    leak_vt = val_students & test_students
    if leak_tv:
        issues.append(f"Student leakage train/val: {len(leak_tv)} students")
    if leak_tt:
        issues.append(f"Student leakage train/test: {len(leak_tt)} students")
    if leak_vt:
        issues.append(f"Student leakage val/test: {len(leak_vt)} students")

    # 3. All themes in each split
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        themes_present = set()
        for eid in ids:
            for ann in essays[eid]["annotations"]:
                themes_present.update(ann)
        missing = set(THEMES) - themes_present
        if missing:
            issues.append(f"{name} missing themes: {missing}")

    # 4. All essays accounted for
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
        # Count class_0 sentences (those with empty annotations)
        counts["class_0"] += sum(1 for a in essays[eid]["annotations"] if not a)
    return dict(counts)


def compute_theme_weights(essays: dict, essay_ids: list) -> list:
    """
    Compute per-theme weights using inverse sqrt frequency.
    weight_i = sqrt(max_count / count_i)
    Returns list of 11 floats in THEMES order.
    """
    counts = np.zeros(NUM_THEMES)
    for eid in essay_ids:
        for ann in essays[eid]["annotations"]:
            for theme in ann:
                if theme in THEME_TO_IDX:
                    counts[THEME_TO_IDX[theme]] += 1

    max_count = counts.max()
    weights = []
    for i, theme in enumerate(THEMES):
        if counts[i] > 0:
            w = math.sqrt(max_count / counts[i])
        else:
            w = 1.0
        weights.append(round(w, 4))
    return weights


def compute_essay_weights(essays: dict, essay_ids: list) -> dict:
    """
    Per-essay sampling weight for weighted random sampling.
    Essays with rarer themes get higher weight.
    weight = max over themes of sqrt(total_themed / count(theme))
    Pure class_0 essays get weight = 0.5 (down-weighted).
    """
    # Count theme occurrences across training essays
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
            weights[eid] = 0.5  # class_0 only
        else:
            w = max(
                math.sqrt(total_themed / max(theme_essay_counts.get(t, 1), 1))
                for t in essay_themes
            )
            weights[eid] = round(w, 4)
    return weights


def create_toy_dataset(essays: dict, essay_ids: list, n: int = 100, seed: int = 42) -> list:
    """
    Create toy dataset of ~n essays, prioritizing essays with rare themes.
    Ensures all 11 themes are represented.
    """
    rng = np.random.RandomState(seed)

    # First, ensure at least 2 essays per theme (from rarest to most common)
    theme_essays = defaultdict(list)
    for eid in essay_ids:
        for ann in essays[eid]["annotations"]:
            for theme in ann:
                if theme in THEME_TO_IDX:
                    theme_essays[theme].append(eid)

    selected = set()
    for theme in sorted(THEMES, key=lambda t: len(theme_essays.get(t, [])))[:]:
        candidates = [e for e in theme_essays.get(theme, []) if e not in selected]
        if candidates:
            picks = rng.choice(candidates, size=min(3, len(candidates)), replace=False)
            selected.update(picks)

    # Fill remaining with random essays
    remaining = [e for e in essay_ids if e not in selected]
    rng.shuffle(remaining)
    needed = max(0, n - len(selected))
    selected.update(remaining[:needed])

    toy_ids = sorted(selected)
    logger.info("Toy dataset: %d essays", len(toy_ids))
    return toy_ids


def build_split_data(essays: dict, essay_ids: list, essay_weights: dict) -> dict:
    """
    Build the output format for a split.

    Returns:
        {
            "essays": {eid: {"sentences": [...], "annotations": [set, ...]}},
            "essay_ids": [...],
            "weights": {eid: float}
        }
    """
    split_essays = {eid: essays[eid] for eid in essay_ids}
    split_weights = {eid: essay_weights.get(eid, 1.0) for eid in essay_ids}
    return {
        "essays": split_essays,
        "essay_ids": list(essay_ids),
        "weights": split_weights,
    }


def build_dapt_corpus(master_data: dict, unannotated_path: Path = None) -> str:
    """
    Build DAPT corpus: all essay text from master_data + unannotated essays.
    Returns full text (one essay per line, double-newline separated).
    """
    texts = []
    for eid, entry in master_data.items():
        if hasattr(entry, "essay_text") and entry.essay_text:
            texts.append(entry.essay_text.strip())
        elif hasattr(entry, "sentences") and entry.sentences:
            texts.append(" ".join(entry.sentences))

    if unannotated_path and unannotated_path.exists():
        with open(unannotated_path, "rb") as f:
            unannotated = pickle.load(f)
        if isinstance(unannotated, dict):
            for eid, entry in unannotated.items():
                if hasattr(entry, "essay_text") and entry.essay_text:
                    texts.append(entry.essay_text.strip())
                elif hasattr(entry, "sentences") and entry.sentences:
                    texts.append(" ".join(entry.sentences))
        logger.info("Added unannotated essays: total corpus = %d essays", len(texts))
    else:
        logger.info("No unannotated data found. DAPT corpus = %d essays (annotated only)", len(texts))

    return "\n\n".join(texts)


def main():
    parser = argparse.ArgumentParser(description="Prepare AWARE training data")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Data_for_modeling/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for split files")
    parser.add_argument("--unannotated_path", type=str, default=None, help="Path to unannotated essays (new_data_master.pkl)")
    parser.add_argument("--master_data_path", type=str, default=None, help="Path to master_data.pkl (for alma_id)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--toy_size", type=int, default=100)
    parser.add_argument("--toy_only", action="store_true", help="Only create toy dataset (skip splits)")
    parser.add_argument("--skip_dapt", action="store_true", help="Skip DAPT corpus generation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    model_data = load_model_data(data_dir / "model_data.pkl")

    master_path = Path(args.master_data_path) if args.master_data_path else data_dir / "master_data.pkl"
    master_data = load_master_data(master_path)

    # ── Group sentences → essays ──
    essays = group_sentences_to_essays(model_data)

    # ── Map essay_id → alma_id ──
    essay_to_alma = build_alma_id_mapping(essays, master_data)

    # ── Stratified split by student ──
    train_ids, val_ids, test_ids = stratified_student_split(
        essays, essay_to_alma,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # ── Verify splits ──
    ok = verify_splits(essays, train_ids, val_ids, test_ids, essay_to_alma)
    if not ok:
        logger.error("Split verification FAILED — check issues above")
        sys.exit(1)

    # ── Compute weights ──
    train_weights = compute_essay_weights(essays, train_ids)
    all_weights = {**train_weights}
    for eid in val_ids + test_ids:
        all_weights[eid] = 1.0  # val/test don't need sampling weights

    # ── Compute theme weights (from training data) ──
    theme_weights = compute_theme_weights(essays, train_ids)
    logger.info("Theme weights (train): %s", dict(zip(THEMES, theme_weights)))

    # ── Build and save splits ──
    if not args.toy_only:
        for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            split_data = build_split_data(essays, ids, all_weights)
            out_path = output_dir / f"{name}_data.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(split_data, f)
            logger.info("Saved %s: %d essays → %s", name, len(ids), out_path)

    # ── Toy dataset ──
    toy_ids = create_toy_dataset(essays, train_ids, n=args.toy_size, seed=args.seed)
    toy_val_ids = create_toy_dataset(essays, val_ids, n=min(30, len(val_ids)), seed=args.seed + 1)
    toy_data = {
        "train": build_split_data(essays, toy_ids, all_weights),
        "val": build_split_data(essays, toy_val_ids, all_weights),
    }
    with open(output_dir / "toy_data.pkl", "wb") as f:
        pickle.dump(toy_data, f)
    logger.info("Saved toy dataset: train=%d, val=%d essays", len(toy_ids), len(toy_val_ids))

    # ── DAPT corpus ──
    if not args.skip_dapt:
        unannotated_path = Path(args.unannotated_path) if args.unannotated_path else None
        if unannotated_path is None:
            # Try default location
            default_unann = data_dir.parent / "Data" / "processing_workspace" / "output" / "new_data_master.pkl"
            if default_unann.exists():
                unannotated_path = default_unann
        corpus = build_dapt_corpus(master_data, unannotated_path)
        dapt_path = output_dir / "dapt_corpus.txt"
        with open(dapt_path, "w") as f:
            f.write(corpus)
        n_essays_corpus = corpus.count("\n\n") + 1
        logger.info("Saved DAPT corpus: %d essays, %.1f MB → %s", n_essays_corpus, len(corpus) / 1e6, dapt_path)

    # ── Stats report ──
    stats = {
        "seed": args.seed,
        "total_essays": len(essays),
        "total_sentences": len(model_data),
        "total_students": len(set(essay_to_alma.values())),
        "theme_order": THEMES,
        "theme_weights": dict(zip(THEMES, theme_weights)),
        "splits": {},
    }
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        n_sents = sum(len(essays[e]["sentences"]) for e in ids)
        n_students = len(set(essay_to_alma[e] for e in ids))
        theme_counts = compute_theme_counts(essays, ids)
        stats["splits"][name] = {
            "essays": len(ids),
            "sentences": n_sents,
            "students": n_students,
            "theme_counts": {t: theme_counts.get(t, 0) for t in THEMES},
            "class_0_sentences": theme_counts.get("class_0", 0),
        }
    stats["leakage_check"] = {
        "train_val_student_overlap": len(
            set(essay_to_alma[e] for e in train_ids) & set(essay_to_alma[e] for e in val_ids)
        ),
        "train_test_student_overlap": len(
            set(essay_to_alma[e] for e in train_ids) & set(essay_to_alma[e] for e in test_ids)
        ),
        "val_test_student_overlap": len(
            set(essay_to_alma[e] for e in val_ids) & set(essay_to_alma[e] for e in test_ids)
        ),
    }

    stats_path = output_dir / "splits_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved split stats → %s", stats_path)

    # ── Print summary ──
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
    print(f"\nFiles saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
