"""
build_dataset.py — CSV → essay-level pkl splits for AWARE v3 (Model_v4).

Reads v4_no_cc.csv, groups by essay_id, splits by alma_id (student-level
stratification), produces train/val/test pkl files + DAPT corpus.

Usage:
    python scripts/build_dataset.py --csv_path v4_no_cc.csv --output_dir data/
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
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

THEMES = [
    "Attainment", "Aspirational", "Navigational", "Resistance",
    "Perseverance", "Social", "Spiritual", "Familial_Capital",
]
THEME_TO_IDX = {t: i for i, t in enumerate(THEMES)}
NUM_THEMES = len(THEMES)


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    assert all(t in df.columns for t in THEMES), f"Missing theme columns. Got: {list(df.columns)}"
    return df


def build_essays(df: pd.DataFrame) -> dict:
    """Group sentences by essay_id → essay-level format for AWARE.

    Returns:
        {essay_id: {"sentences": [str,...], "annotations": [set,...], "alma_id": str}}
    """
    essays = {}
    for eid, grp in df.groupby("essay_id"):
        grp = grp.sort_values("sentence_id")
        sentences = grp["sentence"].tolist()
        annotations = []
        for _, row in grp.iterrows():
            themes = set()
            for t in THEMES:
                if row[t] == 1:
                    themes.add(t)
            annotations.append(themes)
        alma_id = str(grp["alma_id"].iloc[0])
        essays[int(eid)] = {
            "sentences": sentences,
            "annotations": annotations,
            "alma_id": alma_id,
        }
    logger.info("Built %d essays from %d sentences", len(essays), len(df))
    return essays


def stratified_student_split(
    essays: dict,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple:
    """Split essays by alma_id (student), stratified by theme presence.

    Uses iterative stratification: rarest themes assigned first.
    Returns (train_ids, val_ids, test_ids).
    """
    rng = np.random.RandomState(seed)

    # Group essays by student
    student_essays = defaultdict(list)
    for eid, essay in essays.items():
        student_essays[essay["alma_id"]].append(eid)

    students = sorted(student_essays.keys())
    logger.info("Splitting %d students into train/val/test", len(students))

    # Per-student theme vector
    student_vectors = {}
    for sid in students:
        vec = np.zeros(NUM_THEMES, dtype=np.float32)
        for eid in student_essays[sid]:
            for ann in essays[eid]["annotations"]:
                for theme in ann:
                    if theme in THEME_TO_IDX:
                        vec[THEME_TO_IDX[theme]] = 1.0
        student_vectors[sid] = vec

    # Theme rarity order (rarest first for iterative stratification)
    theme_counts = np.zeros(NUM_THEMES)
    for sid in students:
        theme_counts += student_vectors[sid]
    theme_order = np.argsort(theme_counts)

    # Iterative stratification
    train_set, val_set, test_set = set(), set(), set()
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

    # Assign remaining
    remaining = [s for s in students if s not in assigned]
    rng.shuffle(remaining)
    n = len(remaining)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_set.update(remaining[:n_train])
    val_set.update(remaining[n_train:n_train + n_val])
    test_set.update(remaining[n_train + n_val:])

    # Resolve overlaps
    val_set -= train_set
    test_set -= (train_set | val_set)

    # Convert to essay IDs
    train_ids = sorted([eid for s in train_set for eid in student_essays[s]])
    val_ids = sorted([eid for s in val_set for eid in student_essays[s]])
    test_ids = sorted([eid for s in test_set for eid in student_essays[s]])

    logger.info(
        "Split: train=%d essays (%d students), val=%d (%d), test=%d (%d)",
        len(train_ids), len(train_set), len(val_ids), len(val_set),
        len(test_ids), len(test_set),
    )
    return train_ids, val_ids, test_ids


def verify_splits(essays, train_ids, val_ids, test_ids) -> bool:
    issues = []
    sets = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sets[a] & sets[b]
        if overlap:
            issues.append(f"{a}/{b} essay overlap: {len(overlap)}")

    # Student leakage check
    students = {}
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        students[name] = {essays[eid]["alma_id"] for eid in ids}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        leak = students[a] & students[b]
        if leak:
            issues.append(f"Student leakage {a}/{b}: {len(leak)}")

    # Theme coverage
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        present = set()
        for eid in ids:
            for ann in essays[eid]["annotations"]:
                present.update(ann)
        missing = set(THEMES) - present
        if missing:
            issues.append(f"{name} missing themes: {missing}")

    total = len(train_ids) + len(val_ids) + len(test_ids)
    if total != len(essays):
        issues.append(f"Essay count mismatch: {total} vs {len(essays)}")

    if issues:
        for i in issues:
            logger.error("SPLIT ISSUE: %s", i)
        return False
    logger.info("All split verification checks PASSED")
    return True


def compute_theme_weights(essays: dict, essay_ids: list) -> list:
    """Inverse sqrt frequency weights from training data."""
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
    """Per-essay sampling weights: rare-theme essays get higher weight."""
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

    total_themed = max(len(themed_ids), 1)
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


def compute_theme_counts(essays: dict, ids: list) -> dict:
    counts = Counter()
    for eid in ids:
        for ann in essays[eid]["annotations"]:
            for theme in ann:
                counts[theme] += 1
        counts["class_0"] += sum(1 for a in essays[eid]["annotations"] if not a)
    return dict(counts)


def build_split_data(essays: dict, ids: list, weights: dict) -> dict:
    return {
        "essays": {eid: essays[eid] for eid in ids},
        "essay_ids": list(ids),
        "weights": {eid: weights.get(eid, 1.0) for eid in ids},
    }


def build_dapt_corpus(essays: dict) -> str:
    texts = []
    for eid in sorted(essays.keys()):
        sentences = essays[eid]["sentences"]
        texts.append(" ".join(sentences))
    return "\n\n".join(texts)


def create_toy_dataset(essays, essay_ids, n=80, seed=42):
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
    selected.update(remaining[:max(0, n - len(selected))])
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description="Build AWARE v3 dataset splits")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.csv_path)
    essays = build_essays(df)

    train_ids, val_ids, test_ids = stratified_student_split(
        essays, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed,
    )
    ok = verify_splits(essays, train_ids, val_ids, test_ids)
    if not ok:
        sys.exit(1)

    # Weights
    essay_weights = compute_essay_weights(essays, train_ids)
    for eid in val_ids + test_ids:
        essay_weights[eid] = 1.0

    theme_weights = compute_theme_weights(essays, train_ids)
    logger.info("Theme weights: %s", dict(zip(THEMES, theme_weights)))

    # Compute train class counts for losses
    train_counts = np.zeros(NUM_THEMES, dtype=int)
    train_total_sents = 0
    for eid in train_ids:
        for ann in essays[eid]["annotations"]:
            train_total_sents += 1
            for theme in ann:
                if theme in THEME_TO_IDX:
                    train_counts[THEME_TO_IDX[theme]] += 1

    # Save splits
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        data = build_split_data(essays, ids, essay_weights)
        path = out / f"{name}_data.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved %s: %d essays → %s", name, len(ids), path)

    # Toy dataset
    toy_ids = create_toy_dataset(essays, train_ids, n=80, seed=args.seed)
    toy_val_ids = create_toy_dataset(essays, val_ids, n=min(30, len(val_ids)), seed=args.seed + 1)
    toy_data = {
        "train": build_split_data(essays, toy_ids, essay_weights),
        "val": build_split_data(essays, toy_val_ids, essay_weights),
    }
    with open(out / "toy_data.pkl", "wb") as f:
        pickle.dump(toy_data, f)
    logger.info("Saved toy dataset: train=%d, val=%d", len(toy_ids), len(toy_val_ids))

    # DAPT corpus
    corpus = build_dapt_corpus(essays)
    with open(out / "dapt_corpus.txt", "w") as f:
        f.write(corpus)
    logger.info("Saved DAPT corpus: %.1f KB", len(corpus) / 1e3)

    # Stats
    stats = {
        "seed": args.seed,
        "total_essays": len(essays),
        "total_sentences": sum(len(e["sentences"]) for e in essays.values()),
        "total_students": len(set(e["alma_id"] for e in essays.values())),
        "theme_order": THEMES,
        "theme_weights": dict(zip(THEMES, theme_weights)),
        "train_class_counts": dict(zip(THEMES, train_counts.tolist())),
        "train_total_sentences": int(train_total_sents),
    }
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        n_sents = sum(len(essays[e]["sentences"]) for e in ids)
        n_studs = len(set(essays[e]["alma_id"] for e in ids))
        tc = compute_theme_counts(essays, ids)
        stats[name] = {
            "essays": len(ids),
            "sentences": n_sents,
            "students": n_studs,
            "theme_counts": {t: tc.get(t, 0) for t in THEMES},
            "class_0": tc.get("class_0", 0),
        }

    # Leakage check
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        a_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[a]
        b_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[b]
        overlap = set(essays[e]["alma_id"] for e in a_ids) & set(essays[e]["alma_id"] for e in b_ids)
        stats[f"leakage_{a}_{b}"] = len(overlap)

    with open(out / "splits_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)
    for name in ["train", "val", "test"]:
        s = stats[name]
        print(f"\n{name.upper():>5}: {s['essays']:>5} essays, {s['sentences']:>5} sents, {s['students']:>4} students")
        for t in THEMES:
            print(f"        {t:<20} {s['theme_counts'].get(t, 0):>5}")
        print(f"        {'class_0':<20} {s['class_0']:>5}")
    print(f"\nTheme weights: {dict(zip(THEMES, theme_weights))}")
    print("=" * 70)


if __name__ == "__main__":
    main()
