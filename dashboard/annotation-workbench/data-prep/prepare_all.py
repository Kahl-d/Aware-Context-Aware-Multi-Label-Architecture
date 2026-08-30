"""
ALMA Dashboard Data Preparation
Converts all CSVs → unified JSON files for the dashboard.
Tracks every sentence across all dataset versions with proper tagging.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
BASE = Path(__file__).resolve().parent.parent.parent  # Final_Thesis_Folders2/
DATA = BASE / "Data"
FINAL = DATA / "Final_Data"
MASTER = DATA / "ALMA_Master_Dataset"
MODELS_INF = BASE / "Models_inference"
MODELS_LIGHT = BASE / "Models_light"
OUT = Path(__file__).resolve().parent.parent / "public" / "data"

V4_THEMES = [
    "Attainment", "Aspirational", "Navigational", "Resistance",
    "Perseverance", "Social", "Spiritual", "Familial_Capital"
]

V1_THEMES = [
    "Attainment", "First_Gen", "Aspirational", "Navigational", "Resistance",
    "Perseverance", "Filial_Piety", "Familial", "Community_Consciousness",
    "Social", "Spiritual"
]


def load_splits():
    """Load train/val/test essay_id assignments from pkl files."""
    splits = {}
    for split_name in ["train", "val", "test"]:
        pkl_path = MODELS_INF / "data" / f"{split_name}_data.pkl"
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        for eid in d["essay_ids"]:
            splits[int(eid)] = split_name
    return splits


def prepare_sentences():
    """Build unified sentences.json from all dataset versions."""
    print("Loading dataset versions...")
    v1 = pd.read_csv(FINAL / "v1_original_processed.csv")
    v2 = pd.read_csv(FINAL / "v2_merged_cleaned.csv")
    v3 = pd.read_csv(FINAL / "v3_boundary_cleaned.csv")
    v4 = pd.read_csv(FINAL / "v4_no_cc.csv")

    print(f"  V1: {len(v1)} rows")
    print(f"  V2: {len(v2)} rows")
    print(f"  V3: {len(v3)} rows")
    print(f"  V4: {len(v4)} rows")

    # Build lookup sets for version membership
    def make_key(row):
        return (int(row["essay_id"]), int(row["sentence_id"]))

    v1_keys = set(v1.apply(make_key, axis=1))
    v2_keys = set(v2.apply(make_key, axis=1))
    v3_keys = set(v3.apply(make_key, axis=1))
    v4_keys = set(v4.apply(make_key, axis=1))

    # Load splits
    splits = load_splits()
    print(f"  Split assignments: {len(splits)} essays")

    # Build from V1 as the superset of annotated data
    records = []
    meta_cols = ["essay_id", "sentence_id", "sentence", "sentence_length",
                 "alma_id", "course", "semester", "year", "prompt", "source_file", "coder"]

    for _, row in v1.iterrows():
        key = make_key(row)
        eid = int(row["essay_id"])

        # Determine version membership
        versions = []
        if key in v1_keys:
            versions.append("v1")
        if key in v2_keys:
            versions.append("v2")
        if key in v3_keys:
            versions.append("v3")
        if key in v4_keys:
            versions.append("v4")

        used_for_training = key in v4_keys
        split = splits.get(eid) if used_for_training else None

        # Determine drop reason
        dropped_reason = None
        if not used_for_training:
            if key not in v2_keys:
                dropped_reason = "theme_consolidation_v1_to_v2"
            elif key not in v4_keys and key in v2_keys:
                dropped_reason = "community_consciousness_drop"

        # Build labels in V4 schema (8 themes)
        labels = {}
        for theme in V4_THEMES:
            if theme in row.index:
                val = row[theme]
                labels[theme] = int(val) if pd.notna(val) and val != -1 else None
            elif theme == "Familial_Capital":
                # V1 has Familial and Filial_Piety separately
                fam = row.get("Familial", 0)
                fil = row.get("Filial_Piety", 0)
                fam_val = 1 if (fam == 1 or fil == 1) else 0
                labels[theme] = fam_val
            else:
                labels[theme] = None

        # Class_0
        if used_for_training:
            # Use V4's Class_0 column
            v4_row = v4[(v4["essay_id"] == eid) & (v4["sentence_id"] == int(row["sentence_id"]))]
            if len(v4_row) > 0:
                labels["Class_0"] = int(v4_row.iloc[0]["Class_0"])
            else:
                labels["Class_0"] = 1 if all(v == 0 for v in labels.values() if v is not None) else 0
        else:
            labels["Class_0"] = 1 if all(v == 0 or v is None for v in labels.values()) else 0

        record = {
            "essay_id": str(eid),
            "sentence_id": int(row["sentence_id"]),
            "sentence": str(row["sentence"]),
            "sentence_length": int(row["sentence_length"]) if pd.notna(row.get("sentence_length")) else len(str(row["sentence"]).split()),
            "alma_id": str(row.get("alma_id", "")),
            "course": str(row.get("course", "")),
            "semester": str(row.get("semester", "")),
            "year": str(row.get("year", "")),
            "prompt": str(row.get("prompt", "")),
            "source_file": str(row.get("source_file", "")),
            "coder": str(row.get("coder", "")),
            "labels": labels,
            "tags": {
                "annotated": True,
                "dataset_versions": versions,
                "used_for_training": used_for_training,
                "split": split,
                "dropped_reason": dropped_reason,
            },
            "predictions": None,
        }
        records.append(record)

    print(f"  Total annotated sentences: {len(records)}")

    # Verify V4 count
    v4_count = sum(1 for r in records if r["tags"]["used_for_training"])
    print(f"  Used for training (V4): {v4_count}")

    return records


def prepare_unannotated():
    """Process unannotated essays into sentence records."""
    unannotated_dir = MASTER / "unannotated"
    records = []
    essay_records = []

    if not unannotated_dir.exists():
        print("  No unannotated directory found, skipping.")
        return records, essay_records

    print("Loading unannotated essays...")
    essay_counter = 0

    for prompt_dir in sorted(unannotated_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue

        for csv_file in prompt_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, encoding="utf-8")
            except Exception:
                try:
                    df = pd.read_csv(csv_file, encoding="latin-1")
                except Exception as e:
                    print(f"  WARNING: Could not read {csv_file}: {e}")
                    continue

            for _, row in df.iterrows():
                essay_counter += 1
                essay_text = str(row.get("essay", ""))
                if not essay_text or essay_text == "nan":
                    continue

                # Simple sentence segmentation (same approach as model)
                import re
                sentences = re.split(r'(?<=[.!?])\s+', essay_text.strip())
                sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

                essay_id = f"U-{essay_counter:04d}"
                alma_id = str(row.get("alma_id", ""))
                course = str(row.get("course", ""))
                semester = str(row.get("semester", ""))
                year = str(row.get("year", ""))
                prompt = str(row.get("prompt", ""))
                source = str(row.get("source_file", csv_file.name))

                sentence_ids = []
                for si, sent in enumerate(sentences):
                    sid = si + 1
                    sentence_ids.append(sid)
                    null_labels = {t: None for t in V4_THEMES}
                    null_labels["Class_0"] = None

                    records.append({
                        "essay_id": essay_id,
                        "sentence_id": sid,
                        "sentence": sent,
                        "sentence_length": len(sent.split()),
                        "alma_id": alma_id,
                        "course": course,
                        "semester": semester,
                        "year": year,
                        "prompt": prompt,
                        "source_file": source,
                        "coder": "",
                        "labels": null_labels,
                        "tags": {
                            "annotated": False,
                            "dataset_versions": [],
                            "used_for_training": False,
                            "split": None,
                            "dropped_reason": None,
                        },
                        "predictions": None,
                    })

                essay_records.append({
                    "essay_id": essay_id,
                    "alma_id": alma_id,
                    "course": course,
                    "semester": semester,
                    "year": year,
                    "prompt": prompt,
                    "coder": "",
                    "sentence_count": len(sentences),
                    "annotated_count": 0,
                    "sentence_ids": sentence_ids,
                    "tags": {
                        "annotated": False,
                        "used_for_training": False,
                        "split": None,
                        "dataset_versions": [],
                    },
                })

    print(f"  Unannotated essays: {len(essay_records)}")
    print(f"  Unannotated sentences: {len(records)}")
    return records, essay_records


def prepare_essays(annotated_records):
    """Build essay index from annotated sentence records."""
    essays = {}
    for rec in annotated_records:
        eid = rec["essay_id"]
        if eid not in essays:
            essays[eid] = {
                "essay_id": eid,
                "alma_id": rec["alma_id"],
                "course": rec["course"],
                "semester": rec["semester"],
                "year": rec["year"],
                "prompt": rec["prompt"],
                "coder": rec["coder"],
                "sentence_count": 0,
                "annotated_count": 0,
                "sentence_ids": [],
                "tags": {
                    "annotated": rec["tags"]["annotated"],
                    "used_for_training": rec["tags"]["used_for_training"],
                    "split": rec["tags"]["split"],
                    "dataset_versions": rec["tags"]["dataset_versions"],
                },
            }
        essays[eid]["sentence_count"] += 1
        essays[eid]["sentence_ids"].append(rec["sentence_id"])
        if rec["tags"]["annotated"]:
            essays[eid]["annotated_count"] += 1
        # Update tags to be most inclusive
        if rec["tags"]["used_for_training"]:
            essays[eid]["tags"]["used_for_training"] = True
        if rec["tags"]["split"]:
            essays[eid]["tags"]["split"] = rec["tags"]["split"]
        for v in rec["tags"]["dataset_versions"]:
            if v not in essays[eid]["tags"]["dataset_versions"]:
                essays[eid]["tags"]["dataset_versions"].append(v)

    return list(essays.values())


def prepare_results():
    """Merge all model evaluation results into one file."""
    results = {}

    # Large v4
    eval_path = MODELS_LIGHT / "Model_large_v3" / "results" / "final_v4" / "evaluation_test.json"
    if eval_path.exists():
        with open(eval_path) as f:
            results["large_v4"] = json.load(f)

    # Large v3
    eval_path = MODELS_LIGHT / "Model_large_v3" / "results" / "final" / "evaluation_test.json"
    if eval_path.exists():
        with open(eval_path) as f:
            results["large_v3"] = json.load(f)

    # Base
    eval_path = MODELS_LIGHT / "Model_base" / "results" / "final" / "evaluation_test.json"
    if eval_path.exists():
        with open(eval_path) as f:
            results["base"] = json.load(f)

    # Baselines
    baseline_path = MODELS_LIGHT / "baselines" / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            results["baselines"] = json.load(f)

    return results


def prepare_training_history():
    """Merge training histories from all models."""
    histories = {}

    for model_name, hist_path in [
        ("large_v4", MODELS_LIGHT / "Model_large_v3" / "results" / "final_v4" / "history.json"),
        ("large_v3", MODELS_LIGHT / "Model_large_v3" / "results" / "final" / "history.json"),
        ("base", MODELS_LIGHT / "Model_base" / "results" / "final" / "history.json"),
    ]:
        if hist_path.exists():
            with open(hist_path) as f:
                histories[model_name] = json.load(f)

    return histories


def prepare_group_analysis():
    """Load multi-label group analysis results."""
    path = MODELS_LIGHT / "multi_label_group_results" / "group_analysis_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def prepare_splits_stats():
    """Copy splits stats."""
    path = MODELS_INF / "data" / "splits_stats.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def copy_plots():
    """Copy plot images to public/plots/."""
    import shutil
    plots_out = OUT / "plots"
    plots_out.mkdir(parents=True, exist_ok=True)

    # Copy from Models_light/plots_comparison/
    src = MODELS_LIGHT / "plots_comparison"
    if src.exists():
        for f in src.glob("*.png"):
            shutil.copy2(f, plots_out / f.name)
        print(f"  Copied {len(list(src.glob('*.png')))} plots from plots_comparison/")

    # Copy from Data_Processing_v2/plots/
    for subdir in ["plots", "plots_boundary", "plots_single_label"]:
        src = DATA / "Data_Processing_v2" / subdir
        if src.exists():
            for f in src.glob("*.png"):
                shutil.copy2(f, plots_out / f"dp2_{subdir}_{f.name}")
            print(f"  Copied {len(list(src.glob('*.png')))} plots from {subdir}/")

    # Copy from Final_Data/
    src = FINAL
    for f in src.glob("*.png"):
        shutil.copy2(f, plots_out / f"final_{f.name}")


def validate(annotated, unannotated, essays):
    """Validate counts against thesis numbers."""
    print("\n=== VALIDATION ===")
    v4_sents = [r for r in annotated if r["tags"]["used_for_training"]]
    v4_essays = set(r["essay_id"] for r in v4_sents)
    train_sents = [r for r in v4_sents if r["tags"]["split"] == "train"]
    val_sents = [r for r in v4_sents if r["tags"]["split"] == "val"]
    test_sents = [r for r in v4_sents if r["tags"]["split"] == "test"]

    checks = [
        ("V4 sentences", len(v4_sents), 17622),
        ("V4 essays", len(v4_essays), 2636),
        ("Train sentences", len(train_sents), 14023),
        ("Val sentences", len(val_sents), 1757),
        ("Test sentences", len(test_sents), 1842),
        ("V1 total", len(annotated), 18019),
    ]

    all_pass = True
    for name, actual, expected in checks:
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}: {name} = {actual} (expected {expected})")

    # Theme counts in V4
    for theme in V4_THEMES:
        count = sum(1 for r in v4_sents if r["labels"].get(theme) == 1)
        print(f"  {theme}: {count}")

    # Verify no unannotated in training
    unannotated_in_training = [r for r in unannotated if r["tags"]["used_for_training"]]
    status = "PASS" if len(unannotated_in_training) == 0 else "FAIL"
    print(f"  {status}: Unannotated in training = {len(unannotated_in_training)} (expected 0)")

    return all_pass


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Prepare annotated sentences
    annotated = prepare_sentences()

    # 2. Prepare unannotated
    unannotated, unannotated_essays = prepare_unannotated()

    # 3. Build essay index
    annotated_essays = prepare_essays(annotated)
    all_essays = annotated_essays + unannotated_essays

    # 4. Combine all sentences
    all_sentences = annotated + unannotated

    # 5. Validate
    valid = validate(annotated, unannotated, all_essays)

    # 6. Write JSON files
    print("\nWriting JSON files...")

    with open(OUT / "sentences.json", "w") as f:
        json.dump(all_sentences, f, separators=(",", ":"))
    print(f"  sentences.json: {len(all_sentences)} records")

    with open(OUT / "essays.json", "w") as f:
        json.dump(all_essays, f, separators=(",", ":"))
    print(f"  essays.json: {len(all_essays)} records")

    # Results
    results = prepare_results()
    with open(OUT / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  evaluation_results.json")

    # Training history
    history = prepare_training_history()
    with open(OUT / "training_history.json", "w") as f:
        json.dump(history, f, separators=(",", ":"))
    print(f"  training_history.json")

    # Group analysis
    groups = prepare_group_analysis()
    with open(OUT / "group_analysis.json", "w") as f:
        json.dump(groups, f, separators=(",", ":"))
    print(f"  group_analysis.json")

    # Splits stats
    stats = prepare_splits_stats()
    with open(OUT / "splits_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  splits_stats.json")

    # Dataset versions summary
    versions = [
        {"version": "v1", "sentences": 18019, "essays": 2698, "themes": 11,
         "description": "Original processed dataset with all 11 themes"},
        {"version": "v2", "sentences": 17859, "essays": 2675, "themes": 9,
         "description": "Merged Familial+Filial_Piety, dropped First_Gen"},
        {"version": "v3", "sentences": 16395, "essays": 2665, "themes": 9,
         "description": "Aggressive boundary cleaning (-1,464 sentences)"},
        {"version": "v4", "sentences": 17622, "essays": 2636, "themes": 8,
         "description": "Dropped Community_Consciousness — final training dataset"},
    ]
    with open(OUT / "dataset_versions.json", "w") as f:
        json.dump(versions, f, indent=2)

    # Theme colors
    theme_colors = {
        "Attainment": "#7c3aed", "Aspirational": "#2563eb", "Navigational": "#0891b2",
        "Resistance": "#dc2626", "Perseverance": "#ea580c", "Social": "#16a34a",
        "Spiritual": "#a855f7", "Familial_Capital": "#ca8a04", "Class_0": "#6b7280",
    }
    with open(OUT / "theme_colors.json", "w") as f:
        json.dump(theme_colors, f, indent=2)

    # Copy plots
    print("\nCopying plots...")
    copy_plots()

    print(f"\n{'='*50}")
    print(f"DONE. All files written to {OUT}")
    print(f"Total sentences: {len(all_sentences)}")
    print(f"Total essays: {len(all_essays)}")
    if valid:
        print("ALL VALIDATIONS PASSED")
    else:
        print("WARNING: Some validations failed!")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
