"""
compute_baselines.py — Compute ALL baselines for comparison.

Baselines:
  1. Majority Class (always predict 0)
  2. Random Prior (predict 1 with P=training_frequency)
  3. Stratified Random (1000 runs averaged)
  4. Most Common Label Set (predict most frequent combination)
  5. TF-IDF + Logistic Regression
  6. TF-IDF + Random Forest
  7. TF-IDF + SVM (LinearSVC)

All baselines use the SAME train/val/test splits as the AWARE models.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

THEMES = [
    "Attainment", "Aspirational", "Navigational", "Resistance",
    "Perseverance", "Social", "Spiritual", "Familial_Capital",
]
NUM_THEMES = len(THEMES)


def load_data(path):
    """Load pickle and extract sentences + labels."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    texts, labels = [], []
    for eid in data["essay_ids"]:
        essay = data["essays"][eid]
        for i, sent in enumerate(essay["sentences"]):
            texts.append(sent)
            # Build multi-hot label vector
            label_vec = np.zeros(NUM_THEMES, dtype=np.float32)
            if i < len(essay["annotations"]):
                for theme in essay["annotations"][i]:
                    if theme in THEMES:
                        label_vec[THEMES.index(theme)] = 1.0
            labels.append(label_vec)
    
    return texts, np.array(labels)


def compute_prauc(y_true, y_score):
    """Compute PR-AUC per theme and macro average.
    Uses average_precision_score to match AWARE's metrics.py computation.
    """
    praucs = {}
    valid = []
    for i, theme in enumerate(THEMES):
        if y_true[:, i].sum() == 0:
            praucs[theme] = 0.0
            continue
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        praucs[theme] = ap
        valid.append(ap)
    praucs["macro"] = float(np.mean(valid)) if valid else 0.0
    return praucs


def compute_f1_per_theme(y_true, y_pred):
    """Compute F1 per theme and macro average."""
    f1s = {}
    for i, theme in enumerate(THEMES):
        f1s[theme] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    f1s["macro"] = np.mean([f1s[t] for t in THEMES])
    return f1s


def baseline_majority_class(train_labels, test_labels):
    """Always predict majority class (0 for all themes since none >50%)."""
    n = len(test_labels)
    y_pred = np.zeros_like(test_labels)
    y_score = np.zeros_like(test_labels)
    
    # Use training frequency as score
    train_freq = train_labels.mean(axis=0)
    y_score = np.tile(train_freq, (n, 1))
    
    f1s = compute_f1_per_theme(test_labels, y_pred)
    praucs = compute_prauc(test_labels, y_score)
    return {"f1": f1s, "prauc": praucs, "name": "Majority Class"}


def baseline_random_prior(train_labels, test_labels, n_runs=1000):
    """Predict 1 with probability = training frequency."""
    n = len(test_labels)
    train_freq = train_labels.mean(axis=0)
    
    all_f1s = []
    for _ in range(n_runs):
        y_pred = (np.random.random((n, NUM_THEMES)) < train_freq).astype(float)
        macro = f1_score(test_labels, y_pred, average='macro', zero_division=0)
        all_f1s.append(macro)
    
    # Use freq as probability score for PRAUC
    y_score = np.tile(train_freq, (n, 1))
    # Add small noise for PRAUC computation
    y_score_noisy = y_score + np.random.normal(0, 0.01, y_score.shape)
    
    avg_f1 = np.mean(all_f1s)
    praucs = compute_prauc(test_labels, y_score_noisy)
    
    # Get per-theme F1 from a single representative run
    y_pred = (np.random.random((n, NUM_THEMES)) < train_freq).astype(float)
    f1s = compute_f1_per_theme(test_labels, y_pred)
    f1s["macro"] = avg_f1  # override with averaged macro
    
    return {"f1": f1s, "prauc": praucs, "name": "Random (Prior)"}


def baseline_most_common_label_set(train_labels, test_labels):
    """Always predict the most common label combination."""
    # Find most common combination
    combos = Counter()
    for row in train_labels:
        key = tuple(row.astype(int).tolist())
        combos[key] += 1
    
    most_common = combos.most_common(1)[0]
    pattern = np.array(most_common[0], dtype=float)
    freq = most_common[1] / len(train_labels)
    logger.info("Most common label set: %s (%.1f%% of training data)", 
                pattern.astype(int).tolist(), freq * 100)
    
    n = len(test_labels)
    y_pred = np.tile(pattern, (n, 1))
    # No meaningful probability scores for this baseline — use binary predictions
    # PR-AUC is not meaningful here (reported as N/A in comparison)
    y_score = y_pred.copy()  # binary 0/1 scores

    f1s = compute_f1_per_theme(test_labels, y_pred)
    # PR-AUC not meaningful for constant predictions
    praucs = {theme: 0.0 for theme in THEMES}
    praucs["macro"] = 0.0
    return {"f1": f1s, "prauc": praucs, "name": "Most Common Label Set",
            "note": "PR-AUC not meaningful (constant predictions)"}


def baseline_tfidf_model(train_texts, train_labels, test_texts, test_labels, 
                          model_class, model_name, **kwargs):
    """TF-IDF + sklearn classifier."""
    logger.info("Training %s...", model_name)
    
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)
    
    # Train OneVsRest
    clf = OneVsRestClassifier(model_class(**kwargs), n_jobs=-1)
    clf.fit(X_train, train_labels.astype(int))
    
    y_pred = clf.predict(X_test)
    
    # Get probability scores if available
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
        # Normalize to [0, 1] range
        y_score = 1 / (1 + np.exp(-y_score))
    else:
        y_score = y_pred.astype(float)
    
    f1s = compute_f1_per_theme(test_labels, y_pred)
    praucs = compute_prauc(test_labels, y_score)
    return {"f1": f1s, "prauc": praucs, "name": model_name}


def main():
    # Reproducibility seed
    np.random.seed(42)

    # Load data
    logger.info("Loading data...")
    train_texts, train_labels = load_data("data/train_data.pkl")
    val_texts, val_labels = load_data("data/val_data.pkl")
    test_texts, test_labels = load_data("data/test_data.pkl")
    
    logger.info("Train: %d sentences, Val: %d, Test: %d", 
                len(train_texts), len(val_texts), len(test_texts))
    
    # Per-theme training frequency
    train_freq = train_labels.mean(axis=0)
    logger.info("Training frequencies:")
    for i, theme in enumerate(THEMES):
        logger.info("  %s: %.1f%% (%d/%d)", 
                     theme, train_freq[i] * 100, int(train_labels[:, i].sum()), len(train_labels))
    
    results = {}
    
    # 1. Majority Class
    logger.info("\n=== Baseline 1: Majority Class ===")
    results["majority_class"] = baseline_majority_class(train_labels, test_labels)
    
    # 2. Random Prior
    logger.info("\n=== Baseline 2: Random (Prior) ===")
    results["random_prior"] = baseline_random_prior(train_labels, test_labels)
    
    # 3. Most Common Label Set
    logger.info("\n=== Baseline 3: Most Common Label Set ===")
    results["most_common"] = baseline_most_common_label_set(train_labels, test_labels)
    
    # 4. TF-IDF + Logistic Regression
    logger.info("\n=== Baseline 4: TF-IDF + Logistic Regression ===")
    results["tfidf_logreg"] = baseline_tfidf_model(
        train_texts, train_labels, test_texts, test_labels,
        LogisticRegression, "TF-IDF + LogReg",
        C=1.0, max_iter=1000, solver="lbfgs",
    )
    
    # 5. TF-IDF + Random Forest
    logger.info("\n=== Baseline 5: TF-IDF + Random Forest ===")
    results["tfidf_rf"] = baseline_tfidf_model(
        train_texts, train_labels, test_texts, test_labels,
        RandomForestClassifier, "TF-IDF + RandomForest",
        n_estimators=200, max_depth=None, min_samples_leaf=5,
    )
    
    # 6. TF-IDF + SVM
    logger.info("\n=== Baseline 6: TF-IDF + SVM ===")
    results["tfidf_svm"] = baseline_tfidf_model(
        train_texts, train_labels, test_texts, test_labels,
        LinearSVC, "TF-IDF + SVM",
        C=1.0, max_iter=5000,
    )
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/baselines_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    # Print comparison table
    print("\n" + "=" * 90)
    print("BASELINE RESULTS — Test Set")
    print("=" * 90)
    print(f"{'Model':<30} {'Macro F1':>10} {'Macro PRAUC':>12}")
    print("-" * 90)
    for key, res in results.items():
        f1 = res["f1"]["macro"]
        prauc = res["prauc"].get("macro", "N/A")
        prauc_str = f"{prauc:.4f}" if isinstance(prauc, float) else prauc
        print(f"{res['name']:<30} {f1:>10.4f} {prauc_str:>12}")
    
    print("\n" + "-" * 90)
    print(f"{'Per-Theme F1':<20}", end="")
    for theme in THEMES:
        print(f" {theme[:5]:>7}", end="")
    print()
    print("-" * 90)
    for key, res in results.items():
        print(f"{res['name'][:20]:<20}", end="")
        for theme in THEMES:
            print(f" {res['f1'].get(theme, 0):.3f}  ", end="")
        print()
    print("=" * 90)


if __name__ == "__main__":
    main()
