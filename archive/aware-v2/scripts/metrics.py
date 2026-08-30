"""
metrics.py — Evaluation metrics for multi-label sentence classification.

Per-theme F1, macro/micro F1, PR-AUC, Hamming loss, exact match,
threshold optimization, bootstrap CI.
"""

import numpy as np
import logging
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sklearn.metrics import average_precision_score

from config import THEMES, THEME_TO_IDX, NUM_THEMES

logger = logging.getLogger(__name__)


def flatten_masked_preds_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten logits/labels by mask, apply sigmoid + threshold.

    Returns:
        preds: [N_valid, NUM_THEMES] binary
        labels: [N_valid, NUM_THEMES] binary
    """
    probs = torch.sigmoid(logits)
    mask_bool = mask.bool()
    flat_probs = probs[mask_bool].cpu().numpy()
    flat_labels = labels[mask_bool].cpu().numpy()
    flat_preds = (flat_probs >= threshold).astype(np.float32)
    return flat_preds, flat_labels


def flatten_masked_logits_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten logits/labels by mask (keep as tensors for threshold optimization)."""
    mask_bool = mask.bool()
    return logits[mask_bool], labels[mask_bool]


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    per_theme: bool = True,
) -> Dict:
    """
    Compute F1 metrics.

    Args:
        preds: [N, NUM_THEMES] binary predictions
        labels: [N, NUM_THEMES] binary ground truth
    Returns:
        dict with f1_macro, f1_micro, f1_per_theme, precision_macro, recall_macro
    """
    eps = 1e-8
    N, C = preds.shape

    # Per-theme F1
    f1_per_theme = {}
    precision_per_theme = {}
    recall_per_theme = {}
    support_per_theme = {}

    for i in range(C):
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum()
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps) if (p + r) > 0 else 0.0
        theme = THEMES[i] if i < len(THEMES) else f"theme_{i}"
        f1_per_theme[theme] = round(float(f1), 4)
        precision_per_theme[theme] = round(float(p), 4)
        recall_per_theme[theme] = round(float(r), 4)
        support_per_theme[theme] = int(labels[:, i].sum())

    # Macro F1 (average over themes)
    f1_values = list(f1_per_theme.values())
    f1_macro = float(np.mean(f1_values))

    # Micro F1 (global TP/FP/FN)
    tp_total = ((preds == 1) & (labels == 1)).sum()
    fp_total = ((preds == 1) & (labels == 0)).sum()
    fn_total = ((preds == 0) & (labels == 1)).sum()
    micro_p = tp_total / (tp_total + fp_total + eps)
    micro_r = tp_total / (tp_total + fn_total + eps)
    f1_micro = float(2 * micro_p * micro_r / (micro_p + micro_r + eps))

    # Class_0 metrics: sentences with NO theme label (69% of data)
    true_no_theme = (labels.sum(axis=1) == 0)
    pred_no_theme = (preds.sum(axis=1) == 0)
    c0_tp = int((true_no_theme & pred_no_theme).sum())
    c0_fp = int((~true_no_theme & pred_no_theme).sum())  # had theme, predicted none
    c0_fn = int((true_no_theme & ~pred_no_theme).sum())  # no theme, predicted some
    c0_p = c0_tp / (c0_tp + c0_fp + eps)
    c0_r = c0_tp / (c0_tp + c0_fn + eps)
    c0_f1 = float(2 * c0_p * c0_r / (c0_p + c0_r + eps)) if (c0_p + c0_r) > 0 else 0.0
    c0_support = int(true_no_theme.sum())

    result = {
        "f1_macro": round(f1_macro, 4),
        "f1_micro": round(f1_micro, 4),
        "precision_macro": round(float(np.mean(list(precision_per_theme.values()))), 4),
        "recall_macro": round(float(np.mean(list(recall_per_theme.values()))), 4),
        "class_0": {
            "f1": round(c0_f1, 4),
            "precision": round(float(c0_p), 4),
            "recall": round(float(c0_r), 4),
            "support": c0_support,
            "tp": c0_tp, "fp": c0_fp, "fn": c0_fn,
        },
    }
    if per_theme:
        result["f1_per_theme"] = f1_per_theme
        result["precision_per_theme"] = precision_per_theme
        result["recall_per_theme"] = recall_per_theme
        result["support_per_theme"] = support_per_theme

    return result


def compute_prauc(
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """Compute PR-AUC (Average Precision) per theme and macro average.

    Threshold-independent metric, ideal for imbalanced multi-label data.
    Better than F1@0.5 for model selection during training.
    Reference: Davis & Goadrich (2006), ICML.

    Args:
        probs: [N, NUM_THEMES] probability scores (after sigmoid)
        labels: [N, NUM_THEMES] binary ground truth
    Returns:
        dict with prauc_macro, prauc_per_theme
    """
    prauc_per_theme = {}
    valid_praucs = []

    for i in range(probs.shape[1]):
        theme = THEMES[i] if i < len(THEMES) else f"theme_{i}"
        support = int(labels[:, i].sum())
        if support == 0:
            prauc_per_theme[theme] = 0.0
            continue
        ap = average_precision_score(labels[:, i], probs[:, i])
        prauc_per_theme[theme] = round(float(ap), 4)
        valid_praucs.append(float(ap))

    prauc_macro = round(float(np.mean(valid_praucs)), 4) if valid_praucs else 0.0

    return {
        "prauc_macro": prauc_macro,
        "prauc_per_theme": prauc_per_theme,
    }


def compute_hamming_loss(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of incorrect labels across all samples and themes."""
    return round(float((preds != labels).mean()), 4)


def compute_exact_match_ratio(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of samples where ALL theme labels are predicted correctly."""
    exact = np.all(preds == labels, axis=1).mean()
    return round(float(exact), 4)


def optimize_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    search_range: Tuple[float, float] = (0.10, 0.75),
    step: float = 0.01,
    default_threshold: float = 0.5,
    rare_support_cutoff: int = 50,
) -> Dict[str, float]:
    """
    Per-theme threshold optimization via grid search (distribution-aware).

    For common themes (support >= rare_support_cutoff): optimize F1.
    For rare themes (support < rare_support_cutoff): optimize F2 (recall-biased).

    Distribution-aware safety:
    - Computes neg/pos probability means and their separation
    - If separation < 0.03: model can't distinguish the theme → use default 0.5
    - Threshold is floored at neg_mean + 0.02 to prevent predicting everything
      positive (the Q005 catastrophe where thresholds fell below neg_mean)

    Returns:
        {theme_name: optimal_threshold}
    """
    thresholds = {}
    candidates = np.arange(search_range[0], search_range[1] + step, step)

    for i in range(probs.shape[1]):
        theme = THEMES[i] if i < len(THEMES) else f"theme_{i}"
        support = int(labels[:, i].sum())

        if support < 3:
            thresholds[theme] = default_threshold
            logger.info("Theme %s: support=%d, using default threshold %.2f", theme, support, default_threshold)
            continue

        # Compute distribution stats for safety checks
        pos_mask = labels[:, i] == 1
        neg_mask = labels[:, i] == 0
        neg_probs = probs[neg_mask, i]
        pos_probs = probs[pos_mask, i]
        neg_mean = float(neg_probs.mean()) if len(neg_probs) > 0 else 0.3
        pos_mean = float(pos_probs.mean()) if len(pos_probs) > 0 else 0.5
        separation = pos_mean - neg_mean

        # If separation is too small, the model can't distinguish this theme.
        # No threshold can help — use default and move on.
        if separation < 0.03:
            thresholds[theme] = default_threshold
            logger.info(
                "Theme %s: sep=%.3f (p+=%.3f p-=%.3f) too low for optimization, using default %.2f (support=%d)",
                theme, separation, pos_mean, neg_mean, default_threshold, support,
            )
            continue

        # Use F2 (recall-biased) for rare themes, F1 for common themes
        beta = 2.0 if support < rare_support_cutoff else 1.0
        beta_sq = beta ** 2

        best_score = -1
        best_t = default_threshold
        best_p, best_r = 0.0, 0.0
        for t in candidates:
            pred = (probs[:, i] >= t).astype(np.float32)
            tp = ((pred == 1) & (labels[:, i] == 1)).sum()
            fp = ((pred == 1) & (labels[:, i] == 0)).sum()
            fn = ((pred == 0) & (labels[:, i] == 1)).sum()
            if tp + fp + fn == 0:
                continue
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f_beta = (1 + beta_sq) * p * r / (beta_sq * p + r + 1e-8) if (p + r) > 0 else 0.0
            if f_beta > best_score:
                best_score = f_beta
                best_t = t
                best_p, best_r = float(p), float(r)

        # Distribution-aware floor: threshold must sit ABOVE the negative cloud.
        # Without this, aggressive loss weights can inflate neg probabilities above
        # the optimized threshold → recall=1.0, precision≈0 (Q005 catastrophe).
        threshold_floor = neg_mean + 0.02
        if best_t < threshold_floor:
            logger.info(
                "Theme %s: F%d search=%.3f < floor(neg_mean+0.02=%.3f), clamping up",
                theme, int(beta), best_t, threshold_floor,
            )
            best_t = threshold_floor

        thresholds[theme] = round(float(best_t), 3)
        metric_name = f"F{beta:.0f}"
        logger.info(
            "Theme %s: threshold=%.3f, %s=%.4f, P=%.3f, R=%.3f (support=%d, sep=%.3f)",
            theme, best_t, metric_name, best_score, best_p, best_r, support, separation,
        )

    return thresholds


def apply_thresholds(
    probs: np.ndarray,
    thresholds: Dict[str, float],
) -> np.ndarray:
    """Apply per-theme thresholds to probability matrix."""
    preds = np.zeros_like(probs)
    for i in range(probs.shape[1]):
        theme = THEMES[i] if i < len(THEMES) else f"theme_{i}"
        t = thresholds.get(theme, 0.5)
        preds[:, i] = (probs[:, i] >= t).astype(np.float32)
    return preds


def bootstrap_ci(
    preds: np.ndarray,
    labels: np.ndarray,
    metric_fn=None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Bootstrap confidence intervals for macro F1.

    Returns:
        {"f1_macro": (mean, lower, upper), "f1_per_theme": {theme: (mean, lower, upper)}}
    """
    rng = np.random.RandomState(seed)
    N = preds.shape[0]
    macro_scores = []
    theme_scores = defaultdict(list)

    for _ in range(n_bootstrap):
        idx = rng.choice(N, N, replace=True)
        m = compute_metrics(preds[idx], labels[idx])
        macro_scores.append(m["f1_macro"])
        for theme, f1 in m.get("f1_per_theme", {}).items():
            theme_scores[theme].append(f1)

    alpha = (1 - ci) / 2
    result = {
        "f1_macro": (
            round(float(np.mean(macro_scores)), 4),
            round(float(np.percentile(macro_scores, 100 * alpha)), 4),
            round(float(np.percentile(macro_scores, 100 * (1 - alpha))), 4),
        ),
        "f1_per_theme": {},
    }
    for theme in THEMES:
        if theme in theme_scores:
            scores = theme_scores[theme]
            result["f1_per_theme"][theme] = (
                round(float(np.mean(scores)), 4),
                round(float(np.percentile(scores, 100 * alpha)), 4),
                round(float(np.percentile(scores, 100 * (1 - alpha))), 4),
            )
    return result
