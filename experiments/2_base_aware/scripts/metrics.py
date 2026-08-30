"""
metrics.py — Full evaluation suite for multi-label sentence classification.

Includes:
  - Per-theme F1, Precision, Recall
  - Macro/Micro F1
  - PR-AUC (Average Precision) — primary model selection metric
  - ROC-AUC per theme
  - Hamming loss, Exact match ratio
  - Class_0 metrics
  - Per-theme threshold optimization (F1 and F2)
  - Per-class Platt scaling calibration
  - Bootstrap confidence intervals
"""

import numpy as np
import logging
import torch
from typing import Dict, Tuple
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score

from config import THEMES, NUM_THEMES

logger = logging.getLogger(__name__)


def flatten_masked_preds_labels(logits, labels, mask, threshold=0.5):
    probs = torch.sigmoid(logits)
    mask_bool = mask.bool()
    flat_probs = probs[mask_bool].cpu().numpy()
    flat_labels = labels[mask_bool].cpu().numpy()
    return (flat_probs >= threshold).astype(np.float32), flat_labels


def flatten_masked_logits_labels(logits, labels, mask):
    mask_bool = mask.bool()
    return logits[mask_bool], labels[mask_bool]


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute per-theme and aggregate metrics."""
    eps = 1e-8
    N, C = preds.shape

    f1_pt, prec_pt, rec_pt, sup_pt = {}, {}, {}, {}
    for i in range(C):
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum()
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps) if (p + r) > 0 else 0.0
        theme = THEMES[i] if i < len(THEMES) else f"theme_{i}"
        f1_pt[theme] = round(float(f1), 4)
        prec_pt[theme] = round(float(p), 4)
        rec_pt[theme] = round(float(r), 4)
        sup_pt[theme] = int(labels[:, i].sum())

    f1_macro = float(np.mean(list(f1_pt.values())))

    # Micro F1
    tp_all = ((preds == 1) & (labels == 1)).sum()
    fp_all = ((preds == 1) & (labels == 0)).sum()
    fn_all = ((preds == 0) & (labels == 1)).sum()
    micro_p = tp_all / (tp_all + fp_all + eps)
    micro_r = tp_all / (tp_all + fn_all + eps)
    f1_micro = float(2 * micro_p * micro_r / (micro_p + micro_r + eps))

    # Class_0 (no theme)
    true_c0 = (labels.sum(axis=1) == 0)
    pred_c0 = (preds.sum(axis=1) == 0)
    c0_tp = int((true_c0 & pred_c0).sum())
    c0_fp = int((~true_c0 & pred_c0).sum())
    c0_fn = int((true_c0 & ~pred_c0).sum())
    c0_p = c0_tp / (c0_tp + c0_fp + eps)
    c0_r = c0_tp / (c0_tp + c0_fn + eps)
    c0_f1 = float(2 * c0_p * c0_r / (c0_p + c0_r + eps)) if (c0_p + c0_r) > 0 else 0.0

    return {
        "f1_macro": round(f1_macro, 4),
        "f1_micro": round(f1_micro, 4),
        "precision_macro": round(float(np.mean(list(prec_pt.values()))), 4),
        "recall_macro": round(float(np.mean(list(rec_pt.values()))), 4),
        "f1_per_theme": f1_pt,
        "precision_per_theme": prec_pt,
        "recall_per_theme": rec_pt,
        "support_per_theme": sup_pt,
        "class_0": {
            "f1": round(c0_f1, 4), "precision": round(float(c0_p), 4),
            "recall": round(float(c0_r), 4), "support": int(true_c0.sum()),
        },
    }


def compute_prauc(probs: np.ndarray, labels: np.ndarray) -> Dict:
    """PR-AUC (Average Precision) per theme and macro. Primary metric."""
    prauc_pt = {}
    valid = []
    for i in range(probs.shape[1]):
        theme = THEMES[i] if i < len(THEMES) else f"theme_{i}"
        support = int(labels[:, i].sum())
        if support == 0:
            prauc_pt[theme] = 0.0
            continue
        ap = average_precision_score(labels[:, i], probs[:, i])
        prauc_pt[theme] = round(float(ap), 4)
        valid.append(float(ap))
    return {
        "prauc_macro": round(float(np.mean(valid)), 4) if valid else 0.0,
        "prauc_per_theme": prauc_pt,
    }


def compute_rocauc(probs: np.ndarray, labels: np.ndarray) -> Dict:
    """ROC-AUC per theme and macro."""
    auc_pt = {}
    valid = []
    for i in range(probs.shape[1]):
        theme = THEMES[i] if i < len(THEMES) else f"theme_{i}"
        if labels[:, i].sum() == 0 or labels[:, i].sum() == len(labels):
            auc_pt[theme] = 0.0
            continue
        a = roc_auc_score(labels[:, i], probs[:, i])
        auc_pt[theme] = round(float(a), 4)
        valid.append(float(a))
    return {
        "rocauc_macro": round(float(np.mean(valid)), 4) if valid else 0.0,
        "rocauc_per_theme": auc_pt,
    }


def compute_hamming_loss(preds, labels):
    return round(float((preds != labels).mean()), 4)


def compute_exact_match_ratio(preds, labels):
    return round(float(np.all(preds == labels, axis=1).mean()), 4)


def optimize_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    search_range=(0.05, 0.80),
    step=0.01,
    default_threshold=0.5,
    rare_support_cutoff=80,
    min_precision=0.15,
) -> Dict[str, float]:
    """Per-theme threshold optimization with distribution-aware safety.

    min_precision: for rare themes, if the best F2 threshold yields precision < min_precision,
    fall back to F1 optimization. Prevents "predict everything" from being rewarded.
    This fixes the Attainment/Spiritual issue where threshold=0.050 gave P=0.14 — useless in practice.
    """
    thresholds = {}
    candidates = np.arange(search_range[0], search_range[1] + step, step)

    for i in range(probs.shape[1]):
        theme = THEMES[i] if i < len(THEMES) else f"theme_{i}"
        support = int(labels[:, i].sum())

        if support < 3:
            thresholds[theme] = default_threshold
            continue

        pos_mask = labels[:, i] == 1
        neg_probs = probs[~pos_mask, i]
        pos_probs = probs[pos_mask, i]
        neg_mean = float(neg_probs.mean()) if len(neg_probs) > 0 else 0.3
        pos_mean = float(pos_probs.mean()) if len(pos_probs) > 0 else 0.5
        separation = pos_mean - neg_mean

        if separation < 0.03:
            thresholds[theme] = default_threshold
            logger.info("Theme %s: sep=%.3f too low, using default %.2f", theme, separation, default_threshold)
            continue

        is_rare = support < rare_support_cutoff

        def _best_threshold(beta_val):
            bsq = beta_val ** 2
            best_score, best_t, best_p = -1, default_threshold, 0.0
            for t in candidates:
                pred = (probs[:, i] >= t).astype(np.float32)
                tp = ((pred == 1) & (labels[:, i] == 1)).sum()
                fp = ((pred == 1) & (labels[:, i] == 0)).sum()
                fn = ((pred == 0) & (labels[:, i] == 1)).sum()
                if tp + fp + fn == 0:
                    continue
                p = tp / (tp + fp + 1e-8)
                r = tp / (tp + fn + 1e-8)
                fb = (1 + bsq) * p * r / (bsq * p + r + 1e-8) if (p + r) > 0 else 0.0
                if fb > best_score:
                    best_score = fb
                    best_t = t
                    best_p = p
            return best_t, best_score, best_p

        if is_rare:
            # F2 (recall-biased) for rare themes
            best_t, best_score, best_p = _best_threshold(2.0)
            # Precision floor: if F2 optimum gives terrible precision, fall back to F1
            if best_p < min_precision:
                f1_t, f1_score, f1_p = _best_threshold(1.0)
                logger.info(
                    "Theme %s: F2 t=%.3f gave P=%.3f < min_precision=%.2f → "
                    "falling back to F1 t=%.3f (P=%.3f)",
                    theme, best_t, best_p, min_precision, f1_t, f1_p,
                )
                best_t, best_score = f1_t, f1_score
                best_p = f1_p
        else:
            best_t, best_score, best_p = _best_threshold(1.0)

        # Floor: threshold >= neg_mean + 0.02 (never below random baseline)
        floor = neg_mean + 0.02
        if best_t < floor:
            best_t = floor

        beta_used = 2.0 if is_rare and best_p >= min_precision else 1.0
        thresholds[theme] = round(float(best_t), 3)
        logger.info(
            "Theme %s: t=%.3f, F%.0f=%.4f, P=%.3f (support=%d, sep=%.3f)",
            theme, best_t, beta_used, best_score, best_p, support, separation,
        )

    return thresholds


def calibrate_per_class(logits: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class Platt scaling: learns (a, b) per theme for calibrated probabilities.

    p_calibrated = sigmoid(a * z + b)
    Fits via grid search on validation data.
    """
    C = logits.shape[1]
    a_params = np.ones(C)
    b_params = np.zeros(C)

    for c in range(C):
        z = logits[:, c]
        y = labels[:, c]
        n_pos = y.sum()
        if n_pos < 3:
            continue

        best_nll = float("inf")
        best_a, best_b = 1.0, 0.0

        # Grid search over a and b — extended a range to 5.0 so rare themes
        # (Attainment, Resistance) don't hit the upper bound and get miscalibrated
        for a_cand in np.arange(0.3, 5.01, 0.1):
            for b_cand in np.arange(-6.0, 2.01, 0.2):
                p = 1.0 / (1.0 + np.exp(-(a_cand * z + b_cand)))
                p = np.clip(p, 1e-7, 1 - 1e-7)
                nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                if nll < best_nll:
                    best_nll = nll
                    best_a, best_b = a_cand, b_cand

        a_params[c] = best_a
        b_params[c] = best_b
        theme = THEMES[c] if c < len(THEMES) else f"theme_{c}"
        logger.info("  Platt %s: a=%.2f, b=%.2f (NLL=%.4f, n+=%d)", theme, best_a, best_b, best_nll, int(n_pos))

    calibrated = 1.0 / (1.0 + np.exp(-(a_params[np.newaxis, :] * logits + b_params[np.newaxis, :])))
    return np.stack([a_params, b_params], axis=0), calibrated


def apply_thresholds(probs: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    preds = np.zeros_like(probs)
    for i in range(probs.shape[1]):
        theme = THEMES[i] if i < len(THEMES) else f"theme_{i}"
        t = thresholds.get(theme, 0.5)
        preds[:, i] = (probs[:, i] >= t).astype(np.float32)
    return preds


def bootstrap_ci(preds, labels, n_bootstrap=1000, ci=0.95, seed=42):
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
