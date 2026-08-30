"""
losses.py — ASL (Asymmetric Loss) for multi-label classification.

Ridnik et al., "Asymmetric Loss For Multi-Label Classification" (ICCV 2021).

Key insight: in multi-label, negatives vastly outnumber positives.
gamma_neg > gamma_pos down-weights easy negatives while preserving positive signal.
Probability margin (clip) zeros out very confident negatives entirely.

Research-backed defaults for text with 8 themes, max 21.8x imbalance:
  gamma_pos = 0.0  (preserve ALL positive gradient signal, esp. for rare themes)
  gamma_neg = 3.0  (moderate negative suppression — text needs more signal than vision)
  clip = 0.03      (zero out negatives with p < 0.03)
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
        eps: float = 1e-8,
        theme_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.label_smoothing = label_smoothing
        self.eps = eps
        if theme_weights is not None:
            self.register_buffer("theme_weights", theme_weights)
        else:
            self.theme_weights = None

    def forward(self, logits, targets, mask=None):
        """
        logits:  [B, S, C]
        targets: [B, S, C]
        mask:    [B, S]
        """
        probs = torch.sigmoid(logits)

        if self.label_smoothing > 0:
            # Asymmetric smoothing: only smooth negatives (0 → eps), keep positives at 1.0
            targets = targets + (1 - targets) * self.label_smoothing

        # Positive loss: -(1-p)^gamma_pos * log(p)
        pos_loss = (1 - probs) ** self.gamma_pos * torch.log(probs + self.eps)

        # Negative loss with probability margin shift
        probs_neg = (probs - self.clip).clamp(min=0)
        neg_loss = probs_neg ** self.gamma_neg * torch.log(1 - probs_neg + self.eps)

        loss = -targets * pos_loss - (1 - targets) * neg_loss

        if self.theme_weights is not None:
            loss = loss * self.theme_weights.view(1, 1, -1)

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).float()
            loss = loss * mask_exp
            return loss.sum() / mask_exp.sum().clamp(min=1)
        return loss.mean()


def compute_cb_theme_weights(counts: List[int], beta: float = 0.9999) -> torch.Tensor:
    """Class-Balanced weights (Cui et al., CVPR 2019).

    Effective number of samples: E_n = (1 - beta^n) / (1 - beta).
    Weight per class = 1 / E_n, normalized so mean weight = 1.0.

    Better than inverse-sqrt for severe imbalance: Attainment (339 samples)
    gets weight ~10.4x vs Navigational (3368 samples), vs inverse-sqrt's 3.15x.
    This gives Attainment 3x more gradient signal relative to current config.

    Args:
        counts: per-theme positive sentence counts [C]
        beta:   smoothing factor (0.9999 → good balance between uniform and inverse-freq)
    """
    counts_t = torch.tensor(counts, dtype=torch.float64).clamp(min=1)
    effective_n = (1.0 - beta ** counts_t) / (1.0 - beta)
    weights = 1.0 / effective_n
    weights = weights / weights.mean()  # normalize: mean weight = 1.0
    weights = weights.float()
    logger.info(
        "CB theme weights (beta=%.4f): %s",
        beta,
        {f"t{i}": round(float(w), 3) for i, w in enumerate(weights)},
    )
    return weights


def build_loss_from_config(config, train_class_counts=None, train_total_sentences=None):
    """Build ASL loss from config. Optionally compute CB theme weights from data."""
    L = config.loss

    theme_weights = None

    # Priority 1: explicit weights from config
    if L.theme_weights is not None:
        from config import NUM_THEMES
        w = L.theme_weights
        if isinstance(w, (list, tuple)) and len(w) == NUM_THEMES:
            theme_weights = torch.tensor(w, dtype=torch.float32)
            logger.info("Using explicit theme weights: %s", w)

    # Priority 2: auto-compute CB weights from training data (when theme_weights: null)
    elif train_class_counts is not None and getattr(L, "use_cb_weights", True):
        counts_list = list(train_class_counts)  # [C] positive counts per theme
        if len(counts_list) > 0 and all(c > 0 for c in counts_list):
            theme_weights = compute_cb_theme_weights(counts_list)
            logger.info("Auto-computed CB theme weights from training data (n_themes=%d)", len(counts_list))
        else:
            logger.warning("train_class_counts has zeros — using uniform weights")

    loss = AsymmetricLoss(
        gamma_pos=L.asl_gamma_pos,
        gamma_neg=L.asl_gamma_neg,
        clip=L.asl_clip,
        label_smoothing=L.label_smoothing,
        theme_weights=theme_weights,
    )
    logger.info(
        "ASL Loss: gamma_pos=%.1f, gamma_neg=%.1f, clip=%.2f, smoothing=%.2f, weights=%s",
        L.asl_gamma_pos, L.asl_gamma_neg, L.asl_clip, L.label_smoothing,
        "cb_auto" if (theme_weights is not None and L.theme_weights is None) else
        ("explicit" if theme_weights is not None else "none"),
    )
    return loss
