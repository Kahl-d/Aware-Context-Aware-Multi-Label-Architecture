"""
losses.py — Asymmetric Loss (ASL) for multi-label classification.

No SupCon in v2 — removed due to noisy signal with small batches.
ASL with per-theme weights handles class imbalance.
Q011: Added CB Loss weights + logit adjustment.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

# Training set class counts (from splits_stats.json, verified)
# Order: Nav, Att, Per, Asp, Soc, FP, Spi, Fam, Res, CC, FG
TRAIN_CLASS_COUNTS = [7879, 5434, 5163, 4732, 944, 263, 698, 2432, 770, 170, 114]
TOTAL_TRAIN_SENTENCES = 68000  # approximate


def compute_cb_weights(class_counts: List[int], beta: float = 0.999) -> torch.Tensor:
    """Class-Balanced Loss weights via effective number of samples.

    weight_c = 1 / effective_n_c, where effective_n = (1 - beta^n) / (1 - beta).
    Normalized so min weight = 1.0.

    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number
    of Samples" (CVPR 2019)
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)
    effective_n = (1.0 - beta ** counts) / (1.0 - beta)
    weights = 1.0 / effective_n
    weights = weights / weights.min()
    return weights


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification (Ridnik et al., 2021).

    gamma_neg > gamma_pos: down-weight easy negatives more aggressively.
    clip: probability shifting — prevents very easy negatives from dominating.
    theme_weights: per-class weighting for imbalanced themes.
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
        eps: float = 1e-8,
        theme_weights: Optional[torch.Tensor] = None,
        logit_adjustment: Optional[torch.Tensor] = None,
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
        if logit_adjustment is not None:
            self.register_buffer("logit_adjustment", logit_adjustment)
        else:
            self.logit_adjustment = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, max_sentences, num_labels]
            targets: [batch, max_sentences, num_labels] (binary)
            mask: [batch, max_sentences] sentence mask
        Returns:
            Scalar loss.
        """
        # Logit adjustment: shift logits by log(prior/(1-prior)) during training.
        # Rare classes get a positive shift → model more likely to predict them.
        if self.logit_adjustment is not None:
            logits = logits + self.logit_adjustment.view(1, 1, -1)

        probs = torch.sigmoid(logits)

        if self.label_smoothing > 0:
            # Asymmetric smoothing: preserve full positive signal (1.0 stays 1.0),
            # only smooth negatives (0 → label_smoothing).
            # Symmetric smoothing (old: targets * 0.95 + 0.025) weakened positive
            # signal for rare themes where every positive example matters.
            targets = targets + (1 - targets) * self.label_smoothing

        # Positive loss: (1 - p)^gamma_pos * log(p)
        pos_loss = (1 - probs) ** self.gamma_pos * torch.log(probs + self.eps)

        # Negative loss with probability shifting
        probs_neg = (probs - self.clip).clamp(min=0)
        neg_loss = probs_neg ** self.gamma_neg * torch.log(1 - probs_neg + self.eps)

        loss = -targets * pos_loss - (1 - targets) * neg_loss

        # Per-theme weighting
        if self.theme_weights is not None:
            loss = loss * self.theme_weights.view(1, 1, -1)

        # Mask: only compute loss for valid sentences
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            loss = loss * mask_expanded
            num_valid = mask_expanded.sum().clamp(min=1)
            loss = loss.sum() / num_valid
        else:
            loss = loss.mean()

        return loss


def build_loss_from_config(config) -> AsymmetricLoss:
    """Build ASL from AWAREConfig. Supports manual weights, CB Loss, and logit adjustment."""
    L = config.loss
    theme_weights = None

    # CB Loss weights override manual theme_weights (Cui et al., CVPR 2019)
    cb_beta = getattr(L, "cb_beta", None)
    if cb_beta is not None:
        theme_weights = compute_cb_weights(TRAIN_CLASS_COUNTS, beta=cb_beta)
        from config import THEMES
        logger.info("CB Loss weights (beta=%.4f): %s", cb_beta,
            {t: round(w, 2) for t, w in zip(THEMES, theme_weights.tolist())})
    elif getattr(L, "theme_weights", None):
        w = L.theme_weights
        if isinstance(w, (list, tuple)) and len(w) == 11:
            theme_weights = torch.tensor(w, dtype=torch.float32)

    # Logit adjustment: log(prior / (1 - prior)) per class (Menon et al., ICLR 2021)
    logit_adj = None
    if getattr(L, "logit_adjustment", False):
        counts = torch.tensor(TRAIN_CLASS_COUNTS, dtype=torch.float32)
        prior = counts / TOTAL_TRAIN_SENTENCES
        logit_adj = torch.log(prior / (1 - prior))
        from config import THEMES
        logger.info("Logit adjustment: %s",
            {t: round(a, 3) for t, a in zip(THEMES, logit_adj.tolist())})

    return AsymmetricLoss(
        gamma_pos=L.asl_gamma_pos,
        gamma_neg=L.asl_gamma_neg,
        clip=L.asl_clip,
        label_smoothing=L.label_smoothing,
        theme_weights=theme_weights,
        logit_adjustment=logit_adj,
    )
