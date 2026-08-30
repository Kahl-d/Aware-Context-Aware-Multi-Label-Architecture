"""
config.py — Configuration system for AWARE v2.

Loads from YAML, provides typed dataclass access. Supports toy/full modes.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    encoder_name: str = "microsoft/deberta-v3-base"
    encoder_path: Optional[str] = None  # Path to DAPT encoder (if available)
    hidden_size: int = 768  # DeBERTa-v3-base hidden size
    lstm_hidden: int = 256
    lstm_layers: int = 2
    max_sentences: int = 32
    max_seq_length: int = 512
    num_labels: int = 11
    dropout: float = 0.3


@dataclass
class LossConfig:
    asl_gamma_pos: float = 1.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    label_smoothing: float = 0.05
    theme_weights: Optional[List[float]] = None  # Auto-computed if None
    # Class-Balanced Loss (Cui et al., CVPR 2019): replaces manual theme_weights
    # with principled formula. Set to 0.999 for aggressive, 0.99 for mild. None = disabled.
    cb_beta: Optional[float] = None
    # Logit Adjustment (Menon et al., ICLR 2021): add log(prior/(1-prior)) to logits
    # during training. Shifts rare class logits up. Only applied in loss, not at inference.
    logit_adjustment: bool = False


@dataclass
class TrainingConfig:
    phase1_epochs: int = 5
    phase2_epochs: int = 15
    batch_size: int = 8
    gradient_accumulation: int = 4
    encoder_lr: float = 2e-5
    decoder_lr: float = 1e-4
    phase2_encoder_lr_scale: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.06
    early_stopping_patience: int = 5
    phase2_early_stopping_patience: int = 7
    fp16: bool = True
    num_workers: int = 2
    # Per-theme essay sampling boost: multiplies base essay weight for essays
    # containing each rare theme, so they appear more often in training batches.
    # Keys are theme names, values are multiplicative boost factors.
    rare_theme_boost: Optional[Dict[str, float]] = None
    # R-Drop regularization alpha (0.0 = disabled). Forces consistent predictions
    # across dropout masks, stabilizing rare theme predictions. (Liang & Wu, NeurIPS 2021)
    rdrop_alpha: float = 0.0
    # EMA decay for model weight averaging (0.0 = disabled). Reduces prediction
    # variance, especially for rare themes. (arXiv:2411.18704)
    ema_decay: float = 0.0
    # Layer-Wise Learning Rate Decay (LLRD): each encoder layer gets a different LR.
    # Lower layers (pretrained knowledge) get smaller LR, upper layers get larger.
    # LR for layer i = encoder_lr × decay^(num_layers - i). 0.0 = disabled (2-group).
    # Recommended: 0.95 for DeBERTa-v3-large (24 layers).
    llrd_decay: float = 0.0
    # Gradient checkpointing: trade ~30% speed for ~40% memory savings.
    # Required for v3-large at batch_size=8. Recomputes activations during backward.
    gradient_checkpointing: bool = False


@dataclass
class AugmentationConfig:
    aeda_prob: float = 0.3
    aeda_punctuation: str = ".,;:!?"
    enabled: bool = True


@dataclass
class DAPTConfig:
    mlm_probability: float = 0.15
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    fp16: bool = True
    eval_split: float = 0.1
    weight_decay: float = 0.01  # Was hardcoded in dapt.py; configurable for v3-large


@dataclass
class AWAREConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    dapt: DAPTConfig = field(default_factory=DAPTConfig)
    seed: int = 42
    mode: str = "full"  # "toy" or "full"

    @classmethod
    def from_yaml(cls, path: str) -> "AWAREConfig":
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "AWAREConfig":
        config = cls()
        if "model" in d:
            for k, v in d["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        if "loss" in d:
            for k, v in d["loss"].items():
                if hasattr(config.loss, k):
                    setattr(config.loss, k, v)
        if "training" in d:
            for k, v in d["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        if "augmentation" in d:
            for k, v in d["augmentation"].items():
                if hasattr(config.augmentation, k):
                    setattr(config.augmentation, k, v)
        if "dapt" in d:
            for k, v in d["dapt"].items():
                if hasattr(config.dapt, k):
                    setattr(config.dapt, k, v)
        if "seed" in d:
            config.seed = d["seed"]
        if "mode" in d:
            config.mode = d["mode"]
        return config

    def to_dict(self) -> dict:
        """Serialize config to dict (for saving alongside results)."""
        import dataclasses
        return {
            "model": dataclasses.asdict(self.model),
            "loss": dataclasses.asdict(self.loss),
            "training": dataclasses.asdict(self.training),
            "augmentation": dataclasses.asdict(self.augmentation),
            "dapt": dataclasses.asdict(self.dapt),
            "seed": self.seed,
            "mode": self.mode,
        }


# Canonical theme info
THEMES = [
    "Navigational", "Attainment", "Perseverance", "Aspirational",
    "Social", "Filial Piety", "Spiritual", "Familial",
    "Resistance", "Community Consciousness", "First Gen",
]
THEME_TO_IDX = {t: i for i, t in enumerate(THEMES)}
IDX_TO_THEME = {i: t for i, t in enumerate(THEMES)}
NUM_THEMES = len(THEMES)
