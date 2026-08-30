"""
config.py — Configuration system for AWARE v3 (Model_v4).

8 CCW themes (no CC, no First_Gen, Familial+Filial_Piety merged).
Loads from YAML, provides typed dataclass access.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ── Canonical theme info (matches CSV column order) ──────────────────────────
THEMES = [
    "Attainment",
    "Aspirational",
    "Navigational",
    "Resistance",
    "Perseverance",
    "Social",
    "Spiritual",
    "Familial_Capital",
]
THEME_TO_IDX = {t: i for i, t in enumerate(THEMES)}
IDX_TO_THEME = {i: t for i, t in enumerate(THEMES)}
NUM_THEMES = len(THEMES)


@dataclass
class ModelConfig:
    encoder_name: str = "microsoft/deberta-v3-base"
    encoder_path: Optional[str] = None
    hidden_size: int = 768
    lstm_hidden: int = 256
    lstm_layers: int = 2
    max_sentences: int = 32
    max_seq_length: int = 512
    num_labels: int = NUM_THEMES
    dropout: float = 0.15
    use_bilstm: bool = True
    bilstm_dropout: float = 0.0
    # Sentence position embedding: learned 32×H table added before BiLSTM
    # so the model explicitly knows sentence index within essay (intro vs body vs conclusion).
    use_position_embedding: bool = False
    # Prototype head: cosine-similarity to learned theme description embeddings.
    # When True, ClassificationHead is replaced with PrototypeClassificationHead.
    # Requires calling model.initialize_prototypes(tokenizer) after model creation.
    use_prototype_head: bool = False
    # Multi-sample dropout: average K dropout masks in classification head (Inoue 2019)
    n_dropout_samples: int = 1  # 1 = standard single dropout, 3-5 = multi-sample
    # Context encoder type: "bilstm" (original AWARE) or "gated_attention" (v5)
    context_type: str = "bilstm"
    # Gated attention config (only used when context_type="gated_attention")
    context_num_heads: int = 4       # attention heads for sentence self-attention
    gate_init_bias: float = -2.0     # negative = gate starts mostly CLOSED (conservative)
    # Pooling type: "mean" (original) or "attention" (learned token importance)
    pooling_type: str = "mean"


@dataclass
class LossConfig:
    loss_type: str = "asl"
    # ASL (Ridnik et al., ICCV 2021)
    asl_gamma_pos: float = 0.0      # 0 = standard BCE for positives (preserve all signal)
    asl_gamma_neg: float = 2.0      # 2 = moderate suppression of easy negatives
    asl_clip: float = 0.05          # Probability margin shift for negatives
    label_smoothing: float = 0.05   # Asymmetric: only smooth negatives
    theme_weights: Optional[List[float]] = None  # None = auto-compute from data
    use_cb_weights: bool = True     # FIX: was silently dropped before (not in dataclass)


@dataclass
class TrainingConfig:
    phase1_epochs: int = 4
    phase2_epochs: int = 30
    batch_size: int = 8
    gradient_accumulation: int = 4
    encoder_lr: float = 2e-5
    decoder_lr: float = 1e-4
    phase2_encoder_lr_scale: float = 0.25
    weight_decay: float = 0.02
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.10
    early_stopping_patience: int = 7
    phase2_early_stopping_patience: int = 7
    fp16: bool = True
    num_workers: int = 4
    rdrop_alpha: float = 1.0
    phase3_epochs: int = 10
    phase3_lr: float = 1e-4
    phase3_patience: int = 5
    adam_beta2: float = 0.999
    gradient_checkpointing: bool = False
    use_llrd: bool = False
    llrd_decay: float = 0.9
    # Essay-level multi-task loss weight. 0.0 = disabled.
    # When > 0, essay_head predicts essay-level theme presence (OR over sentences).
    # Doubles effective positive examples for rare themes without adding data.
    essay_aux_weight: float = 0.0
    # NEW: Separate Phase 1 LR (grad clipping reduces effective LR by ~7x for large model)
    phase1_lr: Optional[float] = None  # None = use decoder_lr
    # NEW: Phase 1 R-Drop (0.0 = disable R-Drop in Phase 1, head is random)
    phase1_rdrop: Optional[float] = None  # None = use rdrop_alpha
    # NEW: SWA start ratio (fraction of P2 epochs)
    swa_start_ratio: float = 0.5
    # NEW: Progressive unfreezing in Phase 2
    progressive_unfreeze: bool = False
    progressive_unfreeze_layers: int = 12  # unfreeze top N layers first
    progressive_unfreeze_after: int = 6    # unfreeze remaining after N P2 epochs
    # NEW: Phase 3 unfreeze BiLSTM (not just head)
    phase3_unfreeze_bilstm: bool = False
    # NEW: Context dropout — randomly mask sentence embeddings in Phase 2
    # Forces the context mechanism to actually reconstruct from neighbors
    context_dropout: float = 0.0  # 0.0 = disabled, 0.15 = recommended


@dataclass
class AugmentationConfig:
    enabled: bool = False
    aeda_prob: float = 0.3


@dataclass
class DAPTConfig:
    mlm_probability: float = 0.15
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    fp16: bool = True
    eval_split: float = 0.1
    weight_decay: float = 0.01


@dataclass
class AWAREConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    dapt: DAPTConfig = field(default_factory=DAPTConfig)
    seed: int = 42
    mode: str = "full"

    @classmethod
    def from_yaml(cls, path: str) -> "AWAREConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "AWAREConfig":
        config = cls()
        for section in ("model", "loss", "training", "augmentation", "dapt"):
            if section in d:
                obj = getattr(config, section)
                for k, v in d[section].items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
        if "seed" in d:
            config.seed = d["seed"]
        if "mode" in d:
            config.mode = d["mode"]
        return config

    def to_dict(self) -> dict:
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
