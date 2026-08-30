"""
config.py — Configuration for Standard baseline (no AWARE components).
Same theme definitions as AWARE for consistent evaluation.
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional, List

THEMES = [
    "Attainment", "Aspirational", "Navigational", "Resistance",
    "Perseverance", "Social", "Spiritual", "Familial_Capital",
]
THEME_TO_IDX = {t: i for i, t in enumerate(THEMES)}
IDX_TO_THEME = {i: t for i, t in enumerate(THEMES)}
NUM_THEMES = len(THEMES)


@dataclass
class ModelConfig:
    encoder_name: str = "microsoft/deberta-v3-base"
    hidden_size: int = 768
    max_sentences: int = 32
    max_seq_length: int = 512
    num_labels: int = NUM_THEMES
    dropout: float = 0.15


@dataclass
class LossConfig:
    loss_type: str = "bce"
    theme_weights: Optional[List[float]] = None

@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 8
    gradient_accumulation: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.10
    early_stopping_patience: int = 5
    fp16: bool = True
    num_workers: int = 4


@dataclass
class StandardConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "StandardConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        config = cls()
        for section in ("model", "loss", "training"):
            if section in raw:
                obj = getattr(config, section)
                for k, v in raw[section].items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
        if "seed" in raw:
            config.seed = raw["seed"]
        return config

    def to_dict(self):
        import dataclasses
        return {"model":dataclasses.asdict(self.model),"loss":dataclasses.asdict(self.loss),"training":dataclasses.asdict(self.training),"seed":self.seed}
