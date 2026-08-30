"""
train.py — Training entry point for AWARE v3.

Usage:
    python scripts/train.py --config configs/quick.yaml --data_dir data/ --output_dir results/q001/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import AWAREConfig, THEMES, NUM_THEMES
from model import build_model_from_config
from dataset import load_split_data, create_dataloader
from trainer import AWARETrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train AWARE v3")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--encoder_path", type=str, default=None)
    args = parser.parse_args()

    config = AWAREConfig.from_yaml(args.config)
    if args.encoder_path:
        config.model.encoder_path = args.encoder_path

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

    if args.toy:
        import pickle
        with open(data_dir / "toy_data.pkl", "rb") as f:
            toy = pickle.load(f)
        train_data, val_data = toy["train"], toy["val"]
    else:
        train_data = load_split_data(data_dir / "train_data.pkl")
        val_data = load_split_data(data_dir / "val_data.pkl")

    logger.info("Train: %d essays, Val: %d essays",
                len(train_data["essay_ids"]), len(val_data["essay_ids"]))
    n_weighted = sum(1 for w in train_data.get("weights", {}).values() if w != 1.0)
    logger.info("Weighted sampling: %d/%d essays have non-uniform weight",
                n_weighted, len(train_data["essay_ids"]))

    # Load per-theme class counts from stats — used for CB weight computation.
    # CB Loss (Cui et al., CVPR 2019) is better than inverse-sqrt for severe imbalance.
    # We do NOT override config.loss.theme_weights here so that build_loss_from_config
    # can auto-compute CB weights from counts (when theme_weights: null in config).
    train_class_counts = None
    stats_path = data_dir / "splits_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        counts_dict = stats.get("train_class_counts", {})
        if counts_dict:
            train_class_counts = [counts_dict.get(t, 1) for t in THEMES]
            logger.info(
                "Train class counts: %s",
                {t: counts_dict.get(t, 0) for t in THEMES},
            )
        # Only use explicit inverse-sqrt theme weights if config explicitly sets them
        # AND use_cb_weights is not set (backwards compat). Otherwise prefer CB.
        if config.loss.theme_weights is None and not getattr(config.loss, "use_cb_weights", True):
            tw = stats.get("theme_weights", {})
            if tw:
                config.loss.theme_weights = [tw.get(t, 1.0) for t in THEMES]
                logger.info("Auto theme weights (inverse-sqrt): %s", config.loss.theme_weights)

    train_loader = create_dataloader(
        train_data, tokenizer, config,
        shuffle=True, augment=config.augmentation.enabled,
        use_weighted_sampling=True,
    )
    val_loader = create_dataloader(
        val_data, tokenizer, config, shuffle=False, augment=False,
    )

    model = build_model_from_config(config)

    # Initialize prototype vectors from theme description encodings (if enabled).
    # Must happen after model creation (encoder weights loaded) and before training.
    if getattr(config.model, "use_prototype_head", False):
        logger.info("Initializing prototype vectors from theme descriptions...")
        model.initialize_prototypes(tokenizer)

    trainer = AWARETrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        output_dir=str(output_dir), config=config, device=args.device,
        train_class_counts=train_class_counts,
    )
    history = trainer.train()
    logger.info("Done. Best PRAUC: %.4f at epoch %d",
                history.get("best_prauc", 0), history.get("best_epoch", 0))


if __name__ == "__main__":
    main()
