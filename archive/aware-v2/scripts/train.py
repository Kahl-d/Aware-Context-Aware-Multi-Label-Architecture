"""
train.py — Training entry point for AWARE v2.

Usage:
    # Local toy test
    python scripts/train.py --config configs/toy.yaml --data_dir data/ --output_dir results/toy/

    # Full HPC training
    python scripts/train.py --config configs/full.yaml --data_dir data/ --output_dir results/full/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import AWAREConfig
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
    parser = argparse.ArgumentParser(description="Train AWARE v2")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data/ directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--toy", action="store_true", help="Use toy dataset")
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Override config encoder_path (e.g. path to DAPT encoder output)")
    args = parser.parse_args()

    config = AWAREConfig.from_yaml(args.config)
    if args.encoder_path:
        config.model.encoder_path = args.encoder_path
        logger.info("Encoder path overridden from CLI: %s", args.encoder_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Load tokenizer
    encoder = getattr(config.model, "encoder_path", None) or config.model.encoder_name
    logger.info("Loading tokenizer: %s", config.model.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

    # Load data
    if args.toy:
        logger.info("Using toy dataset")
        import pickle
        with open(data_dir / "toy_data.pkl", "rb") as f:
            toy = pickle.load(f)
        train_data = toy["train"]
        val_data = toy["val"]
    else:
        train_data = load_split_data(data_dir / "train_data.pkl")
        val_data = load_split_data(data_dir / "val_data.pkl")

    logger.info("Train: %d essays, Val: %d essays", len(train_data["essay_ids"]), len(val_data["essay_ids"]))

    # Auto-set theme weights from split stats if not in config
    if config.loss.theme_weights is None:
        stats_path = data_dir / "splits_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            tw = stats.get("theme_weights", {})
            if tw:
                from config import THEMES
                config.loss.theme_weights = [tw.get(t, 1.0) for t in THEMES]
                logger.info("Auto-loaded theme weights from splits_stats.json: %s", config.loss.theme_weights)

    # Create data loaders
    train_loader = create_dataloader(
        train_data, tokenizer, config,
        shuffle=True, augment=config.augmentation.enabled,
        use_weighted_sampling=True,
    )
    val_loader = create_dataloader(
        val_data, tokenizer, config,
        shuffle=False, augment=False,
    )

    # Build model
    model = build_model_from_config(config, freeze_encoder=False)

    # Train
    trainer = AWARETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        config=config,
        device=args.device,
    )
    history = trainer.train()

    logger.info("Training finished. Best F1: %.4f at epoch %d", history["best_f1"], history["best_epoch"])
    return history


if __name__ == "__main__":
    main()
