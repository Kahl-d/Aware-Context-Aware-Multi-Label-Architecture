"""
dapt.py — Domain Adaptive Pre-Training (DAPT) for DeBERTa-v3.

MLM pre-training on all available essay text (annotated + unannotated)
before fine-tuning. Following Gururangan et al. 2020.

Usage:
    # Local toy test
    python scripts/dapt.py --corpus data/dapt_corpus.txt --output_dir results/dapt/ --toy

    # Full HPC DAPT
    python scripts/dapt.py --corpus data/dapt_corpus.txt --output_dir results/dapt/ \
        --config configs/full.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

from config import AWAREConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_corpus(path: str, max_essays: int = None) -> list:
    """Load DAPT corpus (one essay per double-newline-separated block)."""
    with open(path) as f:
        text = f.read()
    essays = [e.strip() for e in text.split("\n\n") if e.strip()]
    if max_essays:
        essays = essays[:max_essays]
    logger.info("Loaded DAPT corpus: %d essays, %.1f MB", len(essays), sum(len(e) for e in essays) / 1e6)
    return essays


def main():
    parser = argparse.ArgumentParser(description="DAPT for AWARE v2")
    parser.add_argument("--corpus", type=str, required=True, help="Path to dapt_corpus.txt")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir for DAPT encoder")
    parser.add_argument("--config", type=str, default=None, help="YAML config (optional, uses defaults)")
    parser.add_argument("--encoder_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--toy", action="store_true", help="Toy mode: 100 essays, 1 epoch")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    if args.config:
        config = AWAREConfig.from_yaml(args.config)
        dapt_cfg = config.dapt
        encoder_name = config.model.encoder_name
    else:
        config = AWAREConfig()
        dapt_cfg = config.dapt
        encoder_name = args.encoder_name

    # Toy overrides
    if args.toy:
        dapt_cfg.epochs = 1
        dapt_cfg.batch_size = 4

    # Load corpus
    max_essays = 100 if args.toy else None
    essays = load_corpus(args.corpus, max_essays=max_essays)

    # Split eval
    n_eval = max(1, int(len(essays) * dapt_cfg.eval_split))
    train_essays = essays[n_eval:]
    eval_essays = essays[:n_eval]
    logger.info("DAPT train: %d essays, eval: %d essays", len(train_essays), len(eval_essays))

    # Tokenizer + model
    logger.info("Loading encoder: %s", encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModelForMaskedLM.from_pretrained(encoder_name)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=512,
            truncation=True,
            padding=False,
        )

    train_dataset = Dataset.from_dict({"text": train_essays}).map(
        tokenize_fn, batched=True, remove_columns=["text"],
    )
    eval_dataset = Dataset.from_dict({"text": eval_essays}).map(
        tokenize_fn, batched=True, remove_columns=["text"],
    )

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=dapt_cfg.mlm_probability,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=dapt_cfg.epochs,
        per_device_train_batch_size=dapt_cfg.batch_size,
        per_device_eval_batch_size=dapt_cfg.batch_size,
        learning_rate=dapt_cfg.learning_rate,
        warmup_ratio=dapt_cfg.warmup_ratio,
        weight_decay=getattr(dapt_cfg, 'weight_decay', 0.01),
        fp16=dapt_cfg.fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting DAPT...")
    train_result = trainer.train()
    logger.info("DAPT complete. Train loss: %.4f", train_result.training_loss)

    # Evaluate
    eval_result = trainer.evaluate()
    import math
    perplexity = math.exp(eval_result["eval_loss"])
    logger.info("DAPT eval loss: %.4f, perplexity: %.2f", eval_result["eval_loss"], perplexity)

    # Save encoder (not the MLM head — just the base model for downstream)
    encoder_dir = output_dir / "encoder"
    encoder_dir.mkdir(exist_ok=True)
    model.base_model.save_pretrained(str(encoder_dir))
    tokenizer.save_pretrained(str(encoder_dir))
    logger.info("Saved DAPT encoder to %s", encoder_dir)

    # Save report
    import json
    report = {
        "encoder_name": encoder_name,
        "corpus_size": len(essays),
        "train_size": len(train_essays),
        "eval_size": len(eval_essays),
        "epochs": dapt_cfg.epochs,
        "batch_size": dapt_cfg.batch_size,
        "learning_rate": dapt_cfg.learning_rate,
        "mlm_probability": dapt_cfg.mlm_probability,
        "train_loss": round(train_result.training_loss, 4),
        "eval_loss": round(eval_result["eval_loss"], 4),
        "perplexity": round(perplexity, 2),
        "encoder_path": str(encoder_dir),
    }
    with open(output_dir / "dapt_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("DAPT report saved")


if __name__ == "__main__":
    main()
