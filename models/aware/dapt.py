"""
dapt.py — Domain Adaptive Pre-Training (DAPT) for DeBERTa-v3.

MLM pre-training on essay text before fine-tuning.
Proven +0.040 macro F1 in Q13.

Usage:
    python scripts/dapt.py --corpus data/dapt_corpus.txt --output_dir results/q001/dapt/
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
    TrainingArguments, Trainer,
)
from datasets import Dataset

from config import AWAREConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DAPT for AWARE v3")
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--encoder_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config:
        config = AWAREConfig.from_yaml(args.config)
        dapt_cfg = config.dapt
        encoder_name = config.model.encoder_name
        seed = config.seed
    else:
        config = AWAREConfig()
        dapt_cfg = config.dapt
        encoder_name = args.encoder_name
        seed = 42

    if args.toy:
        dapt_cfg.epochs = 1
        dapt_cfg.batch_size = 4

    # Load corpus
    with open(args.corpus) as f:
        text = f.read()
    essays = [e.strip() for e in text.split("\n\n") if e.strip()]
    if args.toy:
        essays = essays[:100]
    logger.info("DAPT corpus: %d essays, %.1f KB", len(essays), sum(len(e) for e in essays) / 1e3)

    # Split
    rng = random.Random(seed)
    shuffled = essays.copy()
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * dapt_cfg.eval_split))
    eval_essays, train_essays = shuffled[:n_eval], shuffled[n_eval:]

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModelForMaskedLM.from_pretrained(encoder_name)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], max_length=512, truncation=True, padding=False)

    train_ds = Dataset.from_dict({"text": train_essays}).map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds = Dataset.from_dict({"text": eval_essays}).map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=dapt_cfg.mlm_probability)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=dapt_cfg.epochs,
        per_device_train_batch_size=dapt_cfg.batch_size,
        per_device_eval_batch_size=dapt_cfg.batch_size,
        learning_rate=dapt_cfg.learning_rate,
        warmup_ratio=dapt_cfg.warmup_ratio,
        weight_decay=dapt_cfg.weight_decay,
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
        seed=seed,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
    )

    logger.info("Starting DAPT...")
    result = trainer.train()
    logger.info("DAPT done. Loss: %.4f", result.training_loss)

    eval_result = trainer.evaluate()
    ppl = math.exp(eval_result["eval_loss"])
    logger.info("DAPT eval loss: %.4f, perplexity: %.2f", eval_result["eval_loss"], ppl)

    # Save encoder only (not MLM head)
    encoder_dir = output_dir / "encoder"
    encoder_dir.mkdir(exist_ok=True)
    model.base_model.save_pretrained(str(encoder_dir))
    tokenizer.save_pretrained(str(encoder_dir))
    logger.info("Saved DAPT encoder → %s", encoder_dir)

    report = {
        "encoder_name": encoder_name,
        "corpus_essays": len(essays),
        "train_size": len(train_essays),
        "eval_size": len(eval_essays),
        "epochs": dapt_cfg.epochs,
        "train_loss": round(result.training_loss, 4),
        "eval_loss": round(eval_result["eval_loss"], 4),
        "perplexity": round(ppl, 2),
    }
    with open(output_dir / "dapt_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
