"""
evaluate.py — Full evaluation of a trained AWARE v2 model.

Loads best checkpoint + optimized thresholds, evaluates on test set.
Produces detailed per-theme report with bootstrap confidence intervals.

Usage:
    python scripts/evaluate.py --config configs/full.yaml --data_dir data/ --results_dir results/full/
"""

import argparse
import json
import logging
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from config import AWAREConfig, THEMES, THEME_TO_IDX, NUM_THEMES
from model import build_model_from_config
from dataset import load_split_data, create_dataloader
from metrics import (
    flatten_masked_preds_labels,
    flatten_masked_logits_labels,
    compute_metrics,
    compute_prauc,
    compute_hamming_loss,
    compute_exact_match_ratio,
    apply_thresholds,
    bootstrap_ci,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, thresholds, device) -> dict:
    """Run evaluation, collect all predictions."""
    model.eval()
    all_logits_list, all_labels_list = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentence_mask = batch["sentence_mask"].to(device)
            labels = batch["labels"].to(device)
            sentence_boundaries = batch["sentence_boundaries"]

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_boundaries=sentence_boundaries,
                sentence_mask=sentence_mask,
            )
            flat_logits, flat_labels = flatten_masked_logits_labels(
                output["logits"], labels, sentence_mask,
            )
            all_logits_list.append(flat_logits)
            all_labels_list.append(flat_labels)

    all_logits = torch.cat(all_logits_list, dim=0)
    all_labels = torch.cat(all_labels_list, dim=0)
    all_probs = torch.sigmoid(all_logits).cpu().numpy()
    all_labels_np = all_labels.cpu().numpy()

    # Apply per-theme thresholds
    preds = apply_thresholds(all_probs, thresholds)

    # Compute metrics
    metrics = compute_metrics(preds, all_labels_np)

    # PR-AUC (threshold-independent)
    prauc = compute_prauc(all_probs, all_labels_np)
    metrics["prauc_macro"] = prauc["prauc_macro"]
    metrics["prauc_per_theme"] = prauc["prauc_per_theme"]

    # Hamming loss and exact match
    metrics["hamming_loss"] = compute_hamming_loss(preds, all_labels_np)
    metrics["exact_match_ratio"] = compute_exact_match_ratio(preds, all_labels_np)

    # Also compute with default 0.5 threshold for comparison
    preds_default = (all_probs >= 0.5).astype(np.float32)
    metrics_default = compute_metrics(preds_default, all_labels_np)

    # Bootstrap CI
    ci = bootstrap_ci(preds, all_labels_np, n_bootstrap=1000)

    # Confusion analysis: which themes get confused
    confusion = {}
    for i, theme_i in enumerate(THEMES):
        false_pos_themes = {}
        for j, theme_j in enumerate(THEMES):
            if i == j:
                continue
            # Cases where model predicted theme_i but truth had theme_j (not theme_i)
            fp_with_j = ((preds[:, i] == 1) & (all_labels_np[:, i] == 0) & (all_labels_np[:, j] == 1)).sum()
            if fp_with_j > 0:
                false_pos_themes[theme_j] = int(fp_with_j)
        if false_pos_themes:
            confusion[theme_i] = dict(sorted(false_pos_themes.items(), key=lambda x: -x[1])[:5])

    return {
        "metrics_optimized_threshold": metrics,
        "metrics_default_threshold": metrics_default,
        "thresholds_used": thresholds,
        "bootstrap_ci": ci,
        "confusion_analysis": confusion,
        "n_sentences": int(all_probs.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AWARE v2")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = AWAREConfig.from_yaml(args.config)
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    # Load model
    model = build_model_from_config(config)
    best_path = results_dir / "best.pt"
    if not best_path.exists():
        logger.error("No best.pt found in %s", results_dir)
        sys.exit(1)

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model = model.to(device)
    logger.info("Loaded model from %s", best_path)

    # Load thresholds
    thresh_path = results_dir / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresholds = json.load(f)
        logger.info("Loaded thresholds: %s", thresholds)
    else:
        thresholds = {t: 0.30 for t in THEMES}
        logger.warning("No thresholds.json found, using default 0.30")

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
    split_data = load_split_data(data_dir / f"{args.split}_data.pkl")
    dataloader = create_dataloader(split_data, tokenizer, config, shuffle=False, augment=False)

    # Evaluate
    results = evaluate_model(model, dataloader, thresholds, device)

    # Save
    eval_path = results_dir / f"evaluation_{args.split}.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved evaluation to %s", eval_path)

    # Print summary
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS ({args.split.upper()} SET)")
    print("=" * 70)
    m = results["metrics_optimized_threshold"]
    md = results["metrics_default_threshold"]
    ci = results["bootstrap_ci"]
    print(f"\nMacro F1 (optimized thresholds): {m['f1_macro']:.4f}  [{ci['f1_macro'][1]:.4f}, {ci['f1_macro'][2]:.4f}]")
    print(f"Macro F1 (default 0.5):           {md['f1_macro']:.4f}")
    print(f"Micro F1 (optimized):             {m['f1_micro']:.4f}")
    print(f"Precision (macro):                {m['precision_macro']:.4f}")
    print(f"Recall (macro):                   {m['recall_macro']:.4f}")
    print(f"PR-AUC (macro):                   {m.get('prauc_macro', 0):.4f}")
    print(f"Hamming loss:                     {m.get('hamming_loss', 0):.4f}")
    print(f"Exact match ratio:                {m.get('exact_match_ratio', 0):.4f}")
    print(f"\nPer-theme F1 + PR-AUC (optimized thresholds):")
    print(f"  {'Theme':<25} {'F1':>6} {'P':>6} {'R':>6} {'PRAUC':>6} {'Support':>8} {'Threshold':>10}")
    print(f"  {'-' * 75}")
    for theme in THEMES:
        f1 = m["f1_per_theme"].get(theme, 0)
        p = m["precision_per_theme"].get(theme, 0)
        r = m["recall_per_theme"].get(theme, 0)
        ap = m.get("prauc_per_theme", {}).get(theme, 0)
        sup = m["support_per_theme"].get(theme, 0)
        t = thresholds.get(theme, 0.5)
        print(f"  {theme:<25} {f1:>6.4f} {p:>6.4f} {r:>6.4f} {ap:>6.4f} {sup:>8} {t:>10.3f}")
    # Class_0 (no theme) metrics
    c0 = m.get("class_0", {})
    if c0:
        print(f"  {'-' * 65}")
        print(f"  {'class_0 (no theme)':<25} {c0.get('f1',0):>6.4f} {c0.get('precision',0):>6.4f} {c0.get('recall',0):>6.4f} {c0.get('support',0):>8} {'—':>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
