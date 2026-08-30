"""
evaluate.py — Full evaluation of trained AWARE v3 model.

Loads best checkpoint + calibration + thresholds, evaluates on specified split.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from config import AWAREConfig, THEMES, NUM_THEMES
from model import build_model_from_config
from dataset import load_split_data, create_dataloader
from metrics import (
    flatten_masked_logits_labels,
    compute_metrics, compute_prauc, compute_rocauc,
    compute_hamming_loss, compute_exact_match_ratio,
    apply_thresholds, bootstrap_ci,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, thresholds, device, calibration=None):
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            smask = batch["sentence_mask"].to(device)
            labels = batch["labels"].to(device)
            bounds = batch["sentence_boundaries"]
            out = model(ids, mask, bounds, smask)
            fl, ll = flatten_masked_logits_labels(out["logits"], labels, smask)
            all_logits.append(fl)
            all_labels.append(ll)

    logits_np = torch.cat(all_logits).cpu().numpy()
    labels_np = torch.cat(all_labels).cpu().numpy()

    # Apply Platt scaling if available
    if calibration:
        a = np.array([calibration.get(THEMES[i], {}).get("a", 1.0) for i in range(NUM_THEMES)])
        b = np.array([calibration.get(THEMES[i], {}).get("b", 0.0) for i in range(NUM_THEMES)])
        probs = 1.0 / (1.0 + np.exp(-(a[np.newaxis, :] * logits_np + b[np.newaxis, :])))
    else:
        probs = 1.0 / (1.0 + np.exp(-logits_np))

    preds = apply_thresholds(probs, thresholds)
    metrics = compute_metrics(preds, labels_np)

    prauc = compute_prauc(probs, labels_np)
    metrics["prauc_macro"] = prauc["prauc_macro"]
    metrics["prauc_per_theme"] = prauc["prauc_per_theme"]

    rocauc = compute_rocauc(probs, labels_np)
    metrics["rocauc_macro"] = rocauc["rocauc_macro"]
    metrics["rocauc_per_theme"] = rocauc["rocauc_per_theme"]

    metrics["hamming_loss"] = compute_hamming_loss(preds, labels_np)
    metrics["exact_match_ratio"] = compute_exact_match_ratio(preds, labels_np)

    # Default threshold comparison
    preds_default = (probs >= 0.5).astype(np.float32)
    metrics_default = compute_metrics(preds_default, labels_np)

    ci = bootstrap_ci(preds, labels_np, n_bootstrap=1000)

    return {
        "metrics_optimized": metrics,
        "metrics_default_05": metrics_default,
        "thresholds_used": thresholds,
        "bootstrap_ci": ci,
        "n_sentences": int(probs.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Local encoder path (avoids HuggingFace download; weights are "
                             "overwritten by best.pt anyway)")
    args = parser.parse_args()

    config = AWAREConfig.from_yaml(args.config)
    if args.encoder_path:
        config.model.encoder_path = args.encoder_path
    results_dir = Path(args.results_dir)

    model = build_model_from_config(config)
    best_path = results_dir / "best.pt"
    if not best_path.exists():
        logger.error("No best.pt in %s", results_dir)
        sys.exit(1)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model = model.to(device)

    # Load thresholds
    thresh_path = results_dir / "thresholds.json"
    thresholds = json.load(open(thresh_path)) if thresh_path.exists() else {t: 0.3 for t in THEMES}

    # Load calibration
    cal_path = results_dir / "calibration.json"
    calibration = json.load(open(cal_path)) if cal_path.exists() else None

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
    data = load_split_data(Path(args.data_dir) / f"{args.split}_data.pkl")
    loader = create_dataloader(data, tokenizer, config, shuffle=False, augment=False)

    results = evaluate_model(model, loader, thresholds, device, calibration)

    eval_path = results_dir / f"evaluation_{args.split}.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print
    m = results["metrics_optimized"]
    ci = results["bootstrap_ci"]
    print("\n" + "=" * 70)
    print(f"EVALUATION ({args.split.upper()}) — {results['n_sentences']} sentences")
    print("=" * 70)
    print(f"Macro F1 (optimized):  {m['f1_macro']:.4f}  [{ci['f1_macro'][1]:.4f}, {ci['f1_macro'][2]:.4f}]")
    print(f"Macro F1 (default@0.5): {results['metrics_default_05']['f1_macro']:.4f}")
    print(f"PR-AUC (macro):         {m.get('prauc_macro', 0):.4f}")
    print(f"ROC-AUC (macro):        {m.get('rocauc_macro', 0):.4f}")
    print(f"Hamming loss:           {m.get('hamming_loss', 0):.4f}")
    print(f"Exact match:            {m.get('exact_match_ratio', 0):.4f}")
    print(f"\n{'Theme':<20} {'F1':>6} {'P':>6} {'R':>6} {'PRAUC':>6} {'ROCAUC':>6} {'Sup':>5} {'Thr':>5}")
    print("-" * 62)
    for theme in THEMES:
        print(f"{theme:<20} {m['f1_per_theme'].get(theme,0):>6.4f} "
              f"{m['precision_per_theme'].get(theme,0):>6.4f} "
              f"{m['recall_per_theme'].get(theme,0):>6.4f} "
              f"{m.get('prauc_per_theme',{}).get(theme,0):>6.4f} "
              f"{m.get('rocauc_per_theme',{}).get(theme,0):>6.4f} "
              f"{m['support_per_theme'].get(theme,0):>5} "
              f"{thresholds.get(theme,0.5):>5.3f}")
    c0 = m.get("class_0", {})
    if c0:
        print(f"{'class_0':<20} {c0['f1']:>6.4f} {c0['precision']:>6.4f} {c0['recall']:>6.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
