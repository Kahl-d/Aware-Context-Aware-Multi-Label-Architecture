"""
run_pipeline.py — Full AWARE v2 pipeline orchestration.

Runs: [DAPT →] Train → Evaluate (val + test + train)

Usage:
    # Local toy (no DAPT)
    python scripts/run_pipeline.py --mode toy --no-dapt --data_dir data/ --output_dir results/toy/

    # Local toy with DAPT
    python scripts/run_pipeline.py --mode toy --data_dir data/ --output_dir results/toy/

    # Full HPC (DAPT + Train + Eval)
    python scripts/run_pipeline.py --mode full --config configs/full.yaml \
        --data_dir data/ --output_dir results/full/
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_step(cmd: list, step_name: str):
    """Run a pipeline step as subprocess. Exits on failure."""
    logger.info("=" * 60)
    logger.info("PIPELINE STEP: %s", step_name)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("=" * 60)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("Step '%s' FAILED (exit code %d)", step_name, result.returncode)
        sys.exit(result.returncode)
    logger.info("Step '%s' completed successfully", step_name)


def main():
    parser = argparse.ArgumentParser(description="AWARE v2 Pipeline")
    parser.add_argument("--mode", type=str, default="toy", choices=["toy", "full"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no-dapt", action="store_true", help="Skip DAPT step")
    parser.add_argument("--python", type=str, default="python3", help="Python interpreter")
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── File logging (pipeline.log in output dir) ────────────────────────────
    log_path = output_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info("Pipeline log: %s", log_path)

    config_path = args.config
    if config_path is None:
        config_path = str(scripts_dir.parent / "configs" / f"{args.mode}.yaml")

    py = args.python
    is_toy = args.mode == "toy"

    dapt_encoder_path = None

    # ── Step 1: DAPT ─────────────────────────────────────────────────────────
    if not args.no_dapt:
        dapt_dir = str(output_dir / "dapt")
        cmd = [py, str(scripts_dir / "dapt.py"),
               "--corpus", str(data_dir / "dapt_corpus.txt"),
               "--output_dir", dapt_dir,
               "--config", config_path]
        if is_toy:
            cmd.append("--toy")
        run_step(cmd, "DAPT — Domain Adaptive Pre-Training")

        # The DAPT encoder is what we hand off to training
        dapt_encoder_path = str(Path(dapt_dir) / "encoder")
        logger.info("DAPT encoder ready: %s", dapt_encoder_path)
    else:
        logger.info("Skipping DAPT (--no-dapt flag set)")

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    cmd = [py, str(scripts_dir / "train.py"),
           "--config", config_path,
           "--data_dir", str(data_dir),
           "--output_dir", str(output_dir)]
    if dapt_encoder_path is not None:
        # Critical: use the domain-adapted encoder, not the base DeBERTa
        cmd.extend(["--encoder_path", dapt_encoder_path])
        logger.info("Training will use DAPT encoder: %s", dapt_encoder_path)
    else:
        logger.info("Training will use base encoder from config (no DAPT)")
    if is_toy:
        cmd.append("--toy")
    run_step(cmd, "Training (AWARE v2 — two-phase fine-tune)")

    # ── Steps 3-5: Evaluation (skip in toy mode) ──────────────────────────────
    if not is_toy:
        # Val set — used to select best model and optimize thresholds
        cmd = [py, str(scripts_dir / "evaluate.py"),
               "--config", config_path,
               "--data_dir", str(data_dir),
               "--results_dir", str(output_dir),
               "--split", "val"]
        run_step(cmd, "Evaluation — Validation Set")

        # Test set — MAIN THESIS RESULTS
        cmd = [py, str(scripts_dir / "evaluate.py"),
               "--config", config_path,
               "--data_dir", str(data_dir),
               "--results_dir", str(output_dir),
               "--split", "test"]
        run_step(cmd, "Evaluation — Test Set (MAIN RESULTS)")

        # Train set — overfitting diagnostic
        cmd = [py, str(scripts_dir / "evaluate.py"),
               "--config", config_path,
               "--data_dir", str(data_dir),
               "--results_dir", str(output_dir),
               "--split", "train"]
        run_step(cmd, "Evaluation — Train Set (overfitting diagnostic)")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Results saved to: %s", output_dir)
    logger.info("=" * 60)

    # ── Final summary: print readable results to stdout ───────────────────────
    if not is_toy:
        _print_final_summary(output_dir)


def _print_final_summary(output_dir: Path):
    """Print a human-readable results table to stdout after all evaluations."""
    import json as _json

    def _load(name):
        p = output_dir / name
        if p.exists():
            with open(p) as f:
                return _json.load(f)
        return None

    history  = _load("history.json")
    test_res = _load("evaluation_test.json")
    train_res = _load("evaluation_train.json")
    val_res  = _load("evaluation_val.json")

    sep = "=" * 68

    # ── Training curve ────────────────────────────────────────────────────────
    print("\n" + sep)
    print("  TRAINING CURVE")
    print(sep)
    if history:
        print(f"  {'Epoch':>5}  {'Phase':>5}  {'Train Loss':>11}  {'Val F1 Macro':>13}")
        print("  " + "-" * 44)
        for e in history["epochs"]:
            marker = " ← BEST" if e["epoch"] == history.get("best_epoch") else ""
            print(f"  {e['epoch']:>5}  {e['phase']:>5}  {e['train_loss']:>11.4f}  "
                  f"{e['val_f1_macro']:>13.4f}{marker}")
        print(f"\n  Best val F1 macro: {history['best_f1']:.4f}  (epoch {history['best_epoch']})")

    # ── Per-theme test results ────────────────────────────────────────────────
    print("\n" + sep)
    print("  TEST SET RESULTS  (main thesis numbers)")
    print(sep)
    if test_res:
        tm = test_res.get("metrics_optimized_threshold", {})
        tr = (train_res or {}).get("metrics_optimized_threshold", {})
        thresh = _load("thresholds.json") or {}

        test_f1 = tm.get("f1_per_theme", {})
        test_pre = tm.get("precision_per_theme", {})
        test_rec = tm.get("recall_per_theme", {})
        train_f1 = tr.get("f1_per_theme", {})

        print(f"  {'Theme':<28} {'Test F1':>8} {'Prec':>6} {'Rec':>6} "
              f"{'Train F1':>9} {'Gap':>7} {'Thresh':>7}")
        print("  " + "-" * 68)
        for theme in test_f1:
            tf  = test_f1.get(theme, 0.0)
            pre = test_pre.get(theme, 0.0)
            rec = test_rec.get(theme, 0.0)
            trf = train_f1.get(theme, 0.0)
            gap = trf - tf
            t   = thresh.get(theme, 0.5)
            flag = ""
            if tf == 0.0:
                flag = " ← ZERO"
            elif gap > 0.20:
                flag = " ← overfit"
            elif gap > 0.12:
                flag = " ← mild overfit"
            print(f"  {theme:<28} {tf:>8.4f} {pre:>6.4f} {rec:>6.4f} "
                  f"{trf:>9.4f} {gap:>+7.4f} {t:>7.3f}{flag}")
        print("  " + "-" * 68)
        print(f"  {'MACRO':<28} {tm.get('f1_macro', 0):>8.4f} "
              f"{tm.get('precision_macro', 0):>6.4f} {tm.get('recall_macro', 0):>6.4f}")
        print(f"  {'MICRO':<28} {tm.get('f1_micro', 0):>8.4f}")

    # ── Val macro comparison ──────────────────────────────────────────────────
    if val_res:
        vm = val_res.get("metrics_optimized_threshold", {})
        print(f"\n  Val  macro F1 (threshold-optimized): {vm.get('f1_macro', 0):.4f}")
    if test_res:
        print(f"  Test macro F1 (threshold-optimized): "
              f"{test_res.get('metrics_optimized_threshold', {}).get('f1_macro', 0):.4f}")

    print("\n" + sep + "\n")


if __name__ == "__main__":
    main()
