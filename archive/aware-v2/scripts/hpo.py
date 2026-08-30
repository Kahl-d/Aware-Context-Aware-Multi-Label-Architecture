"""
hpo.py -- Optuna-based hyperparameter optimization for AWARE v2.

Runs a shortened training loop (8 total epochs: 2 phase1 + 6 phase2) per trial,
reports intermediate val macro F1 for pruning, and maximizes val macro F1.

Uses SQLite storage so the study persists across SLURM jobs -- you can run
multiple sequential jobs that each add trials to the same study.

Usage:
    # From cModels/ directory:
    python scripts/hpo.py \
        --config configs/quick.yaml \
        --data_dir data/ \
        --output_dir results/hpo/ \
        --n_trials 25

    # With DAPT encoder:
    python scripts/hpo.py \
        --config configs/quick.yaml \
        --data_dir data/ \
        --output_dir results/hpo/ \
        --n_trials 25 \
        --encoder_path results/quick_002/dapt_encoder/

    # Resume a previous study (same output_dir picks up the SQLite DB):
    python scripts/hpo.py \
        --config configs/quick.yaml \
        --data_dir data/ \
        --output_dir results/hpo/ \
        --n_trials 10
"""

import argparse
import json
import logging
import sys
import gc
from pathlib import Path

import numpy as np
import torch
import random
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from config import AWAREConfig, THEMES
from model import build_model_from_config
from dataset import load_split_data, create_dataloader
from losses import build_loss_from_config
from metrics import flatten_masked_preds_labels, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HPO epoch counts -- kept short so each trial finishes quickly.
# Phase 1 (frozen encoder): 2 epochs -- just enough to warm up BiLSTM + head.
# Phase 2 (full fine-tune):  6 epochs -- enough to see if config is viable.
# ---------------------------------------------------------------------------
HPO_PHASE1_EPOCHS = 2
HPO_PHASE2_EPOCHS = 6


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config_for_trial(
    base_config: AWAREConfig,
    trial: optuna.Trial,
    encoder_path: str = None,
) -> AWAREConfig:
    """Create a config with Optuna-suggested hyperparameters.

    The base config provides all non-tuned values (model architecture, loss
    weights, augmentation settings, etc.). Only the six key hyperparameters
    are overridden by Optuna suggestions.

    Args:
        base_config: AWAREConfig loaded from the base YAML file.
        trial: Optuna trial for suggesting parameters.
        encoder_path: Optional path to a DAPT-adapted encoder.

    Returns:
        A new AWAREConfig with suggested hyperparameters and shortened epochs.
    """
    # Suggest hyperparameters
    encoder_lr = trial.suggest_float("encoder_lr", 5e-6, 5e-5, log=True)
    decoder_lr = trial.suggest_float("decoder_lr", 5e-5, 5e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    asl_gamma_neg = trial.suggest_float("asl_gamma_neg", 2.0, 7.0)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    phase2_encoder_lr_scale = trial.suggest_float(
        "phase2_encoder_lr_scale", 0.03, 0.2, log=True
    )

    # Deep-copy the base config by round-tripping through dict
    config = AWAREConfig._from_dict(base_config.to_dict())

    # Override with suggested values
    config.training.encoder_lr = encoder_lr
    config.training.decoder_lr = decoder_lr
    config.model.dropout = dropout
    config.loss.asl_gamma_neg = asl_gamma_neg
    config.training.weight_decay = weight_decay
    config.training.phase2_encoder_lr_scale = phase2_encoder_lr_scale

    # Shortened training for HPO
    config.training.phase1_epochs = HPO_PHASE1_EPOCHS
    config.training.phase2_epochs = HPO_PHASE2_EPOCHS
    # Disable early stopping during HPO -- we use Optuna pruning instead
    config.training.early_stopping_patience = 999
    config.training.phase2_early_stopping_patience = 999

    # Apply DAPT encoder if provided
    if encoder_path:
        config.model.encoder_path = encoder_path

    return config


def detect_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    gradient_accumulation: int,
    max_grad_norm: float,
    use_fp16: bool,
    scaler,
    epoch: int,
    phase: int,
) -> float:
    """Run one training epoch. Returns average loss.

    This is a simplified version of AWARETrainer._train_epoch(), stripped down
    for HPO where we don't need progress bars or detailed step-level logging.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    accum = max(1, gradient_accumulation)

    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sentence_mask = batch["sentence_mask"].to(device)
        labels = batch["labels"].to(device)
        sentence_boundaries = batch["sentence_boundaries"]

        if use_fp16:
            with torch.amp.autocast("cuda"):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentence_boundaries=sentence_boundaries,
                    sentence_mask=sentence_mask,
                )
                loss = criterion(
                    logits=output["logits"],
                    targets=labels,
                    mask=sentence_mask,
                )
            loss = loss / accum
            scaler.scale(loss).backward()
        else:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_boundaries=sentence_boundaries,
                sentence_mask=sentence_mask,
            )
            loss = criterion(
                logits=output["logits"],
                targets=labels,
                mask=sentence_mask,
            )
            loss = loss / accum
            loss.backward()

        total_loss += loss.item() * accum
        n_batches += 1

        if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
            if use_fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Run validation. Returns dict with f1_macro, f1_micro, f1_per_theme, loss.

    Mirrors AWARETrainer._validate() but without probability diagnostics.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
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
        loss = criterion(
            logits=output["logits"],
            targets=labels,
            mask=sentence_mask,
        )
        total_loss += loss.item()
        n_batches += 1

        p, l = flatten_masked_preds_labels(output["logits"], labels, sentence_mask)
        all_preds.append(p)
        all_labels.append(l)

    if all_preds:
        preds = np.vstack(all_preds)
        lab = np.vstack(all_labels)
        metrics = compute_metrics(preds, lab)
    else:
        metrics = {"f1_macro": 0.0, "f1_micro": 0.0, "f1_per_theme": {}}

    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def objective(
    trial: optuna.Trial,
    base_config: AWAREConfig,
    train_data: dict,
    val_data: dict,
    tokenizer,
    output_dir: Path,
    encoder_path: str = None,
) -> float:
    """Optuna objective: train AWARE v2 with suggested hyperparams, return val macro F1.

    Runs a two-phase training (2 phase1 + 6 phase2 = 8 total epochs).
    Reports intermediate val F1 after each epoch for Optuna's median pruner.
    Each trial's outputs are saved to output_dir/trial_NNN/.

    Args:
        trial: Optuna trial object.
        base_config: Base AWAREConfig to override.
        train_data: Training split data dict.
        val_data: Validation split data dict.
        tokenizer: Pre-loaded tokenizer.
        output_dir: Root HPO output directory.
        encoder_path: Optional path to DAPT encoder.

    Returns:
        Best val macro F1 achieved during the trial.
    """
    trial_dir = output_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Build config with suggested hyperparameters
    config = make_config_for_trial(base_config, trial, encoder_path=encoder_path)
    set_seeds(config.seed)

    # Save trial config
    trial_config = config.to_dict()
    trial_config["hpo_params"] = {
        "encoder_lr": config.training.encoder_lr,
        "decoder_lr": config.training.decoder_lr,
        "dropout": config.model.dropout,
        "asl_gamma_neg": config.loss.asl_gamma_neg,
        "weight_decay": config.training.weight_decay,
        "phase2_encoder_lr_scale": config.training.phase2_encoder_lr_scale,
    }
    with open(trial_dir / "config.json", "w") as f:
        json.dump(trial_config, f, indent=2)

    logger.info(
        "Trial %d: encoder_lr=%.2e, decoder_lr=%.2e, dropout=%.3f, "
        "asl_gamma_neg=%.2f, weight_decay=%.4f, phase2_scale=%.4f",
        trial.number,
        config.training.encoder_lr,
        config.training.decoder_lr,
        config.model.dropout,
        config.loss.asl_gamma_neg,
        config.training.weight_decay,
        config.training.phase2_encoder_lr_scale,
    )

    # Setup
    device = detect_device()
    use_fp16 = bool(config.training.fp16) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    accum = max(1, int(config.training.gradient_accumulation))
    max_grad_norm = float(config.training.max_grad_norm)
    warmup_ratio = float(config.training.warmup_ratio)

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

    # Build model and loss
    model = build_model_from_config(config, freeze_encoder=False)
    model = model.to(device)
    criterion = build_loss_from_config(config).to(device)

    best_f1 = 0.0
    history = []
    global_epoch = 0  # Sequential epoch counter for Optuna reporting (0-indexed)

    # ── Phase 1: Frozen encoder -- train BiLSTM + classification head ──
    model._freeze_encoder()
    phase1_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        phase1_params,
        lr=float(config.training.decoder_lr),
        weight_decay=float(config.training.weight_decay),
    )
    steps_per_epoch = max(1, (len(train_loader) + accum - 1) // accum)
    total_steps = steps_per_epoch * HPO_PHASE1_EPOCHS
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    for epoch in range(1, HPO_PHASE1_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, accum, max_grad_norm, use_fp16, scaler,
            epoch=epoch, phase=1,
        )
        val_metrics = validate(model, val_loader, criterion, device)
        val_f1 = val_metrics["f1_macro"]
        if val_f1 > best_f1:
            best_f1 = val_f1

        history.append({
            "epoch": epoch,
            "phase": 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_f1_macro": round(val_f1, 6),
            "val_f1_per_theme": val_metrics.get("f1_per_theme", {}),
        })

        logger.info(
            "Trial %d | P1 epoch %d: train_loss=%.4f, val_f1_macro=%.4f",
            trial.number, epoch, train_loss, val_f1,
        )

        # Report to Optuna for pruning
        trial.report(val_f1, global_epoch)
        global_epoch += 1
        if trial.should_prune():
            logger.info("Trial %d pruned at phase 1 epoch %d", trial.number, epoch)
            _save_trial_results(trial_dir, history, best_f1, trial, pruned=True)
            _cleanup_gpu(model, optimizer, criterion, scaler)
            raise optuna.TrialPruned()

    # ── Phase 2: Unfreeze encoder -- full fine-tune with differential LR ──
    model.unfreeze_encoder()
    encoder_lr_phase2 = float(config.training.encoder_lr) * float(
        config.training.phase2_encoder_lr_scale
    )
    encoder_params = [p for n, p in model.named_parameters() if "encoder" in n]
    other_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr_phase2},
            {"params": other_params, "lr": float(config.training.decoder_lr)},
        ],
        weight_decay=float(config.training.weight_decay),
    )
    steps_per_epoch_p2 = max(1, (len(train_loader) + accum - 1) // accum)
    total_steps_p2 = steps_per_epoch_p2 * HPO_PHASE2_EPOCHS
    warmup_steps_p2 = max(1, int(total_steps_p2 * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps_p2, num_training_steps=total_steps_p2,
    )

    for epoch in range(HPO_PHASE1_EPOCHS + 1, HPO_PHASE1_EPOCHS + HPO_PHASE2_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, accum, max_grad_norm, use_fp16, scaler,
            epoch=epoch, phase=2,
        )
        val_metrics = validate(model, val_loader, criterion, device)
        val_f1 = val_metrics["f1_macro"]
        if val_f1 > best_f1:
            best_f1 = val_f1

        history.append({
            "epoch": epoch,
            "phase": 2,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_f1_macro": round(val_f1, 6),
            "val_f1_per_theme": val_metrics.get("f1_per_theme", {}),
        })

        logger.info(
            "Trial %d | P2 epoch %d: train_loss=%.4f, val_f1_macro=%.4f (best=%.4f)",
            trial.number, epoch, train_loss, val_f1, best_f1,
        )

        # Report to Optuna for pruning
        trial.report(val_f1, global_epoch)
        global_epoch += 1
        if trial.should_prune():
            logger.info("Trial %d pruned at phase 2 epoch %d", trial.number, epoch)
            _save_trial_results(trial_dir, history, best_f1, trial, pruned=True)
            _cleanup_gpu(model, optimizer, criterion, scaler)
            raise optuna.TrialPruned()

    # Save trial results
    _save_trial_results(trial_dir, history, best_f1, trial, pruned=False)
    _cleanup_gpu(model, optimizer, criterion, scaler)

    logger.info("Trial %d complete: best val_f1_macro=%.4f", trial.number, best_f1)
    return best_f1


def _save_trial_results(
    trial_dir: Path,
    history: list,
    best_f1: float,
    trial: optuna.Trial,
    pruned: bool,
) -> None:
    """Save trial history and summary to disk."""
    results = {
        "trial_number": trial.number,
        "best_f1": round(best_f1, 6),
        "pruned": pruned,
        "n_epochs_completed": len(history),
        "params": trial.params,
        "history": history,
    }
    with open(trial_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


def _cleanup_gpu(model, optimizer, criterion, scaler) -> None:
    """Free GPU memory between trials.

    Without this, VRAM accumulates across trials until OOM. Each AWARE model
    uses ~700MB VRAM, so cleanup is essential for 25+ trial runs.
    """
    del model, optimizer, criterion, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Optuna HPO for AWARE v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/hpo.py --config configs/quick.yaml --data_dir data/ --output_dir results/hpo/
  python scripts/hpo.py --config configs/quick.yaml --data_dir data/ --output_dir results/hpo/ --n_trials 10
  python scripts/hpo.py --config configs/quick.yaml --data_dir data/ --output_dir results/hpo/ --encoder_path results/quick_002/dapt_encoder/
        """,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to base YAML config (non-tuned values come from here)",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data/ directory with train_data.pkl, val_data.pkl, splits_stats.json",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for HPO results (SQLite DB, per-trial dirs, best params)",
    )
    parser.add_argument(
        "--n_trials", type=int, default=25,
        help="Number of Optuna trials to run (default: 25)",
    )
    parser.add_argument(
        "--encoder_path", type=str, default=None,
        help="Path to DAPT encoder directory (overrides config.model.encoder_path)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    base_config = AWAREConfig.from_yaml(args.config)
    logger.info("Base config loaded from %s", args.config)

    # Auto-set theme weights from split stats if not in config
    if base_config.loss.theme_weights is None:
        stats_path = data_dir / "splits_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            tw = stats.get("theme_weights", {})
            if tw:
                base_config.loss.theme_weights = [tw.get(t, 1.0) for t in THEMES]
                logger.info(
                    "Auto-loaded theme weights from splits_stats.json: %s",
                    base_config.loss.theme_weights,
                )

    # Load tokenizer (once -- shared across all trials)
    logger.info("Loading tokenizer: %s", base_config.model.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(base_config.model.encoder_name)

    # Load data (once -- shared across all trials)
    logger.info("Loading training and validation data...")
    train_data = load_split_data(data_dir / "train_data.pkl")
    val_data = load_split_data(data_dir / "val_data.pkl")
    logger.info(
        "Data loaded: %d train essays, %d val essays",
        len(train_data["essay_ids"]),
        len(val_data["essay_ids"]),
    )

    # Create Optuna study with SQLite storage for persistence across SLURM jobs.
    # If the study already exists in the DB, this will resume it.
    storage_path = output_dir / "hpo_study.db"
    storage_url = f"sqlite:///{storage_path}"
    study_name = "aware_v2_hpo"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(
            seed=42,
            n_startup_trials=5,     # Random exploration before TPE kicks in
            multivariate=True,      # Model parameter correlations
        ),
        pruner=MedianPruner(
            n_startup_trials=3,     # Don't prune the first 3 trials
            n_warmup_steps=3,       # Don't prune before epoch 4 (0-indexed)
            interval_steps=1,       # Check pruning every epoch
        ),
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        logger.info(
            "Resuming study '%s' with %d existing trials (best F1 so far: %.4f)",
            study_name, n_existing, study.best_value,
        )
    else:
        logger.info("Starting new study '%s'", study_name)

    # Run optimization
    logger.info("Running %d trials...", args.n_trials)
    study.optimize(
        lambda trial: objective(
            trial,
            base_config=base_config,
            train_data=train_data,
            val_data=val_data,
            tokenizer=tokenizer,
            output_dir=output_dir,
            encoder_path=args.encoder_path,
        ),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )

    # ── Report results ──
    logger.info("=" * 70)
    logger.info("HPO COMPLETE")
    logger.info("=" * 70)

    # Best trial
    best_trial = study.best_trial
    logger.info("Best trial: #%d", best_trial.number)
    logger.info("Best val macro F1: %.4f", best_trial.value)
    logger.info("Best params:")
    for param, value in best_trial.params.items():
        logger.info("  %s: %s", param, value)

    # Save best params
    best_params = {
        "best_trial_number": best_trial.number,
        "best_val_f1_macro": round(best_trial.value, 6),
        "best_params": best_trial.params,
        "total_trials": len(study.trials),
        "n_pruned": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]),
        "n_complete": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]),
    }
    best_params_path = output_dir / "best_hpo_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Best params saved to %s", best_params_path)

    # Summary table of all trials
    all_trials_summary = []
    for t in study.trials:
        summary = {
            "number": t.number,
            "state": t.state.name,
            "value": round(t.value, 6) if t.value is not None else None,
            "params": t.params,
        }
        all_trials_summary.append(summary)
    all_trials_path = output_dir / "all_trials.json"
    with open(all_trials_path, "w") as f:
        json.dump(all_trials_summary, f, indent=2)
    logger.info("All trial summaries saved to %s", all_trials_path)

    # Log top 5 trials
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    completed.sort(key=lambda t: t.value or 0, reverse=True)
    logger.info("Top 5 trials:")
    for rank, t in enumerate(completed[:5], 1):
        logger.info(
            "  #%d: trial %d, F1=%.4f | enc_lr=%.2e dec_lr=%.2e drop=%.3f "
            "gamma_neg=%.2f wd=%.4f p2_scale=%.4f",
            rank, t.number, t.value,
            t.params.get("encoder_lr", 0),
            t.params.get("decoder_lr", 0),
            t.params.get("dropout", 0),
            t.params.get("asl_gamma_neg", 0),
            t.params.get("weight_decay", 0),
            t.params.get("phase2_encoder_lr_scale", 0),
        )

    # Print config snippet for easy copy-paste into quick.yaml
    bp = best_trial.params
    logger.info("")
    logger.info("Copy-paste into your YAML config:")
    logger.info("─" * 40)
    logger.info("model:")
    logger.info("  dropout: %.4f", bp["dropout"])
    logger.info("loss:")
    logger.info("  asl_gamma_neg: %.2f", bp["asl_gamma_neg"])
    logger.info("training:")
    logger.info("  encoder_lr: %.2e", bp["encoder_lr"])
    logger.info("  decoder_lr: %.2e", bp["decoder_lr"])
    logger.info("  weight_decay: %.4f", bp["weight_decay"])
    logger.info("  phase2_encoder_lr_scale: %.4f", bp["phase2_encoder_lr_scale"])
    logger.info("─" * 40)


if __name__ == "__main__":
    main()
