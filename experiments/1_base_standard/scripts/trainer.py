"""
trainer.py — Standard single-phase trainer (NO AWARE components).

Simple training loop:
  - AdamW optimizer with flat LR (no LLRD)
  - Linear warmup + cosine decay
  - BCEWithLogitsLoss (no ASL)
  - Early stopping on val PR-AUC
  - NO R-Drop, NO phases, NO SWA
"""

import os
import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from config import THEMES, NUM_THEMES
from losses import build_loss_from_config
from metrics import (
    flatten_masked_logits_labels, compute_metrics, compute_prauc,
    optimize_thresholds, calibrate_per_class,
)

logger = logging.getLogger(__name__)


def set_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


class StandardTrainer:
    def __init__(self, model, train_loader, val_loader, output_dir, config,
                 device=None, theme_weights_list=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        self.criterion = build_loss_from_config(
            config, theme_weights_list=theme_weights_list
        ).to(self.device)

        T = config.training
        self.epochs = T.epochs
        self.grad_accum = T.gradient_accumulation
        self.lr = T.learning_rate
        self.wd = T.weight_decay
        self.max_grad_norm = T.max_grad_norm
        self.warmup_ratio = T.warmup_ratio
        self.patience = T.early_stopping_patience
        self.fp16 = T.fp16 and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.fp16 else None

        self.history = {"epochs": [], "best_epoch": 0, "best_prauc": 0.0}
        self.best_prauc = 0.0
        self.pat_cnt = 0

        logger.info("StandardTrainer: epochs=%d, lr=%.2e, wd=%.4f, patience=%d, fp16=%s",
                     self.epochs, self.lr, self.wd, self.patience, self.fp16)

    def train(self):
        set_seeds(self.config.seed)
        from torch.optim import AdamW
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        acc = max(1, self.grad_accum)
        spe = max(1, (len(self.train_loader) + acc - 1) // acc)
        total_steps = spe * self.epochs
        warmup = max(1, int(total_steps * self.warmup_ratio))
        sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
        logger.info("Training: steps=%d, warmup=%d", total_steps, warmup)

        for epoch in range(1, self.epochs + 1):
            loss_info = self._train_epoch(opt, sched)
            val = self._validate()
            self._log_epoch(epoch, loss_info, val)
            prauc = val["prauc_macro"]
            if prauc > self.best_prauc:
                self.best_prauc = prauc
                self.pat_cnt = 0
                self.history["best_epoch"] = epoch
                self.history["best_prauc"] = prauc
                torch.save(self.model.state_dict(), self.output_dir / "best.pt")
                logger.info("  ** New best PR-AUC: %.4f at epoch %d **", prauc, epoch)
            else:
                self.pat_cnt += 1
                if self.pat_cnt >= self.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.patience)
                    break

        self._post_training()
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        return self.history

    def _train_epoch(self, opt, sched):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        acc = max(1, self.grad_accum)
        opt.zero_grad()

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for step, batch in enumerate(pbar):
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            smask = batch["sentence_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            bounds = batch["sentence_boundaries"]

            if self.fp16:
                with torch.amp.autocast("cuda"):
                    out = self.model(ids, mask, bounds, smask)
                    loss = self.criterion(out["logits"], labels, smask) / acc
                self.scaler.scale(loss).backward()
            else:
                out = self.model(ids, mask, bounds, smask)
                loss = self.criterion(out["logits"], labels, smask) / acc
                loss.backward()

            total_loss += loss.item() * acc
            n_batches += 1

            if (step + 1) % acc == 0 or (step + 1) == len(self.train_loader):
                if self.fp16:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    opt.step()
                sched.step()
                opt.zero_grad()

            pbar.set_postfix(loss=f"{loss.item() * acc:.4f}")

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        all_logits, all_labels = [], []

        for batch in self.val_loader:
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            smask = batch["sentence_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            bounds = batch["sentence_boundaries"]

            if self.fp16:
                with torch.amp.autocast("cuda"):
                    out = self.model(ids, mask, bounds, smask)
            else:
                out = self.model(ids, mask, bounds, smask)

            fl, ll = flatten_masked_logits_labels(out["logits"], labels, smask)
            all_logits.append(fl.cpu())
            all_labels.append(ll.cpu())

        logits_np = torch.cat(all_logits).numpy()
        labels_np = torch.cat(all_labels).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits_np))

        preds_default = (probs >= 0.5).astype(np.float32)
        metrics = compute_metrics(preds_default, labels_np)

        prauc = compute_prauc(probs, labels_np)
        metrics["prauc_macro"] = prauc["prauc_macro"]
        metrics["prauc_per_theme"] = prauc["prauc_per_theme"]
        return metrics

    def _log_epoch(self, epoch, loss_info, val):
        entry = {
            "epoch": epoch,
            "train_loss": round(loss_info["train_loss"], 4),
            "val_f1_macro": val["f1_macro"],
            "val_prauc_macro": val["prauc_macro"],
        }
        self.history["epochs"].append(entry)
        logger.info(
            "Epoch %d: loss=%.4f, val_F1=%.4f, val_PRAUC=%.4f, patience=%d/%d",
            epoch, loss_info["train_loss"], val["f1_macro"], val["prauc_macro"],
            self.pat_cnt, self.patience,
        )
        for theme in THEMES:
            f1 = val["f1_per_theme"].get(theme, 0)
            prauc_t = val.get("prauc_per_theme", {}).get(theme, 0)
            logger.info("  %-20s F1=%.4f  PRAUC=%.4f", theme, f1, prauc_t)

    def _post_training(self):
        """Post-training: load best model, calibrate, optimize thresholds."""
        best_path = self.output_dir / "best.pt"
        if not best_path.exists():
            logger.warning("No best.pt found, skipping post-training")
            return

        self.model.load_state_dict(
            torch.load(best_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                smask = batch["sentence_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                bounds = batch["sentence_boundaries"]
                out = self.model(ids, mask, bounds, smask)
                fl, ll = flatten_masked_logits_labels(out["logits"], labels, smask)
                all_logits.append(fl.cpu())
                all_labels.append(ll.cpu())

        logits_np = torch.cat(all_logits).numpy()
        labels_np = torch.cat(all_labels).numpy()

        logger.info("Calibrating (Platt scaling)...")
        cal_params, cal_probs = calibrate_per_class(logits_np, labels_np)
        calibration = {}
        for i, theme in enumerate(THEMES):
            calibration[theme] = {"a": float(cal_params[0, i]), "b": float(cal_params[1, i])}
        with open(self.output_dir / "calibration.json", "w") as f:
            json.dump(calibration, f, indent=2)

        logger.info("Optimizing thresholds...")
        thresholds = optimize_thresholds(cal_probs, labels_np)
        with open(self.output_dir / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)

        logger.info("Post-training complete. Thresholds: %s", thresholds)
