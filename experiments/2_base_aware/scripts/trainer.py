"""
trainer.py — Three-phase AWARE v3 trainer.

Phase 1: Frozen encoder → train BiLSTM + head
Phase 2: Full fine-tune with differential LR (LLRD) + optional SWA
Phase 3: Decoupled head retraining (Kang et al., ICLR 2020)

Model selection: PR-AUC macro (threshold-independent).

Key improvements in this version:
  - R-Drop: truly symmetric (both p1 and p2 receive KL gradients)
  - Essay-level multi-task loss weighted by essay_aux_weight
  - SWA (Stochastic Weight Averaging): collects weights from epoch swa_start_ratio×P2
    onward, producing flatter minimum → better test generalization
  - CB Loss weights: auto-computed from training class counts (passed at init)
  - Detailed logging: task/KL/essay loss breakdown, per-theme P/R, gradient norms,
    loss component ratios, calibration drift — everything needed to diagnose next run
  - Phase 3 early abort: FIXED (was a no-op pass block)
  - Co-occurrence confusion logging: shows which true themes get predicted as which
"""

import sys
import shutil
import copy
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import logging
from pathlib import Path
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import Dict, Optional, Callable
from collections import defaultdict

from model import AWAREModel, build_model_from_config
from losses import build_loss_from_config
from metrics import (
    flatten_masked_preds_labels,
    flatten_masked_logits_labels,
    compute_metrics,
    compute_prauc,
    optimize_thresholds,
    calibrate_per_class,
)
from config import THEMES, NUM_THEMES

logger = logging.getLogger(__name__)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AWARETrainer:
    def __init__(self, model, train_loader, val_loader, output_dir, config, device=None,
                 train_class_counts=None):
        """
        Args:
            train_class_counts: list of per-theme positive sentence counts [C].
                Used to compute CB theme weights for AsymmetricLoss.
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        logger.info("Using device: %s", self.device)

        self.criterion = build_loss_from_config(
            config,
            train_class_counts=train_class_counts,
        ).to(self.device)

        T = config.training
        self.phase1_epochs = int(T.phase1_epochs)
        self.phase2_epochs = int(T.phase2_epochs)
        self.gradient_accumulation = int(T.gradient_accumulation)
        self.decoder_lr = float(T.decoder_lr)
        self.encoder_lr = float(T.encoder_lr)
        self.phase2_encoder_lr_scale = float(T.phase2_encoder_lr_scale)
        self.weight_decay = float(T.weight_decay)
        self.max_grad_norm = float(T.max_grad_norm)
        self.warmup_ratio = float(T.warmup_ratio)
        self.early_stopping_patience = int(T.early_stopping_patience)
        self.use_fp16 = bool(T.fp16) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_fp16 else None
        self.rdrop_alpha = float(getattr(T, "rdrop_alpha", 0.0))
        self.adam_beta2 = float(getattr(T, "adam_beta2", 0.999))
        self.phase2_patience = int(getattr(T, "phase2_early_stopping_patience", self.early_stopping_patience))
        # Essay-level multi-task weight (0 = disabled)
        self.essay_aux_weight = float(getattr(T, "essay_aux_weight", 0.0))
        self.use_essay_head = self.essay_aux_weight > 0 and getattr(self.model, "use_essay_head", False)
        # SWA: start collecting at this fraction of phase 2 epochs
        self.swa_start_ratio = float(getattr(T, "swa_start_ratio", 0.5))
        # NEW: Phase 1 dedicated LR (None = use decoder_lr)
        p1_lr = getattr(T, "phase1_lr", None)
        self.phase1_lr = float(p1_lr) if p1_lr is not None else self.decoder_lr
        # NEW: Phase 1 R-Drop alpha (None = use rdrop_alpha, 0.0 = disable)
        p1_rdrop = getattr(T, "phase1_rdrop", None)
        self.phase1_rdrop = float(p1_rdrop) if p1_rdrop is not None else self.rdrop_alpha
        # NEW: Progressive unfreezing
        self.progressive_unfreeze = bool(getattr(T, "progressive_unfreeze", False))
        self.progressive_unfreeze_layers = int(getattr(T, "progressive_unfreeze_layers", 12))
        self.progressive_unfreeze_after = int(getattr(T, "progressive_unfreeze_after", 6))
        # NEW: Phase 3 BiLSTM unfreeze
        self.phase3_unfreeze_bilstm = bool(getattr(T, "phase3_unfreeze_bilstm", False))
        # NEW: Context dropout — mask sentence embeddings to force context usage
        self.context_dropout = float(getattr(T, "context_dropout", 0.0))

        self.history = {"epochs": [], "best_epoch": 0, "best_prauc": 0.0, "best_phase": 1}
        self.current_epoch = 0
        self.best_prauc_val = 0.0
        self.patience_counter = 0
        # SWA state: dict of param_name → running sum of param values
        self._swa_params = None
        self._swa_count = 0

        logger.info(
            "AWARETrainer init: phases=(%d, %d), fp16=%s, rdrop=%.1f, essay_aux=%.2f, "
            "swa_start_ratio=%.2f, cb_weights=%s",
            self.phase1_epochs, self.phase2_epochs, self.use_fp16,
            self.rdrop_alpha, self.essay_aux_weight, self.swa_start_ratio,
            train_class_counts is not None,
        )

    def train(self):
        set_seeds(self.config.seed)
        logger.info(
            "Starting 3-phase training (P1=%d, P2=%d, device=%s, fp16=%s, rdrop=%.1f, essay_aux=%.2f)",
            self.phase1_epochs, self.phase2_epochs, self.device, self.use_fp16,
            self.rdrop_alpha, self.essay_aux_weight,
        )

        # Phase 1: frozen encoder
        self.model._freeze_encoder()
        self.patience_counter = 0
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.phase1_lr, weight_decay=self.weight_decay,
            betas=(0.9, self.adam_beta2),
        )
        accum = max(1, self.gradient_accumulation)
        steps_per_ep = max(1, (len(self.train_loader) + accum - 1) // accum)
        total_steps = steps_per_ep * self.phase1_epochs
        warmup = max(1, int(total_steps * self.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

        logger.info("=" * 60)
        logger.info("PHASE 1: Frozen encoder, training BiLSTM + head")
        logger.info("  lr=%.2e, rdrop=%.1f, epochs=%d, steps=%d, warmup=%d",
                    self.phase1_lr, self.phase1_rdrop, self.phase1_epochs, total_steps, warmup)
        logger.info("=" * 60)

        # Save original rdrop_alpha, use phase1_rdrop during Phase 1
        _original_rdrop = self.rdrop_alpha
        self.rdrop_alpha = self.phase1_rdrop

        for epoch in range(1, self.phase1_epochs + 1):
            self.current_epoch = epoch
            loss_info = self._train_epoch(optimizer, scheduler, phase=1)
            val = self._validate()
            self._log_epoch(epoch, loss_info, val, phase=1)
            if self._check_early_stop(val, phase=1):
                logger.info("Phase 1 early stop at epoch %d", epoch)
                break

        # Restore original rdrop_alpha for Phase 2
        self.rdrop_alpha = _original_rdrop

        # Phase 2: unfreeze encoder with LLRD
        if self.progressive_unfreeze:
            # Progressive unfreezing: only unfreeze top N layers first
            self._progressive_unfreeze_top_layers(self.progressive_unfreeze_layers)
        else:
            self.model.unfreeze_encoder()
        self.patience_counter = 0
        optimizer = self._build_phase2_optimizer_llrd()
        steps_per_ep = max(1, (len(self.train_loader) + accum - 1) // accum)
        total_steps = steps_per_ep * self.phase2_epochs
        warmup = max(1, int(total_steps * self.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

        # SWA: start collecting after swa_start_ratio of phase 2 epochs
        swa_start_ep = self.phase1_epochs + max(1, int(self.swa_start_ratio * self.phase2_epochs))
        # Progressive unfreeze: epoch when remaining layers are unfrozen
        full_unfreeze_ep = self.phase1_epochs + self.progressive_unfreeze_after if self.progressive_unfreeze else 0
        _did_full_unfreeze = not self.progressive_unfreeze

        logger.info("=" * 60)
        logger.info("PHASE 2: Full fine-tune with LLRD")
        logger.info("  encoder_lr=%.2e, decoder_lr=%.2e, epochs=%d, patience=%d",
                    self.encoder_lr, self.decoder_lr, self.phase2_epochs, self.phase2_patience)
        logger.info("  SWA starts at epoch %d (ratio=%.2f)", swa_start_ep, self.swa_start_ratio)
        if self.progressive_unfreeze:
            logger.info("  Progressive unfreeze: top %d layers now, all at epoch %d",
                        self.progressive_unfreeze_layers, full_unfreeze_ep)
        logger.info("=" * 60)

        for epoch in range(self.phase1_epochs + 1, self.phase1_epochs + self.phase2_epochs + 1):
            self.current_epoch = epoch

            # Progressive unfreeze: unfreeze remaining layers at the scheduled epoch
            if self.progressive_unfreeze and not _did_full_unfreeze and epoch >= full_unfreeze_ep:
                logger.info("Progressive unfreeze: unfreezing ALL encoder layers at epoch %d", epoch)
                self.model.unfreeze_encoder()
                # Rebuild optimizer to include newly unfrozen parameters
                optimizer = self._build_phase2_optimizer_llrd()
                # Don't reset scheduler — keep the existing cosine schedule position
                _did_full_unfreeze = True

            loss_info = self._train_epoch(optimizer, scheduler, phase=2)
            val = self._validate()
            self._log_epoch(epoch, loss_info, val, phase=2)

            # SWA: accumulate model parameters after swa_start_ep
            if epoch >= swa_start_ep:
                self._swa_accumulate()

            if self._check_early_stop(val, phase=2):
                logger.info("Phase 2 early stop at epoch %d (patience=%d)", epoch, self.phase2_patience)
                break

        # Apply SWA averaged weights if we collected enough
        if self._swa_count >= 3:
            self._apply_swa(optimizer)

        # Phase 3: decoupled head retrain
        phase3_epochs = int(getattr(self.config.training, "phase3_epochs", 10))
        if phase3_epochs > 0:
            self._train_phase3()

        self._save_history()
        self._run_threshold_optimization()
        logger.info("Training complete. Best PRAUC: %.4f at epoch %d",
                     self.history.get("best_prauc", 0.0), self.history["best_epoch"])
        return self.history

    def _swa_accumulate(self):
        """Add current model parameters to the SWA running sum."""
        if self._swa_params is None:
            self._swa_params = {n: p.data.clone().cpu() for n, p in self.model.named_parameters()}
            self._swa_count = 1
        else:
            for n, p in self.model.named_parameters():
                self._swa_params[n] += p.data.cpu()
            self._swa_count += 1
        logger.debug("SWA: accumulated epoch %d (total=%d)", self.current_epoch, self._swa_count)

    def _apply_swa(self, optimizer):
        """Average SWA params and evaluate. Keep if better than current best."""
        if self._swa_params is None or self._swa_count == 0:
            return

        logger.info("=" * 60)
        logger.info("SWA: averaging %d checkpoints...", self._swa_count)

        # Save current best before overwriting
        best_path = self.output_dir / "best.pt"
        swa_path = self.output_dir / "swa_candidate.pt"

        # Build averaged state dict
        avg_state = {}
        for n, param_sum in self._swa_params.items():
            avg_state[n] = (param_sum / self._swa_count).to(self.device)

        # Load averaged weights into model
        self.model.load_state_dict(avg_state)

        # Evaluate SWA model on val set
        val_swa = self._validate()
        prauc_swa = val_swa.get("prauc_macro", 0.0)
        f1_swa = val_swa.get("f1_macro", 0.0)

        logger.info(
            "SWA averaged model: Val PR-AUC=%.4f, F1=%.4f (vs best single=%.4f)",
            prauc_swa, f1_swa, self.best_prauc_val,
        )
        logger.info("  per_theme: %s",
                    " ".join(f"{t[:3]}={val_swa.get('f1_per_theme', {}).get(t, 0):.3f}"
                             for t in sorted(THEMES)))

        if prauc_swa > self.best_prauc_val:
            torch.save(avg_state, best_path)
            self.best_prauc_val = prauc_swa
            self.history["best_prauc"] = round(prauc_swa, 6)
            self.history["best_epoch"] = self.current_epoch
            self.history["best_phase"] = "swa"
            logger.info("SWA IMPROVED: %.4f → %.4f. Saving as best.pt.", self.best_prauc_val, prauc_swa)
        else:
            # Restore original best weights
            self.model.load_state_dict(
                torch.load(best_path, map_location=self.device, weights_only=True)
            )
            logger.info("SWA did not improve (%.4f < best %.4f). Keeping original best.",
                        prauc_swa, self.best_prauc_val)
        logger.info("=" * 60)

    def _build_phase2_optimizer_llrd(self):
        """Build Phase 2 optimizer with LLRD if config.training.use_llrd=True."""
        T = self.config.training
        use_llrd = getattr(T, "use_llrd", False)
        llrd_decay = float(getattr(T, "llrd_decay", 0.9))
        enc_lr = self.encoder_lr * self.phase2_encoder_lr_scale

        if not use_llrd:
            enc_params = [p for n, p in self.model.named_parameters() if "encoder" in n]
            other_params = [p for n, p in self.model.named_parameters() if "encoder" not in n]
            logger.info("Phase 2 LR: encoder=%.2e, decoder=%.2e", enc_lr, self.decoder_lr)
            return AdamW(
                [{"params": enc_params, "lr": enc_lr}, {"params": other_params, "lr": self.decoder_lr}],
                weight_decay=self.weight_decay, betas=(0.9, self.adam_beta2),
            )

        # LLRD: top layer gets enc_lr, each lower layer multiplied by llrd_decay
        all_named = list(self.model.named_parameters())
        max_layer = -1
        for name, _ in all_named:
            if "encoder.encoder.layer." in name:
                try:
                    idx = int(name.split("encoder.encoder.layer.")[1].split(".")[0])
                    if idx > max_layer:
                        max_layer = idx
                except (ValueError, IndexError):
                    pass
        n_layers = max_layer + 1 if max_layer >= 0 else 12

        assigned = set()
        param_groups = []

        # Non-encoder params (BiLSTM, pooling, head, position_embedding) → decoder_lr
        dec = [(n, p) for n, p in all_named if "encoder" not in n]
        param_groups.append({"params": [p for _, p in dec], "lr": self.decoder_lr, "name": "decoder"})
        assigned.update(n for n, _ in dec)

        # Per-layer from top (high LR) to bottom (low LR)
        for layer_idx in range(n_layers - 1, -1, -1):
            layer_key = f"encoder.encoder.layer.{layer_idx}."
            layer = [(n, p) for n, p in all_named if layer_key in n and n not in assigned]
            if layer:
                depth = n_layers - 1 - layer_idx
                lr_i = enc_lr * (llrd_decay ** depth)
                param_groups.append({
                    "params": [p for _, p in layer],
                    "lr": lr_i,
                    "name": f"enc_layer_{layer_idx}",
                })
                assigned.update(n for n, _ in layer)

        # Embeddings and remaining encoder params → lowest LR
        remaining = [(n, p) for n, p in all_named if n not in assigned]
        if remaining:
            embed_lr = enc_lr * (llrd_decay ** n_layers)
            param_groups.append({
                "params": [p for _, p in remaining],
                "lr": embed_lr,
                "name": "enc_embed",
            })

        logger.info(
            "Phase 2 LLRD: n_layers=%d, top_lr=%.2e, embed_lr=%.2e, decoder_lr=%.2e, decay=%.2f",
            n_layers, enc_lr, enc_lr * (llrd_decay ** n_layers), self.decoder_lr, llrd_decay,
        )
        return AdamW(param_groups, weight_decay=self.weight_decay, betas=(0.9, self.adam_beta2))

    def _progressive_unfreeze_top_layers(self, n_top_layers):
        """Unfreeze only the top N encoder layers. Bottom layers stay frozen.

        For DeBERTa-v3-large (24 layers), unfreezing top 12 means layers 12-23 are
        trainable while layers 0-11 remain frozen. This halves effective trainable
        params in early Phase 2, reducing memorization risk.
        """
        # First freeze entire encoder
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        # Find total layers
        max_layer = -1
        for name, _ in self.model.encoder.named_parameters():
            if "encoder.layer." in name:
                try:
                    idx = int(name.split("encoder.layer.")[1].split(".")[0])
                    max_layer = max(max_layer, idx)
                except (ValueError, IndexError):
                    pass
        n_layers = max_layer + 1 if max_layer >= 0 else 12

        # Unfreeze top N layers
        unfreeze_from = max(0, n_layers - n_top_layers)
        unfrozen_count = 0
        for name, p in self.model.encoder.named_parameters():
            if "encoder.layer." in name:
                try:
                    idx = int(name.split("encoder.layer.")[1].split(".")[0])
                    if idx >= unfreeze_from:
                        p.requires_grad = True
                        unfrozen_count += 1
                except (ValueError, IndexError):
                    pass

        # Also unfreeze non-encoder model parts (BiLSTM, heads, position emb)
        for name, p in self.model.named_parameters():
            if "encoder" not in name:
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Progressive unfreeze: layers %d-%d (%d/%d), %.1fM/%.1fM params trainable (%.0f%%)",
            unfreeze_from, n_layers - 1, n_top_layers, n_layers,
            trainable / 1e6, total / 1e6, 100 * trainable / total,
        )

    def _train_phase3(self):
        T = self.config.training
        epochs = int(getattr(T, "phase3_epochs", 10))
        lr = float(getattr(T, "phase3_lr", 1e-4))
        patience_limit = int(getattr(T, "phase3_patience", 5))

        best_path = self.output_dir / "best.pt"
        if not best_path.exists():
            logger.warning("Phase 3: best.pt not found — skipping")
            return

        phase2_path = self.output_dir / "best_phase2.pt"
        shutil.copy2(best_path, phase2_path)
        phase2_prauc = self.best_prauc_val

        self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))

        # Freeze everything, then unfreeze classification heads + optionally BiLSTM
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        if self.use_essay_head and hasattr(self.model, "essay_head"):
            for p in self.model.essay_head.parameters():
                p.requires_grad = True
        # FIX: Optionally unfreeze BiLSTM in Phase 3 (2M params vs 20K head-only)
        # Phase 3 was dead with head-only: 20K params on fixed inputs converged in 1 epoch.
        if self.phase3_unfreeze_bilstm and hasattr(self.model, "context_encoder"):
            for p in self.model.context_encoder.parameters():
                p.requires_grad = True
            # Also unfreeze position embedding if it exists
            if hasattr(self.model, "position_embedding"):
                for p in self.model.position_embedding.parameters():
                    p.requires_grad = True

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01,
                          betas=(0.9, self.adam_beta2))

        logger.info("=" * 60)
        logger.info("PHASE 3: Head%s retraining", " + BiLSTM" if self.phase3_unfreeze_bilstm else "")
        logger.info("  lr=%.1e, epochs=%d, patience=%d, trainable_params=%.1fK",
                    lr, epochs, patience_limit, n_trainable / 1e3)
        logger.info("  Phase 2 best PRAUC=%.4f (must beat this to replace best.pt)", phase2_prauc)
        logger.info("=" * 60)

        phase3_start = self.current_epoch + 1
        best_p3_prauc = 0.0
        patience = patience_limit
        phase3_val_losses = []
        p3_path = self.output_dir / "best_phase3.pt"
        accum = max(1, self.gradient_accumulation)
        p3_steps = max(1, (len(self.train_loader) + accum - 1) // accum) * epochs
        p3_warmup = max(1, int(p3_steps * 0.10))
        scheduler = get_cosine_schedule_with_warmup(optimizer, p3_warmup, p3_steps)

        for ep in range(1, epochs + 1):
            self.current_epoch = phase3_start + ep - 1
            self.model.train()
            total_loss, n = 0.0, 0
            grad_norms = []  # FIX: track grad norms in Phase 3
            optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader):
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                smask = batch["sentence_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                bounds = batch["sentence_boundaries"]

                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    out = self.model(ids, mask, bounds, smask)
                    loss = self.criterion(out["logits"], labels, smask)
                    # Essay-level auxiliary loss in phase 3
                    if self.use_essay_head and "essay_logits" in out:
                        essay_labels = _compute_essay_labels(labels, smask)
                        essay_loss = self.criterion(
                            out["essay_logits"].unsqueeze(1),
                            essay_labels.unsqueeze(1),
                            torch.ones(ids.size(0), 1, device=self.device),
                        )
                        loss = loss + self.essay_aux_weight * essay_loss

                (loss / accum).backward() if not self.use_fp16 else self.scaler.scale(loss / accum).backward()
                total_loss += loss.item()
                n += 1

                if (step + 1) % accum == 0 or (step + 1) == len(self.train_loader):
                    if self.use_fp16:
                        self.scaler.unscale_(optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    grad_norms.append(float(total_norm))
                    if self.use_fp16:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # FIX: Pass grad norm info to _log_epoch so Phase 3 logging works
            loss_info = {
                "total": total_loss / max(n, 1),
                "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
                "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
            }
            val = self._validate()
            self._log_epoch(self.current_epoch, loss_info, val, phase=3)

            prauc = val.get("prauc_macro", 0.0)
            val_loss = val.get("loss", 0.0)
            phase3_val_losses.append(val_loss)

            # FIX: Phase 3 degradation detection — abort if val_loss consistently rising.
            # (Previous code had `pass` here — a no-op. Now actually abort.)
            if ep >= 2:
                rising = all(vl > phase3_val_losses[0] * 1.05 for vl in phase3_val_losses[-2:])
                if rising and prauc < phase2_prauc:
                    logger.info(
                        "Phase 3 ABORT at ep %d: val_loss rising (%.4f→%.4f) and "
                        "PRAUC (%.4f) below phase2 best (%.4f)",
                        ep, phase3_val_losses[0], val_loss, prauc, phase2_prauc,
                    )
                    break

            if prauc > best_p3_prauc:
                best_p3_prauc = prauc
                torch.save(self.model.state_dict(), p3_path)
                patience = patience_limit
                logger.info("  Phase 3 NEW BEST: PRAUC=%.4f at ep %d", prauc, ep)
            else:
                patience -= 1
                logger.info(
                    "  Phase 3 ep %d: PRAUC=%.4f (best=%.4f, patience=%d/%d)",
                    ep, prauc, best_p3_prauc, patience_limit - patience, patience_limit,
                )
                if patience <= 0:
                    logger.info("Phase 3 early stop at ep %d (patience exhausted)", ep)
                    break

        if best_p3_prauc > phase2_prauc:
            shutil.copy2(p3_path, best_path)
            self.best_prauc_val = best_p3_prauc
            self.history["best_prauc"] = round(best_p3_prauc, 6)
            logger.info("Phase 3 IMPROVED: %.4f → %.4f", phase2_prauc, best_p3_prauc)
        else:
            shutil.copy2(phase2_path, best_path)
            logger.info("Phase 3 did NOT improve (%.4f vs %.4f) — keeping phase2 best",
                        best_p3_prauc, phase2_prauc)

        self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))

    def _rdrop_kl_loss(self, p1, p2, mask):
        """Symmetric KL divergence for R-Drop. Both p1 and p2 retain gradients."""
        eps = 1e-7
        p1 = p1.clamp(eps, 1 - eps)
        p2 = p2.clamp(eps, 1 - eps)
        kl1 = p1 * torch.log(p1 / p2) + (1 - p1) * torch.log((1 - p1) / (1 - p2))
        kl2 = p2 * torch.log(p2 / p1) + (1 - p2) * torch.log((1 - p2) / (1 - p1))
        kl = (kl1 + kl2) / 2
        if mask is not None:
            kl = kl * mask.unsqueeze(-1).float()
            return kl.sum() / mask.sum().clamp(1)
        return kl.mean()

    def _train_epoch(self, optimizer, scheduler, phase=1):
        """Train one epoch. Returns dict of loss components for detailed logging."""
        self.model.train()
        total_loss, task_loss_sum, kl_loss_sum, essay_loss_sum = 0.0, 0.0, 0.0, 0.0
        n = 0
        accum = max(1, self.gradient_accumulation)
        optimizer.zero_grad()
        use_rdrop = self.rdrop_alpha > 0

        # Track gradient norms every 50 optimizer steps
        grad_norms = []

        for step, batch in enumerate(self.train_loader):
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            smask = batch["sentence_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            bounds = batch["sentence_boundaries"]
            ctx_drop = getattr(self, "context_dropout", 0.0)
            fwd = dict(input_ids=ids, attention_mask=mask, sentence_boundaries=bounds,
                       sentence_mask=smask, context_dropout=ctx_drop)

            if use_rdrop:
                # Symmetric R-Drop: two forward passes in one autocast block.
                # Both p1 and p2 are live tensors → KL gradients flow to both.
                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    out1 = self.model(**fwd)
                    out2 = self.model(**fwd)

                    loss1 = self.criterion(out1["logits"], labels, smask)
                    loss2 = self.criterion(out2["logits"], labels, smask)
                    task_loss = (loss1 + loss2) / 2

                    p1 = torch.sigmoid(out1["logits"])
                    p2 = torch.sigmoid(out2["logits"])
                    kl = self._rdrop_kl_loss(p1, p2, smask)

                    total = task_loss + self.rdrop_alpha * kl

                    # Essay-level auxiliary loss (multi-task)
                    essay_loss = torch.tensor(0.0, device=self.device)
                    if self.use_essay_head:
                        essay_labels = _compute_essay_labels(labels, smask)
                        essay_loss1 = self.criterion(
                            out1["essay_logits"].unsqueeze(1),
                            essay_labels.unsqueeze(1),
                            torch.ones(ids.size(0), 1, device=self.device),
                        )
                        essay_loss2 = self.criterion(
                            out2["essay_logits"].unsqueeze(1),
                            essay_labels.unsqueeze(1),
                            torch.ones(ids.size(0), 1, device=self.device),
                        )
                        essay_loss = (essay_loss1 + essay_loss2) / 2
                        total = total + self.essay_aux_weight * essay_loss

                task_loss_sum += task_loss.item()
                kl_loss_sum += kl.item()
                essay_loss_sum += essay_loss.item() if hasattr(essay_loss, "item") else float(essay_loss)

            else:
                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    out1 = self.model(**fwd)
                    task_loss = self.criterion(out1["logits"], labels, smask)

                    essay_loss = torch.tensor(0.0, device=self.device)
                    if self.use_essay_head:
                        essay_labels = _compute_essay_labels(labels, smask)
                        essay_loss = self.criterion(
                            out1["essay_logits"].unsqueeze(1),
                            essay_labels.unsqueeze(1),
                            torch.ones(ids.size(0), 1, device=self.device),
                        )
                        task_loss = task_loss + self.essay_aux_weight * essay_loss

                    total = task_loss
                    task_loss_sum += task_loss.item()
                    essay_loss_sum += essay_loss.item()

            total_loss += total.item()
            n += 1

            scaled = total / accum
            if self.use_fp16:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (step + 1) % accum == 0 or (step + 1) == len(self.train_loader):
                if self.use_fp16:
                    self.scaler.unscale_(optimizer)
                # Compute and log gradient norm
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                grad_norms.append(float(total_norm))

                if self.use_fp16:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg = max(n, 1)
        loss_info = {
            "total": total_loss / avg,
            "task": task_loss_sum / avg,
            "kl": kl_loss_sum / avg,
            "essay": essay_loss_sum / avg,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
        }
        return loss_info

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        all_preds, all_labels, all_probs_list, all_logits_list = [], [], [], []
        total_loss, n = 0.0, 0

        for batch in self.val_loader:
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            smask = batch["sentence_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            bounds = batch["sentence_boundaries"]

            out = self.model(ids, mask, bounds, smask)
            loss = self.criterion(out["logits"], labels, smask)
            total_loss += loss.item()
            n += 1
            p, l = flatten_masked_preds_labels(out["logits"], labels, smask)
            all_preds.append(p)
            all_labels.append(l)
            flat_probs = torch.sigmoid(out["logits"])[smask.bool()].cpu().numpy()
            flat_logits = out["logits"][smask.bool()].cpu().numpy()
            all_probs_list.append(flat_probs)
            all_logits_list.append(flat_logits)

        metrics = {"loss": total_loss / max(n, 1), "f1_macro": 0.0, "prauc_macro": 0.0}

        if all_preds:
            preds = np.vstack(all_preds)
            lab = np.vstack(all_labels)
            probs = np.vstack(all_probs_list)
            logits_np = np.vstack(all_logits_list)

            m = compute_metrics(preds, lab)
            prauc = compute_prauc(probs, lab)
            metrics.update(m)
            metrics["prauc_macro"] = prauc["prauc_macro"]
            metrics["prauc_per_theme"] = prauc["prauc_per_theme"]

            # Logit statistics per theme — diagnose overconfidence
            logit_stats = {}
            for i, theme in enumerate(THEMES):
                pos_mask = lab[:, i] == 1
                neg_mask = lab[:, i] == 0
                z = logits_np[:, i]
                logit_stats[theme] = {
                    "mean": round(float(z.mean()), 3),
                    "pos_mean": round(float(z[pos_mask].mean()), 3) if pos_mask.any() else 0.0,
                    "neg_mean": round(float(z[neg_mask].mean()), 3) if neg_mask.any() else 0.0,
                    "separation": round(float(z[pos_mask].mean() - z[neg_mask].mean()), 3)
                                  if (pos_mask.any() and neg_mask.any()) else 0.0,
                }
            metrics["logit_stats"] = logit_stats

            # Mean predicted probability per theme — detect systematic under/over-prediction
            prob_stats = {}
            for i, theme in enumerate(THEMES):
                prob_stats[theme] = {
                    "mean_pred": round(float(probs[:, i].mean()), 4),
                    "mean_true": round(float(lab[:, i].mean()), 4),
                    "overconf": round(float(probs[:, i].mean() - lab[:, i].mean()), 4),
                }
            metrics["prob_stats"] = prob_stats

        return metrics

    def _log_epoch(self, epoch, loss_info, val, phase):
        prauc = val.get("prauc_macro", 0.0)
        f1 = val.get("f1_macro", 0.0)

        if isinstance(loss_info, dict):
            train_loss = loss_info.get("total", 0.0)
        else:
            train_loss = float(loss_info)
            loss_info = {"total": train_loss}

        entry = {
            "epoch": epoch, "phase": phase,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val["loss"], 6),
            "val_f1_macro": round(f1, 6),
            "val_prauc_macro": round(prauc, 6),
            "val_f1_per_theme": val.get("f1_per_theme", {}),
            "val_prauc_per_theme": val.get("prauc_per_theme", {}),
            "val_prec_per_theme": val.get("precision_per_theme", {}),
            "val_rec_per_theme": val.get("recall_per_theme", {}),
            "loss_components": {k: round(v, 6) for k, v in loss_info.items()},
        }
        self.history["epochs"].append(entry)

        # Main epoch line
        logger.info(
            "EPOCH P%d ep=%d train=%.4f val=%.4f f1=%.4f prauc=%.4f",
            phase, epoch, train_loss, val["loss"], f1, prauc,
        )

        # Loss component breakdown
        kl = loss_info.get("kl", 0.0)
        essay = loss_info.get("essay", 0.0)
        task = loss_info.get("task", train_loss)
        if kl > 0 or essay > 0:
            logger.info(
                "  loss_breakdown: task=%.4f kl=%.4f(×%.1f=%.4f) essay=%.4f(×%.2f=%.4f)",
                task, kl, self.rdrop_alpha, kl * self.rdrop_alpha,
                essay, self.essay_aux_weight, essay * self.essay_aux_weight,
            )

        # Gradient norm
        gn_mean = loss_info.get("grad_norm_mean", 0.0)
        gn_max = loss_info.get("grad_norm_max", 0.0)
        if gn_mean > 0:
            logger.info("  grad_norm: mean=%.3f max=%.3f (clip=%.1f)",
                        gn_mean, gn_max, self.max_grad_norm)

        # Per-theme F1
        f1_pt = val.get("f1_per_theme", {})
        prec_pt = val.get("precision_per_theme", {})
        rec_pt = val.get("recall_per_theme", {})
        if f1_pt:
            logger.info(
                "  per_theme F1:   %s",
                " ".join(f"{t[:3]}={f1_pt.get(t, 0):.3f}" for t in sorted(f1_pt)),
            )
        if prec_pt and rec_pt:
            logger.info(
                "  per_theme P:    %s",
                " ".join(f"{t[:3]}={prec_pt.get(t, 0):.3f}" for t in sorted(prec_pt)),
            )
            logger.info(
                "  per_theme R:    %s",
                " ".join(f"{t[:3]}={rec_pt.get(t, 0):.3f}" for t in sorted(rec_pt)),
            )

        # Per-theme PR-AUC
        prauc_pt = val.get("prauc_per_theme", {})
        if prauc_pt:
            logger.info(
                "  per_theme PRAUC:%s",
                " ".join(f"{t[:3]}={prauc_pt.get(t, 0):.3f}" for t in sorted(prauc_pt)),
            )

        # Class_0 (no-theme sentences)
        c0 = val.get("class_0")
        if c0:
            logger.info("  class_0: F1=%.3f P=%.3f R=%.3f (sup=%d)",
                        c0["f1"], c0["precision"], c0["recall"], c0.get("support", 0))

        # Logit separation — key diagnostic for overconfidence
        logit_stats = val.get("logit_stats", {})
        if logit_stats:
            seps = [(t, logit_stats[t]["separation"]) for t in THEMES if t in logit_stats]
            seps.sort(key=lambda x: x[1])
            logger.info(
                "  logit_sep (pos-neg): %s",
                " ".join(f"{t[:3]}={s:.2f}" for t, s in seps),
            )
            # Flag themes where logit separation is dangerously low
            bad_seps = [(t, s) for t, s in seps if s < 0.5]
            if bad_seps:
                logger.warning(
                    "  LOW SEPARATION (< 0.5): %s — model may not distinguish these themes",
                    ", ".join(f"{t}({s:.2f})" for t, s in bad_seps),
                )

        # Prediction calibration: mean predicted prob vs mean true rate
        prob_stats = val.get("prob_stats", {})
        if prob_stats:
            overcalls = [(t, prob_stats[t]["mean_pred"], prob_stats[t]["mean_true"],
                          prob_stats[t]["overconf"])
                         for t in THEMES if t in prob_stats]
            overcalls.sort(key=lambda x: abs(x[3]), reverse=True)
            logger.info(
                "  pred_vs_true (mean): %s",
                " ".join(f"{t[:3]}={pred:.3f}/{true:.3f}" for t, pred, true, _ in overcalls[:5]),
            )

        if prauc > self.best_prauc_val:
            self.best_prauc_val = prauc
            self.history["best_epoch"] = epoch
            self.history["best_prauc"] = round(prauc, 6)
            self.history["best_phase"] = phase
            self.patience_counter = 0
            torch.save(self.model.state_dict(), self.output_dir / "best.pt")
            logger.info("  NEW BEST (PRAUC=%.4f, phase %d)", prauc, phase)
        else:
            self.patience_counter += 1

    def _check_early_stop(self, val, phase=1):
        patience = self.phase2_patience if phase == 2 else self.early_stopping_patience
        return self.patience_counter >= patience

    def _save_history(self):
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def _run_threshold_optimization(self):
        best_path = self.output_dir / "best.pt"
        if not best_path.exists():
            return
        self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
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
                all_logits.append(fl)
                all_labels.append(ll)

        if not all_logits:
            return
        logits_np = torch.cat(all_logits).cpu().numpy()
        labels_np = torch.cat(all_labels).cpu().numpy()

        logger.info("Running per-class Platt scaling calibration...")
        platt_params, calibrated_probs = calibrate_per_class(logits_np, labels_np)

        platt_dict = {THEMES[i]: {"a": round(float(platt_params[0, i]), 4),
                                   "b": round(float(platt_params[1, i]), 4)}
                      for i in range(len(THEMES))}
        with open(self.output_dir / "calibration.json", "w") as f:
            json.dump(platt_dict, f, indent=2)

        # Log calibration params — large negative b = overconfident model
        logger.info("Platt calibration params (b near 0 = well-calibrated):")
        for theme, params in platt_dict.items():
            calibration_quality = "OK" if abs(params["b"]) < 1.0 else ("BIASED" if abs(params["b"]) < 2.0 else "SEVERELY_BIASED")
            logger.info("  %-20s a=%.2f b=%.2f  [%s]", theme, params["a"], params["b"], calibration_quality)

        thresholds = optimize_thresholds(calibrated_probs, labels_np)
        with open(self.output_dir / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)
        logger.info("Thresholds (calibrated): %s", thresholds)

        # Co-occurrence confusion analysis: for each true theme, what does the model predict?
        logger.info("=" * 60)
        logger.info("CO-OCCURRENCE CONFUSION ANALYSIS (val set):")
        logger.info("  For each TRUE theme, shows which other themes are co-predicted.")
        logger.info("  High co-prediction with a wrong theme = theme confusion.")
        from metrics import apply_thresholds
        preds = apply_thresholds(calibrated_probs, thresholds)
        _log_cooccurrence_confusion(preds, labels_np)
        logger.info("=" * 60)


def _compute_essay_labels(labels: torch.Tensor, sentence_mask: torch.Tensor) -> torch.Tensor:
    """Compute essay-level multi-hot labels as OR over valid sentence labels.

    Args:
        labels:        [B, S, C] — per-sentence multi-hot labels
        sentence_mask: [B, S]   — 1.0 for valid sentences, 0.0 for padding

    Returns:
        essay_labels: [B, C] — 1.0 if theme appears in ANY valid sentence
    """
    valid_labels = labels * sentence_mask.unsqueeze(-1).float()  # [B, S, C]
    return valid_labels.max(dim=1).values  # [B, C]


def _log_cooccurrence_confusion(preds: np.ndarray, labels: np.ndarray):
    """Log: for each true positive theme, what fraction of sentences also get OTHER themes predicted?

    This reveals:
    - False positives: theme A predicted when only theme B is true
    - Confusion pairs: Aspirational predicted when Attainment is true (and vice versa)
    """
    N, C = preds.shape
    for i, true_theme in enumerate(THEMES):
        true_pos_idx = np.where(labels[:, i] == 1)[0]  # sentences where theme i is true
        n_true = len(true_pos_idx)
        if n_true == 0:
            continue

        # How often does the model correctly predict true_theme when it's true?
        recall_i = preds[true_pos_idx, i].mean()

        # For sentences where true_theme is true, which other themes get predicted?
        co_preds = []
        for j, other_theme in enumerate(THEMES):
            if j == i:
                continue
            rate = preds[true_pos_idx, j].mean()
            co_preds.append((other_theme, rate))
        co_preds.sort(key=lambda x: x[1], reverse=True)

        top_co = [(t, r) for t, r in co_preds if r > 0.05][:4]
        logger.info(
            "  TRUE %-20s (n=%3d, recall=%.2f): co-predicted: %s",
            true_theme, n_true, recall_i,
            " ".join(f"{t[:3]}={r:.2f}" for t, r in top_co) if top_co else "none",
        )

    # Also: false positive analysis — when theme is predicted but not true
    logger.info("  FALSE POSITIVE ANALYSIS (pred=1 but true=0):")
    for i, theme in enumerate(THEMES):
        fp_idx = np.where((preds[:, i] == 1) & (labels[:, i] == 0))[0]
        n_fp = len(fp_idx)
        n_pred = int(preds[:, i].sum())
        if n_pred == 0:
            logger.info("    %-20s NEVER PREDICTED", theme)
            continue
        precision_i = (n_pred - n_fp) / n_pred
        logger.info("    %-20s n_pred=%3d FP=%3d precision=%.3f", theme, n_pred, n_fp, precision_i)
