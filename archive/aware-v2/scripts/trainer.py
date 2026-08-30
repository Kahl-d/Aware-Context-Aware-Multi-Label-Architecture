"""
Two-phase AWARE v2 trainer.
Phase 1: frozen encoder (train BiLSTM + head).
Phase 2: full fine-tune with differential LR.

Uses linear warmup + cosine decay (standard for transformers).
Logs history.json, saves best checkpoint.
"""

import sys
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
from typing import Callable, Dict, Optional

from model import AWAREModel, build_model_from_config
from losses import AsymmetricLoss, build_loss_from_config
from metrics import (
    flatten_masked_preds_labels,
    flatten_masked_logits_labels,
    compute_metrics,
    compute_prauc,
    optimize_thresholds,
)

logger = logging.getLogger(__name__)


def build_llrd_param_groups(
    model: AWAREModel,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float,
    llrd_decay: float,
) -> list:
    """Build parameter groups with Layer-Wise Learning Rate Decay (LLRD).

    Each encoder layer gets LR = encoder_lr × decay^(num_layers - layer_idx).
    Lower layers (pretrained knowledge) get smaller LR, upper layers adapt faster.
    BiLSTM + classifier head use decoder_lr (unchanged).
    Bias and LayerNorm params get weight_decay=0.0.

    Falls back to 2-group if encoder layer detection fails.

    Reference: Clark et al. (2020), "Electra"; Howard & Ruder (2018), "ULMFiT"
    """
    # Detect encoder layers
    encoder = model.encoder
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
        num_layers = len(layers)
    else:
        logger.warning("LLRD: cannot detect encoder layers, falling back to 2-group")
        return None  # Caller will use original 2-group

    logger.info("LLRD: detected %d encoder layers, decay=%.3f", num_layers, llrd_decay)

    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    param_groups = []

    # Group 1: Embeddings — lowest LR (deepest pretrained knowledge)
    emb_lr = encoder_lr * (llrd_decay ** num_layers)
    emb_params_decay = []
    emb_params_no_decay = []
    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        # Embedding params: anything not in encoder.layer.* (e.g. embeddings, encoder.rel_embeddings)
        if ".layer." not in name:
            if any(nd in name for nd in no_decay):
                emb_params_no_decay.append(param)
            else:
                emb_params_decay.append(param)

    if emb_params_decay:
        param_groups.append({"params": emb_params_decay, "lr": emb_lr, "weight_decay": weight_decay})
    if emb_params_no_decay:
        param_groups.append({"params": emb_params_no_decay, "lr": emb_lr, "weight_decay": 0.0})

    # Group 2: Each encoder layer — LR increases with depth
    for layer_idx in range(num_layers):
        layer_lr = encoder_lr * (llrd_decay ** (num_layers - layer_idx))
        layer_params_decay = []
        layer_params_no_decay = []
        layer_prefix = f".layer.{layer_idx}."

        for name, param in encoder.named_parameters():
            if not param.requires_grad:
                continue
            if layer_prefix in name:
                if any(nd in name for nd in no_decay):
                    layer_params_no_decay.append(param)
                else:
                    layer_params_decay.append(param)

        if layer_params_decay:
            param_groups.append({"params": layer_params_decay, "lr": layer_lr, "weight_decay": weight_decay})
        if layer_params_no_decay:
            param_groups.append({"params": layer_params_no_decay, "lr": layer_lr, "weight_decay": 0.0})

    # Group 3: Non-encoder params (BiLSTM, classifier head) — decoder_lr
    decoder_params_decay = []
    decoder_params_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            continue  # Already handled above
        if any(nd in name for nd in no_decay):
            decoder_params_no_decay.append(param)
        else:
            decoder_params_decay.append(param)

    if decoder_params_decay:
        param_groups.append({"params": decoder_params_decay, "lr": decoder_lr, "weight_decay": weight_decay})
    if decoder_params_no_decay:
        param_groups.append({"params": decoder_params_no_decay, "lr": decoder_lr, "weight_decay": 0.0})

    # Log summary
    total_params = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
    logger.info("LLRD: %d param groups, %d total params", len(param_groups), total_params)
    logger.info("  Embeddings LR: %.2e", emb_lr)
    logger.info("  Layer 0 LR:    %.2e", encoder_lr * (llrd_decay ** num_layers))
    logger.info("  Layer %d LR:   %.2e", num_layers - 1, encoder_lr * llrd_decay)
    logger.info("  Decoder LR:    %.2e", decoder_lr)

    return param_groups


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AWARETrainer:
    def __init__(
        self,
        model: AWAREModel,
        train_loader,
        val_loader,
        output_dir: str,
        config,
        device: str = None,
        train_loader_factory: Optional[Callable[[int], object]] = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader_factory = train_loader_factory
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

        self.criterion = build_loss_from_config(config).to(self.device)
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

        # R-Drop regularization (Liang & Wu, NeurIPS 2021)
        self.rdrop_alpha = float(getattr(T, "rdrop_alpha", 0.0))
        if self.rdrop_alpha > 0:
            logger.info("R-Drop enabled: alpha=%.2f", self.rdrop_alpha)

        # EMA model (reduces prediction variance, especially for rare themes)
        self.ema_decay = float(getattr(T, "ema_decay", 0.0))
        self.ema_state = None
        if self.ema_decay > 0:
            logger.info("EMA enabled: decay=%.4f", self.ema_decay)

        self.history = {"epochs": [], "best_epoch": 0, "best_prauc": 0.0, "best_phase": 1}
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.best_phase = 1
        self.patience_counter = 0
        self.phase2_patience = int(
            getattr(T, "phase2_early_stopping_patience", None) or self.early_stopping_patience
        )

    def train(self):
        set_seeds(self.config.seed)
        logger.info(
            "Starting two-phase training (phase1=%d, phase2=%d, device=%s, fp16=%s)",
            self.phase1_epochs, self.phase2_epochs, self.device, self.use_fp16,
        )

        # ── Phase 1: frozen encoder — train BiLSTM + head ──
        self.model._freeze_encoder()
        self.patience_counter = 0
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.decoder_lr,
            weight_decay=self.weight_decay,
        )
        accum = max(1, self.gradient_accumulation)
        steps_per_epoch = max(1, (len(self.train_loader) + accum - 1) // accum)
        total_steps = steps_per_epoch * self.phase1_epochs
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        logger.info(
            "PHASE_START phase=1 steps_per_epoch=%d total_steps=%d warmup_steps=%d",
            steps_per_epoch, total_steps, warmup_steps,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

        for epoch in range(1, self.phase1_epochs + 1):
            self.current_epoch = epoch
            if self.train_loader_factory is not None:
                self.train_loader = self.train_loader_factory(epoch)
            train_loss = self._train_epoch(optimizer, scheduler, phase=1)
            val_metrics = self._validate()
            self._log_epoch(epoch, train_loss, val_metrics, phase=1)
            if self._check_early_stop(val_metrics, phase=1):
                logger.info("Early stopping at epoch %d (phase 1)", epoch)
                break

        # ── Phase 2: unfreeze encoder — full fine-tune with differential LR ──
        self.model.unfreeze_encoder()
        self.patience_counter = 0
        encoder_lr_phase2 = self.encoder_lr * self.phase2_encoder_lr_scale

        # LLRD: per-layer learning rates (recommended for v3-large, 24 layers)
        llrd_decay = float(getattr(self.config.training, "llrd_decay", 0.0))
        if llrd_decay > 0:
            param_groups = build_llrd_param_groups(
                self.model,
                encoder_lr=encoder_lr_phase2,
                decoder_lr=self.decoder_lr,
                weight_decay=self.weight_decay,
                llrd_decay=llrd_decay,
            )
        else:
            param_groups = None

        if param_groups is not None:
            logger.info(
                "Phase 2: LLRD enabled (decay=%.3f), encoder_lr=%.2e (scale=%.2f), decoder_lr=%.2e",
                llrd_decay, encoder_lr_phase2, self.phase2_encoder_lr_scale, self.decoder_lr,
            )
            optimizer = AdamW(param_groups)
        else:
            # Original 2-group — backward compatible with v3-base configs
            encoder_params = [p for n, p in self.model.named_parameters() if "encoder" in n]
            other_params = [p for n, p in self.model.named_parameters() if "encoder" not in n]
            logger.info(
                "Phase 2: 2-group LR, encoder_lr=%.2e (scale=%.2f), decoder_lr=%.2e",
                encoder_lr_phase2, self.phase2_encoder_lr_scale, self.decoder_lr,
            )
            optimizer = AdamW(
                [
                    {"params": encoder_params, "lr": encoder_lr_phase2},
                    {"params": other_params, "lr": self.decoder_lr},
                ],
                weight_decay=self.weight_decay,
            )
        steps_per_epoch_p2 = max(1, (len(self.train_loader) + accum - 1) // accum)
        total_steps_p2 = steps_per_epoch_p2 * self.phase2_epochs
        warmup_steps_p2 = max(1, int(total_steps_p2 * self.warmup_ratio))
        logger.info(
            "PHASE_START phase=2 steps_per_epoch=%d total_steps=%d warmup_steps=%d",
            steps_per_epoch_p2, total_steps_p2, warmup_steps_p2,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps_p2, num_training_steps=total_steps_p2,
        )

        for epoch in range(self.phase1_epochs + 1, self.phase1_epochs + self.phase2_epochs + 1):
            self.current_epoch = epoch
            if self.train_loader_factory is not None:
                self.train_loader = self.train_loader_factory(epoch)
            train_loss = self._train_epoch(optimizer, scheduler, phase=2)
            val_metrics = self._validate()
            self._log_epoch(epoch, train_loss, val_metrics, phase=2)
            if self._check_early_stop(val_metrics, phase=2):
                logger.info("Early stopping at epoch %d (phase 2)", epoch)
                break

        # ── Tau-normalization: zero-cost post-hoc classifier calibration ──
        self._tau_normalize()

        # ── Phase 3: frozen representation — retrain classifier with balanced weights ──
        self._train_phase3_balanced_head()

        self._save_history()
        self._run_threshold_optimization()
        logger.info(
            "Training complete. Best PRAUC: %.4f at epoch %d (phase %d)",
            self.history.get("best_prauc", 0.0),
            self.history["best_epoch"],
            self.history.get("best_phase", 1),
        )
        return self.history

    def _train_phase3_balanced_head(self, epochs: int = 5):
        """Phase 3: Retrain classifier head with balanced theme weights.

        After Phase 2, the encoder has learned good representations but the
        classifier head is biased toward common themes (97% of gradient came
        from common themes during training). Freezing everything and retraining
        only the head with balanced weights fixes this bias.

        Safe to use aggressive weights here because encoder is frozen — can't
        cause the probability inflation that destroyed Q005.

        Reference: Kang et al., "Decoupling Representation and Classifier for
        Long-Tailed Recognition" (ICLR 2020).
        """
        import shutil

        best_path = self.output_dir / "best.pt"
        if not best_path.exists():
            logger.warning("No best.pt for Phase 3, skipping")
            return

        # Backup Phase 2 model — Phase 3 must BEAT it to replace it
        phase2_path = self.output_dir / "best_phase2.pt"
        shutil.copy2(best_path, phase2_path)
        phase2_prauc = self.best_f1  # best_f1 stores best PR-AUC

        self.model.load_state_dict(
            torch.load(best_path, map_location=self.device, weights_only=True)
        )

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # Reinitialize classification head (fresh weights, requires_grad=True)
        from model import ClassificationHead
        from config import NUM_THEMES
        self.model.classifier = ClassificationHead(
            hidden_size=self.model.hidden_size,
            num_labels=NUM_THEMES,
            dropout=self.config.model.dropout,
        ).to(self.device)

        # Balanced loss: sqrt-inverse-frequency weights.
        # Encoder frozen → aggressive weights are safe (only 8.5k head params affected).
        theme_counts = torch.tensor(
            [7879, 5434, 5163, 4732, 944, 263, 698, 2432, 770, 170, 114],
            dtype=torch.float32,
        )
        balanced_weights = torch.sqrt(theme_counts.max() / theme_counts)
        balanced_criterion = AsymmetricLoss(
            gamma_pos=0.0, gamma_neg=4.0, clip=0.05,
            theme_weights=balanced_weights,
        ).to(self.device)

        optimizer = AdamW(
            self.model.classifier.parameters(), lr=1e-3, weight_decay=0.01,
        )

        logger.info(
            "PHASE_START phase=3 (balanced head retraining, %d epochs)", epochs,
        )
        logger.info(
            "  Phase 2 best PRAUC=%.4f (must beat this to replace)",
            phase2_prauc,
        )
        logger.info(
            "  balanced_weights: %s",
            {t: round(w, 1) for t, w in zip(
                ["Nav", "Att", "Per", "Asp", "Soc", "FP", "Spi", "Fam", "Res", "CC", "FG"],
                balanced_weights.tolist(),
            )},
        )

        phase3_start_epoch = self.current_epoch + 1
        best_p3_prauc = 0.0
        patience = 3
        phase3_path = self.output_dir / "best_phase3.pt"

        for ep in range(1, epochs + 1):
            self.current_epoch = phase3_start_epoch + ep - 1
            self.model.train()
            total_loss, n = 0.0, 0

            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sentence_mask = batch["sentence_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                sentence_boundaries = batch["sentence_boundaries"]

                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        sentence_boundaries=sentence_boundaries,
                        sentence_mask=sentence_mask,
                    )
                    loss = balanced_criterion(
                        logits=output["logits"],
                        targets=labels,
                        mask=sentence_mask,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.classifier.parameters(), 1.0,
                )
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                n += 1

            val_metrics = self._validate()
            train_loss = total_loss / max(n, 1)
            self._log_epoch(self.current_epoch, train_loss, val_metrics, phase=3)

            p3_prauc = val_metrics.get("prauc_macro", 0.0)
            if p3_prauc > best_p3_prauc:
                best_p3_prauc = p3_prauc
                torch.save(self.model.state_dict(), phase3_path)
                logger.info("  Phase 3 new best (PRAUC=%.4f, F1=%.4f)", best_p3_prauc, val_metrics["f1_macro"])
                patience = 3
            else:
                patience -= 1
                if patience <= 0:
                    logger.info("  Phase 3 early stop (no improvement for 3 epochs)")
                    break

        # Only use Phase 3 model if it BEATS Phase 2
        if best_p3_prauc > phase2_prauc:
            shutil.copy2(phase3_path, best_path)
            logger.info(
                "Phase 3 IMPROVED: PRAUC %.4f → %.4f (+%.4f). Using Phase 3 model.",
                phase2_prauc, best_p3_prauc, best_p3_prauc - phase2_prauc,
            )
        else:
            # Restore Phase 2 model
            shutil.copy2(phase2_path, best_path)
            logger.info(
                "Phase 3 did NOT improve (%.4f vs Phase 2 %.4f). Keeping Phase 2 model.",
                best_p3_prauc, phase2_prauc,
            )

        # Load whichever model won
        self.model.load_state_dict(
            torch.load(best_path, map_location=self.device, weights_only=True)
        )
        logger.info("Phase 3 complete. Final model PRAUC=%.4f", max(best_p3_prauc, phase2_prauc))

    # ── Tau-normalization ──────────────────────────────────────────────────
    def _tau_normalize(self, tau_values=None):
        """Post-hoc classifier weight rescaling (Kang et al., ICLR 2020).

        Sweeps tau on val set. tau=0: no change. tau=1: unit-normalize all rows.
        Rare themes tend to have smaller weight norms → tau>0 boosts them.
        Zero training cost. Fully reversible.
        """
        if tau_values is None:
            tau_values = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

        best_path = self.output_dir / "best.pt"
        if not best_path.exists():
            logger.info("Tau-norm: no best.pt, skipping")
            return

        self.model.load_state_dict(
            torch.load(best_path, map_location=self.device, weights_only=True)
        )

        head = self.model.classifier
        if hasattr(head, "classifier"):
            W = head.classifier.weight.data
        elif hasattr(head, "label_queries"):
            W = head.label_queries.data
        else:
            logger.info("Tau-norm: unknown head type, skipping")
            return

        norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
        original_W = W.clone()

        from config import THEMES
        logger.info("Tau-norm: weight norms: %s",
            {THEMES[i][:3]: f"{norms[i].item():.3f}" for i in range(min(len(THEMES), len(norms)))})

        pre_tau_prauc = self.best_f1  # best_f1 now stores best PR-AUC
        best_tau, best_prauc = 0.0, 0.0
        for tau in tau_values:
            W.copy_(original_W / (norms ** tau) if tau > 0 else original_W)
            val_metrics = self._validate()
            prauc = val_metrics.get("prauc_macro", 0.0)
            logger.info("  tau=%.1f → val_prauc=%.4f val_f1=%.4f", tau, prauc, val_metrics["f1_macro"])
            if prauc > best_prauc:
                best_tau = tau
                best_prauc = prauc

        # Apply best tau
        W.copy_(original_W / (norms ** best_tau) if best_tau > 0 else original_W)
        logger.info("Tau-norm: best_tau=%.1f, val_prauc=%.4f (pre-tau=%.4f)",
            best_tau, best_prauc, pre_tau_prauc)

        if best_prauc > pre_tau_prauc:
            torch.save(self.model.state_dict(), best_path)
            self.best_f1 = best_prauc
            self.history["best_prauc"] = round(best_prauc, 6)
            self.history["tau_norm"] = {"best_tau": best_tau, "prauc": round(best_prauc, 6)}
            logger.info("Tau-norm IMPROVED: %.4f → %.4f. Saved.", pre_tau_prauc, best_prauc)
        else:
            # Restore original
            W.copy_(original_W)
            logger.info("Tau-norm did not improve. Keeping original.")

    # ── EMA helpers ────────────────────────────────────────────────────────
    def _update_ema(self):
        """Update exponential moving average of model weights."""
        if self.ema_decay <= 0:
            return
        if self.ema_state is None:
            self.ema_state = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
        else:
            for k, v in self.model.state_dict().items():
                self.ema_state[k].mul_(self.ema_decay).add_(v, alpha=1 - self.ema_decay)

    def _swap_to_ema(self):
        """Swap model weights with EMA for evaluation. Call _swap_from_ema to restore."""
        if self.ema_state is None:
            return False
        self._saved_state = self.model.state_dict()
        self.model.load_state_dict(self.ema_state)
        return True

    def _swap_from_ema(self):
        """Restore training weights after EMA evaluation."""
        if hasattr(self, "_saved_state"):
            self.model.load_state_dict(self._saved_state)
            del self._saved_state

    # ── R-Drop KL loss ───────────────────────────────────────────────────
    def _rdrop_kl_loss(self, p1, p2, mask):
        """Symmetric KL divergence for binary (sigmoid) distributions.

        For each element: KL(p1||p2) = p1*log(p1/p2) + (1-p1)*log((1-p1)/(1-p2))
        Symmetrized: (KL(p1||p2) + KL(p2||p1)) / 2
        """
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

    def _train_epoch(self, optimizer, scheduler, phase: int = 1) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        accum = max(1, self.gradient_accumulation)
        optimizer_steps_per_epoch = max(1, (len(self.train_loader) + accum - 1) // accum)
        log_step_interval = max(1, min(200, optimizer_steps_per_epoch // 4))
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} (P{phase})",
            leave=False,
            file=sys.stdout,
        )
        optimizer.zero_grad()
        opt_step = 0
        use_rdrop = self.rdrop_alpha > 0

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sentence_mask = batch["sentence_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            sentence_boundaries = batch["sentence_boundaries"]

            fwd_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_boundaries=sentence_boundaries,
                sentence_mask=sentence_mask,
            )

            if self.use_fp16:
                with torch.amp.autocast("cuda"):
                    output1 = self.model(**fwd_kwargs)
                    loss1 = self.criterion(output1["logits"], labels, sentence_mask)
                    if use_rdrop:
                        output2 = self.model(**fwd_kwargs)
                        loss2 = self.criterion(output2["logits"], labels, sentence_mask)
                        p1 = torch.sigmoid(output1["logits"])
                        p2 = torch.sigmoid(output2["logits"])
                        kl = self._rdrop_kl_loss(p1, p2, sentence_mask)
                        loss = (loss1 + loss2) / 2 + self.rdrop_alpha * kl
                    else:
                        loss = loss1
                loss = loss / accum
                self.scaler.scale(loss).backward()
            else:
                output1 = self.model(**fwd_kwargs)
                loss1 = self.criterion(output1["logits"], labels, sentence_mask)
                if use_rdrop:
                    output2 = self.model(**fwd_kwargs)
                    loss2 = self.criterion(output2["logits"], labels, sentence_mask)
                    p1 = torch.sigmoid(output1["logits"])
                    p2 = torch.sigmoid(output2["logits"])
                    kl = self._rdrop_kl_loss(p1, p2, sentence_mask)
                    loss = (loss1 + loss2) / 2 + self.rdrop_alpha * kl
                else:
                    loss = loss1
                loss = loss / accum
                loss.backward()

            total_loss += loss.item() * accum
            n += 1

            if (step + 1) % accum == 0 or (step + 1) == len(self.train_loader):
                opt_step += 1
                if opt_step % log_step_interval == 0 or opt_step == optimizer_steps_per_epoch:
                    logger.info(
                        "STEP phase=%d epoch=%d opt_step=%d/%d loss=%.4f",
                        phase, self.current_epoch, opt_step, optimizer_steps_per_epoch,
                        total_loss / n,
                    )

                # Gradient norm logging (per-theme, once per log interval)
                if opt_step % log_step_interval == 0:
                    self._log_gradient_norms()

                if self.use_fp16:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update EMA after every optimizer step (NOT per-epoch).
                # decay=0.999 is calibrated for per-step updates (~1200 steps/epoch).
                # Per-epoch updates caused Q009 failure: 0.999^24 = 97.6% initial weights retained.
                self._update_ema()

            pbar.set_postfix(loss=f"{(total_loss / max(n, 1)):.4f}")

        return total_loss / max(n, 1)

    def _log_gradient_norms(self):
        """Log per-theme gradient norms of classification head weights."""
        try:
            # Works for both ClassificationHead (has .classifier) and future heads
            head = self.model.classifier
            if hasattr(head, "classifier"):
                W_grad = head.classifier.weight.grad
            elif hasattr(head, "label_queries"):
                W_grad = head.label_queries.grad
            else:
                return
            if W_grad is None:
                return
            from config import THEMES
            norms = W_grad.norm(dim=1).detach().cpu()
            parts = [f"{THEMES[i][:3]}={norms[i]:.4f}" for i in range(min(len(THEMES), len(norms)))]
            logger.info("  GRAD_NORMS: %s", " ".join(parts))
        except Exception:
            pass  # diagnostic only, never crash training

    @torch.no_grad()
    def _validate(self) -> Dict:
        # Use EMA model for validation if available
        used_ema = self._swap_to_ema() if self.ema_decay > 0 else False
        self.model.eval()
        all_preds, all_labels = [], []
        all_probs_list = []
        total_loss, n_batches = 0.0, 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sentence_mask = batch["sentence_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            sentence_boundaries = batch["sentence_boundaries"]

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_boundaries=sentence_boundaries,
                sentence_mask=sentence_mask,
            )
            loss = self.criterion(
                logits=output["logits"],
                targets=labels,
                mask=sentence_mask,
            )
            total_loss += loss.item()
            n_batches += 1
            p, l = flatten_masked_preds_labels(output["logits"], labels, sentence_mask)
            all_preds.append(p)
            all_labels.append(l)
            # Collect raw probabilities for diagnostics
            mask_bool = sentence_mask.bool()
            flat_probs = torch.sigmoid(output["logits"])[mask_bool].cpu().numpy()
            all_probs_list.append(flat_probs)

        if all_preds:
            preds = np.vstack(all_preds)
            lab = np.vstack(all_labels)
            all_probs = np.vstack(all_probs_list)
            metrics = compute_metrics(preds, lab)
            # PR-AUC: threshold-independent metric for model selection
            prauc = compute_prauc(all_probs, lab)
            metrics["prauc_macro"] = prauc["prauc_macro"]
            metrics["prauc_per_theme"] = prauc["prauc_per_theme"]
            # Per-theme probability diagnostics (logged every epoch)
            self._log_prob_diagnostics(all_probs, lab)
        else:
            metrics = {"f1_macro": 0.0, "f1_micro": 0.0, "f1_per_theme": {},
                       "prauc_macro": 0.0, "prauc_per_theme": {}}
        metrics["loss"] = total_loss / max(n_batches, 1)

        # Restore training weights after EMA validation
        if used_ema:
            self._swap_from_ema()

        return metrics

    def _log_prob_diagnostics(self, probs: np.ndarray, labels: np.ndarray):
        """Log per-theme probability stats for positive vs negative examples.

        This reveals whether the model produces separated distributions for each
        theme. Good separation = high positive mean, low negative mean.
        Poor separation = overlapping distributions → threshold can't help.
        """
        from config import THEMES
        # Focus on themes that need debugging (rare + mid-tier)
        debug_themes = {"Social", "Filial Piety", "Spiritual", "Resistance",
                        "Community Consciousness", "First Gen"}
        parts = []
        for i, theme in enumerate(THEMES):
            if theme not in debug_themes:
                continue
            pos_mask = labels[:, i] == 1
            neg_mask = labels[:, i] == 0
            pos_probs = probs[pos_mask, i] if pos_mask.any() else np.array([0.0])
            neg_probs = probs[neg_mask, i] if neg_mask.any() else np.array([0.0])
            sep = pos_probs.mean() - neg_probs.mean()  # separation metric
            parts.append(
                "%s: p+%.3f±%.3f p-%.3f sep=%.3f n+=%d" % (
                    theme[:6],
                    pos_probs.mean(), pos_probs.std(),
                    neg_probs.mean(), sep,
                    int(pos_mask.sum()),
                )
            )
        if parts:
            logger.info("  prob_diag: %s", " | ".join(parts))

    def _log_epoch(self, epoch: int, train_loss: float, val_metrics: Dict, phase: int):
        prauc_macro = val_metrics.get("prauc_macro", 0.0)
        entry = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_f1_macro": round(val_metrics["f1_macro"], 6),
            "val_f1_micro": round(val_metrics.get("f1_micro", 0), 6),
            "val_prauc_macro": round(prauc_macro, 6),
            "val_f1_per_theme": val_metrics.get("f1_per_theme", {}),
            "val_prauc_per_theme": val_metrics.get("prauc_per_theme", {}),
        }
        self.history["epochs"].append(entry)
        logger.info(
            "EPOCH phase=%d epoch=%d train_loss=%.4f val_loss=%.4f val_f1_macro=%.4f val_prauc_macro=%.4f",
            phase, epoch, train_loss, val_metrics["loss"], val_metrics["f1_macro"], prauc_macro,
        )
        f1_pt = val_metrics.get("f1_per_theme") or {}
        if f1_pt:
            theme_line = " ".join("%s=%.3f" % (t, f1_pt[t]) for t in sorted(f1_pt.keys()))
            logger.info("  per_theme_f1: %s", theme_line)
        c0 = val_metrics.get("class_0")
        if c0:
            logger.info("  class_0: F1=%.3f P=%.3f R=%.3f", c0["f1"], c0["precision"], c0["recall"])
        # Model selection: use PR-AUC (threshold-independent) instead of F1@0.5
        # F1@0.5 is misleading for rare themes whose probabilities never reach 0.5
        if prauc_macro > self.best_f1:
            self.best_f1 = prauc_macro
            self.best_phase = phase
            self.history["best_epoch"] = epoch
            self.history["best_prauc"] = round(prauc_macro, 6)
            self.history["best_f1_at_best_prauc"] = round(val_metrics["f1_macro"], 6)
            self.history["best_phase"] = phase
            # Reset early stopping counter on improvement.
            # Critical: this must happen HERE because _check_early_stop() runs AFTER
            # this method updates self.best_f1, so its f1 > self.best_f1 check would
            # see f1 == self.best_f1 (just updated) and fail to reset on its own.
            self.patience_counter = 0
            # Save EMA weights if available (since validation used EMA), else training weights
            if self.ema_state is not None:
                torch.save(self.ema_state, self.output_dir / "best.pt")
                logger.info("  New best EMA model saved (PRAUC=%.4f, F1=%.4f, phase %d)", prauc_macro, val_metrics["f1_macro"], phase)
            else:
                torch.save(self.model.state_dict(), self.output_dir / "best.pt")
                logger.info("  New best model saved (PRAUC=%.4f, F1=%.4f, phase %d)", prauc_macro, val_metrics["f1_macro"], phase)

    def _check_early_stop(self, val_metrics: Dict, phase: int = 1) -> bool:
        """Check early stopping using PR-AUC (threshold-independent)."""
        patience = self.phase2_patience if phase == 2 else self.early_stopping_patience
        prauc = val_metrics.get("prauc_macro", 0.0)
        if prauc > self.best_f1:  # best_f1 now stores best PR-AUC
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return self.patience_counter >= patience

    def _save_history(self):
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def _run_threshold_optimization(self):
        """Load best model (EMA or training), collect val logits, optimize per-theme thresholds."""
        best_path = self.output_dir / "best.pt"
        if not best_path.exists():
            logger.warning("No best.pt found, skipping threshold optimization")
            return
        self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        self.model.eval()
        logger.info("Loaded best model for threshold optimization (EMA=%s)", self.ema_state is not None)
        all_logits_list, all_labels_list = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sentence_mask = batch["sentence_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                sentence_boundaries = batch["sentence_boundaries"]
                output = self.model(
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
        if not all_logits_list:
            logger.warning("No val data for threshold optimization")
            return
        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)
        all_probs = torch.sigmoid(all_logits).cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        # Log probability distribution stats before threshold optimization
        from config import THEMES
        logger.info("Probability distributions on val (best model):")
        for i, theme in enumerate(THEMES):
            pos_mask = all_labels_np[:, i] == 1
            pos_p = all_probs[pos_mask, i] if pos_mask.any() else np.array([0.0])
            neg_p = all_probs[~pos_mask, i]
            logger.info(
                "  %s: n+=%d p+=[%.3f ± %.3f, min=%.3f, q25=%.3f, q50=%.3f, max=%.3f] "
                "p-=[%.3f ± %.3f]",
                theme, int(pos_mask.sum()),
                pos_p.mean(), pos_p.std(), pos_p.min(),
                np.percentile(pos_p, 25), np.percentile(pos_p, 50), pos_p.max(),
                neg_p.mean(), neg_p.std(),
            )

        thresholds = optimize_thresholds(all_probs, all_labels_np)
        with open(self.output_dir / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)
        logger.info("Saved per-theme thresholds: %s", thresholds)
