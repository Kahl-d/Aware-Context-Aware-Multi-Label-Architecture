# Q011: "Principled Loss" — Detailed Implementation Guide

## Goal
Replace hand-tuned heuristics with research-backed formulas:
1. **Class-Balanced Loss weights** — replaces 11 manual theme_weights with one formula
2. **Logit Adjustment** — shifts logits by class prior during training
3. **Tau-normalization** — post-hoc classifier weight rescaling (zero training cost)
4. **LWS (Learnable Weight Scaling)** — replaces Phase 3 full head reinit

---

## Problems Being Solved

**Problem 4: Hand-Tuned Theme Weights Are Fragile**
Q005 proved aggressive manual weights (CC=15, FG=20) cause catastrophe. Q006's moderate
weights (CC=9, FG=10) work but are arbitrary. Manual tuning = one run per config.

**Problem 5: Phase 3 Head Retraining Can Be Improved**
Reinitializing head discards Phase 2 knowledge. Kang et al. (ICLR 2020) showed that
fine-tuning from existing weights (cRT) or learning per-class scales (LWS) outperforms
random reinitialization.

---

## Step-by-Step Implementation

### Step 1: Add CB Loss weight computation to losses.py

**Where**: Before `build_loss_from_config()` (line 93), add new function.

```python
def compute_cb_weights(
    class_counts: list,
    beta: float = 0.999,
    normalize: bool = True,
) -> torch.Tensor:
    """Class-Balanced Loss weights via effective number of samples.

    Formula: effective_n = (1 - beta^n) / (1 - beta)
             weight = 1.0 / effective_n

    With beta=0.999 and our class counts:
      Nav: 1.00, Att: 1.45, Per: 1.53, Asp: 1.67, Soc: 8.36
      FP: 26.9, Spi: 11.2, Fam: 3.25, Res: 10.2, CC: 36.5, FG: 50.2

    One hyperparameter (beta) replaces 11 manual weights.

    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number
    of Samples" (CVPR 2019)

    Args:
        class_counts: list of positive sample counts per class (length 11)
        beta: effective number parameter (0.9=mild, 0.999=aggressive, 0.9999=very aggressive)
        normalize: if True, normalize so min weight = 1.0
    Returns:
        Tensor of shape [num_classes] with per-class weights
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)
    effective_n = (1.0 - beta ** counts) / (1.0 - beta)
    weights = 1.0 / effective_n
    if normalize:
        weights = weights / weights.min()  # min weight = 1.0
    return weights
```

### Step 2: Add logit adjustment to LossConfig and AsymmetricLoss

**Where**: `config.py` `LossConfig` dataclass — add fields:

```python
@dataclass
class LossConfig:
    asl_gamma_pos: float = 1.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    label_smoothing: float = 0.05
    theme_weights: Optional[List[float]] = None
    cb_beta: Optional[float] = None        # NEW: if set, compute CB weights (overrides theme_weights)
    logit_adjustment: bool = False         # NEW: add log(prior) to logits during training
```

**Where**: `losses.py` `AsymmetricLoss.__init__` — add logit adjustment buffer:

```python
def __init__(
    self,
    gamma_pos=1.0, gamma_neg=4.0, clip=0.05,
    label_smoothing=0.0, eps=1e-8,
    theme_weights=None,
    logit_adjustment=None,  # NEW: tensor of shape [num_labels]
):
    super().__init__()
    # ... existing code ...
    if logit_adjustment is not None:
        self.register_buffer("logit_adjustment", logit_adjustment)
    else:
        self.logit_adjustment = None
```

**Where**: `losses.py` `AsymmetricLoss.forward` — apply adjustment before sigmoid:

```python
def forward(self, logits, targets, mask=None):
    # Apply logit adjustment: shift logits by log(prior / (1-prior))
    # This encourages the model to predict rare classes more often during training.
    # At test time, the adjustment is NOT applied (thresholds handle calibration).
    if self.logit_adjustment is not None:
        logits = logits + self.logit_adjustment.view(1, 1, -1)

    probs = torch.sigmoid(logits)
    # ... rest of existing code unchanged ...
```

### Step 3: Update build_loss_from_config

**Where**: `losses.py` `build_loss_from_config()` — handle cb_beta and logit_adjustment:

```python
def build_loss_from_config(config) -> AsymmetricLoss:
    """Build ASL from AWAREConfig. Theme weights auto-computed if not in config."""
    L = config.loss
    theme_weights = None

    # CB Loss weights (overrides manual theme_weights if set)
    cb_beta = getattr(L, "cb_beta", None)
    if cb_beta is not None:
        # Training set class counts (from splits_stats.json)
        class_counts = [7879, 5434, 5163, 4732, 944, 263, 698, 2432, 770, 170, 114]
        theme_weights = compute_cb_weights(class_counts, beta=cb_beta)
        from config import THEMES
        logger.info("CB Loss weights (beta=%.4f): %s", cb_beta,
            {t: round(w, 2) for t, w in zip(THEMES, theme_weights.tolist())})
    elif getattr(L, "theme_weights", None):
        w = L.theme_weights
        if isinstance(w, (list, tuple)) and len(w) == 11:
            theme_weights = torch.tensor(w, dtype=torch.float32)

    # Logit adjustment: log(prior / (1-prior)) per class
    logit_adj = None
    if getattr(L, "logit_adjustment", False):
        total_sentences = 68000  # approximate total training sentences
        class_counts = torch.tensor(
            [7879, 5434, 5163, 4732, 944, 263, 698, 2432, 770, 170, 114],
            dtype=torch.float32,
        )
        prior = class_counts / total_sentences
        logit_adj = torch.log(prior / (1 - prior))
        from config import THEMES
        logger.info("Logit adjustment: %s",
            {t: round(a, 3) for t, a in zip(THEMES, logit_adj.tolist())})

    return AsymmetricLoss(
        gamma_pos=L.asl_gamma_pos,
        gamma_neg=L.asl_gamma_neg,
        clip=L.asl_clip,
        label_smoothing=L.label_smoothing,
        theme_weights=theme_weights,
        logit_adjustment=logit_adj,
    )
```

### Step 4: Add tau-normalization to trainer.py

**Where**: In `train()` method, AFTER Phase 2 ends (line 186), BEFORE Phase 3 call (line 188).

```python
    # ── Tau-normalization (zero-cost post-hoc calibration) ──
    tau_f1 = self._tau_normalize()

    # ── Phase 3 (if tau-norm didn't already beat Phase 2 significantly) ──
    self._train_phase3_balanced_head()
```

Add the method:

```python
def _tau_normalize(self, tau_values=None):
    """Post-hoc classifier weight rescaling. Sweeps tau on val set.

    For each tau in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
    - Rescale W → W / ||W||^tau (per-row)
    - Evaluate on val set
    - Keep best tau

    tau=0: no change. tau=1: unit-normalize all rows (equalizes norms).
    Rare themes tend to have smaller weight norms → tau>0 boosts them.

    Zero training cost. Fully reversible.

    Reference: Kang et al., "Decoupling Representation and Classifier
    for Long-Tailed Recognition" (ICLR 2020)
    """
    if tau_values is None:
        tau_values = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

    best_path = self.output_dir / "best.pt"
    if not best_path.exists():
        return 0.0

    self.model.load_state_dict(
        torch.load(best_path, map_location=self.device, weights_only=True)
    )

    # Get classifier weight matrix
    head = self.model.classifier
    if hasattr(head, "classifier"):
        W = head.classifier.weight.data          # Linear head: [11, 768]
    elif hasattr(head, "label_queries"):
        W = head.label_queries.data               # LSAN head: [11, 768]
    else:
        logger.info("Tau-norm: unknown head type, skipping")
        return 0.0

    norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    original_W = W.clone()
    from config import THEMES

    logger.info("Tau-norm: weight norms before: %s",
        {THEMES[i][:3]: f"{norms[i].item():.3f}" for i in range(min(len(THEMES), len(norms)))})

    best_tau, best_f1 = 0.0, 0.0
    for tau in tau_values:
        if tau == 0.0:
            W.copy_(original_W)
        else:
            # W_new = W / ||W||^tau → rows with small norms get boosted
            W.copy_(original_W / (norms ** tau))

        val_metrics = self._validate()
        logger.info("  tau=%.1f → val_f1_macro=%.4f", tau, val_metrics["f1_macro"])

        if val_metrics["f1_macro"] > best_f1:
            best_tau = tau
            best_f1 = val_metrics["f1_macro"]

    # Apply best tau
    if best_tau == 0.0:
        W.copy_(original_W)
    else:
        W.copy_(original_W / (norms ** best_tau))

    # Save tau-normalized model
    tau_path = self.output_dir / "best_tau_norm.pt"
    torch.save(self.model.state_dict(), tau_path)

    logger.info("Tau-norm complete: best_tau=%.1f, val_f1=%.4f (saved to %s)",
        best_tau, best_f1, tau_path)

    # Also update best.pt if tau-norm improved
    if best_f1 > self.best_f1:
        import shutil
        shutil.copy2(tau_path, best_path)
        self.best_f1 = best_f1
        logger.info("Tau-norm IMPROVED over Phase 2: %.4f → %.4f",
            self.history["best_f1"], best_f1)
        self.history["best_f1"] = round(best_f1, 6)
        self.history["tau_norm"] = {"best_tau": best_tau, "f1": round(best_f1, 6)}

    return best_f1
```

### Step 5: Replace Phase 3 head reinit with LWS

**Where**: `trainer.py` `_train_phase3_balanced_head()` — major rewrite.

Instead of reinitializing the full classification head (discards Phase 2 knowledge),
freeze it and learn only 11 per-class scaling factors.

```python
def _train_phase3_balanced_head(self, epochs: int = 10):
    """Phase 3: Learnable Weight Scaling (LWS).

    Instead of reinitializing the classification head (Q008 approach),
    freeze the entire model and learn only 11 per-class scaling factors.
    Much more stable — preserves all Phase 2 knowledge.

    Reference: Kang et al., ICLR 2020
    """
    import shutil

    best_path = self.output_dir / "best.pt"
    if not best_path.exists():
        logger.warning("No best.pt for Phase 3 LWS, skipping")
        return

    # Load best model (may be tau-normalized)
    phase2_f1 = self.best_f1
    self.model.load_state_dict(
        torch.load(best_path, map_location=self.device, weights_only=True)
    )

    # Backup pre-Phase-3 model
    phase2_path = self.output_dir / "best_pre_phase3.pt"
    shutil.copy2(best_path, phase2_path)

    # Freeze everything
    for p in self.model.parameters():
        p.requires_grad = False

    # Add learnable per-class scales (11 parameters)
    from config import NUM_THEMES
    lws_scale = nn.Parameter(torch.ones(NUM_THEMES, device=self.device))

    # Balanced loss for LWS training
    theme_counts = torch.tensor(
        [7879, 5434, 5163, 4732, 944, 263, 698, 2432, 770, 170, 114],
        dtype=torch.float32,
    )
    balanced_weights = torch.sqrt(theme_counts.max() / theme_counts)
    balanced_criterion = AsymmetricLoss(
        gamma_pos=0.0, gamma_neg=4.0, clip=0.05,
        theme_weights=balanced_weights,
    ).to(self.device)

    optimizer = torch.optim.AdamW([lws_scale], lr=1e-2, weight_decay=0.0)

    logger.info("PHASE_START phase=3 (LWS: learning 11 per-class scales, %d epochs)", epochs)
    logger.info("  Pre-Phase-3 best F1=%.4f (must beat this to replace)", phase2_f1)

    phase3_start_epoch = self.current_epoch + 1
    best_p3_f1 = 0.0
    best_scales = lws_scale.data.clone()
    patience = 5

    for ep in range(1, epochs + 1):
        self.current_epoch = phase3_start_epoch + ep - 1
        self.model.train()  # for dropout in head
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
                # Apply learnable scales to logits
                scaled_logits = output["logits"] * lws_scale.view(1, 1, -1)
                loss = balanced_criterion(scaled_logits, labels, sentence_mask)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n += 1

        # Validate with scaled logits
        # Temporarily modify forward to include scaling
        val_metrics = self._validate_with_scales(lws_scale)
        train_loss = total_loss / max(n, 1)
        self._log_epoch(self.current_epoch, train_loss, val_metrics, phase=3)

        from config import THEMES
        logger.info("  LWS scales: %s",
            {THEMES[i][:3]: f"{lws_scale.data[i]:.3f}" for i in range(len(THEMES))})

        if val_metrics["f1_macro"] > best_p3_f1:
            best_p3_f1 = val_metrics["f1_macro"]
            best_scales = lws_scale.data.clone()
            patience = 5
        else:
            patience -= 1
            if patience <= 0:
                logger.info("  LWS early stop (no improvement for 5 epochs)")
                break

    # Bake scales into classifier weights permanently
    head = self.model.classifier
    if hasattr(head, "classifier"):
        # Linear head: scale each row of weight matrix
        head.classifier.weight.data *= best_scales.unsqueeze(1)
        if head.classifier.bias is not None:
            head.classifier.bias.data *= best_scales
    elif hasattr(head, "label_queries"):
        # LSAN head: scale each label query
        head.label_queries.data *= best_scales.unsqueeze(1)

    # Only use Phase 3 model if it BEATS pre-Phase-3
    if best_p3_f1 > phase2_f1:
        torch.save(self.model.state_dict(), best_path)
        logger.info("Phase 3 LWS IMPROVED: %.4f → %.4f (+%.4f). Using LWS model.",
            phase2_f1, best_p3_f1, best_p3_f1 - phase2_f1)
    else:
        shutil.copy2(phase2_path, best_path)
        self.model.load_state_dict(
            torch.load(best_path, map_location=self.device, weights_only=True)
        )
        logger.info("Phase 3 LWS did NOT improve (%.4f vs pre-Phase-3 %.4f). Keeping pre-Phase-3.",
            best_p3_f1, phase2_f1)

    logger.info("Phase 3 complete. Final model F1=%.4f", max(best_p3_f1, phase2_f1))
```

**Also add helper** `_validate_with_scales`:
```python
@torch.no_grad()
def _validate_with_scales(self, scales) -> Dict:
    """Validate with LWS scaling applied to logits."""
    self.model.eval()
    all_preds, all_labels = [], []
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
        # Apply scales
        scaled_logits = output["logits"] * scales.view(1, 1, -1)
        p, l = flatten_masked_preds_labels(scaled_logits, labels, sentence_mask)
        all_preds.append(p)
        all_labels.append(l)

    if all_preds:
        preds = np.vstack(all_preds)
        lab = np.vstack(all_labels)
        metrics = compute_metrics(preds, lab)
    else:
        metrics = {"f1_macro": 0.0}
    metrics["loss"] = 0.0
    return metrics
```

### Step 6: Update quick.yaml

```yaml
loss:
  asl_gamma_pos: 1.0
  asl_gamma_neg: 4.0
  asl_clip: 0.05
  label_smoothing: 0.05
  cb_beta: 0.999                   # CB Loss (Cui et al., CVPR 2019). Replaces manual weights.
  logit_adjustment: true           # Menon et al., ICLR 2021. Shifts logits by log(prior).
  # theme_weights removed — cb_beta overrides it
```

### Step 7: Update config.py

Add to `LossConfig`:
```python
cb_beta: Optional[float] = None
logit_adjustment: bool = False
```

### Step 8: Update run_job.sh

```bash
RUN_NUMBER="011"
```

---

## Integration Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CB weights too aggressive (FG=50x) | Q005-style catastrophe | Distribution-aware threshold floor still in place |
| Logit adjustment shifts distributions | Threshold optimization breaks | Thresholds are optimized AFTER training, will adapt |
| Tau-norm + LWS redundant | Wasted compute | Compare: tau-norm alone vs LWS alone vs both |
| LWS doesn't converge | No Phase 3 benefit | Keep pre-Phase-3 model (comparison built in) |
| CB beta too high/low | Under/over-weighting | Can try beta=0.99 (milder) if 0.999 fails |

## Key Decision: CB weights vs Manual weights

| Theme | Manual (Q006) | CB beta=0.99 | CB beta=0.999 | CB beta=0.9999 |
|-------|--------------|--------------|---------------|----------------|
| Nav | 1.00 | 1.00 | 1.00 | 1.00 |
| Att | 1.20 | 1.06 | 1.45 | 4.36 |
| Per | 1.24 | 1.07 | 1.53 | 4.97 |
| Asp | 1.29 | 1.09 | 1.67 | 5.89 |
| Soc | 2.89 | 1.50 | 8.36 | 68.3 |
| FP | 7.00 | 2.53 | 26.9 | 626 |
| Spi | 3.36 | 1.72 | 11.2 | 113 |
| Fam | 1.80 | 1.16 | 3.25 | 15.1 |
| Res | 3.20 | 1.62 | 10.2 | 95.1 |
| CC | 9.00 | 3.11 | 36.5 | 1196 |
| FG | 10.00 | 3.83 | 50.2 | 2282 |

**Recommendation**: Start with beta=0.999. If it causes Q005-style issues (check prob_diag
for neg_mean inflation), fall back to beta=0.99.

---

## Verification After Run

1. **CB weights**: Log shows computed weights — are they reasonable?
2. **Logit adjustment**: Log shows adjustment values — rare classes should be large negative
3. **Tau-norm sweep**: Which tau won? What were the F1s at each tau?
4. **LWS scales**: What did the learned scales converge to? (should boost rare themes)
5. **Tau-norm vs LWS**: Which gave better final F1?
6. **Overall macro F1**: Must be >= best of Q009/Q010
7. **No catastrophe**: Check prob_diag — neg_mean must not inflate above 0.30
8. **Per-theme comparison**: FG/CC/FP should improve from better calibration

---

## Files Changed Summary

| File | Changes |
|------|---------|
| `config.py` | Add `cb_beta`, `logit_adjustment` to `LossConfig` |
| `losses.py` | Add `compute_cb_weights()`, logit adjustment in `AsymmetricLoss`, update `build_loss_from_config` |
| `trainer.py` | Add `_tau_normalize()`, replace Phase 3 with LWS, add `_validate_with_scales()` |
| `quick.yaml` | Replace `theme_weights` with `cb_beta: 0.999`, add `logit_adjustment: true` |
| `run_job.sh` | `RUN_NUMBER="011"` |
