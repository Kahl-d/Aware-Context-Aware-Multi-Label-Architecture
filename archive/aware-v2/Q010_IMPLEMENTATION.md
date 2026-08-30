# Q010: "Smart Head" — Detailed Implementation Guide

## Goal
Replace the shared linear classification head and mean pooling with:
1. **SentenceAttentionPooling** — learnable token-level attention (replaces uniform mean)
2. **LabelAttentionHead** — per-theme query vectors (replaces single Linear(768,11))

Both are architecture changes to the classification pipeline. They're paired together
because LSAN benefits from attention pooling giving it better sentence representations.

---

## Problem Being Solved

**Problem 1: Classifier Head Bias**
Current `ClassificationHead` is `Linear(768, 11)` — all 11 themes share one hyperplane.
Gradient from Navigational (7,879 examples) overwhelms First Gen (114 examples).
Evidence: FG train F1=0.89 but test F1=0.32 (overfits to patterns, not concepts).

**Problem 2: Mean Pooling Dilutes Signals**
`SentenceMeanPooling` averages ALL tokens equally. Theme-relevant tokens like "first",
"family" get diluted by generic tokens like "I", "and", "to".

---

## Step-by-Step Implementation

### Step 1: Add SentenceAttentionPooling to model.py

**Where**: After current `SentenceMeanPooling` class (line 53), add new class.
Keep `SentenceMeanPooling` as fallback.

```python
class SentenceAttentionPooling(nn.Module):
    """Learnable attention-weighted pooling: tokens that matter get higher weight.

    Instead of uniform mean over all tokens in a sentence span, learns a query
    vector that scores each token's importance. Theme-relevant tokens (e.g., "first",
    "family" for First Gen) get higher weight than generic tokens ("I", "and").

    Drop-in replacement for SentenceMeanPooling — same interface, same output shape.
    Only 768 new parameters (the query vector).

    Reference: "Pool Me Wisely" (NeurIPS 2025)
    """

    def __init__(self, hidden_size: int = 768, max_sentences: int = 32):
        super().__init__()
        self.max_sentences = max_sentences
        self.query = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_boundaries: List[List[Tuple[int, int]]],
        sentence_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = token_embeddings.shape
        device = token_embeddings.device
        sentence_embeddings = torch.zeros(
            batch_size, self.max_sentences, hidden_size,
            device=device, dtype=token_embeddings.dtype,
        )
        for b in range(batch_size):
            boundaries = sentence_boundaries[b]
            for s, (start, end) in enumerate(boundaries):
                if s >= self.max_sentences or start >= end:
                    break
                start = max(0, min(start, seq_len - 1))
                end = max(start + 1, min(end, seq_len))
                tokens = token_embeddings[b, start:end, :]         # [span, hidden]
                mask = attention_mask[b, start:end].float()         # [span]
                # Attention scores: learn which tokens are important
                scores = self.query(tokens).squeeze(-1)             # [span]
                scores = scores.masked_fill(mask == 0, -1e9)
                weights = torch.softmax(scores, dim=0).unsqueeze(-1)  # [span, 1]
                sentence_embeddings[b, s] = (tokens * weights).sum(dim=0)
        return sentence_embeddings
```

### Step 2: Add LabelAttentionHead to model.py

**Where**: After current `ClassificationHead` class (line 113), add new class.
Keep `ClassificationHead` as fallback.

```python
class LabelAttentionHead(nn.Module):
    """Label-Aware Attention: each theme gets its own learned query vector.

    Instead of a single Linear(768, 11) where all themes share the same weight
    space, each theme has a 768-dim query that computes a dot-product score
    against the sentence embedding. This gives each theme independent gradient
    flow — rare themes (FG, CC) don't compete with common themes (Nav, Att).

    Adds only 11 * 768 = 8,448 parameters (the label queries).

    Reference: Xiao et al., "Label-Specific Document Representation for
    Multi-Label Text Classification" (EMNLP 2019)
    """

    def __init__(self, hidden_size: int = 768, num_labels: int = 11, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Each label has its own query vector
        self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)
        self.scale = hidden_size ** -0.5

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: [batch, max_sent, hidden_size]
        x = self.dropout(embeddings)
        x = self.layer_norm(x)
        # Scaled dot-product: each sentence gets a score per label
        # [batch, max_sent, hidden] @ [hidden, num_labels] → [batch, max_sent, num_labels]
        logits = torch.matmul(x, self.label_queries.T) * self.scale
        return logits
```

**IMPORTANT**: The `label_queries` attribute name must match what `_log_gradient_norms()`
in trainer.py already checks for:
```python
elif hasattr(head, "label_queries"):
    W_grad = head.label_queries.grad
```
This was added in Q009 specifically for forward-compatibility.

### Step 3: Update AWAREModel.__init__ to use new classes

**Where**: `model.py` lines 144 and 151-153.

Change:
```python
self.pooling = SentenceMeanPooling(max_sentences)
```
To:
```python
self.pooling = SentenceAttentionPooling(hidden_size, max_sentences)
```

Change:
```python
self.classifier = ClassificationHead(
    hidden_size=hidden_size, num_labels=num_labels, dropout=dropout,
)
```
To:
```python
self.classifier = LabelAttentionHead(
    hidden_size=hidden_size, num_labels=num_labels, dropout=dropout,
)
```

### Step 4: Update Phase 3 in trainer.py

**Where**: `trainer.py` `_train_phase3_balanced_head()` lines 234-241.

Phase 3 currently reinitializes `ClassificationHead`. Must update to use `LabelAttentionHead`:

Change:
```python
from model import ClassificationHead
from config import NUM_THEMES
self.model.classifier = ClassificationHead(
    hidden_size=self.model.hidden_size,
    num_labels=NUM_THEMES,
    dropout=self.config.model.dropout,
).to(self.device)
```
To:
```python
from model import LabelAttentionHead
from config import NUM_THEMES
self.model.classifier = LabelAttentionHead(
    hidden_size=self.model.hidden_size,
    num_labels=NUM_THEMES,
    dropout=self.config.model.dropout,
).to(self.device)
```

### Step 5: Update quick.yaml

Add Q010 to run history. No new config parameters needed (architecture change is in code).

### Step 6: Update run_job.sh

```bash
RUN_NUMBER="010"
```

---

## Integration Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LabelAttentionHead doesn't learn | No improvement over linear | Keep ClassificationHead code, can revert with one-line change |
| Attention pooling breaks batch loop | Training crash | Same loop structure as mean pooling, just adds softmax |
| Phase 3 reinit of LabelAttentionHead | Wastes label queries | Q011 may replace Phase 3 with LWS anyway |
| EMA/R-Drop interaction | Unknown | Both are orthogonal to head architecture |
| Gradient logging breaks | Silent fail | Already handles `label_queries` attribute (Q009) |

## Parameter Count Change

| Component | Before | After | Delta |
|-----------|--------|-------|-------|
| SentenceMeanPooling | 0 | — | — |
| SentenceAttentionPooling | — | 768 | +768 |
| ClassificationHead | 768*11 + 11 + 768 + 768 = 10,027 | — | — |
| LabelAttentionHead | 11*768 + 768 + 768 = 9,984 | 9,984 | -43 |
| **Total change** | — | — | **+725** |

Negligible parameter change. Same model size.

---

## Verification After Run

1. Check `evaluation_test.json` → `f1_per_theme` for FG, CC, FP specifically
2. Check gradient norms — should show more balanced per-theme values with independent queries
3. Check class_0 — should be comparable to Q009
4. Check Phase 3 — does reinitializing LabelAttentionHead still help?
5. Compare bootstrap CI width to Q009 — should be similar or narrower
6. Check timing — should be similar to Q009 (no extra forward passes)

---

## Files Changed Summary

| File | Changes |
|------|---------|
| `model.py` | Add `SentenceAttentionPooling`, `LabelAttentionHead`; update `AWAREModel.__init__` |
| `trainer.py` | Update Phase 3 to use `LabelAttentionHead` instead of `ClassificationHead` |
| `quick.yaml` | Update run history comments |
| `run_job.sh` | `RUN_NUMBER="010"` |
