"""
model.py — AWARE v3 model: DeBERTa-v3 → SentenceMeanPooling → BiLSTM → ClassificationHead.

Essay-aware sentence classification: each sentence classified in context of full essay.

Improvements over v3-base:
  - ClassificationHead fixed: LN → dropout → linear (was dropout → LN → linear, which
    weakened dropout because LN re-normalized the dropped activations)
  - SentenceMeanPooling vectorized: inner sentence loop replaced with tensor matmul
    (GPU-parallelized per essay instead of 32 Python iterations per batch item)
  - SentencePositionEmbedding: learned 32×H position table added before BiLSTM so
    the model explicitly knows sentence position within the essay
  - EssayLevelHead: separate ClassificationHead for essay-level multi-task supervision.
    Essay logits are the OR-union of sentence-level labels — much richer training signal
    for rare themes (Attainment: 2.4% → ~8% at essay level).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
import logging
from typing import Dict, List, Tuple, Optional

from config import NUM_THEMES, THEMES

logger = logging.getLogger(__name__)


# Semantic descriptions for each CCW theme — used to initialize prototype vectors.
# Descriptions written to capture the core CCW construct, not surface keywords.
THEME_DESCRIPTIONS = {
    "Attainment":       "pursuing a college degree credential as a concrete personal achievement goal",
    "Aspirational":     "aspiring toward future career goals life achievements and professional ambitions",
    "Navigational":     "navigating academic requirements prerequisites and institutional bureaucratic systems",
    "Resistance":       "facing and overcoming systemic obstacles challenges barriers and adversity",
    "Familial_Capital": "motivated by family support cultural heritage familial relationships and obligations",
    "Spiritual":        "spiritual calling sense of purpose inner meaning personal identity and life mission",
    "Perseverance":     "persisting with determination through difficulty hard work and personal dedication",
    "Social":           "community connections social capital peer relationships and collective support networks",
}


class PrototypeClassificationHead(nn.Module):
    """Cosine-similarity prototype head for label-aware semantic classification.

    Instead of a learned linear projection, each theme is represented by a
    prototype vector in the same embedding space as the sentence embeddings.
    Classification is cosine similarity × per-theme temperature + bias.

    Advantages over linear head:
    - Prototypes can be initialized from semantic descriptions of each theme,
      giving the head a strong prior even before any gradient steps.
    - Cosine similarity is scale-invariant — less sensitive to embedding norm.
    - Per-theme temperature allows the model to control confidence per class.
    - Addresses Attainment/Spiritual confusion: prototypes start in semantically
      separated positions rather than random linear directions.
    """

    def __init__(self, hidden_size, num_labels=NUM_THEMES, dropout=0.15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        # Prototype vectors: one per theme, in the hidden embedding space
        self.prototypes = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)
        # Per-theme log-temperature: exp(0.0) = 1.0 — identity scaling at init.
        # Starts at cosine similarity range [-1, 1]; learned during training.
        # exp(2.0)=7.4 was too aggressive: overconfident logits → loss spike on ep1.
        self.log_temperature = nn.Parameter(torch.zeros(num_labels))
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, embeddings):
        """
        embeddings: [..., H] — works for [B, S, H] or [B, H]
        Returns:    [..., C] logits
        """
        x = self.dropout(self.layer_norm(embeddings))
        x_norm = F.normalize(x, p=2, dim=-1)                     # [..., H]
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)    # [C, H]
        similarity = torch.matmul(x_norm, proto_norm.t())         # [..., C]
        temperature = self.log_temperature.exp()                  # [C]
        return similarity * temperature + self.bias               # [..., C]


class SentenceMeanPooling(nn.Module):
    """Pool token embeddings to sentence embeddings via masked mean per sentence span.

    Vectorized implementation: builds [n_valid, T] span masks per essay item using
    tensor comparisons, then computes all sentence embeddings via a single matmul.
    Reduces Python iterations from batch×sentences to just batch (4 vs 128 for bs=4).
    """

    def __init__(self, max_sentences: int = 32):
        super().__init__()
        self.max_sentences = max_sentences

    def forward(self, token_embeddings, attention_mask, sentence_boundaries, sentence_mask):
        B, T, H = token_embeddings.shape
        device = token_embeddings.device
        sent_emb = torch.zeros(B, self.max_sentences, H, device=device, dtype=token_embeddings.dtype)

        tok_idx = torch.arange(T, device=device)  # [T]

        for b in range(B):
            n_valid = int(sentence_mask[b].sum().item())
            if n_valid == 0:
                continue

            # Extract boundaries for valid sentences
            starts, ends = [], []
            for s in range(n_valid):
                st, en = sentence_boundaries[b][s]
                st = max(0, min(int(st), T - 1))
                en = max(st + 1, min(int(en), T))
                starts.append(st)
                ends.append(en)

            # Build span mask [n_valid, T] via vectorized range comparison
            sidx = torch.tensor(starts, device=device).unsqueeze(1)   # [n_valid, 1]
            eidx = torch.tensor(ends, device=device).unsqueeze(1)     # [n_valid, 1]
            span_mask = (tok_idx.unsqueeze(0) >= sidx) & (tok_idx.unsqueeze(0) < eidx)  # [n_valid, T]

            # Apply attention mask to zero out padding tokens within spans
            am = attention_mask[b].unsqueeze(0).float()  # [1, T]
            masked = span_mask.float() * am              # [n_valid, T]

            denom = masked.sum(dim=1, keepdim=True).clamp(min=1e-9)   # [n_valid, 1]
            # Matrix multiply: [n_valid, T] × [T, H] = [n_valid, H], then divide by count
            sent_emb[b, :n_valid] = torch.matmul(masked, token_embeddings[b]) / denom

        return sent_emb


class SentencePositionEmbedding(nn.Module):
    """Learned position embedding for sentence index within an essay.

    Adds a position-specific vector to each sentence embedding before the BiLSTM,
    so the model explicitly learns that sentence 0 (intro) often sets the essay theme
    while later sentences elaborate. Position is independent of the BiLSTM's implicit
    sequential encoding — this is an additive bias, not a replacement.

    Cost: max_sentences × hidden_size learnable parameters (~32k for large model).
    """

    def __init__(self, max_sentences: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(max_sentences, hidden_size)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, sent_emb: torch.Tensor, sentence_mask: torch.Tensor) -> torch.Tensor:
        """
        sent_emb:      [B, S, H]
        sentence_mask: [B, S]
        Returns:       [B, S, H] with position vectors added to valid positions only
        """
        B, S, H = sent_emb.shape
        positions = torch.arange(S, device=sent_emb.device).unsqueeze(0).expand(B, S)
        pos_emb = self.embedding(positions)  # [B, S, H]
        # Only add to valid (non-padding) sentence slots
        return sent_emb + pos_emb * sentence_mask.unsqueeze(-1).float()


class AttentionPooling(nn.Module):
    """Attention-weighted pooling: learn which tokens matter per sentence.

    Instead of uniform mean over all tokens, compute attention weights
    a_i = softmax(W * h_i + b) and return weighted sum. This lets the model
    focus on content words (e.g., "grandmother" for Familial_Capital) vs
    function words ("the", "is").

    Cost: hidden_size + 1 parameters (~1K). Negligible.
    """

    def __init__(self, hidden_size, max_sentences=32):
        super().__init__()
        self.max_sentences = max_sentences
        self.attention = nn.Linear(hidden_size, 1, bias=True)
        # Initialize with small weights so attention starts near uniform (mean pooling)
        nn.init.xavier_uniform_(self.attention.weight, gain=0.1)
        nn.init.zeros_(self.attention.bias)

    def forward(self, token_embeddings, attention_mask, sentence_boundaries, sentence_mask):
        B, T, H = token_embeddings.shape
        device = token_embeddings.device
        sent_emb = torch.zeros(B, self.max_sentences, H, device=device,
                               dtype=token_embeddings.dtype)

        # Compute attention scores for all tokens at once
        attn_scores = self.attention(token_embeddings).squeeze(-1)  # [B, T]
        tok_idx = torch.arange(T, device=device)

        for b in range(B):
            n_valid = int(sentence_mask[b].sum().item())
            if n_valid == 0:
                continue

            starts, ends = [], []
            for s in range(n_valid):
                st, en = sentence_boundaries[b][s]
                st = max(0, min(int(st), T - 1))
                en = max(st + 1, min(int(en), T))
                starts.append(st)
                ends.append(en)

            sidx = torch.tensor(starts, device=device).unsqueeze(1)
            eidx = torch.tensor(ends, device=device).unsqueeze(1)
            span_mask = (tok_idx.unsqueeze(0) >= sidx) & (tok_idx.unsqueeze(0) < eidx)

            am = attention_mask[b].unsqueeze(0).float()
            masked = span_mask.float() * am  # [n_valid, T]

            # Attention-weighted pooling within each sentence span
            scores = attn_scores[b].unsqueeze(0).expand(n_valid, -1)
            scores = scores.masked_fill(masked == 0, -1e4)
            weights = torch.softmax(scores, dim=-1) * masked

            sent_emb[b, :n_valid] = torch.matmul(weights, token_embeddings[b])

        return sent_emb


class GatedSentenceAttention(nn.Module):
    """Multi-head self-attention over sentences with learned gating.

    Replaces BiLSTM for short sequences (~7 sentences/essay). Key innovation:
    gating mechanism initialized near ZERO so the model starts as "standard"
    (no context) and LEARNS to add context only where it helps.

    The gate g = sigmoid(W_g * [h_i; c_i] + b_g):
      - b_g initialized to -2.0 → sigmoid(-2.0) = 0.12 → mostly pass-through
      - Model must learn to OPEN the gate — implicit regularizer
      - Gate stays closed for themes where context hurts (Spiritual, Familial)
      - Gate opens for themes where context helps (Attainment, Resistance)

    For 7-sentence essays, self-attention is more expressive than BiLSTM
    because each sentence directly attends to every other sentence.
    """

    def __init__(self, hidden_size, num_heads=4, dropout=0.1,
                 output_dropout=0.0, gate_init_bias=-2.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Gating: sigmoid(W * [h; c] + b) controls how much context to add
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_init_bias)

        self.output_dropout = nn.Dropout(output_dropout) if output_dropout > 0 else None

    def forward(self, sentence_embeddings, sentence_mask):
        B, S, H = sentence_embeddings.shape

        # True = IGNORE in PyTorch attention
        key_padding_mask = (sentence_mask == 0)

        # Self-attention over sentences
        normed = self.layer_norm(sentence_embeddings)
        context, _ = self.attention(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
        )

        # Gated fusion: learn when to use context vs original
        gate_input = torch.cat([sentence_embeddings, context], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # [B, S, H]

        # output = gate * context + (1 - gate) * original
        output = gate * context + (1 - gate) * sentence_embeddings

        if self.output_dropout is not None:
            output = self.output_dropout(output)

        return output * sentence_mask.unsqueeze(-1).float()


class BiLSTMContextEncoder(nn.Module):
    """BiLSTM over sentence embeddings with residual connection."""

    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.1,
                 output_dropout=0.0):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.projection = nn.Linear(hidden_size * 2, input_size)
        self.output_dropout = nn.Dropout(output_dropout) if output_dropout > 0 else None

    def forward(self, sentence_embeddings, sentence_mask):
        lengths = sentence_mask.sum(dim=1).cpu().long().clamp(min=1)
        packed = pack_padded_sequence(
            sentence_embeddings, lengths, batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.bilstm(packed)
        context, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=sentence_embeddings.shape[1],
        )
        context = self.projection(context) + sentence_embeddings  # residual add
        if self.output_dropout is not None:
            context = self.output_dropout(context)
        return context * sentence_mask.unsqueeze(-1).float()


class ClassificationHead(nn.Module):
    """LayerNorm → Multi-Sample Dropout → Linear → logits.

    FIX: original code was Dropout → LayerNorm → Linear. LayerNorm after dropout
    re-normalizes the sparse (zeroed-out) activations, partially defeating the
    regularization. Correct order: normalize first for stable scale, then drop,
    then project to logit space.

    NEW: Multi-sample dropout (Inoue 2019) — average logits across K dropout masks
    during training. Free regularization: zero extra parameters, negligible compute.
    At inference, uses standard single pass (no dropout).
    """

    def __init__(self, hidden_size, num_labels=NUM_THEMES, dropout=0.15, n_dropout_samples=1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.n_dropout_samples = n_dropout_samples
        if n_dropout_samples > 1:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_dropout_samples)])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout)])
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, embeddings):
        x = self.layer_norm(embeddings)
        if self.training and self.n_dropout_samples > 1:
            # Average logits across multiple dropout masks
            logits = torch.stack([self.classifier(d(x)) for d in self.dropouts], dim=0)
            return logits.mean(dim=0)
        return self.classifier(self.dropouts[0](x))


class EssayConditioningLayer(nn.Module):
    """Top-down essay-to-sentence information flow.

    The key AWARE innovation: first predict what themes the ESSAY contains,
    then use that as a prior for sentence-level classification.

    This is fundamentally different from BiLSTM (bottom-up neighbor context):
      - BiLSTM: "what do nearby sentences say?" → noisy, redundant with DeBERTa
      - Essay conditioning: "what is this essay about?" → strong top-down prior

    For Attainment (2.4% sentence, ~8% essay): the essay prior says "this essay
    mentions degree attainment" — a much stronger signal than any neighbor sentence.

    Architecture:
      1. Mean-pool sentence embeddings → essay embedding [B, H]
      2. Essay classifier → essay_logits [B, C] (what themes does this essay contain?)
      3. Project essay_logits to conditioning vector [B, C] → [B, H_cond]
      4. Broadcast to each sentence: [B, 1, H_cond] + sentence_emb [B, S, H]
      5. Gated fusion: sentence decides how much essay prior to use
    """

    def __init__(self, hidden_size, num_labels=NUM_THEMES, dropout=0.15):
        super().__init__()
        # Essay-level classifier (predicts essay themes)
        self.essay_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
        # Project essay logits to conditioning space
        self.condition_proj = nn.Linear(num_labels, hidden_size)
        # Gated fusion: sigmoid gate controls how much essay prior to use
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        # Initialize gate bias negative so conditioning starts mild
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.5)

    def forward(self, sentence_embeddings, sentence_mask):
        """
        sentence_embeddings: [B, S, H]
        sentence_mask: [B, S]
        Returns: (conditioned_embeddings [B, S, H], essay_logits [B, C])
        """
        B, S, H = sentence_embeddings.shape

        # Step 1: Mean-pool to essay embedding
        mask_f = sentence_mask.unsqueeze(-1).float()  # [B, S, 1]
        essay_emb = (sentence_embeddings * mask_f).sum(dim=1)  # [B, H]
        n_valid = sentence_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        essay_emb = essay_emb / n_valid  # [B, H]

        # Step 2: Essay-level prediction
        essay_logits = self.essay_classifier(essay_emb)  # [B, C]

        # Step 3: Project to conditioning vector
        essay_prior = self.condition_proj(essay_logits.detach())  # [B, H] — detach to avoid shortcut
        essay_prior = essay_prior.unsqueeze(1).expand(-1, S, -1)  # [B, S, H]

        # Step 4: Gated fusion
        gate_input = torch.cat([sentence_embeddings, essay_prior], dim=-1)  # [B, S, 2H]
        g = torch.sigmoid(self.gate(gate_input))  # [B, S, H]

        # Step 5: Conditioned output
        output = sentence_embeddings + g * essay_prior
        output = output * mask_f  # zero padding

        return output, essay_logits


class AWAREModel(nn.Module):
    """DeBERTa → SentenceMeanPooling → [PositionEmbedding] → ContextEncoder → ClassificationHead.

    AWARE = Essay-Aware Sentence Classification. Core innovation:
      1. Sentence pooling: tokens → sentence embeddings
      2. Context encoding: BiLSTM (v4) or GatedAttention (v5) or EssayConditioning (v5)
      3. Essay-level top-down prior: what themes does this essay contain?

    The essay conditioning layer provides TOP-DOWN context:
      "This essay is about family and navigation" → bias sentence predictions accordingly.
    This is more useful than bottom-up BiLSTM context for short essays (~7 sentences).
    """

    def __init__(
        self,
        encoder_name="microsoft/deberta-v3-base",
        encoder_path=None,
        hidden_size=None,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        num_labels=NUM_THEMES,
        max_sentences=32,
        dropout=0.15,
        freeze_encoder=False,
        use_bilstm=True,
        bilstm_dropout=0.0,
        gradient_checkpointing=False,
        use_position_embedding=False,
        use_essay_head=False,
        use_prototype_head=False,
        n_dropout_samples=1,
        context_type="bilstm",
        context_num_heads=4,
        gate_init_bias=-2.0,
        pooling_type="mean",
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path or encoder_name)
        if hidden_size is None:
            hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.use_bilstm = use_bilstm
        self.use_position_embedding = use_position_embedding
        self.use_essay_head = use_essay_head
        self.use_prototype_head = use_prototype_head
        self.context_type = context_type

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (saves ~40%% memory, costs ~30%% speed)")

        # Pooling: mean (original) or attention-weighted (v5)
        if pooling_type == "attention":
            self.pooling = AttentionPooling(hidden_size, max_sentences)
            logger.info("Using AttentionPooling (learned token importance)")
        else:
            self.pooling = SentenceMeanPooling(max_sentences)

        if use_position_embedding:
            self.position_embedding = SentencePositionEmbedding(max_sentences, hidden_size)

        # Context encoder: BiLSTM (v4) or GatedSentenceAttention (v5) or None
        if use_bilstm:
            if context_type == "gated_attention":
                self.context_encoder = GatedSentenceAttention(
                    hidden_size=hidden_size,
                    num_heads=context_num_heads,
                    dropout=dropout,
                    output_dropout=bilstm_dropout,
                    gate_init_bias=gate_init_bias,
                )
                logger.info("Using GatedSentenceAttention (heads=%d, gate_bias=%.1f)",
                            context_num_heads, gate_init_bias)
            else:
                self.context_encoder = BiLSTMContextEncoder(
                    input_size=hidden_size, hidden_size=lstm_hidden_size,
                    num_layers=lstm_num_layers, dropout=dropout,
                    output_dropout=bilstm_dropout,
                )
        else:
            self.context_encoder = None

        # Essay conditioning: top-down essay→sentence information flow
        # This is the core AWARE innovation: use essay-level theme predictions
        # as a prior for sentence-level classification
        self.use_essay_conditioning = use_essay_head  # reuse the essay_head flag
        if self.use_essay_conditioning:
            self.essay_conditioning = EssayConditioningLayer(
                hidden_size, num_labels, dropout
            )
            logger.info("Using EssayConditioningLayer (top-down essay prior)")

        # Sentence-level classification head (primary task)
        # PrototypeClassificationHead: cosine-similarity to learned theme prototypes
        # ClassificationHead: standard LN → multi-sample dropout → linear projection
        if use_prototype_head:
            self.classifier = PrototypeClassificationHead(hidden_size, num_labels, dropout)
        else:
            self.classifier = ClassificationHead(
                hidden_size, num_labels, dropout,
                n_dropout_samples=n_dropout_samples,
            )

        # Essay-level classification head (auxiliary multi-task)
        if use_essay_head:
            self.essay_head = ClassificationHead(hidden_size, num_labels, dropout)

        if freeze_encoder:
            self._freeze_encoder()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "AWAREModel: %s, %.1fM params (%.1fM trainable), bilstm=%s, pos_emb=%s, "
            "essay_head=%s, prototype_head=%s",
            encoder_path or encoder_name, total / 1e6, trainable / 1e6,
            use_bilstm, use_position_embedding, use_essay_head, use_prototype_head,
        )

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen")

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        logger.info("Encoder unfrozen")

    @torch.no_grad()
    def initialize_prototypes(self, tokenizer):
        """Initialize prototype vectors from encoded THEME_DESCRIPTIONS.

        Runs each theme description through the (already loaded) DeBERTa encoder,
        mean-pools over non-padding tokens, and writes the result into
        self.classifier.prototypes. Call this once after build_model_from_config()
        and before training begins.

        Skips silently if the classifier is not a PrototypeClassificationHead.
        """
        if not isinstance(self.classifier, PrototypeClassificationHead):
            logger.info("initialize_prototypes: not a prototype head — skipping")
            return

        device = next(self.encoder.parameters()).device
        was_training = self.encoder.training
        self.encoder.eval()

        logger.info("Initializing %d prototype vectors from theme descriptions...", len(THEMES))
        for i, theme in enumerate(THEMES):
            description = THEME_DESCRIPTIONS[theme]
            enc = tokenizer(
                description, return_tensors="pt",
                padding=True, truncation=True, max_length=64,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = self.encoder(**enc)
            last_hidden = out.last_hidden_state   # [1, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()   # [1, T, 1]
            proto = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [1, H]
            self.classifier.prototypes.data[i] = proto.squeeze(0)
            logger.info("  [%d] %-20s ← \"%s\"", i, theme, description[:60])

        if was_training:
            self.encoder.train()
        logger.info("Prototype initialization complete.")

    def forward(self, input_ids, attention_mask, sentence_boundaries, sentence_mask,
                return_embeddings=False, context_dropout=0.0):
        token_emb = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        sent_emb = self.pooling(token_emb, attention_mask, sentence_boundaries, sentence_mask)

        # Optional: add learned sentence position bias before context encoder
        if self.use_position_embedding:
            sent_emb = self.position_embedding(sent_emb, sentence_mask)

        # Context dropout: randomly zero some sentence embeddings to force
        # the context mechanism to reconstruct from neighbors
        if self.training and context_dropout > 0 and self.context_encoder is not None:
            n_valid = sentence_mask.sum(dim=1)  # [B]
            # Only apply to essays with >2 sentences (need neighbors)
            can_mask = (n_valid > 2).float()  # [B]
            # Random drop mask per sentence
            drop_mask = (torch.rand_like(sentence_mask) > context_dropout).float()
            drop_mask = drop_mask * sentence_mask  # respect padding
            # Ensure at least 2 sentences survive per essay
            surviving = drop_mask.sum(dim=1)
            too_few = (surviving < 2).float().unsqueeze(-1)
            drop_mask = drop_mask + too_few * sentence_mask
            drop_mask = drop_mask.clamp(0, 1)
            # Apply: zero out dropped sentences, keep originals for non-maskable essays
            sent_emb_masked = sent_emb * drop_mask.unsqueeze(-1)
            sent_emb = torch.where(
                can_mask.unsqueeze(-1).unsqueeze(-1).bool().expand_as(sent_emb),
                sent_emb_masked, sent_emb
            )

        context = self.context_encoder(sent_emb, sentence_mask) if self.context_encoder else sent_emb

        # Essay conditioning: top-down essay prior → sentence embeddings
        essay_logits = None
        if self.use_essay_conditioning:
            context, essay_logits = self.essay_conditioning(context, sentence_mask)

        # Sentence-level logits [B, S, C]
        logits = self.classifier(context)
        out = {"logits": logits}

        # Essay-level logits from conditioning layer (or separate head as fallback)
        if essay_logits is not None:
            out["essay_logits"] = essay_logits
        elif self.use_essay_head:
            mask_f = sentence_mask.unsqueeze(-1).float()  # [B, S, 1]
            essay_emb = (context * mask_f).sum(dim=1)     # [B, H]
            n_valid = sentence_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # [B, 1]
            essay_emb = essay_emb / n_valid
            out["essay_logits"] = self.essay_head(essay_emb)  # [B, C]

        if return_embeddings:
            out["sentence_embeddings"] = sent_emb
            out["context_embeddings"] = context
        return out


def build_model_from_config(config, freeze_encoder=False):
    m = config.model
    t = config.training
    return AWAREModel(
        encoder_name=m.encoder_name,
        encoder_path=getattr(m, "encoder_path", None) or None,
        hidden_size=m.hidden_size,
        lstm_hidden_size=getattr(m, "lstm_hidden", 256),
        lstm_num_layers=getattr(m, "lstm_layers", 2),
        num_labels=NUM_THEMES,
        max_sentences=m.max_sentences,
        dropout=m.dropout,
        freeze_encoder=freeze_encoder,
        use_bilstm=getattr(m, "use_bilstm", True),
        bilstm_dropout=getattr(m, "bilstm_dropout", 0.0),
        gradient_checkpointing=getattr(t, "gradient_checkpointing", False),
        use_position_embedding=getattr(m, "use_position_embedding", False),
        use_essay_head=getattr(t, "essay_aux_weight", 0.0) > 0,
        use_prototype_head=getattr(m, "use_prototype_head", False),
        n_dropout_samples=getattr(m, "n_dropout_samples", 1),
        context_type=getattr(m, "context_type", "bilstm"),
        context_num_heads=getattr(m, "context_num_heads", 4),
        gate_init_bias=getattr(m, "gate_init_bias", -2.0),
        pooling_type=getattr(m, "pooling_type", "mean"),
    )
