"""
AWARE v2 Model: DeBERTa-v3 → Sentence Attention Pooling → BiLSTM → Label Attention Head.

Essay-aware sentence classification: each sentence is classified in the context
of the full essay via BiLSTM over sentence embeddings.

Q010: SentenceMeanPooling → SentenceAttentionPooling (learnable token weights)
      ClassificationHead → LabelAttentionHead (per-theme query vectors, LSAN)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
import logging
from typing import Dict, List, Tuple, Optional

from config import NUM_THEMES

logger = logging.getLogger(__name__)


class SentenceMeanPooling(nn.Module):
    """Pool token embeddings to sentence embeddings via masked mean per sentence span."""

    def __init__(self, max_sentences: int = 32):
        super().__init__()
        self.max_sentences = max_sentences

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
                tokens = token_embeddings[b, start:end, :]
                mask = attention_mask[b, start:end].unsqueeze(-1).float()
                masked_tokens = tokens * mask
                sum_tokens = masked_tokens.sum(dim=0)
                sum_mask = mask.sum().clamp(min=1e-9)
                sentence_embeddings[b, s] = sum_tokens / sum_mask
        return sentence_embeddings


class SentenceAttentionPooling(nn.Module):
    """Learnable attention-weighted pooling over token spans.

    Instead of uniform mean over all tokens, learns a query vector that scores
    each token's importance via softmax. Theme-relevant tokens get higher weight.
    Drop-in replacement for SentenceMeanPooling — same interface, same output shape.

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
                tokens = token_embeddings[b, start:end, :]
                mask = attention_mask[b, start:end].float()
                scores = self.query(tokens).squeeze(-1)
                scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores, dim=0).unsqueeze(-1)
                sentence_embeddings[b, s] = (tokens * weights).sum(dim=0)
        return sentence_embeddings


class BiLSTMContextEncoder(nn.Module):
    """BiLSTM over sentence embeddings; project back to encoder dim."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.projection = nn.Linear(hidden_size * 2, input_size)

    def forward(
        self,
        sentence_embeddings: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_sentences, _ = sentence_embeddings.shape
        lengths = sentence_mask.sum(dim=1).cpu().long().clamp(min=1)
        packed = pack_padded_sequence(
            sentence_embeddings, lengths, batch_first=True, enforce_sorted=False,
        )
        packed_output, _ = self.bilstm(packed)
        context_embeddings, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_sentences,
        )
        context_embeddings = self.projection(context_embeddings)
        # Residual connection: preserve original sentence embeddings through the
        # LSTM bottleneck (768 → 256*2 → 768). This lets the classifier use both
        # the essay-context from BiLSTM AND the original token-level sentence info.
        # Improves generalization by preventing information loss in projection.
        context_embeddings = context_embeddings + sentence_embeddings
        context_embeddings = context_embeddings * sentence_mask.unsqueeze(-1).float()
        return context_embeddings


class ClassificationHead(nn.Module):
    """Dropout → LayerNorm → Linear → logits."""

    def __init__(self, hidden_size: int, num_labels: int = 11, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self.dropout(embeddings)
        x = self.layer_norm(x)
        return self.classifier(x)


class LabelAttentionHead(nn.Module):
    """Label-Aware Attention: each theme gets its own learned query vector.

    Instead of a single Linear(768, 11) where all themes share the same weight
    space, each theme has a 768-dim query that computes a scaled dot-product
    against sentence embeddings. This gives each theme independent gradient flow.

    Reference: Xiao et al., EMNLP 2019 (LSAN)
    """

    def __init__(self, hidden_size: int = 768, num_labels: int = 11, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)
        self.scale = hidden_size ** -0.5

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self.dropout(embeddings)
        x = self.layer_norm(x)
        logits = torch.matmul(x, self.label_queries.T) * self.scale
        return logits


class AWAREModel(nn.Module):
    """
    DeBERTa → sentence mean pooling → BiLSTM → classification head.
    Q011: Reverted to Q006 proven architecture (LSAN/attn pooling regressed in Q010).

    forward() returns dict with "logits" and optionally embeddings.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        encoder_path: Optional[str] = None,
        hidden_size: Optional[int] = None,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        num_labels: int = 11,
        max_sentences: int = 32,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if encoder_path:
            self.encoder = AutoModel.from_pretrained(encoder_path)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        if hidden_size is None:
            hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size

        # Gradient checkpointing: trade ~30% speed for ~40% memory savings.
        # Critical for DeBERTa-v3-large (304M params) at batch_size=8.
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        self.pooling = SentenceMeanPooling(max_sentences)
        self.context_encoder = BiLSTMContextEncoder(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
        )
        self.classifier = ClassificationHead(
            hidden_size=hidden_size, num_labels=num_labels, dropout=dropout,
        )

        if freeze_encoder:
            self._freeze_encoder()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "AWAREModel: %s, params %.1fM trainable %.1fM",
            encoder_path or encoder_name, total / 1e6, trainable / 1e6,
        )

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen")

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        logger.info("Encoder unfrozen")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_boundaries: List[List[Tuple[int, int]]],
        sentence_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = encoder_out.last_hidden_state
        sentence_embeddings = self.pooling(
            token_embeddings, attention_mask, sentence_boundaries, sentence_mask,
        )
        context_embeddings = self.context_encoder(sentence_embeddings, sentence_mask)
        logits = self.classifier(context_embeddings)
        out = {"logits": logits}
        if return_embeddings:
            out["sentence_embeddings"] = sentence_embeddings
            out["context_embeddings"] = context_embeddings
        return out


def build_model_from_config(config, freeze_encoder: bool = False) -> AWAREModel:
    """Build AWAREModel from AWAREConfig."""
    m = config.model
    t = config.training
    return AWAREModel(
        encoder_name=m.encoder_name,
        encoder_path=getattr(m, "encoder_path", None) or None,
        hidden_size=m.hidden_size,
        lstm_hidden_size=m.lstm_hidden,
        lstm_num_layers=m.lstm_layers,
        num_labels=NUM_THEMES,
        max_sentences=m.max_sentences,
        dropout=m.dropout,
        freeze_encoder=freeze_encoder,
        gradient_checkpointing=getattr(t, "gradient_checkpointing", False),
    )
