"""
model.py — Standard fine-tuning model (NO AWARE components).

Architecture: DeBERTa encoder → SentenceMeanPooling → Dropout → Linear
NO BiLSTM, NO position embeddings, NO essay head, NO multi-sample dropout.
"""

import torch
import torch.nn as nn
import logging
from transformers import AutoModel, AutoConfig

from config import NUM_THEMES

logger = logging.getLogger(__name__)


class SentenceMeanPooling(nn.Module):
    """Mean-pool token embeddings within each sentence's character span.
    Identical to AWARE — required because data is essay-level with sentence boundaries.
    """

    def __init__(self, max_sentences=32):
        super().__init__()
        self.max_sentences = max_sentences

    def forward(self, token_embeddings, attention_mask, sentence_boundaries):
        B, S, H = token_embeddings.shape[0], self.max_sentences, token_embeddings.shape[-1]
        device = token_embeddings.device
        sentence_embeddings = torch.zeros(B, S, H, device=device)

        for b in range(B):
            for s in range(S):
                start, end = sentence_boundaries[b][s]
                if start == end:
                    break
                end = min(end, token_embeddings.shape[1])
                mask = attention_mask[b, start:end].unsqueeze(-1).float()
                if mask.sum() > 0:
                    sentence_embeddings[b, s] = (
                        token_embeddings[b, start:end] * mask
                    ).sum(0) / mask.sum().clamp(min=1)
        return sentence_embeddings


class StandardModel(nn.Module):
    """DeBERTa → SentenceMeanPooling → Dropout → Linear.
    
    Simple standard fine-tuning baseline. No BiLSTM context, no essay head,
    no position embeddings. Each sentence classified independently based on
    its token embeddings only.
    """

    def __init__(self, encoder_name, hidden_size=768, dropout=0.15,
                 max_sentences=32, num_labels=NUM_THEMES):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.pooling = SentenceMeanPooling(max_sentences)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "StandardModel: %s, %.1fM params (%.1fM trainable)",
            encoder_name, total / 1e6, trainable / 1e6,
        )

    def forward(self, input_ids, attention_mask, sentence_boundaries, sentence_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state  # [B, T, H]

        sentence_embs = self.pooling(token_embs, attention_mask, sentence_boundaries)  # [B, S, H]
        logits = self.classifier(self.dropout(sentence_embs))  # [B, S, num_labels]
        
        return {"logits": logits, "sentence_embeddings": sentence_embs}

    @classmethod
    def from_config(cls, config):
        m = config.model
        return cls(
            encoder_name=m.encoder_name,
            hidden_size=m.hidden_size,
            dropout=m.dropout,
            max_sentences=m.max_sentences,
            num_labels=m.num_labels,
        )

def build_model_from_config(config):
    return StandardModel.from_config(config)
