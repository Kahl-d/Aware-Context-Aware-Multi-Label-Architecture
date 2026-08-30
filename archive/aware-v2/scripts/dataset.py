"""
dataset.py — Essay-level dataset for AWARE v2.

Each sample is one essay. Sentences are joined, tokenized as one sequence,
then sentence boundaries are mapped to token positions for pooling.
AEDA augmentation for training.
"""

import math
import pickle
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
from collections import Counter

from config import THEMES, THEME_TO_IDX, NUM_THEMES

logger = logging.getLogger(__name__)

SENTENCE_SEP = " "

# Indices of rare themes (support < 300 in training) — get higher augmentation rate
RARE_THEME_INDICES = [
    THEME_TO_IDX["Filial Piety"],            # 263 train examples
    THEME_TO_IDX["Community Consciousness"],  # 170
    THEME_TO_IDX["First Gen"],               # 114
]


class AWAREDataset(Dataset):
    """
    Essay-level dataset.

    Each item: one essay → tokenized text + sentence boundaries + multi-label annotations.
    """

    def __init__(
        self,
        data: dict,
        tokenizer,
        max_seq_length: int = 512,
        max_sentences: int = 32,
        augment: bool = False,
        aeda_prob: float = 0.3,
        cache_tokenization: bool = False,
    ):
        self.essays = data["essays"]
        self.essay_ids = data["essay_ids"]
        self.weights = data.get("weights", {})
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_sentences = max_sentences
        self.augment = augment
        self.aeda_prob = aeda_prob
        self.aeda_punctuation = list(".,;:!?")

        self.cache_tokenization = cache_tokenization and not augment
        self._cache = {}

        logger.info(
            "AWAREDataset: %d essays, augment=%s, max_seq=%d, max_sent=%d",
            len(self.essay_ids), augment, max_seq_length, max_sentences,
        )

    def __len__(self):
        return len(self.essay_ids)

    def __getitem__(self, idx) -> Dict:
        eid = self.essay_ids[idx]

        if self.cache_tokenization and eid in self._cache:
            return self._cache[eid]

        essay = self.essays[eid]
        sentences = essay["sentences"][:self.max_sentences]
        annotations = essay["annotations"][:self.max_sentences]

        if self.augment:
            # Apply higher augmentation rate (0.6) for sentences with rare theme labels
            augmented = []
            for i, s in enumerate(sentences):
                if i < len(annotations) and any(
                    t in annotations[i] for t in ("Filial Piety", "Community Consciousness", "First Gen")
                ):
                    augmented.append(self._aeda_augment(s, prob_override=0.6))
                else:
                    augmented.append(self._aeda_augment(s))
            sentences = augmented

        item = self._process_essay(eid, sentences, annotations)

        if self.cache_tokenization:
            self._cache[eid] = item
        return item

    def _process_essay(
        self, eid: str, sentences: List[str], annotations: List[set],
    ) -> Dict:
        """Tokenize essay, compute sentence boundaries in token space, build labels."""
        num_sentences = len(sentences)

        # Join sentences with separator, track char positions
        char_starts = []
        joined_parts = []
        offset = 0
        for i, s in enumerate(sentences):
            char_starts.append(offset)
            joined_parts.append(s)
            offset += len(s)
            if i < num_sentences - 1:
                joined_parts.append(SENTENCE_SEP)
                offset += len(SENTENCE_SEP)
        joined_text = "".join(joined_parts)
        char_ends = []
        for i in range(num_sentences):
            if i < num_sentences - 1:
                char_ends.append(char_starts[i + 1] - len(SENTENCE_SEP))
            else:
                char_ends.append(len(joined_text))

        # Tokenize with offset_mapping
        encoding = self.tokenizer(
            joined_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)  # (seq_len, 2)

        # Map char boundaries → token boundaries
        sentence_boundaries = []
        for i in range(num_sentences):
            cs, ce = char_starts[i], char_ends[i]
            tok_start, tok_end = self._char_to_token_span(offset_mapping, cs, ce)
            if tok_start is not None and tok_end is not None:
                sentence_boundaries.append((tok_start, tok_end))
            else:
                break  # Sentence got truncated

        actual_sentences = len(sentence_boundaries)

        # Sentence mask
        sentence_mask = torch.zeros(self.max_sentences, dtype=torch.float32)
        sentence_mask[:actual_sentences] = 1.0

        # Labels: multi-hot [max_sentences, NUM_THEMES]
        labels = torch.zeros(self.max_sentences, NUM_THEMES, dtype=torch.float32)
        for i in range(actual_sentences):
            if i < len(annotations):
                for theme in annotations[i]:
                    if theme in THEME_TO_IDX:
                        labels[i, THEME_TO_IDX[theme]] = 1.0

        # Pad boundaries to max_sentences
        while len(sentence_boundaries) < self.max_sentences:
            sentence_boundaries.append((0, 0))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentence_boundaries": sentence_boundaries,
            "sentence_mask": sentence_mask,
            "labels": labels,
            "essay_id": eid,
        }

    def _char_to_token_span(
        self, offset_mapping: torch.Tensor, char_start: int, char_end: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find token indices covering the character span [char_start, char_end)."""
        tok_start = None
        tok_end = None
        for t in range(offset_mapping.shape[0]):
            ts, te = offset_mapping[t].tolist()
            if ts == 0 and te == 0:
                continue  # Special token
            if ts < char_end and te > char_start:
                if tok_start is None:
                    tok_start = t
                tok_end = t + 1
        return tok_start, tok_end

    def _aeda_augment(self, text: str, prob_override: float = None) -> str:
        """AEDA: insert random punctuation at word boundaries."""
        prob = prob_override if prob_override is not None else self.aeda_prob
        if random.random() > prob:
            return text
        words = text.split()
        if len(words) < 3:
            return text
        n_insert = max(1, len(words) // 5)
        positions = sorted(random.sample(range(1, len(words)), min(n_insert, len(words) - 1)))
        result = []
        for i, word in enumerate(words):
            result.append(word)
            if i + 1 in positions:
                result.append(random.choice(self.aeda_punctuation))
        return " ".join(result)


def get_sampling_weights(data: dict, rare_theme_boost: dict = None) -> List[float]:
    """Get per-essay sampling weights for WeightedRandomSampler.

    Applies optional theme-aware boost so essays containing rare themes appear
    more often in training batches. Without this, First Gen essays appear in
    only ~5% of batches (64 essays / 9664 total), starving training of signal.

    Args:
        data: Split data dict with 'weights', 'essay_ids', 'essays' keys.
        rare_theme_boost: Dict mapping theme name → multiplicative boost factor.
            E.g. {"First Gen": 8.0} multiplies that essay's base weight by 8.
            The max boost across all rare themes in the essay is applied.

    Returns:
        List of float sampling weights, one per essay.
    """
    base_weights = data.get("weights", {})
    if not rare_theme_boost:
        return [base_weights.get(eid, 1.0) for eid in data["essay_ids"]]

    essays = data.get("essays", {})
    result = []
    for eid in data["essay_ids"]:
        w = base_weights.get(eid, 1.0)
        essay_themes = set()
        for ann in essays.get(eid, {}).get("annotations", []):
            essay_themes.update(ann)
        boost = max((rare_theme_boost.get(t, 1.0) for t in essay_themes), default=1.0)
        result.append(w * boost)
    return result


def create_dataloader(
    data: dict,
    tokenizer,
    config,
    shuffle: bool = True,
    augment: bool = False,
    use_weighted_sampling: bool = False,
) -> DataLoader:
    """Create DataLoader from split data dict."""
    dataset = AWAREDataset(
        data=data,
        tokenizer=tokenizer,
        max_seq_length=config.model.max_seq_length,
        max_sentences=config.model.max_sentences,
        augment=augment,
        aeda_prob=config.augmentation.aeda_prob if augment else 0,
        cache_tokenization=not augment,
    )

    sampler = None
    if use_weighted_sampling and shuffle:
        rare_boost = getattr(config.training, "rare_theme_boost", None) or {}
        weights = get_sampling_weights(data, rare_theme_boost=rare_boost)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False  # Can't use both sampler and shuffle

    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.training.num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=False,
    )


def _collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate: stack tensors, keep boundaries as list."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "sentence_boundaries": [b["sentence_boundaries"] for b in batch],
        "sentence_mask": torch.stack([b["sentence_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "essay_ids": [b["essay_id"] for b in batch],
    }


def load_split_data(path: str) -> dict:
    """Load a split data file (train_data.pkl, val_data.pkl, etc.)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded %d essays from %s", len(data["essay_ids"]), path)
    return data
