"""
dataset.py — Dataset for Standard baseline.
Same data format as AWARE but NO AEDA augmentation.
Uses ". " sentence separator (same as AWARE for fair comparison).
"""

import pickle
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import THEMES, THEME_TO_IDX, NUM_THEMES

logger = logging.getLogger(__name__)

SENTENCE_SEP = ". "


class StandardDataset(Dataset):
    """Essay-level dataset — identical to AWARE but no augmentation."""

    def __init__(self, data, tokenizer, max_seq_length=512, max_sentences=32):
        self.essays = data["essays"]
        self.essay_ids = data["essay_ids"]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_sentences = max_sentences
        logger.info("StandardDataset: %d essays, max_seq=%d, max_sent=%d",
                     len(self.essay_ids), max_seq_length, max_sentences)

    def __len__(self):
        return len(self.essay_ids)

    def __getitem__(self, idx):
        eid = self.essay_ids[idx]
        essay = self.essays[eid]
        sentences = essay["sentences"][:self.max_sentences]
        annotations = essay["annotations"][:self.max_sentences]
        return self._process_essay(eid, sentences, annotations)

    def _process_essay(self, eid, sentences, annotations):
        num_sentences = len(sentences)

        # Join sentences with proper separator
        char_starts, joined_parts = [], []
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

        # Tokenize
        encoding = self.tokenizer(
            joined_text, max_length=self.max_seq_length, truncation=True,
            padding="max_length", return_tensors="pt", return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        # Map char → token boundaries
        sentence_boundaries = []
        for i in range(num_sentences):
            cs, ce = char_starts[i], char_ends[i]
            tok_start, tok_end = None, None
            for t in range(offset_mapping.shape[0]):
                ts, te = offset_mapping[t].tolist()
                if ts == 0 and te == 0:
                    continue
                if ts < ce and te > cs:
                    if tok_start is None:
                        tok_start = t
                    tok_end = t + 1
            if tok_start is not None and tok_end is not None:
                sentence_boundaries.append((tok_start, tok_end))
            else:
                break

        actual = len(sentence_boundaries)
        sentence_mask = torch.zeros(self.max_sentences, dtype=torch.float32)
        sentence_mask[:actual] = 1.0

        labels = torch.zeros(self.max_sentences, NUM_THEMES, dtype=torch.float32)
        for i in range(actual):
            if i < len(annotations):
                for theme in annotations[i]:
                    if theme in THEME_TO_IDX:
                        labels[i, THEME_TO_IDX[theme]] = 1.0

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


def create_dataloader(data, tokenizer, config, shuffle=True):
    dataset = StandardDataset(
        data=data, tokenizer=tokenizer,
        max_seq_length=config.model.max_seq_length,
        max_sentences=config.model.max_sentences,
    )
    return DataLoader(
        dataset, batch_size=config.training.batch_size,
        shuffle=shuffle, num_workers=config.training.num_workers,
        collate_fn=_collate_fn, pin_memory=True, drop_last=False,
    )


def _collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "sentence_boundaries": [b["sentence_boundaries"] for b in batch],
        "sentence_mask": torch.stack([b["sentence_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "essay_ids": [b["essay_id"] for b in batch],
    }


def load_split_data(path: str) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded %d essays from %s", len(data["essay_ids"]), path)
    return data
