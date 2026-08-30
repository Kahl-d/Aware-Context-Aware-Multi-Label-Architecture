"""AWARE model wrapper for inference."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


class AWAREInference:
    """Loads and runs an AWARE model for sentence-level CCW theme classification."""

    def __init__(self, model_config: dict):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.thresholds = None
        self.calibration = None
        self.themes = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        cfg = self.model_config
        scripts_path = cfg["scripts_path"]
        # Ensure model scripts path is FIRST to avoid name collisions
        if scripts_path in sys.path:
            sys.path.remove(scripts_path)
        sys.path.insert(0, scripts_path)

        # Force reimport from model scripts path
        import importlib
        import config as config_module
        importlib.reload(config_module)
        from config import AWAREConfig, THEMES
        from model import build_model_from_config

        self.themes = list(THEMES)

        with open(cfg["config_path"]) as f:
            config_dict = json.load(f)
        config_dict["model"]["encoder_path"] = cfg["encoder_path"]
        config = AWAREConfig._from_dict(config_dict)

        self.model = build_model_from_config(config)
        state_dict = torch.load(cfg["weights_path"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_path"])

        with open(cfg["thresholds_path"]) as f:
            self.thresholds = json.load(f)
        with open(cfg["calibration_path"]) as f:
            self.calibration = json.load(f)

        self._loaded = True
        print(f"Model loaded: {cfg['name']} on {self.device}")

    def segment_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 3]

    def predict(self, text: str) -> dict:
        self.load()

        sentences = self.segment_sentences(text)
        if not sentences:
            return {"sentences": [], "model": self.model_config["name"]}

        max_sents = 32
        sentences = sentences[:max_sents]
        n_sents = len(sentences)

        # Join with ". " separator (must match training)
        essay_text = ". ".join(sentences)

        # Tokenize
        enc = self.tokenizer(
            essay_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offsets = enc["offset_mapping"][0].numpy()  # [T, 2]

        # Find character-level sentence boundaries
        char_positions = []
        pos = 0
        for sent in sentences:
            start = essay_text.find(sent, pos)
            end = start + len(sent)
            char_positions.append((start, end))
            pos = end

        # Map char positions to token positions
        boundaries = []
        for char_start, char_end in char_positions:
            tok_start, tok_end = None, None
            for idx, (os, oe) in enumerate(offsets):
                if os == 0 and oe == 0:
                    continue
                if tok_start is None and oe > char_start:
                    tok_start = idx
                if os < char_end:
                    tok_end = idx + 1
            boundaries.append((tok_start or 0, tok_end or 1))

        # Pad to max_sentences
        while len(boundaries) < max_sents:
            boundaries.append((0, 0))

        # Build tensors: sentence_boundaries [B, S, 2], sentence_mask [B, S]
        sentence_boundaries = torch.tensor([boundaries], dtype=torch.long).to(self.device)
        sentence_mask = torch.zeros(1, max_sents, dtype=torch.float).to(self.device)
        sentence_mask[0, :n_sents] = 1.0

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_boundaries=sentence_boundaries,
                sentence_mask=sentence_mask,
            )

        logits = out["logits"][0, :n_sents, :].cpu().numpy()

        # Apply Platt calibration and thresholds
        results = []
        for i, sent in enumerate(sentences):
            preds = {}
            for j, theme in enumerate(self.themes):
                logit = float(logits[i, j])
                cal = self.calibration.get(theme, {"a": 1.0, "b": 0.0})
                a = cal.get("a", 1.0)
                b = cal.get("b", 0.0)
                prob = 1.0 / (1.0 + np.exp(-(a * logit + b)))
                threshold = self.thresholds.get(theme, 0.5)

                preds[theme] = {
                    "probability": round(float(prob), 4),
                    "predicted": bool(prob >= threshold),
                    "threshold": threshold,
                }

            results.append({
                "index": i,
                "text": sent,
                "predictions": preds,
            })

        return {"sentences": results, "model": self.model_config["name"]}
