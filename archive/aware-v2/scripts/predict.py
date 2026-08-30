"""
predict.py — Inference for AWARE v2.

Single essay prediction and batch prediction (CSV/Excel → annotated output).

Usage:
    # Single essay
    python scripts/predict.py --model_dir results/full/ --config configs/full.yaml \
        --text "I came to this country when I was young. My family always supported me."

    # Batch prediction from CSV
    python scripts/predict.py --model_dir results/full/ --config configs/full.yaml \
        --input_file essays.csv --output_file annotated_essays.xlsx

    # Batch from Excel (expects 'essay_text' or 'text' column)
    python scripts/predict.py --model_dir results/full/ --config configs/full.yaml \
        --input_file essays.xlsx --output_file annotated_output.xlsx
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer

from config import AWAREConfig, THEMES, THEME_TO_IDX, NUM_THEMES
from model import build_model_from_config
from metrics import apply_thresholds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class AWAREPredictor:
    """Load trained model + thresholds and predict themes for essays."""

    def __init__(self, model_dir: str, config: AWAREConfig, device: str = None):
        self.config = config
        model_dir = Path(model_dir)

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Load model
        self.model = build_model_from_config(config)
        best_path = model_dir / "best.pt"
        self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded model from %s on %s", best_path, self.device)

        # Load thresholds
        thresh_path = model_dir / "thresholds.json"
        if thresh_path.exists():
            with open(thresh_path) as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = {t: 0.30 for t in THEMES}
            logger.warning("No thresholds.json, using default 0.30")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
        self.max_seq_length = config.model.max_seq_length
        self.max_sentences = config.model.max_sentences

    def predict_essay(self, text: str) -> List[Dict]:
        """
        Predict themes for a single essay.

        Args:
            text: Full essay text (will be split into sentences).
        Returns:
            List of dicts, one per sentence:
            [{"sentence": str, "themes": [str, ...], "probabilities": {theme: float}}]
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        sentences = sentences[:self.max_sentences]
        item = self._prepare_input(sentences)

        with torch.no_grad():
            output = self.model(
                input_ids=item["input_ids"].unsqueeze(0).to(self.device),
                attention_mask=item["attention_mask"].unsqueeze(0).to(self.device),
                sentence_boundaries=[item["sentence_boundaries"]],
                sentence_mask=item["sentence_mask"].unsqueeze(0).to(self.device),
            )

        logits = output["logits"][0]  # [max_sentences, NUM_THEMES]
        probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for i, sent in enumerate(sentences):
            sent_probs = probs[i]
            # Apply thresholds
            themes = []
            prob_dict = {}
            for j, theme in enumerate(THEMES):
                t = self.thresholds.get(theme, 0.30)
                prob_dict[theme] = round(float(sent_probs[j]), 4)
                if sent_probs[j] >= t:
                    themes.append(theme)
            results.append({
                "sentence": sent,
                "themes": themes if themes else ["class_0"],
                "probabilities": prob_dict,
            })

        return results

    def predict_batch(self, essays: List[str]) -> List[List[Dict]]:
        """Predict themes for multiple essays."""
        return [self.predict_essay(essay) for essay in essays]

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting (period/question/exclamation followed by space+capital or end)."""
        import re
        text = text.strip()
        if not text:
            return []
        # Split on sentence-ending punctuation followed by space
        raw = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in raw if s.strip()]
        return sentences

    def _prepare_input(self, sentences: List[str]) -> Dict:
        """Tokenize essay, compute sentence boundaries."""
        num_sentences = len(sentences)
        char_starts = []
        joined_parts = []
        offset = 0
        for i, s in enumerate(sentences):
            char_starts.append(offset)
            joined_parts.append(s)
            offset += len(s)
            if i < num_sentences - 1:
                joined_parts.append(" ")
                offset += 1
        joined_text = "".join(joined_parts)
        char_ends = []
        for i in range(num_sentences):
            if i < num_sentences - 1:
                char_ends.append(char_starts[i + 1] - 1)
            else:
                char_ends.append(len(joined_text))

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
        offset_mapping = encoding["offset_mapping"].squeeze(0)

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
        while len(sentence_boundaries) < self.max_sentences:
            sentence_boundaries.append((0, 0))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentence_boundaries": sentence_boundaries,
            "sentence_mask": sentence_mask,
        }


def predict_from_file(predictor, input_path: str, output_path: str):
    """Read essays from CSV/Excel, predict, write annotated output."""
    import pandas as pd

    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    # Find text column
    text_col = None
    for col in ["essay_text", "text", "essay", "Essay", "Text", "content"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        raise ValueError(f"No text column found. Available: {list(df.columns)}")

    logger.info("Processing %d essays from column '%s'", len(df), text_col)

    # Predict
    rows = []
    for idx, row in df.iterrows():
        essay_text = str(row[text_col])
        predictions = predictor.predict_essay(essay_text)
        for pred in predictions:
            r = {"essay_idx": idx, "sentence": pred["sentence"], "themes": ", ".join(pred["themes"])}
            for theme in THEMES:
                r[f"prob_{theme}"] = pred["probabilities"].get(theme, 0)
            rows.append(r)

    result_df = pd.DataFrame(rows)

    if output_path.suffix in [".xlsx", ".xls"]:
        result_df.to_excel(output_path, index=False)
    else:
        result_df.to_csv(output_path, index=False)

    logger.info("Saved %d sentence predictions to %s", len(result_df), output_path)


def main():
    parser = argparse.ArgumentParser(description="AWARE v2 Prediction")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with best.pt and thresholds.json")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--text", type=str, default=None, help="Single essay text to predict")
    parser.add_argument("--input_file", type=str, default=None, help="CSV/Excel file with essays")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for batch predictions")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = AWAREConfig.from_yaml(args.config)
    predictor = AWAREPredictor(args.model_dir, config, device=args.device)

    if args.text:
        results = predictor.predict_essay(args.text)
        print("\n" + "=" * 70)
        print("PREDICTIONS")
        print("=" * 70)
        for r in results:
            themes = ", ".join(r["themes"])
            print(f"\n  [{themes}]")
            print(f"  {r['sentence']}")
            top_probs = sorted(r["probabilities"].items(), key=lambda x: -x[1])[:3]
            prob_str = "  ".join(f"{t}: {p:.3f}" for t, p in top_probs)
            print(f"  Top probs: {prob_str}")
        print("=" * 70)

    elif args.input_file:
        if not args.output_file:
            args.output_file = str(Path(args.input_file).stem) + "_annotated.xlsx"
        predict_from_file(predictor, args.input_file, args.output_file)

    else:
        parser.error("Provide either --text or --input_file")


if __name__ == "__main__":
    main()
