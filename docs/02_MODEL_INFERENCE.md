# Model Inference — How to Use the Trained Models

## Available Models

| Model | Architecture | Params | Test F1 | Test PR-AUC | Weights Path |
|-------|-------------|--------|---------|-------------|-------------|
| **Large v4** | DeBERTa-v3-large | 360M | **0.494** | 0.484 | `Models_inference/Model_large_v3/results/final_v4/best.pt` (1.7GB) |
| Base | DeBERTa-v3-base | 125M | 0.474 | 0.473 | `Models_inference/Model_base/results/final/best.pt` |
| TF-IDF | LogReg + TF-IDF | ~10K | 0.378 | 0.350 | `Models_light/baselines/` |

## Critical Inference Parameters

### Per-Theme Thresholds (NOT 0.5!)

**Large v4:**
```json
{
  "Attainment": 0.05,
  "Aspirational": 0.23,
  "Navigational": 0.31,
  "Resistance": 0.09,
  "Perseverance": 0.35,
  "Social": 0.22,
  "Spiritual": 0.053,
  "Familial_Capital": 0.14
}
```

**Base:**
```json
{
  "Attainment": 0.06,
  "Aspirational": 0.16,
  "Navigational": 0.32,
  "Resistance": 0.09,
  "Perseverance": 0.18,
  "Social": 0.17,
  "Spiritual": 0.05,
  "Familial_Capital": 0.18
}
```

### Platt Calibration: prob = sigmoid(a * logit + b)

**Large v4:**
| Theme | a | b |
|-------|---|---|
| Attainment | 1.2 | -3.0 |
| Aspirational | 1.7 | -1.8 |
| Navigational | 1.6 | -1.6 |
| Resistance | 1.6 | -1.8 |
| Perseverance | 1.3 | -2.0 |
| Social | 1.4 | -2.2 |
| Spiritual | 1.1 | -2.2 |
| Familial_Capital | 1.3 | -2.6 |

**Base:**
| Theme | a | b |
|-------|---|---|
| Attainment | 0.7 | -2.4 |
| Aspirational | 1.1 | -2.0 |
| Navigational | 0.8 | -1.4 |
| Resistance | 0.7 | -1.8 |
| Perseverance | 0.7 | -2.0 |
| Social | 1.0 | -2.2 |
| Spiritual | 0.9 | -1.2 |
| Familial_Capital | 0.8 | -1.6 |

## Inference Pipeline Steps

1. **Segment essay into sentences** (spaCy or regex)
2. **Join sentences** with ". " separator (period + space)
3. **Tokenize** with DeBERTa tokenizer (max 512 tokens)
4. **Map sentence boundaries** to token positions via offset_mapping
5. **Pad** to max_sentences=32
6. **Forward pass** (no grad): DeBERTa → SentenceMeanPooling → PositionEmbedding → BiLSTM → ClassificationHead
7. **Apply Platt calibration**: prob = sigmoid(a * logit + b) per theme
8. **Apply per-theme thresholds** → binary predictions
9. **Return** per-sentence predictions with confidence scores

## Key Files for Inference Server

```
Models_inference/
├── README.md                          ← Complete inference code reference
├── Model_large_v3/
│   ├── dapt_encoder/                  ← DAPT-adapted encoder + tokenizer
│   │   ├── model.safetensors (1.7GB)
│   │   ├── config.json
│   │   └── tokenizer files (spm.model, tokenizer.json, etc.)
│   ├── results/final_v4/
│   │   ├── best.pt (1.7GB)           ← Trained weights
│   │   ├── config.json               ← Model config
│   │   ├── thresholds.json           ← Per-theme thresholds
│   │   └── calibration.json          ← Platt parameters
│   └── scripts/
│       ├── model.py                   ← AWAREModel class
│       ├── config.py                  ← AWAREConfig, THEMES, NUM_THEMES
│       ├── dataset.py                 ← Data loading
│       └── evaluate.py                ← Evaluation pipeline
├── Model_base/
│   ├── dapt_encoder/
│   ├── results/final/
│   └── scripts/
└── data/
    ├── train_data.pkl
    ├── val_data.pkl
    ├── test_data.pkl
    └── splits_stats.json
```

## Inference API Design

### POST /api/infer/single
```
Request:  { "text": "essay text...", "model_id": "large_v4" }
Response: {
  "model_id": "large_v4",
  "sentences": [
    {
      "index": 0,
      "text": "First sentence.",
      "predictions": {
        "Attainment": { "probability": 0.72, "predicted": true, "threshold": 0.05 },
        ...8 themes...
      }
    }
  ],
  "processing_time_ms": 342
}
```

### POST /api/infer/batch
```
Request: multipart/form-data { file: CSV, model_id: "large_v4" }
Response: SSE stream with progress events, then download URL
CSV columns expected: essay_id, essay_text
```

### GET /api/models
Returns list of available models with metadata (F1, params, loaded status).

### GET /api/health
Returns server status and which models are loaded.

## Technical Notes

- CPU inference works (~1-3 sec/essay for large model, ~0.5 sec for base)
- GPU auto-detected via torch.cuda.is_available()
- Lazy loading: large v4 on startup, base on first request
- Tokenizer: DebertaV2Tokenizer, vocab 128,100, SentencePiece model
- SENTENCE_SEP = ". " (period + space, not just space)
- Max 32 sentences per essay, max 512 tokens total

## Dependencies for Inference Server

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pyyaml>=6.0
spacy>=3.7.0
python-multipart>=0.0.9
pydantic>=2.0.0
scikit-learn>=1.3.0
```
