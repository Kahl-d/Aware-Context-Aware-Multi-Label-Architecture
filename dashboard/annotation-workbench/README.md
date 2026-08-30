# ALMA Research Dashboard

A React + TypeScript (Vite) frontend with a FastAPI inference backend for
exploring the ALMA reflective-essay dataset and running the **AWARE** CCW
theme classifier on new text.

Pages: dataset explorer, EDA/exploration charts, live model inference, and a
results/paper view.

```
dashboard/
├── src/                  # React app (TypeScript)
│   ├── pages/            # one component per route
│   ├── components/       # data-explorer, exploration, inference, paper, layout, shared
│   ├── hooks/            # useDataLoader, useFilteredData, useDebounce
│   ├── stores/           # Zustand state
│   ├── lib/              # api client, csv + stats helpers
│   ├── constants/        # themes, routes
│   └── types/
├── public/
│   ├── data/             # JSON the app fetches (see public/data/README.md)
│   └── plots/            # (gitignored — large PNGs)
├── inference-server/     # FastAPI backend
│   ├── main.py           # API: POST /infer, /infer/batch, etc.
│   ├── server_config.py  # model paths + theme list  ← edit paths here
│   ├── models/aware_model.py
│   └── requirements.txt
└── data-prep/
    └── prepare_all.py    # builds public/data/*.json from the processed CSVs
```

## 1. Frontend (works without the model)

```bash
npm install
# one-time: provide demo data so the app boots (see public/data/README.md)
cp public/data/essays.demo.json    public/data/essays.json
cp public/data/sentences.demo.json public/data/sentences.json
npm run dev          # http://localhost:5173
```

The explorer and EDA pages run entirely on the static JSON in `public/data/`.
**No Python/model is needed** for these.

## 2. Inference backend (optional — needs the trained model)

Live inference requires the trained AWARE weights, which are **~4.7 GB and are
NOT in this repo**. `inference-server/server_config.py` expects them under a
sibling `Models_inference/` folder:

```
Models_inference/Model_large_v3/
  dapt_encoder/                       # domain-adapted DeBERTa-v3-large encoder
  results/final_v4/{best.pt, config.json, thresholds.json, calibration.json}
```

Get those from Khalid / the ALMA team (or retrain — see `../docs/02_MODEL_INFERENCE.md`),
update the paths in `server_config.py`, then:

```bash
cd inference-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The frontend's inference page calls this API (`src/lib/api.ts`). On Apple
Silicon it runs on MPS; CPU works too (slower).

## Stack

- React 19 · TypeScript · Vite · Zustand · Recharts
- FastAPI · PyTorch · HuggingFace Transformers (DeBERTa-v3)

## The 8 themes

Attainment · Aspirational · Navigational · Resistance · Perseverance · Social ·
Spiritual · Familial_Capital (+ Class_0 = no theme). See `../docs/` for meaning
and `../sample-data/` for the data schema.
