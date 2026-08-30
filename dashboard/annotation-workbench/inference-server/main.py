"""FastAPI inference server for AWARE CCW theme classification."""
import time
import io
import csv
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server_config import MODELS, THEMES
from models.aware_model import AWAREInference

app = FastAPI(title="AWARE Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model instances (lazy loaded)
_models: dict[str, AWAREInference] = {}


def get_model(model_id: str) -> AWAREInference:
    """Get or create model instance."""
    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    if model_id not in _models:
        _models[model_id] = AWAREInference(MODELS[model_id])
    return _models[model_id]


class InferenceRequest(BaseModel):
    text: str
    model_id: str = "large_v4"


class ModelInfo(BaseModel):
    id: str
    name: str
    f1_macro: float
    params: str
    loaded: bool


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_available": list(MODELS.keys()),
        "models_loaded": [k for k, v in _models.items() if v._loaded],
    }


@app.get("/api/models")
def list_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            id=model_id,
            name=cfg["name"],
            f1_macro=cfg["f1_macro"],
            params=cfg["params"],
            loaded=model_id in _models and _models[model_id]._loaded,
        )
        for model_id, cfg in MODELS.items()
    ]


@app.post("/api/infer/single")
def infer_single(req: InferenceRequest):
    start = time.time()
    model = get_model(req.model_id)
    result = model.predict(req.text)
    elapsed = round((time.time() - start) * 1000)
    return {
        "model_id": req.model_id,
        "sentences": result["sentences"],
        "processing_time_ms": elapsed,
    }


@app.post("/api/infer/batch")
async def infer_batch(
    file: UploadFile = File(...),
    model_id: str = Form("large_v4"),
):
    """Process a CSV of essays. Expected columns: essay_id, essay_text."""
    model = get_model(model_id)

    content = await file.read()
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))

    if "essay_text" not in (reader.fieldnames or []):
        raise HTTPException(
            status_code=400,
            detail="CSV must have 'essay_id' and 'essay_text' columns",
        )

    results = []
    for row in reader:
        essay_id = row.get("essay_id", "")
        essay_text = row.get("essay_text", "")
        if not essay_text.strip():
            continue

        prediction = model.predict(essay_text)
        for sent in prediction["sentences"]:
            results.append({
                "essay_id": essay_id,
                "sentence_index": sent["index"],
                "sentence_text": sent["text"],
                **{
                    f"{theme}_prob": sent["predictions"][theme]["probability"]
                    for theme in THEMES
                },
                **{
                    f"{theme}_pred": int(sent["predictions"][theme]["predicted"])
                    for theme in THEMES
                },
            })

    # Return as CSV
    if not results:
        raise HTTPException(status_code=400, detail="No valid essays found in CSV")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=aware_predictions.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
