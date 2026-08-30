"""Paths and configuration for the inference server."""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent  # Final_Thesis_Folders2/
MODELS_DIR = BASE / "Models_inference"

MODELS = {
    "large_v4": {
        "name": "AWARE Large v4 (DeBERTa-v3-large)",
        "encoder_path": str(MODELS_DIR / "Model_large_v3" / "dapt_encoder"),
        "weights_path": str(MODELS_DIR / "Model_large_v3" / "results" / "final_v4" / "best.pt"),
        "config_path": str(MODELS_DIR / "Model_large_v3" / "results" / "final_v4" / "config.json"),
        "thresholds_path": str(MODELS_DIR / "Model_large_v3" / "results" / "final_v4" / "thresholds.json"),
        "calibration_path": str(MODELS_DIR / "Model_large_v3" / "results" / "final_v4" / "calibration.json"),
        "scripts_path": str(MODELS_DIR / "Model_large_v3" / "scripts"),
        "f1_macro": 0.494,
        "params": "360M",
    },
    "base": {
        "name": "AWARE Base (DeBERTa-v3-base)",
        "encoder_path": str(MODELS_DIR / "Model_base" / "dapt_encoder"),
        "weights_path": str(MODELS_DIR / "Model_base" / "results" / "final" / "best.pt"),
        "config_path": str(MODELS_DIR / "Model_base" / "results" / "final" / "config.json"),
        "thresholds_path": str(MODELS_DIR / "Model_base" / "results" / "final" / "thresholds.json"),
        "calibration_path": str(MODELS_DIR / "Model_base" / "results" / "final" / "calibration.json"),
        "scripts_path": str(MODELS_DIR / "Model_base" / "scripts"),
        "f1_macro": 0.474,
        "params": "125M",
    },
}

THEMES = [
    "Attainment", "Aspirational", "Navigational", "Resistance",
    "Perseverance", "Social", "Spiritual", "Familial_Capital",
]
