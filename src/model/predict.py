import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"

def load_model():
    # choose model in metrics.json
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            info = json.load(f)
            name = info.get("chosen", "random_forest")
    else:
        name = "random_forest"
    model = joblib.load(MODELS_DIR / f"{name}.joblib")
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)
    return model, feature_cols

def predict_proba(feat_row: pd.Series) -> float:
    model, cols = load_model()
    x = feat_row[cols].values.reshape(1, -1)
    proba = float(model.predict_proba(x)[0,1])
    return proba
