import json, joblib, pandas as pd, os
from .features import FEATURES

# lazy single-load
_MODEL = None
_SCHEMA = {"features": FEATURES}

def get_model(path="backend/ml/model.pkl"):
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(path)
    return _MODEL

def predict_df(df: pd.DataFrame):
    # ensure all expected columns exist
    for c in FEATURES:
        if c not in df.columns:
            df[c] = None
    df = df[FEATURES]
    proba = get_model().predict_proba(df)[:, 1]
    return proba
