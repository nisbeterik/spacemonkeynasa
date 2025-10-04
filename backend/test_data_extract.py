'''
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)
from sklearn.impute import SimpleImputer
import joblib

# Default feature set; the script will auto-drop ones not present in the CSV.
FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_steff", "koi_slogg", "koi_srad",
    "koi_teq", "koi_insol",
]

def load_koi_data(file_path: str) -> pd.DataFrame:
    """Load KOI CSV (NASA Kepler cumulative table) with # comments allowed."""
    return pd.read_csv(file_path, comment="#")

def make_labels(df: pd.DataFrame, positive_mode: str) -> pd.Series:
    """
    Map dispositions to binary labels.
    positive_mode:
      - 'confirmed+candidate': CONFIRMED and CANDIDATE are positive (1)
      - 'confirmed': only CONFIRMED is positive (1)
    """
    if positive_mode == "confirmed+candidate":
        mapping = {"CONFIRMED": 1, "CANDIDATE": 1, "FALSE POSITIVE": 0}
    elif positive_mode == "confirmed":
        mapping = {"CONFIRMED": 1, "CANDIDATE": 0, "FALSE POSITIVE": 0}
    else:
        raise ValueError("positive_mode must be 'confirmed+candidate' or 'confirmed'")
    return df["koi_disposition"].map(mapping)

def select_available_features(df: pd.DataFrame) -> list[str]:
    """Use only features that actually exist in the CSV."""
    feats = [f for f in FEATURES if f in df.columns]
    if not feats:
        raise ValueError(
            "None of the expected feature columns found.\n"
            f"Expected any of: {FEATURES}\nGot columns like: {list(df.columns)[:30]}"
        )
    return feats

def preprocess_data(df: pd.DataFrame, positive_mode: str):
    """Create labels, select features, median-impute missing values."""
    y = make_labels(df, positive_mode)
    df = df.loc[~y.isna()].copy()
    y = y.loc[df.index].astype(int)

    feats = select_available_features(df)
    X = df[feats].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    return X_imputed, y.values, feats, imputer, df

def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified split to keep class balance similar across splits."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_random_forest(X_train, y_train, n_estimators=300, random_state=42):
    """RandomForest with class_weight to handle imbalance."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """Print metrics and return a dict."""
    proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {ap:.4f}")

    return {"report": report, "cm": cm, "roc": roc, "ap": ap}

def save_artifacts(clf, imputer, feats, df_raw, out_dir: Path):
    """Save model, feature importances, and per-object probabilities."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature importance
    fi = pd.DataFrame({"feature": feats, "importance": clf.feature_importances_}) \
            .sort_values("importance", ascending=False)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)

    # Predictions for ALL rows (handy for triage)
    X_all = df_raw[feats]
    X_all_imp = imputer.transform(X_all)
    proba_all = clf.predict_proba(X_all_imp)[:, 1]

    id_col = "kepoi_name" if "kepoi_name" in df_raw.columns else ("kepid" if "kepid" in df_raw.columns else None)
    preds = pd.DataFrame({
        "object_id": df_raw[id_col] if id_col else pd.RangeIndex(len(df_raw)),
        "koi_disposition": df_raw.get("koi_disposition"),
        "prob_exoplanet": proba_all,
    })
    preds.sort_values("prob_exoplanet", ascending=False).to_csv(out_dir / "predictions.csv", index=False)

    # Save the trained model + imputer together
    joblib.dump({"model": clf, "imputer": imputer, "features": feats}, out_dir / "rf_exoplanet_model.joblib")

def run_pipeline(file_path: str, positive_mode: str, out_dir: str = "outputs"):
    df = load_koi_data(file_path)
    X, y, feats, imputer, df_kept = preprocess_data(df, positive_mode)
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_random_forest(X_train, y_train)
    _ = evaluate_model(clf, X_test, y_test)
    save_artifacts(clf, imputer, feats, df_kept, Path(out_dir))
    return clf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="./data/cumulative_2025.10.04_01.59.40.csv", help="Path to KOI cumulative CSV")
    parser.add_argument("--positive", choices=["confirmed+candidate", "confirmed"], default="confirmed+candidate",
                        help="Which dispositions count as positive (1).")
    parser.add_argument("--out", default="outputs", help="Directory to save results")
    args = parser.parse_args()

    run_pipeline(args.csv, args.positive, args.out)
'''

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)
from sklearn.impute import SimpleImputer
import joblib

# Default feature set
FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_steff", "koi_slogg", "koi_srad",
    "koi_teq", "koi_insol",
]

def load_koi_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, comment="#")

def make_labels(df: pd.DataFrame, positive_mode: str) -> pd.Series:
    if positive_mode == "confirmed+candidate":
        mapping = {"CONFIRMED": 1, "CANDIDATE": 1, "FALSE POSITIVE": 0}
    elif positive_mode == "confirmed":
        mapping = {"CONFIRMED": 1, "CANDIDATE": 0, "FALSE POSITIVE": 0}
    else:
        raise ValueError("positive_mode must be 'confirmed+candidate' or 'confirmed'")
    return df["koi_disposition"].map(mapping)

def select_available_features(df: pd.DataFrame) -> list[str]:
    feats = [f for f in FEATURES if f in df.columns]
    if not feats:
        raise ValueError(
            "None of the expected feature columns found.\n"
            f"Expected any of: {FEATURES}\nGot columns like: {list(df.columns)[:30]}"
        )
    return feats

def preprocess_data(df: pd.DataFrame, positive_mode: str):
    y = make_labels(df, positive_mode)
    df = df.loc[~y.isna()].copy()
    y = y.loc[df.index].astype(int)

    feats = select_available_features(df)
    X = df[feats].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    return X_imputed, y.values, feats, imputer, df

def split_data_train_val_test(X, y, train_size=0.6, val_size=0.2, random_state=42):
    """Split data into train (60%), validation (20%), test (20%) with stratification."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    val_relative_size = val_size / (1 - 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_random_forest_with_validation(X_train, y_train, X_val, y_val, random_state=42):
    """Manual hyperparameter tuning using validation set, avoiding multiprocessing issues."""
    param_grid = [
        {"n_estimators": n, "max_depth": d}
        for n in [100, 200, 300] for d in [None, 5, 10]
    ]

    best_model = None
    best_ap = -1
    for params in param_grid:
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        proba_val = clf.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, proba_val)
        if ap > best_ap:
            best_ap = ap
            best_model = clf
    print(f"Selected model with AP on validation: {best_ap:.4f}")
    return best_model

def evaluate_model(clf, X_test, y_test):
    proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {ap:.4f}")

    return {"report": report, "cm": cm, "roc": roc, "ap": ap}

def save_artifacts(clf, imputer, feats, df_raw, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fi = pd.DataFrame({"feature": feats, "importance": clf.feature_importances_}) \
            .sort_values("importance", ascending=False)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)

    X_all = df_raw[feats]
    X_all_imp = imputer.transform(X_all)
    proba_all = clf.predict_proba(X_all_imp)[:, 1]

    id_col = "kepoi_name" if "kepoi_name" in df_raw.columns else ("kepid" if "kepid" in df_raw.columns else None)
    preds = pd.DataFrame({
        "object_id": df_raw[id_col] if id_col else pd.RangeIndex(len(df_raw)),
        "koi_disposition": df_raw.get("koi_disposition"),
        "prob_exoplanet": proba_all,
    })
    preds.sort_values("prob_exoplanet", ascending=False).to_csv(out_dir / "predictions.csv", index=False)

    joblib.dump({"model": clf, "imputer": imputer, "features": feats}, out_dir / "rf_exoplanet_model.joblib")

def run_pipeline(file_path: str, positive_mode: str, out_dir: str = "outputs"):
    df = load_koi_data(file_path)
    X, y, feats, imputer, df_kept = preprocess_data(df, positive_mode)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_train_val_test(X, y)
    clf = train_random_forest_with_validation(X_train, y_train, X_val, y_val)
    _ = evaluate_model(clf, X_test, y_test)
    save_artifacts(clf, imputer, feats, df_kept, Path(out_dir))
    return clf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="./data/cumulative_2025.10.04_01.59.40.csv", help="Path to KOI cumulative CSV")
    parser.add_argument("--positive", choices=["confirmed+candidate", "confirmed"], default="confirmed+candidate",
                        help="Which dispositions count as positive (1).")
    parser.add_argument("--out", default="outputs", help="Directory to save results")
    args = parser.parse_args()

    run_pipeline(args.csv, args.positive, args.out)
