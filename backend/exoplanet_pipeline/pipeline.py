import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
import joblib

FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_steff", "koi_slogg", "koi_srad",
    "koi_teq", "koi_insol",
]

def load_koi_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, comment="#")

def make_labels(df: pd.DataFrame, positive_mode: str):
    if positive_mode == "confirmed+candidate":
        mapping = {"CONFIRMED": 1, "CANDIDATE": 1, "FALSE POSITIVE": 0}
    elif positive_mode == "confirmed":
        mapping = {"CONFIRMED": 1, "CANDIDATE": 0, "FALSE POSITIVE": 0}
    else:
        raise ValueError("positive_mode must be 'confirmed+candidate' or 'confirmed'")
    return df["koi_disposition"].map(mapping)

def preprocess_data(df, positive_mode):
    y = make_labels(df, positive_mode)
    df = df.loc[~y.isna()].copy()
    y = y.loc[df.index].astype(int)

    feats = [f for f in FEATURES if f in df.columns]
    if not feats:
        raise ValueError("No expected features found in the CSV.")

    X = df[feats].copy()
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    return X_imputed, y.values, feats, imputer, df

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def train_random_forest(X_train, y_train, n_estimators=300, random_state=42):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    return {"report": report, "cm": cm, "roc": roc, "ap": ap}

def save_artifacts(clf, imputer, feats, df_raw, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # feature importances
    importances = pd.DataFrame({"feature": feats, "importance": clf.feature_importances_}) \
        .sort_values("importance", ascending=False)
    importances.to_csv(out_dir / "feature_importance.csv", index=False)

    # predictions
    X_all_imp = imputer.transform(df_raw[feats])
    proba_all = clf.predict_proba(X_all_imp)[:, 1]
    id_col = "kepoi_name" if "kepoi_name" in df_raw.columns else ("kepid" if "kepid" in df_raw.columns else None)

    preds = pd.DataFrame({
        "object_id": df_raw[id_col] if id_col else pd.RangeIndex(len(df_raw)),
        "koi_disposition": df_raw.get("koi_disposition"),
        "prob_exoplanet": proba_all,
    })
    preds.to_csv(out_dir / "predictions.csv", index=False)

    joblib.dump({"model": clf, "imputer": imputer, "features": feats}, out_dir / "rf_exoplanet_model.joblib")

def run_pipeline(file_path: str, positive_mode: str = "confirmed+candidate", out_dir: str = "outputs"):
    df = load_koi_data(file_path)
    X, y, feats, imputer, df_kept = preprocess_data(df, positive_mode)
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_random_forest(X_train, y_train)
    metrics = evaluate_model(clf, X_test, y_test)
    save_artifacts(clf, imputer, feats, df_kept, Path(out_dir))
    return clf, metrics

