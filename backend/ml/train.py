import argparse, os, json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from .features import FEATURES, LABEL_COL

def load_and_prepare(csv_path: str):
    import pandas as pd

    # Skip NASA’s commented prologue; use the first non-comment line as header
    df = pd.read_csv(
        csv_path,
        comment="#",        # <-- key line
        low_memory=False,   # avoids mixed dtypes warnings
    )

    FEATURES = [
        "koi_period","koi_duration","koi_model_snr",
        "koi_prad","koi_srad","koi_steff","koi_slogg",
        "koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec"
    ]
    LABEL_COL = "koi_disposition"

    # keep only rows with known disposition
    keep = df[LABEL_COL].isin(["CONFIRMED","CANDIDATE","FALSE POSITIVE"])
    df = df[keep].copy()
    y = (df[LABEL_COL].isin(["CONFIRMED","CANDIDATE"])).astype(int)

    X = df.reindex(columns=FEATURES).copy()
    for c in ["koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec"]:
        if c in X.columns:
            X[c] = X[c].fillna(0).astype(int)
        else:
            X[c] = 0
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y


def train_model(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])

    clf = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
        n_jobs=4, random_state=42
    )

    pipe = Pipeline([("num", num_pipe), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    print(classification_report(yte, pred, digits=3))
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to KOI CSV")
    ap.add_argument("--out", default="backend/ml/model.pkl", help="Where to write model.pkl")
    ap.add_argument("--schema", default="backend/ml/schema.json", help="Where to write feature schema")
    args = ap.parse_args()

    X, y = load_and_prepare(args.csv)
    model = train_model(X, y)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(model, args.out)
    with open(args.schema, "w") as f:
        json.dump({"features": FEATURES}, f)

    print(f"✅ Saved model to {args.out}")
    print(f"✅ Saved schema to {args.schema}")

if __name__ == "__main__":
    main()
