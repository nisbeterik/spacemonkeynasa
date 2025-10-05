# backend/mlapi/views.py

import io
import os
import json
import time
import logging
import tempfile
from typing import List, Optional
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# ---- import your ML helpers ----
from ml.infer import predict_df, get_model
from ml.features import FEATURES

log = logging.getLogger(__name__)

# ---- API tunables ----
THRESHOLD: float = 0.50
MAX_ROWS_JSON: int = 10_000
MAX_ROWS_CSV: int = 500_000

# ---------- TAP helpers ----------
def _escape_sql(s: str) -> str:
    return s.replace("'", "''")

def _tap_sync(sql: str, fmt: str = "json"):
    base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    url = f"{base}?query={urllib.parse.quote(sql)}&format={fmt}"
    with urllib.request.urlopen(url, timeout=45) as resp:
        data = resp.read()
    if fmt == "json":
        return json.loads(data.decode("utf-8"))
    return data.decode("utf-8")

def _fetch_koi_meta_by_kepoi_names(kepoi_names: List[str]) -> pd.DataFrame:
    """
    Batch-fetch KOI metadata for kepoi_name values.
    Returns cols: kepid, kepoi_name, kepler_name, koi_disposition, koi_pdisposition.
    """
    if not kepoi_names:
        return pd.DataFrame(columns=["kepid","kepoi_name","kepler_name","koi_disposition","koi_pdisposition"])

    out_frames = []
    BATCH = 200
    for i in range(0, len(kepoi_names), BATCH):
        chunk = kepoi_names[i:i+BATCH]
        vals = ",".join(f"'{_escape_sql(v)}'" for v in chunk)
        sql = (
            "select kepid,kepoi_name,kepler_name,koi_disposition,koi_pdisposition "
            f"from cumulative where kepoi_name IN ({vals})"
        )
        rows = _tap_sync(sql, fmt="json")
        out_frames.append(pd.DataFrame(rows))

    if not out_frames:
        return pd.DataFrame(columns=["kepid","kepoi_name","kepler_name","koi_disposition","koi_pdisposition"])

    df = pd.concat(out_frames, ignore_index=True)
    if "koi_pdisposition" not in df.columns:
        df["koi_pdisposition"] = None
    return df

# ---------- small helpers ----------
def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURES:
        if c not in df.columns:
            df[c] = None
    return df[FEATURES]

def _missing_features(rows: List[dict]) -> List[str]:
    return [c for c in FEATURES if all(c not in r for r in rows)]

# ---------- index / health ----------
@require_http_methods(["GET"])
def index(_request):
    return JsonResponse({
        "ok": True,
        "service": "SpaceMonkeyNASA backend",
        "endpoints": [
            "/api/health",
            "/api/predict",
            "/api/predict_csv",
            "/api/feature_importances",
            "/api/train_csv",
            "/api/check_exo_csv",
            "/api/koi_status",
        ],
    })

@require_http_methods(["GET"])
def health(_request):
    try:
        model = get_model("backend/ml/model.pkl")
        clf = getattr(model, "named_steps", {}).get("clf", None)
        return JsonResponse({
            "ok": True,
            "model_loaded": model is not None,
            "has_clf_step": clf is not None,
            "n_features": len(FEATURES),
        })
    except Exception as e:
        log.exception("health failed")
        return JsonResponse({"ok": False, "detail": str(e)}, status=500)

# ---------- predict (JSON rows) ----------
@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    """
    Body: {"rows":[{feature: value, ...}, ...]}
    Returns: predictions, labels and metadata.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON body"}, status=400)

    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return JsonResponse({"detail": "Provide non-empty 'rows' list"}, status=400)

    if len(rows) > MAX_ROWS_JSON:
        return JsonResponse({"detail": f"Too many rows (> {MAX_ROWS_JSON})"}, status=413)

    miss = _missing_features(rows)
    try:
        df = pd.DataFrame(rows)
        probs = predict_df(df)
        labels = [int(p >= THRESHOLD) for p in probs]
        return JsonResponse({
            "predictions": [float(p) for p in probs],
            "labels": labels,
            "threshold": THRESHOLD,
            "num_rows": len(rows),
            "missing_features": miss,
        })
    except Exception as e:
        log.exception("predict failed")
        return JsonResponse({"detail": str(e)}, status=400)

# ---------- predict CSV (returns CSV) ----------
@csrf_exempt
@require_http_methods(["POST"])
def predict_csv(request):
    """
    Upload: multipart/form-data with 'file' (CSV).
    Returns: CSV with planet_like_prob, planet_like_label.
    """
    f = request.FILES.get("file")
    if not f:
        return JsonResponse({"detail": "Upload a CSV in form field 'file'."}, status=400)

    try:
        df = pd.read_csv(f, comment="#", low_memory=False)
        if len(df) > MAX_ROWS_CSV:
            return JsonResponse({"detail": f"CSV too large (> {MAX_ROWS_CSV} rows)"}, status=413)

        model = get_model("backend/ml/model.pkl")
        X = _ensure_feature_columns(df.copy())
        probs = model.predict_proba(X)[:, 1]
        labels = (probs >= THRESHOLD).astype(int)

        out = df.copy()
        out["planet_like_prob"] = probs
        out["planet_like_label"] = labels

        buf = io.StringIO()
        out.to_csv(buf, index=False)
        resp = HttpResponse(buf.getvalue(), content_type="text/csv")
        resp["Content-Disposition"] = 'attachment; filename="predictions.csv"'
        return resp
    except Exception as e:
        log.exception("predict_csv failed")
        return JsonResponse({"detail": str(e)}, status=400)

# ---------- feature importances ----------
@require_http_methods(["GET"])
def feature_importances(_request):
    try:
        model = get_model("backend/ml/model.pkl")
        clf = getattr(model, "named_steps", {}).get("clf", model)

        importances = None
        source_attr = None
        if hasattr(clf, "feature_importances_"):
            importances = np.asarray(clf.feature_importances_).ravel()
            source_attr = "feature_importances_"
        elif hasattr(clf, "coef_"):
            importances = np.asarray(clf.coef_).ravel()
            source_attr = "coef_"

        if importances is None:
            return JsonResponse({"detail": "Model has no importances/coefs"}, status=400)

        items = sorted(
            [{"feature": f, "importance": float(v)} for f, v in zip(FEATURES, importances)],
            key=lambda x: abs(x["importance"]),
            reverse=True,
        )
        return JsonResponse({"source_attr": source_attr, "items": items})
    except Exception as e:
        log.exception("feature_importances failed")
        return JsonResponse({"detail": str(e)}, status=400)

# ---------- train from CSV ----------
@csrf_exempt
@require_http_methods(["POST"])
def train_csv(request):
    """
    Upload: 'file' (KOI CSV).
    Trains and saves backend/ml/model.pkl & backend/ml/schema.json.
    Returns metrics.
    """
    f = request.FILES.get("file")
    if not f:
        return JsonResponse({"detail": "Upload a CSV file in form field 'file'."}, status=400)

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        for chunk in f.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        raw = pd.read_csv(tmp_path, comment="#", low_memory=False)

        if "koi_disposition" not in raw.columns:
            return JsonResponse({"detail": "CSV must include 'koi_disposition'."}, status=400)

        y = raw["koi_disposition"].isin(["CONFIRMED", "CANDIDATE"]).astype(int)

        X = raw.copy()
        for c in FEATURES:
            if c not in X.columns:
                X[c] = None
        X = X[FEATURES].apply(pd.to_numeric, errors="coerce")

        from backend.ml.train import train_model
        model = train_model(X, y)

        # quick eval
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        pred = model.predict(Xte)
        rep = classification_report(yte, pred, output_dict=True, digits=3)

        os.makedirs("backend/ml", exist_ok=True)
        joblib.dump(model, "backend/ml/model.pkl")
        with open("backend/ml/schema.json", "w") as fh:
            json.dump({"features": FEATURES}, fh)

        return JsonResponse({
            "ok": True,
            "saved_model": "backend/ml/model.pkl",
            "saved_schema": "backend/ml/schema.json",
            "metrics": rep
        })
    except Exception as e:
        log.exception("train_csv failed")
        return JsonResponse({"detail": str(e)}, status=400)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ---------- check CSV (predict + KOI enrichment) ----------
@csrf_exempt
@require_http_methods(["POST"])
def check_exo_csv(request):
    """
    Upload CSV; returns predictions plus KOI metadata columns and a unified 'status':
      - CONFIRMED_EXOPLANET (KOI confirmed)
      - CANDIDATE (KOI candidate)
      - MODEL_POSITIVE (model predicts positive but KOI not confirmed/candidate)
      - NOT_EXOPLANET (otherwise)
    Output highlights:
      kepid, kepoi_name, kepler_name,
      exoplanet_archive_disposition, disposition_using_kepler_data,
      status, planet_like_prob, planet_like_label
    """
    f = request.FILES.get("file")
    if not f:
        return JsonResponse({"detail": "Upload a CSV file in form field 'file'."}, status=400)

    try:
        df_in = pd.read_csv(f, comment="#", low_memory=False)

        # Predict
        X = _ensure_feature_columns(df_in.copy())
        model = get_model("backend/ml/model.pkl")
        probs = model.predict_proba(X)[:, 1]
        labels = (probs >= THRESHOLD).astype(int)

        # Enrich with KOI metadata if a KOI name column exists
        kepoi_col_candidates = [c for c in df_in.columns if c.lower() in ("kepoi_name","koi_name","kepoi")]
        if kepoi_col_candidates:
            kepoi_col = kepoi_col_candidates[0]
            kepois = [str(x) for x in df_in[kepoi_col].dropna().unique().tolist()]
            meta = _fetch_koi_meta_by_kepoi_names(kepois)
            df_out = df_in.copy()
            if kepoi_col != "kepoi_name":
                df_out = df_out.rename(columns={kepoi_col: "kepoi_name"})
            merged = df_out.merge(meta, on="kepoi_name", how="left")
        else:
            merged = df_in.copy()
            for c in ["kepid","kepoi_name","kepler_name","koi_disposition","koi_pdisposition"]:
                if c not in merged.columns:
                    merged[c] = None

        # Predictions
        merged["planet_like_prob"] = probs
        merged["planet_like_label"] = labels

        # Rename dispositions to the names you asked for
        merged = merged.assign(
            exoplanet_archive_disposition = merged.get("koi_disposition"),
            disposition_using_kepler_data = merged.get("koi_pdisposition")
        )

        # ---- NEW: unified 'status' column ----
        disp = merged["exoplanet_archive_disposition"].fillna("")
        merged["status"] = np.where(
            disp.eq("CONFIRMED"), "CONFIRMED_EXOPLANET",
            np.where(
                disp.eq("CANDIDATE"), "CANDIDATE",
                np.where(merged["planet_like_label"].eq(1), "MODEL_POSITIVE", "NOT_EXOPLANET")
            )
        )

        # Put key columns first (includes new 'status')
        preferred = [
            "kepid","kepoi_name","kepler_name",
            "exoplanet_archive_disposition","disposition_using_kepler_data",
            "status",                          # <-- here
            "planet_like_prob","planet_like_label"
        ]
        front = [c for c in preferred if c in merged.columns]
        rest = [c for c in merged.columns if c not in front]
        merged = merged[front + rest]

        # Return as CSV
        buf = io.StringIO()
        merged.to_csv(buf, index=False)
        resp = HttpResponse(buf.getvalue(), content_type="text/csv")
        resp["Content-Disposition"] = 'attachment; filename="exo_check_results.csv"'
        return resp
    except Exception as e:
        log.exception("check_exo_csv failed")
        return JsonResponse({"detail": str(e)}, status=400)


# ---------- KOI / Kepler status lookup ----------
@csrf_exempt
@require_http_methods(["POST"])
def koi_status(request):
    """
    Body can be:
      {"koi":"K00010.01"} | {"kois":[...]} | {"kepler_name":"Kepler-10 b"} |
      {"kepler_names":[...]} | {"koi_like":"K00010"}
    Returns: {"count":N,"items":[{"kepoi_name","kepler_name","koi_disposition"}...]}
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON body"}, status=400)

    koi: Optional[str] = payload.get("koi")
    kois: Optional[List[str]] = payload.get("kois")
    kepler_name: Optional[str] = payload.get("kepler_name")
    kepler_names: Optional[List[str]] = payload.get("kepler_names")
    koi_like: Optional[str] = payload.get("koi_like")

    if not any([koi, kois, kepler_name, kepler_names, koi_like]):
        return JsonResponse({"detail": "Provide one of koi | kois | kepler_name | kepler_names | koi_like"}, status=400)

    if koi:
        where = f"kepoi_name='{_escape_sql(koi)}'"
    elif kois:
        vals = ",".join(f"'{_escape_sql(v)}'" for v in kois)
        where = f"kepoi_name IN ({vals})"
    elif kepler_name:
        where = f"kepler_name='{_escape_sql(kepler_name)}'"
    elif kepler_names:
        vals = ",".join(f"'{_escape_sql(v)}'" for v in kepler_names)
        where = f"kepler_name IN ({vals})"
    else:  # koi_like
        where = f"kepoi_name LIKE '{_escape_sql(koi_like)}%'"

    sql = "select kepoi_name, kepler_name, koi_disposition from cumulative where " + where
    try:
        rows = _tap_sync(sql, fmt="json")
        return JsonResponse({"count": len(rows), "items": rows})
    except Exception as e:
        log.exception("koi_status failed")
        return JsonResponse({"detail": str(e)}, status=400)
    
@csrf_exempt
@require_http_methods(["POST"])
def check_exo_status_csv(request):
    """
    Upload CSV (multipart 'file' with a KOI name column).
    Returns a minimal CSV with columns:
        kepoi_name, status
    where status is derived from NASA KOI disposition:
        CONFIRMED, CANDIDATE, FALSE_POSITIVE, UNKNOWN
    """
    f = request.FILES.get("file")
    if not f:
        return JsonResponse({"detail": "Upload a CSV file in form field 'file'."}, status=400)

    try:
        df_in = pd.read_csv(f, comment="#", low_memory=False)

        # Find a KOI name column
        koi_cols = [c for c in df_in.columns if c.lower() in ("kepoi_name","koi_name","kepoi")]
        if not koi_cols:
            return JsonResponse({"detail": "CSV must include a KOI name column (kepoi_name/koi_name/kepoi)."}, status=400)

        koi_col = koi_cols[0]
        # Normalize to 'kepoi_name' for merge
        df_norm = df_in.rename(columns={koi_col: "kepoi_name"})
        # Unique KOIs to look up
        kois = [str(x) for x in df_norm["kepoi_name"].dropna().unique().tolist()]

        # Pull disposition from NASA KOI table
        meta = _fetch_koi_meta_by_kepoi_names(kois)  # has kepoi_name, koi_disposition
        # Ensure column exists even if archive didn’t return a row
        if "koi_disposition" not in meta.columns:
            meta["koi_disposition"] = None

        # Merge back (left join keeps all input rows)
        merged = df_norm[["kepoi_name"]].merge(
            meta[["kepoi_name", "koi_disposition"]],
            on="kepoi_name",
            how="left"
        )

        # Map dispositions to status
        disp = merged["koi_disposition"].fillna("").str.upper()
        merged["status"] = np.select(
            [
                disp.eq("CONFIRMED"),
                disp.eq("CANDIDATE"),
                disp.eq("FALSE POSITIVE"),
            ],
            ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"],
            default="UNKNOWN"
        )

        # Output only the two requested columns
        out = merged[["kepoi_name", "status"]]

        buf = io.StringIO()
        out.to_csv(buf, index=False)
        resp = HttpResponse(buf.getvalue(), content_type="text/csv")
        resp["Content-Disposition"] = 'attachment; filename="koi_status.csv"'
        return resp

    except Exception as e:
        log.exception("check_exo_status_csv failed")
        return JsonResponse({"detail": str(e)}, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def evaluate_pair_csv(request):
    """
    Compare an actual labeled KOI CSV vs your exo_check_results.csv (predictions).
    Form-data fields:
      - actual: the original KOI CSV (must include koi_disposition)
      - pred:   exo_check_results.csv (must include planet_like_label, ideally kepoi_name)
    Returns JSON: compared_rows, correct, accuracy_percent, confusion matrix, and merge diagnostics.
    """
    f_actual = request.FILES.get("actual")
    f_pred   = request.FILES.get("pred")
    if not f_actual or not f_pred:
        return JsonResponse({"detail": "Upload two files: 'actual' and 'pred'."}, status=400)

    try:
        df_a = pd.read_csv(f_actual, comment="#", low_memory=False)
        df_p = pd.read_csv(f_pred,   comment="#", low_memory=False)

        # ---- choose a join key: prefer kepoi_name, else kepid ----
        def _pick_key(df):
            koi_cols = [c for c in df.columns if c.lower() in ("kepoi_name","koi_name","kepoi")]
            if koi_cols:
                return koi_cols[0], "kepoi_name"
            if "kepid" in df.columns:
                return "kepid", "kepid"
            return None, None

        a_key_src, a_key_norm = _pick_key(df_a)
        p_key_src, p_key_norm = _pick_key(df_p)
        if not a_key_src or not p_key_src:
            return JsonResponse({
                "detail": "Need a common key. Provide kepoi_name (preferred) or kepid in both files."
            }, status=400)

        # normalize key names for merge
        if a_key_src != a_key_norm:
            df_a = df_a.rename(columns={a_key_src: a_key_norm})
        if p_key_src != p_key_norm:
            df_p = df_p.rename(columns={p_key_src: p_key_norm})

        # ---- ground truth from 'actual' ----
        if "koi_disposition" not in df_a.columns:
            return JsonResponse({"detail": "Actual CSV must include 'koi_disposition'."}, status=400)
        disp = df_a["koi_disposition"].astype(str).str.upper()
        # accepted truth labels
        is_valid = disp.isin(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])
        df_a = df_a.loc[is_valid, [a_key_norm, "koi_disposition"]].copy()
        df_a["y_true"] = df_a["koi_disposition"].str.upper().isin(["CONFIRMED", "CANDIDATE"]).astype(int)

        # ---- predictions from 'pred' ----
        # Use planet_like_label if present; else threshold planet_like_prob
        if "planet_like_label" in df_p.columns:
            df_p["y_pred"] = pd.to_numeric(df_p["planet_like_label"], errors="coerce").fillna(-1).astype(int)
        elif "planet_like_prob" in df_p.columns:
            probs = pd.to_numeric(df_p["planet_like_prob"], errors="coerce")
            df_p["y_pred"] = (probs >= THRESHOLD).astype(int)
        else:
            return JsonResponse({
                "detail": "Pred CSV must include 'planet_like_label' or 'planet_like_prob'."
            }, status=400)

        df_p = df_p[[p_key_norm, "y_pred"]].copy()

        # ---- merge and evaluate ----
        merged = df_a.merge(df_p, left_on=a_key_norm, right_on=p_key_norm, how="inner")
        # If keys differ (kepoi_name vs kepid), we still merged via the normalized names; drop duplicate
        if a_key_norm != p_key_norm and p_key_norm in merged.columns:
            merged.drop(columns=[p_key_norm], inplace=True)

        merged = merged.dropna(subset=["y_true", "y_pred"])
        merged = merged[(merged["y_pred"] == 0) | (merged["y_pred"] == 1)]

        n = int(len(merged))
        correct = int((merged["y_true"] == merged["y_pred"]).sum())
        acc = float(correct / n) if n else 0.0

        # confusion matrix
        tp = int(((merged["y_pred"] == 1) & (merged["y_true"] == 1)).sum())
        tn = int(((merged["y_pred"] == 0) & (merged["y_true"] == 0)).sum())
        fp = int(((merged["y_pred"] == 1) & (merged["y_true"] == 0)).sum())
        fn = int(((merged["y_pred"] == 0) & (merged["y_true"] == 1)).sum())

        # diagnostics
        a_keys = set(df_a[a_key_norm].dropna().astype(str))
        p_keys = set(df_p[p_key_norm].dropna().astype(str))
        common_keys = len(a_keys & p_keys)

        return JsonResponse({
            "ok": True,
            "key_used": a_key_norm,
            "compared_rows": n,
            "correct": correct,
            "accuracy": acc,
            "accuracy_percent": round(acc * 100, 2),
            "threshold": THRESHOLD,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "diagnostics": {
                "actual_unique_keys": len(a_keys),
                "pred_unique_keys": len(p_keys),
                "common_keys": common_keys,
                "actual_only": len(a_keys - p_keys),
                "pred_only": len(p_keys - a_keys),
            }
        })
    except Exception as e:
        log.exception("evaluate_pair_csv failed")
        return JsonResponse({"detail": str(e)}, status=400)
