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
'''

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve
)
from sklearn.impute import SimpleImputer
import joblib

# -----------------------------
# Physics helpers (Transit + RV)
# -----------------------------
G_CGS = 6.67430e-8        # cm^3 g^-1 s^-2
R_SUN_CM = 6.957e10       # cm
R_EARTH_CM = 6.371e8      # cm
M_SUN_G = 1.98847e33      # g
SECONDS_PER_DAY = 86400.0
RE_PER_RSUN = R_SUN_CM / R_EARTH_CM  # ~109.2


def estimate_stellar_mass_from_logg_radius(logg_cgs: pd.Series, r_star_rsun: pd.Series) -> pd.Series:
    """M = g R^2 / G. logg is cgs, R in R_sun. Return mass in M_sun with safe NaN handling."""
    g = (10 ** logg_cgs.astype(float))
    R_cm = r_star_rsun.astype(float) * R_SUN_CM
    M_g = g * (R_cm ** 2) / G_CGS
    return (M_g / M_SUN_G).replace([np.inf, -np.inf], np.nan)


def estimate_rho_star_from_logg_radius(logg_cgs: pd.Series, r_star_rsun: pd.Series) -> pd.Series:
    """rho = M / (4/3*pi*R^3) with M from gR^2/G. Returns g/cm^3."""
    g = (10 ** logg_cgs.astype(float))
    R_cm = r_star_rsun.astype(float) * R_SUN_CM
    rho = g / ((4.0 / 3.0) * np.pi * G_CGS * R_cm)
    return rho.replace([np.inf, -np.inf], np.nan)


def a_over_rstar_from_density(period_days: pd.Series, rho_star_cgs: pd.Series) -> pd.Series:
    """From rho* ≈ (3π)/(G P^2) (a/R*)^3 => a/R* = [G rho* P^2 / (3π)]^(1/3)."""
    P = period_days.astype(float) * SECONDS_PER_DAY
    term = (G_CGS * rho_star_cgs.astype(float) * (P ** 2)) / (3.0 * np.pi)
    with np.errstate(invalid="ignore"):
        a_over_r = np.power(term.clip(lower=0), 1.0 / 3.0)
    return a_over_r.replace([np.inf, -np.inf], np.nan)


def simple_mass_radius_proxy(rp_re: pd.Series) -> pd.Series:
    """Very rough Mp proxy in Earth masses (piecewise). Only for relative RV features."""
    rp = rp_re.astype(float)
    mp = np.where(rp <= 1.5, rp ** 3.0,  # rocky
                  np.where(rp <= 4.0, 2.69 * (rp ** 0.93),  # sub-Neptune
                           17.0 * (rp / 4.0) ** 2.0))  # Neptune+
    return pd.Series(mp, index=rp.index)


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Transit + RV-inspired features from KOI-style columns."""
    out = pd.DataFrame(index=df.index)

    # Base columns (safe get)
    P_days = df.get("koi_period")
    T_hours = df.get("koi_duration")
    depth_ppm = df.get("koi_depth")
    rp_re = df.get("koi_prad")
    rs_rsun = df.get("koi_srad")
    logg = df.get("koi_slogg")
    impact = df.get("koi_impact")
    snr = df.get("koi_model_snr")
    kepmag = df.get("koi_kepmag")

    # Transit geometry: Rp/Rs and predicted depth
    if rp_re is not None and rs_rsun is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            rprs = (rp_re.astype(float) / (rs_rsun.astype(float) * RE_PER_RSUN))
        out["feat_rprs"] = rprs
        out["feat_depth_pred_ppm"] = 1e6 * np.square(rprs)
    if depth_ppm is not None and "feat_depth_pred_ppm" in out:
        out["feat_depth_residual_ppm"] = depth_ppm.astype(float) - out["feat_depth_pred_ppm"]
        out["feat_depth_ratio"] = depth_ppm.astype(float) / (out["feat_depth_pred_ppm"].replace(0, np.nan))

    # Stellar density and a/R*
    if (logg is not None) and (rs_rsun is not None):
        rho_star = estimate_rho_star_from_logg_radius(logg, rs_rsun)
        out["feat_rho_star_cgs"] = rho_star
        if P_days is not None:
            a_over_r = a_over_rstar_from_density(P_days, rho_star)
            out["feat_a_over_rstar"] = a_over_r

    # Duration consistency (approx formula; assumes near-circular, sin i ~ 1)
    if (P_days is not None) and (impact is not None) and ("feat_a_over_rstar" in out):
        k = out.get("feat_rprs")
        b = impact.astype(float)
        factor = np.sqrt(np.clip(((1.0 + (k.fillna(0))) ** 2) - (b ** 2), a_min=0, a_max=None))
        dur_pred_days = (P_days.astype(float) / np.pi) * (1.0 / out["feat_a_over_rstar"]) * factor
        dur_pred_hours = dur_pred_days * 24.0
        out["feat_duration_pred_hr"] = dur_pred_hours
        if T_hours is not None:
            out["feat_duration_ratio"] = T_hours.astype(float) / dur_pred_hours
            out["feat_duration_residual_hr"] = T_hours.astype(float) - dur_pred_hours

    # Photometric SNR + brightness
    if snr is not None:
        out["feat_model_snr"] = snr.astype(float)
    if kepmag is not None:
        out["feat_kepmag"] = kepmag.astype(float)

    # KOI vetting / FP flags if present
    for col in ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]:
        if col in df.columns:
            out[f"feat_{col}"] = df[col].astype(float)

    # RV-inspired proxies: mass-radius -> K proxy
    if (rp_re is not None) and (P_days is not None) and (logg is not None) and (rs_rsun is not None):
        ms_msun = estimate_stellar_mass_from_logg_radius(logg, rs_rsun)
        mp_re = simple_mass_radius_proxy(rp_re)  # in Earth masses (proxy)
        # K ∝ (Mp) * P^{-1/3} * M_*^{-2/3} (ignoring constants and sin i, (1-e^2)^-1/2)
        with np.errstate(invalid="ignore", divide="ignore"):
            K_proxy = (mp_re.astype(float)) * (np.power(P_days.astype(float), -1.0/3.0)) * (np.power(ms_msun.astype(float), -2.0/3.0))
        out["feat_rv_K_proxy"] = K_proxy

    return out


# -----------------------------
# Core pipeline
# -----------------------------
DEFAULT_BASE_FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_steff", "koi_slogg", "koi_srad", "koi_teq", "koi_insol",
    "koi_impact", "koi_model_snr", "koi_kepmag",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
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


def select_available_features(df: pd.DataFrame, base_features=None) -> list[str]:
    feats = [f for f in (base_features or DEFAULT_BASE_FEATURES) if f in df.columns]
    if not feats:
        raise ValueError(
            "None of the expected feature columns found.\n"
            f"Expected any of: {DEFAULT_BASE_FEATURES}\nGot columns like: {list(df.columns)[:30]}"
        )
    return feats


def preprocess_data(df: pd.DataFrame, positive_mode: str):
    y = make_labels(df, positive_mode)
    df = df.loc[~y.isna()].copy()
    y = y.loc[df.index].astype(int)

    # Physics features
    physics = engineer_physics_features(df)

    # Base feature set (whatever is present)
    base_feats = select_available_features(df)
    X_base = df[base_feats].copy()

    # Build combined feature matrix (base + physics)
    X = pd.concat([X_base, physics], axis=1)

    # Sanitize before imputation
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.clip(lower=-1e12, upper=1e12)

    bad_counts = np.isinf(X).sum().sum() + np.isnan(X).sum().sum()
    if bad_counts:
        cols_inf = [c for c in X.columns if np.isinf(X[c]).any()]
        cols_nan = [c for c in X.columns if X[c].isna().any()]
        print(f"[sanitize] Replaced non-finite values. inf-cols={cols_inf} | nan-cols(sample)={cols_nan[:10]}")

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    return X_imputed, y.values, list(X.columns), imputer, df


def group_split_data(X, y, groups, train_size=0.6, val_size=0.2, random_state=42):
    """Group-aware split so that all KOIs from the same star (kepid) stay together."""
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(X, y, groups=groups))
    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    groups_temp = groups[temp_idx]

    val_rel = val_size / (1 - train_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_rel, random_state=random_state)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))

    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data_train_val_test(X, y, train_size=0.6, val_size=0.2, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    val_relative_size = val_size / (1 - 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_models_with_validation(X_train, y_train, X_val, y_val, random_state=42):
    """Train RF and HGB; pick best AP on validation. Calibrate afterward."""
    candidates = []

    for n in [200, 400]:
        for d in [None, 10]:
            rf = RandomForestClassifier(
                n_estimators=n, max_depth=d, class_weight="balanced",
                random_state=random_state, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            proba = rf.predict_proba(X_val)[:, 1]
            ap = average_precision_score(y_val, proba)
            candidates.append((ap, f"rf(n={n}, d={d})", rf))

    for lr in [0.05, 0.1]:
        for md in [None, 8]:
            hgb = HistGradientBoostingClassifier(
                learning_rate=lr, max_depth=md, max_iter=400,
                l2_regularization=1.0,
                class_weight={0: 1, 1: (sum(y_train == 0) / max(sum(y_train == 1), 1))}
            )
            hgb.fit(X_train, y_train)
            proba = hgb.predict_proba(X_val)[:, 1]
            ap = average_precision_score(y_val, proba)
            candidates.append((ap, f"hgb(lr={lr}, md={md})", hgb))

    best_ap, best_name, best_model = max(candidates, key=lambda x: x[0])

    calibrator = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
    calibrator.fit(X_val, y_val)

    print(f"Selected {best_name} with AP on validation: {best_ap:.4f}")
    return calibrator, best_name, best_ap


def choose_threshold(proba_val, y_val, target_precision=None):
    p, r, t = precision_recall_curve(y_val, proba_val)
    t = np.r_[t, 1.0]  # align lengths

    if target_precision is not None:
        ok = np.where(p >= target_precision)[0]
        if len(ok) > 0:
            idx = ok[np.argmax(r[ok])]
            return float(t[idx]), float(p[idx]), float(r[idx])

    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0)
    idx = int(np.argmax(f1))
    return float(t[idx]), float(p[idx]), float(r[idx])


def evaluate_model(clf, X_test, y_test, threshold=0.5):
    proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {ap:.4f} | thr={threshold:.3f}")

    return {"report": report, "cm": cm, "roc": roc, "ap": ap, "threshold": threshold}


def save_artifacts(clf, imputer, feats, df_raw, out_dir: Path, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature importances for tree-based models where available
    try:
        importances = getattr(clf.base_estimator, "feature_importances_", None)
    except Exception:
        importances = None
    if importances is None and hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_

    if importances is not None:
        fi = pd.DataFrame({"feature": feats, "importance": importances}).sort_values("importance", ascending=False)
        fi.to_csv(out_dir / "feature_importance.csv", index=False)

    # Rebuild features exactly like training and in the same order
    base_feats = [f for f in DEFAULT_BASE_FEATURES if f in df_raw.columns]
    physics = engineer_physics_features(df_raw)
    X_full = pd.concat([df_raw[base_feats], physics], axis=1)
    X_full = X_full.reindex(columns=feats)

    # Sanitize
    X_full = X_full.apply(pd.to_numeric, errors="coerce")
    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_full = X_full.clip(lower=-1e12, upper=1e12)

    X_all_imp = imputer.transform(X_full)
    proba_all = clf.predict_proba(X_all_imp)[:, 1]

    id_col = "kepoi_name" if "kepoi_name" in df_raw.columns else ("kepid" if "kepid" in df_raw.columns else None)
    preds = pd.DataFrame({
        "object_id": df_raw[id_col] if id_col else pd.RangeIndex(len(df_raw)),
        "koi_disposition": df_raw.get("koi_disposition"),
        "prob_exoplanet": proba_all,
    })
    preds.sort_values("prob_exoplanet", ascending=False).to_csv(out_dir / "predictions.csv", index=False)

    joblib.dump({
        "model": clf,
        "imputer": imputer,
        "features": feats,
        "meta": meta,
    }, out_dir / "exoplanet_model_transit_rv.joblib")


def run_pipeline(file_path: str, positive_mode: str, out_dir: str = "outputs", target_precision: float | None = None, group_by_star: bool = True, random_state: int = 42):
    df = load_koi_data(file_path)
    X, y, feats, imputer, df_kept = preprocess_data(df, positive_mode)

    if group_by_star and ("kepid" in df_kept.columns):
        groups = df_kept["kepid"].values
        X_train, X_val, X_test, y_train, y_val, y_test = group_split_data(X, y, groups)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_train_val_test(X, y)

    clf, model_name, ap_val = train_models_with_validation(X_train, y_train, X_val, y_val, random_state=random_state)

    proba_val = clf.predict_proba(X_val)[:, 1]
    thr, prec_at_thr, rec_at_thr = choose_threshold(proba_val, y_val, target_precision=target_precision)
    print(f"Chosen threshold: {thr:.3f} (Precision {prec_at_thr:.3f}, Recall {rec_at_thr:.3f} on validation)")

    _ = evaluate_model(clf, X_test, y_test, threshold=thr)

    meta = {
        "model_name": model_name,
        "ap_val": ap_val,
        "threshold": thr,
        "precision_val": prec_at_thr,
        "recall_val": rec_at_thr,
        "positive_mode": positive_mode,
        "group_by_star": group_by_star,
    }
    save_artifacts(clf, imputer, feats, df_kept, Path(out_dir), meta)
    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="./data/cumulative_2025.10.04_01.59.40.csv", help="Path to KOI cumulative CSV")
    parser.add_argument("--positive", choices=["confirmed+candidate", "confirmed"], default="confirmed+candidate",
                        help="Which dispositions count as positive (1).")
    parser.add_argument("--out", default="outputs", help="Directory to save results")
    parser.add_argument("--target_precision", type=float, default=None, help="Pick threshold to achieve >= this precision on validation")
    parser.add_argument("--no_group_by_star", action="store_true", help="Disable group-aware splitting by kepid")
    args = parser.parse_args()

    run_pipeline(args.csv, args.positive, args.out, target_precision=args.target_precision, group_by_star=not args.no_group_by_star)
