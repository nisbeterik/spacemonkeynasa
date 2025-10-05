import sys, pandas as pd, joblib, json
FEATURES = [
    "koi_period","koi_duration","koi_model_snr",
    "koi_prad","koi_srad","koi_steff","koi_slogg",
    "koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec"
]
clf = joblib.load("backend/ml/model.pkl")
df = pd.read_csv(sys.argv[1], comment="#")
X = df.reindex(columns=FEATURES)
proba = clf.predict_proba(X)[:,1][:10]  # show first 10 to keep it short
print(json.dumps({"sample_probs": [float(p) for p in proba]}, indent=2))
