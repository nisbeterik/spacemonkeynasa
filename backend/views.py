from django.http import JsonResponse
from exoplanet_pipeline import run_pipeline
import argparse

def ping(request):
    return JsonResponse({"message": "Hello, world!"})

def run_model(request):
    csv_path = "./data/cumulative_2025.10.04_01.59.40.csv"
    positive_mode = "confirmed+candidate"
    out_dir = "outputs"

    try:
        clf, metrics = run_pipeline(csv_path, positive_mode, out_dir)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({
        "status": "ok",
        "inputs": {"csv": csv_path, "positive": positive_mode, "out": out_dir},
        "metrics": {
            "roc_auc": float(metrics["roc"]),
            "pr_auc": float(metrics["ap"]),
            "confusion_matrix": metrics["cm"].tolist(),
            "classification_report": metrics["report"],
        },
    })
