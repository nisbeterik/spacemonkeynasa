import os
from uuid import uuid4
from pathlib import Path

from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from exoplanet_pipeline import run_pipeline
import argparse

def ping(request):
    return JsonResponse({"message": "Hello, world!"})

@csrf_exempt
@require_POST
def run_model(request):
    """
    POST /run/
      form-data:
        - file: <CSV file>          (required)
        - positive: confirmed+candidate | confirmed   (optional, default: confirmed+candidate)
    """
    # 1) Validate upload
    uploaded = request.FILES.get("file")
    if not uploaded:
        return JsonResponse({"error": "No file uploaded. Use form-data field named 'file'."}, status=400)

    if not uploaded.name.lower().endswith(".csv"):
        return JsonResponse({"error": "Only .csv files are supported."}, status=400)

    # 2) Where to store things
    media_root = Path(settings.MEDIA_ROOT)  # e.g., <project>/media
    uploads_dir = media_root / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid4().hex[:8]
    saved_csv_path = uploads_dir / f"{run_id}_{uploaded.name}"

    # 3) Save the uploaded file
    with open(saved_csv_path, "wb+") as dest:
        for chunk in uploaded.chunks():
            dest.write(chunk)

    # 4) Read optional params (or use defaults)
    positive_mode = request.POST.get("positive", "confirmed+candidate")
    # Put outputs under media so we can serve them
    out_dir = media_root / "pipeline_outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        clf, metrics = run_pipeline(
            file_path=str(saved_csv_path),
            positive_mode=positive_mode,
            out_dir=str(out_dir),
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    # 5) Build URLs (served via MEDIA_URL in dev)
    base_url = settings.MEDIA_URL.rstrip("/")  # e.g., /media
    outputs_rel = f"pipeline_outputs/{run_id}"
    response = {
        "status": "ok",
        "run_id": run_id,
        "inputs": {
            "uploaded_csv_path": f"{base_url}/uploads/{saved_csv_path.name}",
            "positive_mode": positive_mode,
        },
        "metrics": {
            "roc_auc": float(metrics["roc"]),
            "pr_auc": float(metrics["ap"]),
            "confusion_matrix": metrics["cm"].tolist(),
            "classification_report": metrics["report"],
        },
        "artifacts": {
            "dir": f"{base_url}/{outputs_rel}",
            "predictions_csv": f"{base_url}/{outputs_rel}/predictions.csv",
            "feature_importance_csv": f"{base_url}/{outputs_rel}/feature_importance.csv",
            "model_file": f"{base_url}/{outputs_rel}/rf_exoplanet_model.joblib",
        },
    }
    return JsonResponse(response, status=200)
