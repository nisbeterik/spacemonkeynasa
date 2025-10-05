# backend/urls.py
from django.urls import path
from mlapi.views import (index, health, predict, predict_csv,
                         feature_importances, train_csv, check_exo_csv,
                         check_exo_status_csv, koi_status, evaluate_pair_csv)

urlpatterns = [
    path("", index),
    path("api/health", health),
    path("api/predict", predict),
    path("api/predict_csv", predict_csv),
    path("api/feature_importances", feature_importances),
    path("api/train_csv", train_csv),
    path("api/check_exo_csv", check_exo_csv),
    path("api/check_exo_status_csv", check_exo_status_csv),
    path("api/koi_status", koi_status),
    path("api/evaluate_pair_csv", evaluate_pair_csv),
]
