import argparse
from exoplanet_pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="./data/cumulative_2025.10.04_01.59.40.csv",
        help="Path to KOI cumulative CSV"
    )
    parser.add_argument(
        "--positive",
        choices=["confirmed+candidate", "confirmed"],
        default="confirmed+candidate",
        help="Which dispositions count as positive (1)."
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Directory to save results"
    )
    args = parser.parse_args()

    clf, metrics = run_pipeline(args.csv, args.positive, args.out)

    print("Pipeline complete.")
    print("ROC-AUC:", metrics["roc"])
    print("PR-AUC:", metrics["ap"])
    print("Confusion Matrix:", metrics["cm"])
    print(metrics["report"])

