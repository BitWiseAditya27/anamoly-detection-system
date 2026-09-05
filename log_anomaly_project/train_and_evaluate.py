"""
train_and_evaluate.py
End-to-end training pipeline:
  1. Load labeled synthetic logs (data/train_logs.csv)
  2. Parse + extract features
  3. Train & compare IsolationForest / LOF / OneClassSVM
  4. Print a metrics comparison table
  5. Save the best model + scaler + training stats (for explainability) to results/
"""

import csv
import json
from datetime import datetime
import numpy as np

from core.feature_extractor import extract_features
from core.model import train_and_evaluate, pick_best_model, save_artifacts


def load_labeled_csv(path):
    records = []
    labels = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            records.append({
                "line_no": i,
                "timestamp": ts,
                "level": row["level"],
                "message": row["message"],
                "raw": f'{row["timestamp"]} {row["level"]} {row["message"]}',
            })
            labels.append(int(row["label"]))
    return records, np.array(labels)


def main():
    print("Loading labeled synthetic dataset...")
    records, y_true = load_labeled_csv("data/train_logs.csv")
    print(f"  {len(records)} log lines "
          f"({np.sum(y_true == 1)} normal / {np.sum(y_true == -1)} anomalies)")

    print("Extracting features...")
    X, feature_names, _ = extract_features(records)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features: {feature_names}")

    print("\nTraining & evaluating models (labels used ONLY for scoring, not fitting)...")
    scaler, results = train_and_evaluate(X, y_true, feature_names)

    print("\n%-20s %-10s %-10s %-10s %-10s" % ("Model", "Precision", "Recall", "F1", "ROC-AUC"))
    print("-" * 65)
    metrics_report = {}
    for name, res in results.items():
        m = res["metrics"]
        auc_str = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "N/A"
        print("%-20s %-10.3f %-10.3f %-10.3f %-10s" % (
            name, m["precision"], m["recall"], m["f1"], auc_str))
        metrics_report[name] = m

    best_name = pick_best_model(results, metric="f1")
    print(f"\nBest model by F1 score: {best_name}")

    # Save training stats for explainability (z-score against training distribution)
    X_scaled = scaler.transform(X)
    train_mean = X_scaled.mean(axis=0)
    train_std = X_scaled.std(axis=0)

    save_artifacts(scaler, results, best_name, feature_names)
    np.savez("results/train_stats.npz", mean=train_mean, std=train_std)

    with open("results/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "metrics": metrics_report}, f, indent=2)

    print("\nSaved: results/best_model.joblib, results/scaler.joblib, "
          "results/meta.joblib, results/train_stats.npz, results/metrics.json")


if __name__ == "__main__":
    main()
