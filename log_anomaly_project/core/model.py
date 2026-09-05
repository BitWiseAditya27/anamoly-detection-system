"""
model.py
Trains and compares three unsupervised anomaly detection models:
  - Isolation Forest
  - Local Outlier Factor (novelty mode, so it can score new/unseen data)
  - One-Class SVM

All three follow the sklearn convention: predict() returns 1 for normal, -1 for anomaly.
This lets us fit on unlabeled data (as a real deployment would) and only use labels
at evaluation time to report precision/recall/F1 — the labels are NEVER used during fit().
"""

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

CONTAMINATION = 0.10  # expected fraction of anomalies; tune per-dataset

MODEL_DIR = "results"


def build_models(contamination=CONTAMINATION):
    return {
        "IsolationForest": IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42
        ),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=20, contamination=contamination, novelty=True
        ),
        "OneClassSVM": OneClassSVM(nu=contamination, kernel="rbf", gamma="scale"),
    }


def train_and_evaluate(X, y_true, feature_names):
    """
    X: raw feature matrix (n_samples, n_features)
    y_true: ground-truth labels, 1=normal, -1=anomaly (used ONLY for evaluation)
    Returns: dict of {model_name: {"model": fitted_model, "metrics": {...}, "scores": anomaly_scores}}
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = build_models()
    results = {}

    for name, model in models.items():
        model.fit(X_scaled)
        preds = model.predict(X_scaled)  # 1 normal, -1 anomaly

        # anomaly "score": higher = more anomalous, used for ranking / explainability
        if hasattr(model, "decision_function"):
            raw_score = model.decision_function(X_scaled)
            anomaly_score = -raw_score  # decision_function: higher = more normal, so flip it
        else:
            anomaly_score = (preds == -1).astype(float)

        metrics = {
            "precision": precision_score(y_true, preds, pos_label=-1, zero_division=0),
            "recall": recall_score(y_true, preds, pos_label=-1, zero_division=0),
            "f1": f1_score(y_true, preds, pos_label=-1, zero_division=0),
        }
        try:
            # roc_auc needs a continuous score and binary labels; flip sign convention
            metrics["roc_auc"] = roc_auc_score((y_true == -1).astype(int), anomaly_score)
        except ValueError:
            metrics["roc_auc"] = None

        results[name] = {
            "model": model,
            "metrics": metrics,
            "preds": preds,
            "scores": anomaly_score,
        }

    return scaler, results


def pick_best_model(results, metric="f1"):
    return max(results.items(), key=lambda kv: kv[1]["metrics"][metric])[0]


def save_artifacts(scaler, results, best_name, feature_names, path_prefix="results/"):
    joblib.dump(scaler, path_prefix + "scaler.joblib")
    joblib.dump(results[best_name]["model"], path_prefix + "best_model.joblib")
    joblib.dump({"best_name": best_name, "feature_names": feature_names},
                path_prefix + "meta.joblib")


def load_artifacts(path_prefix="results/"):
    scaler = joblib.load(path_prefix + "scaler.joblib")
    model = joblib.load(path_prefix + "best_model.joblib")
    meta = joblib.load(path_prefix + "meta.joblib")
    return scaler, model, meta


def explain_row(x_row, feature_names, train_mean, train_std, top_k=3):
    """
    Very lightweight explainability: for a given feature row, report which
    features deviate the most (in standard deviations) from the training mean.
    This gives a human-readable 'why was this flagged' without needing SHAP.
    """
    z = (x_row - train_mean) / np.where(train_std == 0, 1, train_std)
    idx_sorted = np.argsort(-np.abs(z))[:top_k]
    reasons = []
    for idx in idx_sorted:
        reasons.append(f"{feature_names[idx]} (z={z[idx]:+.1f})")
    return reasons
