"""
main.py
Flask web application.

Flow:
  1. User uploads a .log/.txt file
  2. Server parses it, extracts the same features used in training
  3. The pre-trained best model (Isolation Forest, chosen by F1 during training)
     scores every line as normal or anomaly
  4. Results are shown in a table (anomalies highlighted), a summary chart,
     lightweight "why was this flagged" reasons, and a CSV download link
"""

import os
import io
import csv
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, render_template, send_file, redirect, url_for, flash

from core.parser import parse_log_text
from core.feature_extractor import extract_features
from core.model import load_artifacts, explain_row

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"

UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
RESULTS_PREFIX = "results/"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Load trained model artifacts once at startup
try:
    scaler, model, meta = load_artifacts(RESULTS_PREFIX)
    stats = np.load(RESULTS_PREFIX + "train_stats.npz")
    train_mean, train_std = stats["mean"], stats["std"]
    feature_names = meta["feature_names"]
    best_model_name = meta["best_name"]
    MODEL_READY = True
except FileNotFoundError:
    MODEL_READY = False
    best_model_name = None
    feature_names = []


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_ready=MODEL_READY, model_name=best_model_name)


@app.route("/analyze", methods=["POST"])
def analyze():
    if not MODEL_READY:
        flash("Model not trained yet. Run train_and_evaluate.py first.")
        return redirect(url_for("index"))

    file = request.files.get("logfile")
    if not file or file.filename == "":
        flash("Please choose a log file to upload.")
        return redirect(url_for("index"))

    text = file.read().decode("utf-8", errors="ignore")
    records = parse_log_text(text)

    if not records:
        flash("No valid log lines found in the uploaded file.")
        return redirect(url_for("index"))

    X, f_names, _ = extract_features(records)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)  # 1 normal, -1 anomaly

    if hasattr(model, "decision_function"):
        anomaly_score = -model.decision_function(X_scaled)
    else:
        anomaly_score = (preds == -1).astype(float)

    rows = []
    for i, rec in enumerate(records):
        is_anomaly = preds[i] == -1
        reasons = explain_row(X_scaled[i], f_names, train_mean, train_std) if is_anomaly else []
        rows.append({
            "line_no": rec["line_no"],
            "timestamp": rec["timestamp_raw"] or "N/A",
            "level": rec["level"],
            "message": rec["message"],
            "is_anomaly": bool(is_anomaly),
            "score": round(float(anomaly_score[i]), 3),
            "reasons": ", ".join(reasons),
        })

    total = len(rows)
    n_anomalies = sum(r["is_anomaly"] for r in rows)

    # Save a CSV export
    result_id = uuid.uuid4().hex[:8]
    csv_path = os.path.join(UPLOAD_DIR, f"results_{result_id}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "line_no", "timestamp", "level", "message", "is_anomaly", "score", "reasons"
        ])
        writer.writeheader()
        writer.writerows(rows)

    chart_path = _make_chart(rows, result_id)

    return render_template(
        "results.html",
        rows=rows,
        total=total,
        n_anomalies=n_anomalies,
        pct=round(100 * n_anomalies / total, 1) if total else 0,
        model_name=best_model_name,
        csv_filename=os.path.basename(csv_path),
        chart_filename=os.path.basename(chart_path),
    )


def _make_chart(rows, result_id):
    """Bar chart: anomaly count per log level, saved to static/ for display."""
    levels = ["INFO", "WARNING", "ERROR", "CRITICAL", "UNKNOWN"]
    normal_counts = {lvl: 0 for lvl in levels}
    anomaly_counts = {lvl: 0 for lvl in levels}

    for r in rows:
        lvl = r["level"] if r["level"] in levels else "UNKNOWN"
        if r["is_anomaly"]:
            anomaly_counts[lvl] += 1
        else:
            normal_counts[lvl] += 1

    x = np.arange(len(levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, [normal_counts[l] for l in levels], width, label="Normal", color="#4C72B0")
    ax.bar(x + width / 2, [anomaly_counts[l] for l in levels], width, label="Anomaly", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Count")
    ax.set_title("Log Lines by Level: Normal vs Anomaly")
    ax.legend()
    fig.tight_layout()

    chart_filename = f"chart_{result_id}.png"
    chart_path = os.path.join(STATIC_DIR, chart_filename)
    fig.savefig(chart_path)
    plt.close(fig)
    return chart_path


@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(UPLOAD_DIR, filename)
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
