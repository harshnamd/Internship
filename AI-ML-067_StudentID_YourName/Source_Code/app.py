"""Flask app for wind speed prediction: single, batch and auto-generated reports."""
import io
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request, send_from_directory

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload cap

MODEL_DIR, OUT_DIR = "models", "outputs"
if not os.path.exists(f"{MODEL_DIR}/meta.json"):
    raise SystemExit("Models not found. Run `python train.py` first.")

scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
MODELS = {"Random Forest": joblib.load(f"{MODEL_DIR}/random_forest.pkl"),
          "Linear Regression": joblib.load(f"{MODEL_DIR}/linear_regression.pkl")}
META = json.load(open(f"{MODEL_DIR}/meta.json"))
METRICS = json.load(open(f"{OUT_DIR}/metrics.json"))
FEATURES = META["features"]
BEST = META["best_model"]
FORM_FIELDS = ["T.MAX", "T.MIN", "T.MIN.G", "RAIN", "YEAR", "MONTH", "DAY"]


def predict(df, model_name):
    X = scaler.transform(df[FEATURES])
    return np.clip(MODELS[model_name].predict(X), 0, None)  # wind speed cannot be negative


@app.route("/")
def index():
    return render_template("index.html", fields=FORM_FIELDS, best=BEST, models=list(MODELS), result=None,
                           values={}, error=None)


@app.route("/predict", methods=["POST"])
def predict_single():
    values = {f: request.form.get(f, "").strip() for f in FORM_FIELDS}
    model_name = request.form.get("model", BEST)
    try:
        row = {f: float(values[f]) for f in FORM_FIELDS}
        if not (1 <= row["MONTH"] <= 12 and 1 <= row["DAY"] <= 31):
            raise ValueError("MONTH must be 1-12 and DAY 1-31.")
        if row["RAIN"] < 0:
            raise ValueError("RAIN cannot be negative.")
        if row["T.MIN"] > row["T.MAX"]:
            raise ValueError("T.MIN cannot exceed T.MAX.")
        pred = float(predict(pd.DataFrame([row]), model_name)[0])
        result = {"model": model_name, "wind": round(pred, 2)}
        error = None
    except (ValueError, KeyError) as e:
        result, error = None, f"Invalid input: {e}"
    return render_template("index.html", fields=FORM_FIELDS, best=BEST, models=list(MODELS),
                           result=result, values=values, error=error)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", table=None, error=None, models=list(MODELS), best=BEST)
    f = request.files.get("file")
    model_name = request.form.get("model", BEST)
    try:
        if not f or not f.filename.lower().endswith(".csv"):
            raise ValueError("Please upload a .csv file.")
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        if "DATE" in df.columns and not {"YEAR", "MONTH", "DAY"} <= set(df.columns):
            d = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
            df["YEAR"], df["MONTH"], df["DAY"] = d.dt.year, d.dt.month, d.dt.day
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")
        df = df.dropna(subset=FEATURES)
        if df.empty:
            raise ValueError("No valid rows found.")
        df["PREDICTED_WIND"] = predict(df, model_name).round(2)
        app.config["LAST_BATCH"] = df
        table = df.head(100).to_html(classes="tbl", index=False, border=0)
        return render_template("batch.html", table=table, n=len(df), error=None,
                               models=list(MODELS), best=BEST, model_used=model_name)
    except Exception as e:
        return render_template("batch.html", table=None, error=str(e), models=list(MODELS), best=BEST)


@app.route("/batch/download")
def batch_download():
    df = app.config.get("LAST_BATCH")
    if df is None:
        return "No batch results yet.", 404
    buf = io.StringIO(); df.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=wind_predictions.csv"})


@app.route("/report/<fmt>")
def report(fmt):
    r = METRICS["results"]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if fmt == "txt":
        lines = [f"WIND SPEED PREDICTION REPORT ({stamp})", "=" * 50,
                 f"Best model: {BEST}", f"Train/Test samples: {METRICS['n_train']}/{METRICS['n_test']}", ""]
        for n, v in r.items():
            t = v["test"]
            lines += [n, f"  Test R2={t['R2']:.4f} MAE={t['MAE']:.3f} MSE={t['MSE']:.3f} RMSE={t['RMSE']:.3f}",
                      f"  Train R2={v['train']['R2']:.4f}", f"  5-fold CV R2={v['cv_r2_mean']:.4f} +/- {v['cv_r2_std']:.4f}", ""]
        lines.append("Feature importance (Random Forest):")
        lines += [f"  {k}: {v:.4f}" for k, v in META["feature_importance"].items()]
        return Response("\n".join(lines), mimetype="text/plain",
                        headers={"Content-Disposition": "attachment; filename=wind_report.txt"})
    if fmt == "html":
        figs = sorted(p for p in os.listdir(OUT_DIR) if p.endswith(".png"))
        return render_template("report.html", r=r, best=BEST, stamp=stamp, meta=META, m=METRICS, figs=figs)
    return "Unknown format", 404


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUT_DIR, name)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
