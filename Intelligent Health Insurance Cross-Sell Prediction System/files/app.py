"""Flask app: single + batch cross-sell prediction with risk tier, top-3 factors, HTML/TXT reports."""
import io
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request, send_from_directory

from features import add_features

app = Flask(__name__, template_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
MODEL_DIR, OUT_DIR = "models", "outputs"
if not os.path.exists(f"{MODEL_DIR}/meta.json"):
    raise SystemExit("Models not found. Run `python train.py` first.")

PIPES = {"Ridge": joblib.load(f"{MODEL_DIR}/ridge_pipeline.pkl"), "Lasso": joblib.load(f"{MODEL_DIR}/lasso_pipeline.pkl")}
BINS = json.load(open(f"{MODEL_DIR}/bins.json"))
META = json.load(open(f"{MODEL_DIR}/meta.json"))
METRICS = json.load(open(f"{OUT_DIR}/metrics.json"))
BEST, COLS, CODES = META["best_model"], META["input_cols"], META["include_codes"]
CATS = ["Gender", "Vehicle_Age", "Vehicle_Damage"]


def tier(p):
    # NOTE: scores come from a model trained on SMOTE-balanced data, so they are propensity SCORES, not true purchase rates.
    return "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"


def score(raw, model):
    pipe = PIPES[model]
    X = add_features(raw[COLS], BINS, CODES)
    p = pipe.predict_proba(X)[:, 1]
    xt = pipe.named_steps["prep"].transform(X)
    names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    contrib = xt * pipe.named_steps["clf"].coef_[0]
    top = []
    for row in contrib:
        idx = np.argsort(-np.abs(row))[:3]
        top.append([(names[i], float(row[i])) for i in idx])
    return p, top


def page(**kw):
    return render_template("index.html", cols=COLS, cats=CATS, choices=META["choices"], models=list(PIPES), best=BEST,
                           samples=META["samples"], result=kw.get("result"), values=kw.get("values", {}), error=kw.get("error"))


@app.route("/")
def index():
    return page()


@app.route("/predict", methods=["POST"])
def predict_single():
    values = {c: request.form.get(c, "").strip() for c in COLS}
    model = request.form.get("model", BEST)
    try:
        row = {c: (values[c] if c in CATS else float(values[c])) for c in COLS}
        if not 18 <= row["Age"] <= 100:
            raise ValueError("Age must be between 18 and 100.")
        if row["Annual_Premium"] <= 0 or row["Vintage"] < 0:
            raise ValueError("Annual_Premium must be positive and Vintage non-negative.")
        p, top = score(pd.DataFrame([row]), model)
        res = {"model": model, "score": round(float(p[0]) * 100, 1), "tier": tier(p[0]),
               "factors": [{"name": n, "effect": "raises" if v > 0 else "lowers", "val": round(v, 2)} for n, v in top[0]]}
        return page(result=res, values=values)
    except (ValueError, KeyError) as e:
        return page(error=f"Invalid input: {e}", values=values)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    ctx = dict(models=list(PIPES), best=BEST, cols=COLS, table=None, error=None, n=0, summary=None, model_used=None)
    if request.method == "GET":
        return render_template("batch.html", **ctx)
    try:
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith(".csv"):
            raise ValueError("Please upload a .csv file.")
        df = pd.read_csv(f)
        miss = [c for c in COLS if c not in df.columns]
        if miss:
            raise ValueError("Missing columns: " + ", ".join(miss))
        df = df.dropna(subset=COLS).reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid rows found.")
        model = request.form.get("model", BEST)
        p, top = score(df, model)
        out = df.copy()
        out["PURCHASE_SCORE_%"] = (p * 100).round(1)
        out["TIER"] = [tier(x) for x in p]
        out["TOP_FACTORS"] = ["; ".join(n for n, _ in t) for t in top]
        app.config["LAST_BATCH"] = out
        ctx.update(table=out.head(100).to_html(classes="tbl", index=False, border=0), n=len(out), model_used=model,
                   summary=out["TIER"].value_counts().to_dict())
    except Exception as e:
        ctx["error"] = str(e)
    return render_template("batch.html", **ctx)


@app.route("/batch/download")
def batch_download():
    df = app.config.get("LAST_BATCH")
    if df is None:
        return "No batch results yet.", 404
    buf = io.StringIO(); df.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=cross_sell_predictions.csv"})


@app.route("/report/<fmt>")
def report(fmt):
    r = METRICS["results"]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if fmt == "txt":
        L = [f"HEALTH INSURANCE CROSS-SELL REPORT ({stamp})", "=" * 55, f"Best model (AUC): {BEST}",
             f"Train/Test: {METRICS['n_train']}/{METRICS['n_test']} | outliers removed: {METRICS['outliers_removed']}", ""]
        for n, v in r.items():
            t = v["test_at_0.5"]; cv = v["cv"]
            L += [f"{n} (alpha={v['alpha']}, C={v['C']:.4g}) test@0.5: Acc={t['Accuracy']:.4f} P={t['Precision']:.4f} R={t['Recall']:.4f} F1={t['F1']:.4f} AUC={t['ROC_AUC']:.4f}",
                  f"   5-fold CV: AUC {cv['ROC_AUC']['mean']:.4f}+/-{cv['ROC_AUC']['std']:.4f}, F1 {cv['F1']['mean']:.4f}+/-{cv['F1']['std']:.4f}", ""]
        L.append("Top coefficients:")
        for n, c in META["coefficients"].items():
            L.append(f" {n}: " + ", ".join(f"{k}={v:.3f}" for k, v in list(c.items())[:8]))
        return Response("\n".join(L), mimetype="text/plain", headers={"Content-Disposition": "attachment; filename=cross_sell_report.txt"})
    if fmt == "html":
        figs = sorted(p for p in os.listdir(OUT_DIR) if p.endswith(".png"))
        return render_template("report.html", r=r, best=BEST, m=METRICS, figs=figs, stamp=stamp)
    return "Unknown format", 404


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUT_DIR, name)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
