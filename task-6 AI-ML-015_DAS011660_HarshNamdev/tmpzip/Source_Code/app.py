"""LoanDefend Flask app: single + batch loan sanction prediction and HTML/TXT reports."""
import io
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, Response, render_template, request, send_from_directory

from features import CATEGORICAL_RAW, DEPENDENTS, RAW_COLS, prepare

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
MODEL_DIR, OUT_DIR = "models", "outputs"
if not os.path.exists(f"{MODEL_DIR}/meta.json"):
    raise SystemExit("Models not found. Run `python train.py` first.")

PIPES = {"Random Forest": joblib.load(f"{MODEL_DIR}/random_forest.pkl"), "Gradient Boosting": joblib.load(f"{MODEL_DIR}/gradient_boosting.pkl")}
ENC = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
META = json.load(open(f"{MODEL_DIR}/meta.json"))
METRICS = json.load(open(f"{OUT_DIR}/metrics.json"))
BEST, STATS = META["best_model"], META["impute_stats"]
CHOICES = {**META["cat_classes"], "Dependents": DEPENDENTS, "Credit_History": ["1", "0"]}


def decide(df, model):
    X = prepare(df, STATS, ENC)
    p = PIPES[model].predict_proba(X)[:, 1]
    return p


def page(**kw):
    return render_template("index.html", cols=RAW_COLS, choices=CHOICES, cats=list(CHOICES), models=list(PIPES), best=BEST,
                           samples=META["samples"], result=kw.get("result"), values=kw.get("values", {}), error=kw.get("error"))


@app.route("/")
def index():
    return page()


@app.route("/predict", methods=["POST"])
def predict_single():
    values = {c: request.form.get(c, "").strip() for c in RAW_COLS}
    model = request.form.get("model", BEST)
    try:
        row = {c: values[c] if c in CHOICES else float(values[c]) for c in RAW_COLS}
        if row["ApplicantIncome"] < 0 or row["CoapplicantIncome"] < 0:
            raise ValueError("Incomes cannot be negative.")
        if row["LoanAmount"] <= 0 or row["Loan_Amount_Term"] <= 0:
            raise ValueError("LoanAmount (in thousands) and Loan_Amount_Term (months) must be positive.")
        p = float(decide(pd.DataFrame([row]), model)[0])
        ok = p >= 0.5
        res = {"model": model, "approved": ok, "confidence": round((p if ok else 1 - p) * 100, 1), "p": round(p * 100, 1)}
        return page(result=res, values=values)
    except (ValueError, KeyError) as e:
        return page(error=f"Invalid input: {e}", values=values)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    ctx = dict(models=list(PIPES), best=BEST, cols=RAW_COLS, table=None, error=None, n=0, n_ok=0, model_used=None)
    if request.method == "GET":
        return render_template("batch.html", **ctx)
    try:
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith(".csv"):
            raise ValueError("Please upload a .csv file.")
        df = pd.read_csv(f)
        miss = [c for c in RAW_COLS if c not in df.columns]
        if miss:
            raise ValueError("Missing columns: " + ", ".join(miss))
        if df.empty:
            raise ValueError("The file has no rows.")
        model = request.form.get("model", BEST)
        p = decide(df, model)                                 # missing cells are imputed with the training medians/modes
        out = df.copy()
        out["DECISION"] = ["Loan Approved" if x >= .5 else "Not Approved" for x in p]
        out["CONFIDENCE_%"] = [round((x if x >= .5 else 1 - x) * 100, 1) for x in p]
        app.config["LAST_BATCH"] = out
        ctx.update(table=out.head(100).to_html(classes="tbl", index=False, border=0), n=len(out),
                   n_ok=int((p >= .5).sum()), model_used=model)
    except Exception as e:
        ctx["error"] = str(e)
    return render_template("batch.html", **ctx)


@app.route("/batch/download")
def batch_download():
    df = app.config.get("LAST_BATCH")
    if df is None:
        return "No batch results yet.", 404
    buf = io.StringIO(); df.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=loan_decisions.csv"})


@app.route("/report/<fmt>")
def report(fmt):
    r = METRICS["results"]; stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if fmt == "txt":
        L = [f"LOANDEFEND REPORT ({stamp})", "=" * 50, f"Best model (test accuracy): {BEST}",
             f"Rows: raw {METRICS['n_raw']} -> after outliers {METRICS['n_after_outliers']} | train/test {METRICS['n_train']}/{METRICS['n_test']} | features {METRICS['n_features']}", ""]
        for n, v in r.items():
            t, cv = v["test"], v["cv"]
            L += [f"{n}: Acc={t['Accuracy']:.4f} P={t['Precision']:.4f} R={t['Recall']:.4f} F1={t['F1']:.4f} AUC={t['ROC_AUC']:.4f} (train acc {v['train']['Accuracy']:.4f})",
                  f"   5-fold CV: acc {cv['accuracy']['mean']:.4f}+/-{cv['accuracy']['std']:.4f}, F1 {cv['f1']['mean']:.4f}+/-{cv['f1']['std']:.4f}, AUC {cv['roc_auc']['mean']:.4f}+/-{cv['roc_auc']['std']:.4f}", ""]
        for n, imp in META["importance"].items():
            L.append(f"Top features ({n}): " + ", ".join(f"{k}={v:.3f}" for k, v in list(imp.items())[:6]))
        return Response("\n".join(L), mimetype="text/plain", headers={"Content-Disposition": "attachment; filename=loandefend_report.txt"})
    if fmt == "html":
        figs = sorted(p for p in os.listdir(OUT_DIR) if p.endswith(".png"))
        return render_template("report.html", r=r, best=BEST, m=METRICS, figs=figs, stamp=stamp)
    return "Unknown format", 404


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUT_DIR, name)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
