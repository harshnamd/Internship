"""CHIPGUARD Flask app: upload a customer CSV -> batch predictions -> download CSV / HTML report.
Run:  python app.py   then open http://127.0.0.1:5000
"""
import io
import os
import uuid
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, abort, render_template, request, send_file

from features import FEATURE_COLUMNS, RegressionClassifier, preprocess  # noqa: F401 (needed for unpickling)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    "ridge": joblib.load("models/ridge_pipeline.joblib"),
    "lasso": joblib.load("models/lasso_pipeline.joblib"),
}
BEST = joblib.load("models/model_info.joblib")["best_model"].lower()
HIGH_PROB = 0.70


def score_dataframe(raw: pd.DataFrame, model_key: str) -> pd.DataFrame:
    df = preprocess(raw)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    pipe = MODELS[model_key]
    X = df[FEATURE_COLUMNS]
    prob = pipe.predict_proba(X)[:, 1]
    out = raw.copy()
    out["Loan_Probability"] = prob.round(4)
    out["Predicted_Loan"] = pipe.predict(X)
    out["Campaign_Priority"] = pd.cut(prob, [-0.01, 0.4, HIGH_PROB, 1.01],
                                      labels=["Low", "Medium", "High"])
    return out.sort_values("Loan_Probability", ascending=False)


def build_report_html(res: pd.DataFrame, model_key: str) -> str:
    n, pos = len(res), int(res["Predicted_Loan"].sum())
    high = res[res["Campaign_Priority"] == "High"]
    return (
        f"<html><head><meta charset='utf-8'><title>CHIPGUARD Report</title></head><body style='font-family:Arial'>"
        f"<h1>CHIPGUARD Campaign Report</h1><p>Generated {datetime.now():%Y-%m-%d %H:%M} | Model: {model_key.title()}</p>"
        f"<ul><li>Customers scored: {n}</li><li>Predicted loan acceptors: {pos} ({pos / n:.1%})</li>"
        f"<li>High-priority (probability &ge; {HIGH_PROB:.0%}): {len(high)}</li></ul>"
        f"<h2>Top 20 customers to target</h2>{res.head(20).to_html(index=False, border=1)}</body></html>"
    )


@app.route("/")
def index():
    return render_template("index.html", best=BEST)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    model_key = request.form.get("model", BEST)
    if not file or not file.filename.lower().endswith(".csv") or model_key not in MODELS:
        return render_template("index.html", best=BEST, error="Please upload a .csv file."), 400
    try:
        res = score_dataframe(pd.read_csv(file), model_key)
    except Exception as e:  # bad columns / unreadable file
        return render_template("index.html", best=BEST, error=str(e)), 400
    rid = uuid.uuid4().hex
    res.to_csv(f"{RESULTS_DIR}/{rid}.csv", index=False)
    with open(f"{RESULTS_DIR}/{rid}.html", "w", encoding="utf-8") as f:
        f.write(build_report_html(res, model_key))
    summary = {"total": len(res), "positives": int(res["Predicted_Loan"].sum()),
               "high": int((res["Campaign_Priority"] == "High").sum())}
    return render_template("result.html", rid=rid, model=model_key.title(), summary=summary,
                           rows=res.head(50).to_dict("records"), cols=list(res.columns))


@app.route("/download/<kind>/<rid>")
def download(kind, rid):
    if kind not in ("csv", "html") or not rid.isalnum():
        abort(404)
    path = f"{RESULTS_DIR}/{rid}.{kind}"
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"chipguard_{kind}_report.{kind}")


if __name__ == "__main__":
    app.run(debug=False)
