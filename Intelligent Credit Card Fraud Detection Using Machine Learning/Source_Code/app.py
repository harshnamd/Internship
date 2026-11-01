"""Flask app: single + batch fraud prediction and auto-generated HTML/TXT reports."""
import io
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, Response, abort, render_template, request, send_from_directory

from features import FEATURES, RAW_COLS, V_COLS, build_features

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
MODEL_DIR, OUT_DIR = "models", "outputs"
if not os.path.exists(f"{MODEL_DIR}/meta.json"):
    raise SystemExit("Models not found. Run `python train.py` first.")

scaler = joblib.load(f"{MODEL_DIR}/robust_scaler.pkl")
MODELS = {"XGBoost": joblib.load(f"{MODEL_DIR}/xgboost.pkl"),
          "Random Forest": joblib.load(f"{MODEL_DIR}/random_forest.pkl"),
          "Logistic Regression": joblib.load(f"{MODEL_DIR}/logistic_regression.pkl")}
META = json.load(open(f"{MODEL_DIR}/meta.json"))
METRICS = json.load(open(f"{OUT_DIR}/metrics.json"))
BEST = META["best_model"]
POLICIES = {
    "privacy-policy": {
        "title": "Privacy Policy",
        "sections": [
            {
                "heading": "Information this app processes",
                "paragraphs": [
                    "When you use a prediction form, the transaction values you submit are sent to this app to calculate a result. For batch prediction, the app reads the CSV you upload and processes the required transaction columns. Do not submit payment-card numbers, names, account details, or other information that this app does not need.",
                    "The app uses submitted data to produce and display fraud predictions and, for batch uploads, to provide a downloadable results CSV. The app code does not configure advertising, analytics, or user profiling.",
                ],
            },
            {
                "heading": "Storage and retention",
                "paragraphs": [
                    "The app does not save submitted transactions to a database or permanent file. Single predictions are processed for the request. The latest batch results are kept in the running app's memory so they can be downloaded; a later batch replaces them, and restarting the app clears them.",
                    "The batch results are shared by the current app instance, rather than isolated by user. Anyone with access to that instance's batch-download page may be able to download its latest results. Do not use this build with sensitive real-world transaction data or expose it publicly without adding appropriate access controls and per-user isolation.",
                    "The app itself does not provide user accounts or a self-service data deletion feature. If you are using a hosted copy, its operator or hosting provider may retain connection logs or other operational data under their own practices.",
                ],
            },
            {
                "heading": "Service providers and security",
                "paragraphs": [
                    "No third-party analytics or advertising SDK is configured in this app. The app's Python libraries perform prediction and web serving. If the app is hosted by another provider, that provider may process network traffic and operational logs; review the provider's privacy terms.",
                    "The app's deployment determines whether browser-to-server traffic is encrypted. Use a trusted, HTTPS-enabled deployment for any non-public data. This prototype is not designed to handle sensitive cardholder information.",
                ],
            },
            {
                "heading": "Your choices and contact",
                "paragraphs": [
                    "You can choose not to submit transaction data. Because this build has no user accounts or persistent transaction database, it has no account-based deletion process. Batch results remain in memory until replaced or the app restarts.",
                    "This project does not identify a business operator or provide a privacy contact address. Before public or commercial use, the operator should add its legal name and a working privacy contact, and update this policy to describe the actual hosting and retention setup.",
                ],
            },
        ],
    },
    "terms-of-service": {
        "title": "Terms of Service",
        "sections": [
            {
                "heading": "Purpose and acceptable use",
                "paragraphs": [
                    "This credit-card fraud detection app is provided as an educational and demonstration tool. Use it only with data you are authorized to submit and for lawful purposes. Do not submit payment-card numbers or other sensitive personal information.",
                    "You are responsible for your inputs and for independently assessing any result before acting on it. Do not use the app as the sole basis for denying transactions, accounts, services, credit, or other consequential decisions.",
                ],
            },
            {
                "heading": "Predictions and availability",
                "paragraphs": [
                    "Predictions are generated by statistical models and can be incorrect, incomplete, or out of date. A prediction or probability is not a guarantee, professional advice, or a determination that a transaction is fraudulent. The app is supplied without a promise of accuracy, uninterrupted availability, or suitability for a particular purpose.",
                    "The app may be changed or unavailable without notice. The operator should obtain appropriate legal advice and publish any additional terms required for the deployment in which the app is used.",
                ],
            },
            {
                "heading": "Operator information",
                "paragraphs": [
                    "No business operator, registered address, or support contact is configured in this project. Those details and any applicable governing-law terms must be supplied by the operator before offering the app publicly or commercially.",
                ],
            },
        ],
    },
    "refund-policy": {
        "title": "Refund Policy",
        "sections": [
            {
                "heading": "No in-app purchases",
                "paragraphs": [
                    "This app does not offer paid products, subscriptions, or payment processing. It does not collect fees, so there are no in-app purchases to refund.",
                    "If you paid a third party to access a hosted or bundled version, contact that seller or service provider about its charges and refund terms. This app's operator cannot promise or process refunds for third-party transactions.",
                ],
            },
            {
                "heading": "Commercial use",
                "paragraphs": [
                    "If paid features or services are added later, the responsible operator must publish clear pricing, billing, cancellation, and refund terms before accepting payment. Nothing on this page overrides rights that cannot be excluded under applicable law.",
                ],
            },
        ],
    },
    "cookies-policy": {
        "title": "Cookies Policy",
        "sections": [
            {
                "heading": "Cookies used by this app",
                "paragraphs": [
                    "This build does not intentionally set or use cookies for analytics, advertising, or personalization. It does not use Flask sessions. The cookie notice can be dismissed for the current page view; that choice is not saved in a cookie or other persistent browser storage.",
                    "A hosting provider, reverse proxy, or other infrastructure used to serve a deployed copy may set its own cookies or collect technical data. Those practices are controlled by that provider, not by this app.",
                ],
            },
            {
                "heading": "Managing cookies",
                "paragraphs": [
                    "You can manage or delete cookies through your browser settings. This app does not currently require optional cookies for its prediction forms. If non-essential cookies are introduced in a future version, the operator should update this policy and obtain any consent required before using them.",
                ],
            },
        ],
    },
}


def score(raw_df, model_name):
    X = build_features(raw_df, scaler)[FEATURES]
    p = MODELS[model_name].predict_proba(X)[:, 1]
    return (p >= 0.5).astype(int), p


def page(**kw):
    return render_template("index.html", raw_cols=RAW_COLS, models=list(MODELS), best=BEST,
                           samples=META["samples"], result=kw.pop("result", None),
                           values=kw.pop("values", {}), error=kw.pop("error", None))


@app.route("/")
def index():
    return page()


@app.route("/<policy_name>")
def policy(policy_name):
    if policy_name not in POLICIES:
        abort(404)
    return render_template("policy.html", policy=POLICIES[policy_name])


@app.route("/predict", methods=["POST"])
def predict_single():
    values = {c: request.form.get(c, "").strip() for c in RAW_COLS}
    model_name = request.form.get("model", BEST)
    try:
        row = {c: float(values[c]) for c in RAW_COLS}
        if row["Amount"] < 0 or row["Time"] < 0:
            raise ValueError("Time and Amount must be non-negative.")
        pred, p = score(pd.DataFrame([row]), model_name)
        result = {"model": model_name, "label": int(pred[0]), "prob": round(float(p[0]) * 100, 2)}
        return page(result=result, values=values)
    except (ValueError, KeyError) as e:
        return page(error=f"Invalid input: {e}", values=values)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    ctx = dict(models=list(MODELS), best=BEST, table=None, error=None, n=0, n_fraud=0, model_used=None)
    if request.method == "GET":
        return render_template("batch.html", **ctx)
    try:
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith(".csv"):
            raise ValueError("Please upload a .csv file.")
        df = pd.read_csv(f)
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        df = df.dropna(subset=RAW_COLS).reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid rows found.")
        model_name = request.form.get("model", BEST)
        pred, p = score(df, model_name)
        out = df.copy()
        out["PREDICTION"] = ["Fraudulent (1)" if x else "Legitimate (0)" for x in pred]
        out["FRAUD_PROBABILITY_%"] = (p * 100).round(2)
        app.config["LAST_BATCH"] = out
        show = out[["Time", "Amount", "PREDICTION", "FRAUD_PROBABILITY_%"]].head(100)
        ctx.update(table=show.to_html(classes="tbl", index=False, border=0), n=len(out),
                   n_fraud=int(pred.sum()), model_used=model_name)
    except Exception as e:
        ctx["error"] = str(e)
    return render_template("batch.html", **ctx)


@app.route("/batch/download")
def batch_download():
    df = app.config.get("LAST_BATCH")
    if df is None:
        return "No batch results yet.", 404
    buf = io.StringIO(); df.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=fraud_predictions.csv"})


@app.route("/report/<fmt>")
def report(fmt):
    r, cv = METRICS["results"], METRICS["cv_xgboost"]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if fmt == "txt":
        L = [f"CREDIT CARD FRAUD DETECTION REPORT ({stamp})", "=" * 55, f"Best model (F1): {BEST}",
             f"Train/Test: {METRICS['n_train']}/{METRICS['n_test']} | duplicates removed: {METRICS['duplicates_removed']}", ""]
        for n, v in r.items():
            L.append(f"{n}: P={v['Precision']:.4f} R={v['Recall']:.4f} F1={v['F1']:.4f} "
                     f"ROC-AUC={v['ROC_AUC']:.4f} Acc={v['Accuracy']:.4f}")
        L += ["", "XGBoost 5-fold CV (mean +/- std):"]
        L += [f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}" for k, v in cv.items()]
        return Response("\n".join(L), mimetype="text/plain",
                        headers={"Content-Disposition": "attachment; filename=fraud_report.txt"})
    if fmt == "html":
        figs = sorted(p for p in os.listdir(OUT_DIR) if p.endswith(".png"))
        return render_template("report.html", r=r, cv=cv, best=BEST, m=METRICS, figs=figs, stamp=stamp)
    return "Unknown format", 404


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUT_DIR, name)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
