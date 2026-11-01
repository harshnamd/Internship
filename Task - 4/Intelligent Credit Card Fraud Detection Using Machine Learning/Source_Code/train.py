"""
Intelligent Credit Card Fraud Detection (AI-ML-008)
EDA + preprocessing + SMOTE + XGBoost / Random Forest / Logistic Regression + evaluation.

Usage:  python train.py [--data data/creditcard.csv] [--smote-ratio 0.1]
Outputs: models/*.pkl, models/meta.json, outputs/*.png, outputs/metrics.json, outputs/classification_reports.txt
"""
import argparse
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

from features import FEATURES, RAW_COLS, build_features

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
OUT_DIR, MODEL_DIR, SEED = "outputs", "models", 42
_n = 0


def save_fig(name):
    global _n
    _n += 1
    path = f"{OUT_DIR}/fig{_n:02d}_{name}.png"
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print("  saved", path)


# ----------------------------- EDA -----------------------------
def run_eda(df):
    print("Generating EDA figures...")
    counts = df["Class"].value_counts()
    pct = counts / counts.sum() * 100
    plt.figure(figsize=(6, 4.5))
    ax = sns.barplot(x=["Legitimate (0)", "Fraud (1)"], y=counts.values, palette=["steelblue", "tomato"])
    ax.set_yscale("log"); ax.set_ylabel("Transactions (log scale)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    plt.title("Target Distribution"); save_fig("target_bar")

    plt.figure(figsize=(5.5, 5.5))
    plt.pie(counts.values, labels=["Legitimate", "Fraud"], colors=["steelblue", "tomato"],
            autopct="%1.3f%%", startangle=90, explode=(0, 0.1))
    plt.title(f"Class Imbalance ({pct[1]:.3f}% fraud)"); save_fig("target_pie")

    plt.figure(figsize=(8, 4.5))
    for c, col in [(0, "steelblue"), (1, "tomato")]:
        sns.histplot(np.log1p(df.loc[df.Class == c, "Amount"]), stat="density", common_norm=False,
                     label=f"Class {c}", color=col, alpha=0.5, bins=50)
    plt.xlabel("log(1 + Amount)"); plt.legend(); plt.title("Amount Distribution by Class"); save_fig("amount_by_class")

    plt.figure(figsize=(8, 4.5))
    for c, col in [(0, "steelblue"), (1, "tomato")]:
        sns.histplot(df.loc[df.Class == c, "Time"] / 3600, stat="density", common_norm=False,
                     label=f"Class {c}", color=col, alpha=0.5, bins=48)
    plt.xlabel("Hours since first transaction"); plt.legend(); plt.title("Transaction Time by Class"); save_fig("time_by_class")

    top_v = ["V14", "V12", "V10", "V17"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for a, v in zip(axes.ravel(), top_v):
        for c, col in [(0, "steelblue"), (1, "tomato")]:
            sns.kdeplot(df.loc[df.Class == c, v], ax=a, label=f"Class {c}", color=col, fill=True, alpha=0.3)
        a.set_title(f"{v} by class"); a.legend()
    plt.suptitle("Key V-feature Distributions by Class"); save_fig("v_features_by_class")

    corr = df.corr()
    top30 = corr["Class"].abs().sort_values(ascending=False).head(30).index
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[top30].corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap (Top 30 features by |corr| with Class)"); save_fig("corr_heatmap")

    c = corr["Class"].drop("Class").sort_values()
    plt.figure(figsize=(8, 8))
    c.plot(kind="barh", color=np.where(c > 0, "tomato", "steelblue"))
    plt.title("Correlation of Each Feature with Class"); save_fig("corr_with_target")

    plt.figure(figsize=(6, 5))
    sns.boxplot(x="Class", y="Amount", data=df[df.Amount < df.Amount.quantile(0.99)], palette=["steelblue", "tomato"])
    plt.title("Amount by Class (below 99th percentile)"); save_fig("amount_box")

    hr = df.assign(H=(df.Time // 3600) % 24).groupby("H")["Class"].mean() * 100
    plt.figure(figsize=(8, 4.5)); hr.plot(marker="o", color="tomato")
    plt.xlabel("Hour of day"); plt.ylabel("Fraud rate (%)"); plt.title("Fraud Rate by Hour"); save_fig("fraud_rate_by_hour")


# ----------------------------- metrics -----------------------------
def evaluate(y, pred, proba):
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "ROC_AUC": roc_auc_score(y, proba),
            "PR_AUC": average_precision_score(y, proba)}


def main(path, smote_ratio):
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(path)
    print("Loaded:", df.shape, "| missing values:", int(df.isna().sum().sum()))
    n_dup = int(df.duplicated().sum())
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"Removed {n_dup} duplicate rows -> {df.shape}; fraud = {df.Class.sum()} ({df.Class.mean()*100:.3f}%)")
    run_eda(df)

    # split FIRST (stratified), then fit scaler on training data only, then SMOTE on training only
    tr, te = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=SEED)
    amount_scaler = RobustScaler().fit(tr[["Amount"]])
    X_tr, y_tr = build_features(tr, amount_scaler)[FEATURES], tr["Class"].values
    X_te, y_te = build_features(te, amount_scaler)[FEATURES], te["Class"].values
    print(f"Train {X_tr.shape} (fraud {y_tr.sum()}) | Test {X_te.shape} (fraud {y_te.sum()})")

    sm = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
    X_bal, y_bal = sm.fit_resample(X_tr, y_tr)
    neg, pos = int((y_bal == 0).sum()), int((y_bal == 1).sum())
    spw = neg / pos   # "auto": negatives/positives of the data XGBoost actually trains on
    print(f"After SMOTE: legit={neg}, fraud={pos} -> scale_pos_weight={spw:.2f}")

    def make_xgb():
        return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=spw,
                             eval_metric="logloss", random_state=SEED, n_jobs=-1, tree_method="hist")
    models = {
        "XGBoost": make_xgb(),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced",
                                                random_state=SEED, n_jobs=-1),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    }
    res, probas, cms, reports = {}, {}, {}, {}
    for name, m in models.items():
        print(f"Training {name}...")
        m.fit(X_bal, y_bal)
        proba = m.predict_proba(X_te)[:, 1]; pred = (proba >= 0.5).astype(int)
        res[name] = evaluate(y_te, pred, proba); probas[name] = proba
        cms[name] = confusion_matrix(y_te, pred)
        reports[name] = classification_report(y_te, pred, target_names=["Legitimate", "Fraud"], digits=4)
        print("  ", {k: round(v, 4) for k, v in res[name].items()})

    # 5-fold stratified CV for XGBoost; SMOTE sits INSIDE the pipeline so it is applied to training folds only
    print("Running 5-fold stratified CV for XGBoost (SMOTE inside each training fold)...")
    pipe = ImbPipeline([("smote", SMOTE(sampling_strategy=smote_ratio, random_state=SEED)), ("xgb", make_xgb())])
    cv = cross_validate(pipe, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        scoring=["precision", "recall", "f1", "roc_auc"], n_jobs=1)
    cv_out = {k: {"mean": float(cv[f"test_{k}"].mean()), "std": float(cv[f"test_{k}"].std()),
                  "folds": cv[f"test_{k}"].tolist()} for k in ["precision", "recall", "f1", "roc_auc"]}
    print("  CV F1: %.4f +/- %.4f" % (cv_out["f1"]["mean"], cv_out["f1"]["std"]))

    # ---------------- evaluation figures ----------------
    print("Generating evaluation figures...")
    plt.figure(figsize=(7, 6))
    for n in models:
        fpr, tpr, _ = roc_curve(y_te, probas[n]); plt.plot(fpr, tpr, label=f"{n} (AUC={res[n]['ROC_AUC']:.4f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.title("ROC Curves"); save_fig("roc_curves")

    for n in models:
        plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(cms[n], annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix - {n}")
        save_fig("confusion_" + n.lower().replace(" ", "_"))

    imps = {}
    for n in ["XGBoost", "Random Forest"]:
        s = pd.Series(models[n].feature_importances_, index=FEATURES).sort_values(ascending=False)
        imps[n] = s
        plt.figure(figsize=(8, 7)); s.head(20)[::-1].plot(kind="barh", color="seagreen")
        plt.title(f"Top 20 Feature Importances - {n}"); save_fig("feature_importance_" + n.lower().replace(" ", "_"))

    comp = pd.DataFrame(res).T[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]
    comp.plot(kind="bar", figsize=(10, 5)); plt.xticks(rotation=0); plt.ylim(0, 1.05)
    plt.title("Performance Comparison (test set)"); save_fig("model_comparison")

    comp[["Precision", "Recall", "F1"]].plot(kind="bar", figsize=(8, 5), color=["#4c72b0", "#dd8452", "#55a868"])
    plt.xticks(rotation=0); plt.ylim(0, 1.05); plt.title("Precision / Recall / F1 (fraud class)"); save_fig("prf_chart")

    plt.figure(figsize=(7, 4.5))
    for k in ["precision", "recall", "f1"]:
        plt.plot(range(1, 6), cv_out[k]["folds"], marker="o", label=k)
    plt.xlabel("Fold"); plt.legend(); plt.title("XGBoost 5-Fold CV (fraud class)"); save_fig("cv_folds")

    # ---------------- save artefacts ----------------
    sample_fraud = te[te.Class == 1][RAW_COLS].iloc[0].to_dict()
    sample_legit = te[te.Class == 0][RAW_COLS].iloc[0].to_dict()
    best = max(res, key=lambda n: res[n]["F1"])
    joblib.dump(amount_scaler, f"{MODEL_DIR}/robust_scaler.pkl")
    for n, m in models.items():
        joblib.dump(m, f"{MODEL_DIR}/{n.lower().replace(' ', '_')}.pkl")
    json.dump({"best_model": best, "features": FEATURES, "samples": {"fraud": sample_fraud, "legit": sample_legit},
               "importance": {n: s.head(20).to_dict() for n, s in imps.items()}},
              open(f"{MODEL_DIR}/meta.json", "w"), indent=2)
    json.dump({"results": res, "cv_xgboost": cv_out, "confusion": {n: c.tolist() for n, c in cms.items()},
               "best_model": best, "duplicates_removed": n_dup, "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
               "smote_ratio": smote_ratio, "scale_pos_weight": spw, "n_train_after_smote": int(len(X_bal))},
              open(f"{OUT_DIR}/metrics.json", "w"), indent=2)
    with open(f"{OUT_DIR}/classification_reports.txt", "w") as f:
        for n, r in reports.items():
            f.write(f"=== {n} ===\n{r}\n")

    print("\n=== SUMMARY (test set, fraud class) ===")
    for n, r in res.items():
        print(f"{n:20s} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f} AUC={r['ROC_AUC']:.4f} PR-AUC={r['PR_AUC']:.4f}")
    print(f"Best by F1: {best}")
    if res[best]["F1"] < 0.85:
        print("NOTE: best F1 < 0.85. Report the real value and discuss it; see README notes on threshold/SMOTE ratio.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--smote-ratio", type=float, default=0.1,
                    help="minority/majority ratio after SMOTE (1.0 = fully balanced)")
    a = ap.parse_args(); main(a.data, a.smote_ratio)
