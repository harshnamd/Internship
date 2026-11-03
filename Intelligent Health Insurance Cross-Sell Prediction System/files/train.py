"""
Intelligent Health Insurance Cross-Sell Prediction (AI-ML-011)
Ridge (L2) vs Lasso (L1) logistic regression, with SMOTE INSIDE the pipeline (leak-free CV).

Usage:  python train.py --data data/train.csv [--alpha-ridge 1 --alpha-lasso 1] [--tune] [--include-codes]
Outputs: models/, outputs/ (figures, metrics.json)
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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import add_features, feature_lists, fit_bins, raw_input_cols

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
OUT_DIR, MODEL_DIR, SEED = "outputs", "models", 42
TARGET = "Response"
_n = 0


def save_fig(name):
    global _n
    _n += 1
    p = f"{OUT_DIR}/fig{_n:02d}_{name}.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close(); print("  saved", p)


def load_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower() == "id"])
    print("Loaded:", df.shape, "| missing:", int(df.isna().sum().sum()))
    for c in df.columns:                                   # handle missing values
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median() if df[c].dtype != object else df[c].mode()[0])
    before = len(df)
    for c in ["Annual_Premium", "Vintage"]:                # IQR outlier removal
        q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
        df = df[(df[c] >= q1 - 1.5 * i) & (df[c] <= q3 + 1.5 * i)]
    print(f"IQR outliers removed: {before - len(df)} -> {df.shape}")
    return df.reset_index(drop=True), before - len(df)


def run_eda(df):
    print("Generating EDA figures...")
    c = df[TARGET].value_counts().sort_index()
    plt.figure(figsize=(5.5, 5.5)); plt.pie(c.values, labels=["No purchase", "Purchase"], autopct="%1.1f%%",
        colors=["steelblue", "tomato"], startangle=90, explode=(0, .08)); plt.title("Target Distribution"); save_fig("target_pie")
    plt.figure(figsize=(6, 4.5)); sns.barplot(x=["No purchase", "Purchase"], y=c.values, palette=["steelblue", "tomato"])
    plt.ylabel("Customers"); plt.title("Target Counts"); save_fig("target_bar")
    for col, nm in [("Age", "age"), ("Annual_Premium", "premium"), ("Vintage", "vintage")]:
        plt.figure(figsize=(7.5, 4.5))
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, stat="density", common_norm=False, bins=40, palette=["steelblue", "tomato"])
        plt.title(f"{col} Distribution by Response"); save_fig(f"{nm}_dist")
    for col in ["Gender", "Vehicle_Age", "Vehicle_Damage", "Previously_Insured"]:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        sns.countplot(data=df, x=col, ax=ax[0], color="steelblue"); ax[0].set_title(f"{col} counts")
        rate = df.groupby(col)[TARGET].mean() * 100
        rate.plot(kind="bar", ax=ax[1], color="tomato"); ax[1].set_ylabel("Purchase rate (%)"); ax[1].set_title(f"Purchase rate by {col}")
        plt.setp(ax[1].get_xticklabels(), rotation=0); save_fig(f"{col.lower()}_dist")
    num = df.copy()
    for col in ["Gender", "Vehicle_Age", "Vehicle_Damage"]:
        num[col] = num[col].astype("category").cat.codes
    corr = num.corr(numeric_only=True)
    plt.figure(figsize=(10, 8)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
    plt.title("Correlation Heatmap"); save_fig("corr_heatmap")
    s = corr[TARGET].drop(TARGET).sort_values()
    plt.figure(figsize=(7, 5)); s.plot(kind="barh", color=np.where(s > 0, "tomato", "steelblue"))
    plt.title("Correlation of Features with Response"); save_fig("corr_with_target")
    g = pd.cut(df["Age"], [0, 25, 35, 45, 55, 65, 200], labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
    plt.figure(figsize=(7, 4.5)); (df.groupby(g)[TARGET].mean() * 100).plot(kind="bar", color="seagreen")
    plt.ylabel("Purchase rate (%)"); plt.xticks(rotation=0); plt.title("Purchase Rate by Age Group"); save_fig("age_group_rate")


def make_pipe(penalty, alpha, cats, nums, codes):
    tf = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats), ("num", StandardScaler(), nums)]
    if codes:
        tf.append(("code", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False, min_frequency=500), codes))
    clf = LogisticRegression(penalty=penalty, C=1.0 / alpha, solver="liblinear", class_weight="balanced",
                             max_iter=1000, random_state=SEED)
    # SMOTE sits INSIDE the pipeline -> only ever applied to training folds
    return ImbPipeline([("prep", ColumnTransformer(tf)), ("smote", SMOTE(random_state=SEED)), ("clf", clf)])


def metr(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "ROC_AUC": roc_auc_score(y, p)}


def main(a):
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)
    df, n_out = load_clean(a.data)
    run_eda(df)
    cats, nums, codes = feature_lists(a.include_codes)
    tr, te = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    bins = fit_bins(tr)                                    # bin edges from training data only
    Xtr, ytr = add_features(tr.drop(columns=TARGET), bins, a.include_codes), tr[TARGET].values
    Xte, yte = add_features(te.drop(columns=TARGET), bins, a.include_codes), te[TARGET].values
    print(f"Train {Xtr.shape} (positives {ytr.mean()*100:.2f}%) | Test {Xte.shape}")

    alphas = {"Ridge": a.alpha_ridge, "Lasso": a.alpha_lasso}
    pen = {"Ridge": "l2", "Lasso": "l1"}
    if a.tune:
        print("Tuning alpha on a 60k training subsample (3-fold, ROC-AUC, SMOTE inside folds)...")
        idx = np.random.RandomState(SEED).choice(len(Xtr), min(60000, len(Xtr)), replace=False)
        for n in alphas:
            gs = GridSearchCV(make_pipe(pen[n], 1, cats, nums, codes), {"clf__C": [100, 10, 1, 0.1, 0.01]},
                              cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="roc_auc", n_jobs=1)
            gs.fit(Xtr.iloc[idx], ytr[idx]); alphas[n] = 1.0 / gs.best_params_["clf__C"]
            print(f"  {n}: best alpha = {alphas[n]}")

    kf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res, probs, cms, pipes, oof_thr = {}, {}, {}, {}, {}
    for n in ["Ridge", "Lasso"]:
        print(f"{n}: 5-fold stratified CV (SMOTE inside each fold)...")
        base = make_pipe(pen[n], alphas[n], cats, nums, codes)
        folds, oof = [], np.zeros(len(ytr))
        for tri, vai in kf.split(Xtr, ytr):
            m = clone(base).fit(Xtr.iloc[tri], ytr[tri]); p = m.predict_proba(Xtr.iloc[vai])[:, 1]
            oof[vai] = p; folds.append(metr(ytr[vai], p))
        cv = {k: {"mean": float(np.mean([f[k] for f in folds])), "std": float(np.std([f[k] for f in folds])),
                  "folds": [float(f[k]) for f in folds]} for k in folds[0]}
        pr, rc, th = precision_recall_curve(ytr, oof)       # threshold from OUT-OF-FOLD train predictions
        f1s = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
        thr = float(th[np.argmax(f1s)]); oof_thr[n] = thr
        print(f"  CV AUC {cv['ROC_AUC']['mean']:.4f}+/-{cv['ROC_AUC']['std']:.4f}; F1-optimal threshold (OOF) = {thr:.3f}")
        print(f"{n}: fitting final model...")
        pipe = clone(base).fit(Xtr, ytr); pipes[n] = pipe
        p = pipe.predict_proba(Xte)[:, 1]; probs[n] = p
        res[n] = {"alpha": alphas[n], "C": 1.0 / alphas[n], "test_at_0.5": metr(yte, p), "test_at_tuned_threshold": metr(yte, p, thr),
                  "tuned_threshold": thr, "cv": cv, "train_at_0.5": metr(ytr, pipe.predict_proba(Xtr)[:, 1])}
        cms[n] = confusion_matrix(yte, (p >= .5).astype(int))
        print("  test@0.5:", {k: round(v, 4) for k, v in res[n]["test_at_0.5"].items()})

    print("Generating evaluation figures...")
    plt.figure(figsize=(7, 6))
    for n in pipes:
        fpr, tpr, _ = roc_curve(yte, probs[n]); plt.plot(fpr, tpr, label=f"{n} (AUC={res[n]['test_at_0.5']['ROC_AUC']:.4f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC Curves"); save_fig("roc_curves")
    plt.figure(figsize=(7, 6))
    for n in pipes:
        pr, rc, _ = precision_recall_curve(yte, probs[n]); plt.plot(rc, pr, label=n)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("Precision-Recall Curves"); save_fig("pr_curves")
    for n in pipes:
        plt.figure(figsize=(5.5, 4.5)); sns.heatmap(cms[n], annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"]); plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {n} (threshold 0.5)"); save_fig(f"confusion_{n.lower()}")
    coefs = {}
    for n, pipe in pipes.items():
        names = [x.split("__", 1)[1] for x in pipe.named_steps["prep"].get_feature_names_out()]
        s = pd.Series(pipe.named_steps["clf"].coef_[0], index=names); coefs[n] = s
        t = s.reindex(s.abs().sort_values(ascending=False).index).head(15)[::-1]
        plt.figure(figsize=(8, 6)); t.plot(kind="barh", color=np.where(t > 0, "tomato", "steelblue"))
        plt.title(f"Top 15 Features by |Coefficient| - {n}"); save_fig(f"coef_{n.lower()}")
    comp = pd.DataFrame({n: res[n]["test_at_0.5"] for n in pipes}).T
    comp.plot(kind="bar", figsize=(9, 5)); plt.xticks(rotation=0); plt.ylim(0, 1.05); plt.title("Test Performance Comparison"); save_fig("model_comparison")
    comp[["Precision", "Recall", "F1"]].plot(kind="bar", figsize=(7, 5), color=["#4c72b0", "#dd8452", "#55a868"])
    plt.xticks(rotation=0); plt.ylim(0, 1.05); plt.title("Precision / Recall / F1"); save_fig("prf_chart")
    plt.figure(figsize=(7.5, 4.5))
    for n, col in [("Ridge", "C0"), ("Lasso", "C1")]:
        plt.plot(range(1, 6), res[n]["cv"]["ROC_AUC"]["folds"], marker="o", label=f"{n} AUC", color=col)
    plt.xlabel("Fold"); plt.legend(); plt.title("5-Fold Stratified CV ROC-AUC (SMOTE inside folds)"); save_fig("cv_folds")
    print("Learning curves (60k subsample, 3-fold)...")
    idx = np.random.RandomState(SEED).choice(len(Xtr), min(60000, len(Xtr)), replace=False)
    for n in pipes:
        sz, trs, vas = learning_curve(make_pipe(pen[n], alphas[n], cats, nums, codes), Xtr.iloc[idx], ytr[idx],
                                      train_sizes=[.1, .25, .5, .75, 1.0], cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                      scoring="roc_auc", n_jobs=1)
        plt.figure(figsize=(7, 4.5)); plt.plot(sz, trs.mean(1), "o-", label="Training AUC"); plt.plot(sz, vas.mean(1), "o-", label="Validation AUC")
        plt.xlabel("Training samples"); plt.ylabel("ROC-AUC"); plt.legend(); plt.title(f"Learning Curve - {n}"); save_fig(f"learning_curve_{n.lower()}")

    best = max(pipes, key=lambda n: res[n]["test_at_0.5"]["ROC_AUC"])
    joblib.dump(pipes["Ridge"], f"{MODEL_DIR}/ridge_pipeline.pkl"); joblib.dump(pipes["Lasso"], f"{MODEL_DIR}/lasso_pipeline.pkl")
    json.dump(bins, open(f"{MODEL_DIR}/bins.json", "w"))
    raw = raw_input_cols(a.include_codes)
    pos, neg = te[te[TARGET] == 1].iloc[0], te[te[TARGET] == 0].iloc[0]
    json.dump({"best_model": best, "include_codes": a.include_codes, "input_cols": raw,
               "choices": {c: sorted(df[c].unique().tolist()) for c in ["Gender", "Vehicle_Age", "Vehicle_Damage"]},
               "samples": {"likely buyer": json.loads(pos[raw].to_json()), "unlikely buyer": json.loads(neg[raw].to_json())},
               "coefficients": {n: s.sort_values(key=abs, ascending=False).head(15).to_dict() for n, s in coefs.items()}},
              open(f"{MODEL_DIR}/meta.json", "w"), indent=2)
    json.dump({"results": res, "confusion": {n: c.tolist() for n, c in cms.items()}, "best_model": best,
               "outliers_removed": int(n_out), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
               "lasso_zero_coefficients": int((coefs["Lasso"] == 0).sum()), "n_features_after_encoding": int(len(coefs["Lasso"]))},
              open(f"{OUT_DIR}/metrics.json", "w"), indent=2)

    print("\n=== SUMMARY (test set) ===")
    for n in pipes:
        r = res[n]["test_at_0.5"]; t = res[n]["test_at_tuned_threshold"]
        print(f"{n:6s} @0.5  Acc={r['Accuracy']:.4f} P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1']:.4f} AUC={r['ROC_AUC']:.4f}")
        print(f"{'':6s} @{res[n]['tuned_threshold']:.2f} Acc={t['Accuracy']:.4f} P={t['Precision']:.4f} R={t['Recall']:.4f} F1={t['F1']:.4f}")
        print(f"{'':6s} CV AUC={res[n]['cv']['ROC_AUC']['mean']:.4f}+/-{res[n]['cv']['ROC_AUC']['std']:.4f}  CV F1={res[n]['cv']['F1']['mean']:.4f}+/-{res[n]['cv']['F1']['std']:.4f}")
    print(f"Lasso zeroed {int((coefs['Lasso']==0).sum())} of {len(coefs['Lasso'])} coefficients. Best by AUC: {best}")
    b = res[best]["test_at_0.5"]
    if b["Accuracy"] < .85 or b["ROC_AUC"] < .90:
        print("NOTE: targets (accuracy>=0.85, AUC>=0.90) not met. Report the real values and explain (see chat notes).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--alpha-ridge", type=float, default=1.0)
    ap.add_argument("--alpha-lasso", type=float, default=1.0)
    ap.add_argument("--tune", action="store_true", help="grid-search alpha (C=1/alpha) on a training subsample")
    ap.add_argument("--include-codes", action="store_true", help="also one-hot Region_Code and Policy_Sales_Channel")
    main(ap.parse_args())
