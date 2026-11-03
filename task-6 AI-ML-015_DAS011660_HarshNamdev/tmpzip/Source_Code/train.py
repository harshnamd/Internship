"""
LoanDefend - Intelligent Loan Sanction Prediction (AI-ML-015)
Random Forest vs Gradient Boosting; pipeline = StandardScaler -> SMOTE -> classifier (leak-free CV).

Usage:  python train.py --data data/loan_data.csv [--keep-outliers]
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler

from features import ENGINEERED, FEATURES, RAW_COLS, fit_impute_stats, make_encoders, prepare

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
OUT_DIR, MODEL_DIR, SEED, TARGET = "outputs", "models", 42, "Loan_Status"
_n = 0


def save_fig(name):
    global _n
    _n += 1
    p = f"{OUT_DIR}/fig{_n:02d}_{name}.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close(); print("  saved", p)


def load(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower() == "loan_id"])
    for c in df.select_dtypes("object"):
        df[c] = df[c].str.strip()
    if TARGET not in df.columns:
        raise SystemExit("Loan_Status column not found - use the TRAINING file, not the test file.")
    df[TARGET] = df[TARGET].map({"Y": 1, "N": 0})
    print("Loaded:", df.shape, "| missing cells:", int(df.isna().sum().sum()), f"| approved {df[TARGET].mean()*100:.1f}%")
    return df


def iqr_filter(df):
    keep = pd.Series(True, index=df.index)
    for c in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
        keep &= df[c].isna() | df[c].between(q1 - 1.5 * i, q3 + 1.5 * i)   # keep rows where value is missing
    return df[keep].reset_index(drop=True)


def run_eda(raw, df):
    print("Generating EDA figures...")
    miss = raw.isna().sum(); miss = miss[miss > 0].sort_values()
    plt.figure(figsize=(7, 4.5)); miss.plot(kind="barh", color="slateblue"); plt.title("Missing Values per Column (raw data)"); save_fig("missing_values")
    c = df[TARGET].value_counts().sort_index()
    plt.figure(figsize=(6, 4.5)); ax = sns.barplot(x=["Not approved (0)", "Approved (1)"], y=c.values, palette=["tomato", "seagreen"])
    for i, v in enumerate(c.values): ax.text(i, v, f"{v} ({v/c.sum()*100:.1f}%)", ha="center", va="bottom")
    plt.title("Target Distribution"); save_fig("target_bar")
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]:
        plt.figure(figsize=(7.5, 4.5))
        sns.histplot(data=df, x=col, hue=TARGET, kde=col != "Loan_Amount_Term", bins=30, palette=["tomato", "seagreen"], stat="density", common_norm=False)
        plt.title(f"{col} Distribution by Loan Status"); save_fig(col.lower() + "_dist")
    for col in ["Credit_History", "Gender", "Married", "Education", "Self_Employed", "Property_Area", "Dependents"]:
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
        sns.countplot(data=df, x=col, ax=ax[0], color="steelblue"); ax[0].set_title(f"{col} counts")
        (df.groupby(col)[TARGET].mean() * 100).plot(kind="bar", ax=ax[1], color="seagreen"); ax[1].set_ylabel("Approval rate (%)")
        ax[1].set_title(f"Approval rate by {col}"); plt.setp(ax[1].get_xticklabels(), rotation=0); save_fig(col.lower() + "_vs_status")
    plt.figure(figsize=(7, 5)); sns.scatterplot(data=df, x="ApplicantIncome", y="LoanAmount", hue=TARGET, palette=["tomato", "seagreen"], alpha=.7)
    plt.title("Income vs Loan Amount"); save_fig("income_vs_loan")
    num = df.copy()
    for cat in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Dependents"]:
        num[cat] = num[cat].astype("category").cat.codes
    corr = num.corr(numeric_only=True)
    plt.figure(figsize=(9, 7.5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7}); plt.title("Correlation Heatmap"); save_fig("corr_heatmap")
    s = corr[TARGET].drop(TARGET).sort_values()
    plt.figure(figsize=(7, 5)); s.plot(kind="barh", color=np.where(s > 0, "seagreen", "tomato")); plt.title("Correlation of Features with Loan Status"); save_fig("corr_with_target")


def metr(y, pred, proba):
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "ROC_AUC": roc_auc_score(y, proba)}


def pipelines():
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=2,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, min_samples_split=5,
                                    min_samples_leaf=2, subsample=0.8, random_state=SEED)
    # SMOTE sits INSIDE the pipeline, so cross-validation applies it to training folds only
    mk = lambda clf: ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=SEED)), ("clf", clf)])
    return {"Random Forest": mk(rf), "Gradient Boosting": mk(gb)}


def main(a):
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)
    raw = load(a.data)
    df = raw if a.keep_outliers else iqr_filter(raw)
    print(f"After IQR outlier removal: {df.shape} (removed {len(raw) - len(df)})")
    run_eda(raw, df)

    tr, te = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    stats, enc = fit_impute_stats(tr), make_encoders()      # imputation values learned from training rows only
    Xtr, ytr = prepare(tr, stats, enc), tr[TARGET].values
    Xte, yte = prepare(te, stats, enc), te[TARGET].values
    print(f"Features: {len(RAW_COLS)} original + {len(ENGINEERED)} engineered = {Xtr.shape[1]} | train {Xtr.shape} test {Xte.shape}")

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res, probs, cms, pipes = {}, {}, {}, {}
    for n, pipe in pipelines().items():
        print(f"{n}: 5-fold stratified CV (SMOTE inside each fold)...")
        cv = cross_validate(pipe, Xtr, ytr, cv=skf, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"], n_jobs=1)
        cvo = {k: {"mean": float(cv[f"test_{k}"].mean()), "std": float(cv[f"test_{k}"].std()), "folds": cv[f"test_{k}"].tolist()}
               for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
        pipe.fit(Xtr, ytr); pipes[n] = pipe
        p = pipe.predict_proba(Xte)[:, 1]; probs[n] = p; pred = (p >= .5).astype(int)
        res[n] = {"test": metr(yte, pred, p), "train": metr(ytr, pipe.predict(Xtr), pipe.predict_proba(Xtr)[:, 1]), "cv": cvo}
        cms[n] = confusion_matrix(yte, pred)
        print("  test:", {k: round(v, 4) for k, v in res[n]["test"].items()}, f"| CV acc {cvo['accuracy']['mean']:.4f}+/-{cvo['accuracy']['std']:.4f}")

    print("Generating evaluation figures...")
    plt.figure(figsize=(7, 6))
    for n in pipes:
        fpr, tpr, _ = roc_curve(yte, probs[n]); plt.plot(fpr, tpr, label=f"{n} (AUC={res[n]['test']['ROC_AUC']:.4f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC Curves"); save_fig("roc_curves")
    for n in pipes:
        plt.figure(figsize=(5.5, 4.5)); sns.heatmap(cms[n], annot=True, fmt="d", cmap="Blues", xticklabels=["Not approved", "Approved"], yticklabels=["Not approved", "Approved"])
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix - {n}"); save_fig("confusion_" + n.lower().replace(" ", "_"))
    imps = {}
    for n, pipe in pipes.items():
        s = pd.Series(pipe.named_steps["clf"].feature_importances_, index=FEATURES).sort_values(ascending=False); imps[n] = s
        plt.figure(figsize=(8, 6)); s.head(15)[::-1].plot(kind="barh", color="seagreen"); plt.title(f"Top 15 Feature Importances - {n}"); save_fig("feature_importance_" + n.lower().replace(" ", "_"))
    comp = pd.DataFrame({n: res[n]["test"] for n in pipes}).T
    comp.plot(kind="bar", figsize=(9, 5)); plt.xticks(rotation=0); plt.ylim(0, 1.05); plt.title("Test Performance Comparison"); save_fig("model_comparison")
    comp[["Precision", "Recall", "F1"]].plot(kind="bar", figsize=(7, 5), color=["#4c72b0", "#dd8452", "#55a868"]); plt.xticks(rotation=0); plt.ylim(0, 1.05)
    plt.title("Precision / Recall / F1 (approved class)"); save_fig("prf_chart")
    plt.figure(figsize=(7.5, 4.5))
    for n in pipes: plt.plot(range(1, 6), res[n]["cv"]["accuracy"]["folds"], marker="o", label=n)
    plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.legend(); plt.title("5-Fold Stratified CV Accuracy (SMOTE inside folds)"); save_fig("cv_folds")
    for n, pipe in pipelines().items():
        sz, trs, vas = learning_curve(pipe, Xtr, ytr, train_sizes=[.2, .4, .6, .8, 1.0], cv=skf, scoring="accuracy", n_jobs=1)
        plt.figure(figsize=(7, 4.5)); plt.plot(sz, trs.mean(1), "o-", label="Training accuracy"); plt.plot(sz, vas.mean(1), "o-", label="Validation accuracy")
        plt.xlabel("Training samples"); plt.ylabel("Accuracy"); plt.legend(); plt.title(f"Learning Curve - {n}"); save_fig("learning_curve_" + n.lower().replace(" ", "_"))

    best = max(res, key=lambda n: res[n]["test"]["Accuracy"])
    for n, pipe in pipes.items(): joblib.dump(pipe, f"{MODEL_DIR}/{n.lower().replace(' ', '_')}.pkl")
    joblib.dump(enc, f"{MODEL_DIR}/label_encoders.pkl")
    full = te.dropna(subset=RAW_COLS)                       # demo samples must have no missing cells
    ap, na = full[full[TARGET] == 1].iloc[0], full[full[TARGET] == 0].iloc[0]
    fix = lambda r: {**json.loads(r[RAW_COLS].to_json()), "Credit_History": str(int(r["Credit_History"]))}
    json.dump({"best_model": best, "impute_stats": stats, "raw_cols": RAW_COLS, "cat_classes": {**{k: v for k, v in __import__("features").CAT_CLASSES.items()}},
               "samples": {"approved-like": fix(ap), "rejected-like": fix(na)},
               "importance": {n: s.head(15).to_dict() for n, s in imps.items()}}, open(f"{MODEL_DIR}/meta.json", "w"), indent=2)
    json.dump({"results": res, "confusion": {n: c.tolist() for n, c in cms.items()}, "best_model": best, "n_raw": int(len(raw)),
               "n_after_outliers": int(len(df)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "n_features": int(Xtr.shape[1]),
               "n_engineered": len(ENGINEERED), "outliers_removed": int(len(raw) - len(df))}, open(f"{OUT_DIR}/metrics.json", "w"), indent=2)

    print("\n=== SUMMARY (test set, approved class) ===")
    for n in pipes:
        t, tr_, cv = res[n]["test"], res[n]["train"], res[n]["cv"]
        print(f"{n:18s} Acc={t['Accuracy']:.4f} P={t['Precision']:.4f} R={t['Recall']:.4f} F1={t['F1']:.4f} AUC={t['ROC_AUC']:.4f} | train acc={tr_['Accuracy']:.4f}")
        print(f"{'':18s} CV acc={cv['accuracy']['mean']:.4f}+/-{cv['accuracy']['std']:.4f} F1={cv['f1']['mean']:.4f}+/-{cv['f1']['std']:.4f} AUC={cv['roc_auc']['mean']:.4f}+/-{cv['roc_auc']['std']:.4f}")
    print(f"Best by test accuracy: {best}. Test set has only {len(yte)} rows, so one prediction = {100/len(yte):.2f} accuracy points; trust CV more.")
    b = res[best]["test"]
    if b["Accuracy"] < .80 or b["F1"] < .80 or b["ROC_AUC"] < .81:
        print("NOTE: a target (acc>=0.80, F1>=0.80, AUC>=0.81) was not met. Report real values and discuss.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/loan_data.csv")
    ap.add_argument("--keep-outliers", action="store_true", help="skip IQR outlier removal (keeps all 614 rows)")
    main(ap.parse_args())
