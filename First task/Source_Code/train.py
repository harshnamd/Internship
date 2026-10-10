"""CHIPGUARD training script: EDA, leak-free SMOTE pipelines, Lasso vs Ridge, evaluation, plots.

Usage:  python train.py [path/to/Bank_Personal_Loan_Modelling.csv]
Outputs: models/*.joblib, outputs/*.png, outputs/metrics.json, outputs/metrics_table.csv
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, TARGET, RegressionClassifier, preprocess

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/Bank_Personal_Loan_Modelling.csv"
OUT, MODELS = "outputs", "models"
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------- 1. load + inspect
raw = pd.read_csv(CSV)
print("Shape:", raw.shape)
print("Missing values:\n", raw.isnull().sum().sum())
raw.columns = [c.strip().replace(" ", "_") for c in raw.columns]
raw[TARGET] = raw[TARGET].astype(int)
print("Class balance:\n", raw[TARGET].value_counts(normalize=True).round(4))

# ---------------------------------------------------------------- 2-3. preprocess + features
df = preprocess(raw)

# ---------------------------------------------------------------- 4. EDA charts
counts = df[TARGET].value_counts().sort_index()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].pie(counts, labels=["No Loan", "Loan"], autopct="%1.1f%%", colors=["#4c72b0", "#dd8452"])
ax[0].set_title("Target distribution")
ax[1].bar(["No Loan", "Loan"], counts.values, color=["#4c72b0", "#dd8452"])
ax[1].set_title("Class counts")
save("eda_target_distribution")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for a, col in zip(ax.ravel(), ["Income", "Age", "Education", "CCAvg"]):
    if col == "Education":
        sns.countplot(data=df, x=col, hue=TARGET, ax=a)
    else:
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, stat="density", common_norm=False, ax=a)
    a.set_title(f"{col} by loan status")
save("eda_feature_distributions")

plt.figure(figsize=(11, 9))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation heatmap")
save("eda_correlation_heatmap")

corr_t = df.corr()[TARGET].drop(TARGET).sort_values()
plt.figure(figsize=(8, 6))
corr_t.plot(kind="barh", color=np.where(corr_t > 0, "#55a868", "#c44e52"))
plt.title("Feature correlation with Personal_Loan")
save("eda_feature_target_correlation")

# ---------------------------------------------------------------- 5. split (NO SMOTE here)
X, y = df[FEATURE_COLUMNS], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train {X_train.shape}, Test {X_test.shape}, "
      f"acceptance train={y_train.mean():.4f} test={y_test.mean():.4f}")

# ---------------------------------------------------------------- 6. leak-free pipelines
def make_pipeline(reg):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(k_neighbors=5, random_state=42)),   # only ever applied to training folds
        ("clf", RegressionClassifier(reg)),
    ])


pipelines = {
    "Lasso": make_pipeline(Lasso(alpha=0.01, max_iter=10000)),
    "Ridge": make_pipeline(Ridge(alpha=1.0, max_iter=10000)),
}

# ---------------------------------------------------------------- 7-8. CV, training, evaluation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results, roc_data, cms, importances = {}, {}, {}, {}


def metrics(y_true, pred, score):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "f1_weighted": f1_score(y_true, pred, average="weighted"),
        "roc_auc": roc_auc_score(y_true, score),
    }


for name, pipe in pipelines.items():
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    tr_pred, te_pred = pipe.predict(X_train), pipe.predict(X_test)
    tr_score, te_score = pipe.decision_function(X_train), pipe.decision_function(X_test)
    results[name] = {
        "cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std(),
        "train": metrics(y_train, tr_pred, tr_score),
        "test": metrics(y_test, te_pred, te_score),
    }
    cms[name] = confusion_matrix(y_test, te_pred)
    roc_data[name] = roc_curve(y_test, te_score)
    coef = pipe.named_steps["clf"].coef_
    importances[name] = pd.Series(coef, index=FEATURE_COLUMNS)
    print(f"\n=== {name} ===  CV AUC {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(classification_report(y_test, te_pred, target_names=["No Loan", "Loan"], digits=4))
    joblib.dump(pipe, f"{MODELS}/{name.lower()}_pipeline.joblib")
    joblib.dump(pipe.named_steps["scaler"], f"{MODELS}/{name.lower()}_scaler.joblib")

best = max(results, key=lambda n: results[n]["test"]["roc_auc"])
joblib.dump({"best_model": best, "features": FEATURE_COLUMNS}, f"{MODELS}/model_info.joblib")
print("\nBest model by test ROC-AUC:", best)

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({"best_model": best, **results}, f, indent=2)
rows = []
for n, r in results.items():
    for split in ("train", "test"):
        rows.append({"model": n, "split": split, **r[split]})
    rows.append({"model": n, "split": "cv_3fold", "roc_auc": r["cv_roc_auc_mean"]})
pd.DataFrame(rows).round(4).to_csv(f"{OUT}/metrics_table.csv", index=False)

# ---------------------------------------------------------------- plots for the report
plt.figure(figsize=(7, 6))
for n, (fpr, tpr, _) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{n} (AUC={results[n]['test']['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test set)")
plt.legend()
save("roc_curves")

for n, cm in cms.items():
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Loan", "Loan"],
                yticklabels=["No Loan", "Loan"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {n}")
    save(f"confusion_matrix_{n.lower()}")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for a, (n, imp) in zip(ax, importances.items()):
    top = imp.reindex(imp.abs().sort_values(ascending=False).index)[:10][::-1]
    top.plot(kind="barh", ax=a, color=np.where(top > 0, "#55a868", "#c44e52"))
    a.set_title(f"Top 10 features by |coefficient| - {n}")
save("feature_importance")

comp = pd.DataFrame({n: r["test"] for n, r in results.items()}).T
comp[["accuracy"]].plot(kind="bar", figsize=(6, 5), ylim=(0, 1), rot=0, legend=False)
plt.title("Accuracy comparison (test set)")
save("accuracy_comparison")
comp[["precision", "recall", "f1"]].plot(kind="bar", figsize=(7, 5), ylim=(0, 1), rot=0)
plt.title("Precision / Recall / F1 (Loan class, test set)")
save("precision_recall_f1")

plt.figure(figsize=(7, 5))
for n, pipe in pipelines.items():
    sizes, tr, va = learning_curve(pipe, X_train, y_train, cv=cv, scoring="roc_auc",
                                   train_sizes=np.linspace(0.2, 1.0, 6), random_state=42)
    plt.plot(sizes, tr.mean(1), "o-", label=f"{n} train")
    plt.plot(sizes, va.mean(1), "s--", label=f"{n} validation")
plt.xlabel("Training samples"); plt.ylabel("ROC-AUC"); plt.title("Learning curve (training vs validation)")
plt.legend()
save("learning_curve")
print(f"\nCharts and metrics saved in ./{OUT}, models in ./{MODELS}")
