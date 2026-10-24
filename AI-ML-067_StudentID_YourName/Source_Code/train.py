"""
Intelligent Wind Speed Prediction System (AI-ML-067)
Training + EDA + evaluation pipeline.

Usage:
    python train.py                      # expects data/weather.csv
    python train.py --data path/to.csv

Outputs:
    models/      scaler.pkl, random_forest.pkl, linear_regression.pkl, best_model.pkl, meta.json
    outputs/     all EDA + evaluation figures (PNG) and metrics.json
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

FEATURES = ["RAIN", "T.MAX", "T.MIN", "T.MIN.G", "YEAR", "MONTH", "DAY"]
TARGET = "WIND"
OUT_DIR, MODEL_DIR = "outputs", "models"
RANDOM_STATE = 42
_fig_counter = 0


def save_fig(name):
    """Save the current figure with an auto-incremented figure number."""
    global _fig_counter
    _fig_counter += 1
    path = os.path.join(OUT_DIR, f"fig{_fig_counter:02d}_{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    # clean column names: strip whitespace / stray characters
    df.columns = [c.strip().replace(" ", "").replace("\ufeff", "") for c in df.columns]
    # keep only the 6 columns we need (the raw Irish dataset also has IND flag columns)
    keep = ["DATE", "WIND", "RAIN", "T.MAX", "T.MIN", "T.MIN.G"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")
    df = df[keep].copy()
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------
def preprocess(df):
    info = {"rows_loaded": int(len(df)), "missing_before": df.isna().sum().to_dict()}
    df = df.dropna(subset=["DATE"]).copy()
    num_cols = ["WIND", "RAIN", "T.MAX", "T.MIN", "T.MIN.G"]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())          # median imputation
    info["missing_after"] = df.isna().sum().to_dict()

    # IQR outlier removal (applied sequentially to the 5 numeric columns)
    before = len(df)
    for c in num_cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df = df[(df[c] >= lo) & (df[c] <= hi)]
    info["outliers_removed"] = int(before - len(df))
    info["rows_after_cleaning"] = int(len(df))
    return df.reset_index(drop=True), info


# --------------------------------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------------------------------
def add_time_features(df):
    df = df.copy()
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    df["DAY"] = df["DATE"].dt.day
    return df


# --------------------------------------------------------------------------
# 4. EDA (20 figures)
# --------------------------------------------------------------------------
def run_eda(df):
    print("Generating EDA figures...")
    # 1-5 distributions
    for col, color in [("WIND", "steelblue"), ("T.MAX", "tomato"), ("T.MIN", "seagreen"),
                       ("T.MIN.G", "darkorange"), ("RAIN", "purple")]:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True, color=color, bins=40)
        plt.title(f"Distribution of {col}")
        save_fig(f"dist_{col.replace('.', '_')}")
    # 6 boxplots
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=(df[["WIND", "RAIN", "T.MAX", "T.MIN", "T.MIN.G"]] - df[["WIND", "RAIN", "T.MAX", "T.MIN", "T.MIN.G"]].mean())
                / df[["WIND", "RAIN", "T.MAX", "T.MIN", "T.MIN.G"]].std())
    plt.title("Boxplots of Standardised Numeric Features")
    save_fig("boxplots")
    # 7 heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[["WIND", "RAIN", "T.MAX", "T.MIN", "T.MIN.G", "YEAR", "MONTH", "DAY"]].corr(),
                annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    save_fig("corr_heatmap")
    # 8-11 scatter vs WIND
    sample = df.sample(min(2000, len(df)), random_state=RANDOM_STATE)
    for col in ["T.MAX", "T.MIN", "T.MIN.G", "RAIN"]:
        plt.figure(figsize=(7, 4.5))
        sns.scatterplot(data=sample, x=col, y="WIND", alpha=0.4)
        sns.regplot(data=sample, x=col, y="WIND", scatter=False, color="red")
        plt.title(f"WIND vs {col}")
        save_fig(f"wind_vs_{col.replace('.', '_')}")
    # 12 monthly mean wind
    plt.figure(figsize=(8, 4.5))
    df.groupby("MONTH")["WIND"].mean().plot(marker="o")
    plt.xlabel("Month"); plt.ylabel("Mean WIND"); plt.title("Mean Wind Speed by Month")
    save_fig("monthly_trend")
    # 13 yearly mean wind
    plt.figure(figsize=(8, 4.5))
    df.groupby("YEAR")["WIND"].mean().plot(marker="o", color="darkgreen")
    plt.xlabel("Year"); plt.ylabel("Mean WIND"); plt.title("Mean Wind Speed by Year")
    save_fig("yearly_trend")
    # 14 wind by month boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="MONTH", y="WIND")
    plt.title("Wind Speed Distribution by Month")
    save_fig("wind_by_month_box")
    # 15 T.MAX vs T.MIN
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=sample, x="T.MAX", y="T.MIN", alpha=0.4, color="teal")
    plt.title("T.MAX vs T.MIN (multicollinearity)")
    save_fig("tmax_vs_tmin")
    # 16 wind time series with rolling mean
    ts = df.set_index("DATE")["WIND"]
    plt.figure(figsize=(12, 4.5))
    ts.plot(alpha=0.3, label="Daily")
    ts.rolling(30).mean().plot(color="red", label="30-day rolling mean")
    plt.legend(); plt.title("Wind Speed Time Series")
    save_fig("wind_timeseries")
    # 17 monthly rainfall
    plt.figure(figsize=(8, 4.5))
    df.groupby("MONTH")["RAIN"].mean().plot(kind="bar", color="slateblue")
    plt.ylabel("Mean RAIN"); plt.title("Mean Rainfall by Month")
    save_fig("monthly_rain")
    # 18 wind KDE by season
    season = df["MONTH"].map(lambda m: "Winter" if m in (12, 1, 2) else "Spring" if m in (3, 4, 5)
                             else "Summer" if m in (6, 7, 8) else "Autumn")
    plt.figure(figsize=(8, 4.5))
    for s in ["Winter", "Spring", "Summer", "Autumn"]:
        sns.kdeplot(df.loc[season == s, "WIND"], label=s)
    plt.legend(); plt.title("Wind Speed Density by Season")
    save_fig("wind_by_season")
    # 19 pairplot
    g = sns.pairplot(sample[["WIND", "T.MAX", "T.MIN", "T.MIN.G", "RAIN"]].sample(min(600, len(sample)), random_state=1),
                     corner=True, plot_kws={"alpha": 0.4, "s": 12})
    g.fig.suptitle("Pairplot of Key Variables", y=1.02)
    save_fig("pairplot")
    # 20 rain vs no-rain wind
    plt.figure(figsize=(6, 4.5))
    sns.boxplot(x=(df["RAIN"] > 0).map({True: "Rain", False: "No rain"}), y=df["WIND"])
    plt.title("Wind Speed on Rainy vs Dry Days")
    save_fig("wind_rain_vs_dry")


# --------------------------------------------------------------------------
# 5. Model evaluation helpers
# --------------------------------------------------------------------------
def metrics(y, p):
    mse = mean_squared_error(y, p)
    return {"R2": float(r2_score(y, p)), "MAE": float(mean_absolute_error(y, p)),
            "MSE": float(mse), "RMSE": float(np.sqrt(mse))}


def main(data_path):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(data_path)
    print(f"Loaded shape: {df.shape}")
    df, prep_info = preprocess(df)
    df = add_time_features(df)
    print(f"After cleaning: {df.shape} (removed {prep_info['outliers_removed']} outlier rows)")
    run_eda(df)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler().fit(X_train)            # fit on training data only
    Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Linear Regression": LinearRegression(),
    }
    results, preds = {}, {}
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(Xs_train, y_train)
        p_tr, p_te = model.predict(Xs_train), model.predict(Xs_test)
        cv = cross_val_score(model, Xs_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
        results[name] = {"train": metrics(y_train, p_tr), "test": metrics(y_test, p_te),
                         "cv_r2_mean": float(cv.mean()), "cv_r2_std": float(cv.std()),
                         "cv_scores": cv.tolist()}
        preds[name] = p_te
        print(f"  Test R2={results[name]['test']['R2']:.4f}  CV R2={cv.mean():.4f} +/- {cv.std():.4f}")

    # ---- evaluation plots ----
    print("Generating evaluation figures...")
    for name in models:
        p = preds[name]
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, p, alpha=0.4)
        lim = [min(y_test.min(), p.min()), max(y_test.max(), p.max())]
        plt.plot(lim, lim, "r--"); plt.xlabel("Actual WIND"); plt.ylabel("Predicted WIND")
        plt.title(f"Actual vs Predicted - {name}")
        save_fig("actual_vs_pred_" + name.lower().replace(" ", "_"))
    for name in models:
        res = y_test.values - preds[name]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].scatter(preds[name], res, alpha=0.4); ax[0].axhline(0, color="r", ls="--")
        ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Residual"); ax[0].set_title(f"Residuals vs Predicted - {name}")
        sns.histplot(res, kde=True, ax=ax[1]); ax[1].set_title("Residual Distribution")
        save_fig("residuals_" + name.lower().replace(" ", "_"))

    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    plt.figure(figsize=(7, 4.5))
    imp.head(5)[::-1].plot(kind="barh", color="seagreen")
    plt.title("Top 5 Feature Importances (Random Forest)"); plt.xlabel("Importance")
    save_fig("feature_importance")

    comp = pd.DataFrame({n: results[n]["test"] for n in models}).T
    comp.plot(kind="bar", figsize=(9, 5)); plt.xticks(rotation=0)
    plt.title("Test-set Metric Comparison"); save_fig("metric_comparison")

    plt.figure(figsize=(7, 4.5))
    for n in models:
        plt.plot(range(1, 6), results[n]["cv_scores"], marker="o", label=n)
    plt.xlabel("Fold"); plt.ylabel("R2"); plt.legend(); plt.title("5-Fold Cross-Validation R2")
    save_fig("cv_scores")

    # ---- save artefacts ----
    best = max(results, key=lambda n: results[n]["test"]["R2"])
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(models["Random Forest"], f"{MODEL_DIR}/random_forest.pkl")
    joblib.dump(models["Linear Regression"], f"{MODEL_DIR}/linear_regression.pkl")
    joblib.dump(models[best], f"{MODEL_DIR}/best_model.pkl")
    meta = {"features": FEATURES, "best_model": best,
            "feature_importance": imp.to_dict(),
            "lr_coefficients": dict(zip(FEATURES, models["Linear Regression"].coef_.tolist()))}
    json.dump(meta, open(f"{MODEL_DIR}/meta.json", "w"), indent=2)
    out = {"preprocessing": prep_info, "results": results, "best_model": best,
           "n_train": int(len(X_train)), "n_test": int(len(X_test)),
           "target_mean": float(y.mean()), "target_std": float(y.std())}
    json.dump(out, open(f"{OUT_DIR}/metrics.json", "w"), indent=2, default=str)

    print("\n=== SUMMARY ===")
    for n, r in results.items():
        t = r["test"]
        print(f"{n:18s} R2={t['R2']:.4f} MAE={t['MAE']:.3f} MSE={t['MSE']:.3f} RMSE={t['RMSE']:.3f} "
              f"| train R2={r['train']['R2']:.4f} | CV={r['cv_r2_mean']:.4f}+/-{r['cv_r2_std']:.4f}")
    print(f"Best model: {best}")
    if results[best]["test"]["R2"] < 0.85:
        print("NOTE: best R2 is below the 0.85 target. Report the honest value and discuss why.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/weather.csv")
    main(ap.parse_args().data)
