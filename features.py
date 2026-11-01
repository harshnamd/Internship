"""Shared feature engineering used by BOTH train.py and app.py (avoids train/serve skew)."""
import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time", "Amount"] + V_COLS


def build_features(df, amount_scaler):
    """Return the model-ready feature frame from raw columns Time, Amount, V1..V28.

    Time_Hour   : hour of day (0-23) derived from seconds elapsed
    Time_Day    : day index (0 or 1) - the dataset spans ~48 hours
    V_Mean/V_Std/V_Abs_Sum : cross-feature summaries of the PCA components
    Amount_Scaled : RobustScaler(Amount), fitted on training data only
    """
    out = df[V_COLS].copy()
    out["Time_Hour"] = ((df["Time"] // 3600) % 24).astype(int)
    out["Time_Day"] = (df["Time"] // 86400).astype(int)
    out["V_Mean"] = df[V_COLS].mean(axis=1)
    out["V_Std"] = df[V_COLS].std(axis=1)
    out["V_Abs_Sum"] = df[V_COLS].abs().sum(axis=1)
    out["Amount_Scaled"] = amount_scaler.transform(df[["Amount"]]).ravel()
    return out


FEATURES = V_COLS + ["Time_Hour", "Time_Day", "V_Mean", "V_Std", "V_Abs_Sum", "Amount_Scaled"]
