"""Shared feature engineering for train.py and app.py (prevents train/serve skew)."""
import numpy as np
import pandas as pd

AGE_BINS = [0, 25, 35, 45, 55, 65, 200]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
PREMIUM_LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]
VINTAGE_LABELS = ["Short", "Medium", "Long", "Very Long"]

BASE_CATS = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
GROUP_CATS = ["Age_Group", "Premium_Group", "Vintage_Group"]
NUMS = ["Age", "Driving_License", "Previously_Insured", "Annual_Premium", "Vintage", "Age_Premium_Ratio"]
CODE_COLS = ["Region_Code", "Policy_Sales_Channel"]   # optional high-cardinality codes (--include-codes)


def fit_bins(train_df):
    """Quantile edges for Premium_Group / Vintage_Group, learned from TRAINING data only."""
    return {"premium": train_df["Annual_Premium"].quantile([.2, .4, .6, .8]).tolist(),
            "vintage": train_df["Vintage"].quantile([.25, .5, .75]).tolist()}


def _cut(x, edges, labels):
    return pd.cut(x, [-np.inf] + list(edges) + [np.inf], labels=labels).astype(str)


def add_features(df, bins, include_codes=False):
    out = df.copy()
    out["Age_Group"] = pd.cut(out["Age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    out["Premium_Group"] = _cut(out["Annual_Premium"], bins["premium"], PREMIUM_LABELS)
    out["Vintage_Group"] = _cut(out["Vintage"], bins["vintage"], VINTAGE_LABELS)
    out["Age_Premium_Ratio"] = out["Age"] / out["Annual_Premium"].clip(lower=1)
    if include_codes:
        for c in CODE_COLS:
            out[c] = out[c].astype(int).astype(str)
    return out


def feature_lists(include_codes):
    cats = BASE_CATS + GROUP_CATS
    return cats, NUMS, (CODE_COLS if include_codes else [])


def raw_input_cols(include_codes):
    cols = ["Gender", "Age", "Driving_License", "Previously_Insured", "Vehicle_Age",
            "Vehicle_Damage", "Annual_Premium", "Vintage"]
    return cols + (CODE_COLS if include_codes else [])
