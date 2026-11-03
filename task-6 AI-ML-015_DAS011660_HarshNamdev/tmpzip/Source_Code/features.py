"""Shared preprocessing + feature engineering (25 engineered + 11 original = 36 features).
Used by BOTH train.py and app.py so training and serving can never drift apart."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CAT_CLASSES = {"Gender": ["Female", "Male"], "Married": ["No", "Yes"], "Education": ["Graduate", "Not Graduate"],
               "Self_Employed": ["No", "Yes"], "Property_Area": ["Rural", "Semiurban", "Urban"]}
DEPENDENTS = ["0", "1", "2", "3+"]
RAW_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome",
            "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"]
NUMERIC_RAW = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
CATEGORICAL_RAW = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

ENGINEERED = ["total_income", "income_diff", "income_ratio", "income_log", "applicant_income_log",
              "coapplicant_income_log", "loan_log", "loan_sqrt", "loan_to_income_ratio", "loan_to_total_income",
              "credit_with_income", "credit_with_loan", "loan_term_years", "has_dependents", "has_coapplicant",
              "emi", "emi_to_income", "balance_income", "income_per_dependent", "family_size",
              "income_per_family_member", "credit_with_emi", "edu_credit", "semiurban_flag", "long_term_loan"]
FEATURES = RAW_COLS + ENGINEERED


def make_encoders():
    """LabelEncoders (as required by the task) fitted on the known category lists."""
    return {c: LabelEncoder().fit(v) for c, v in CAT_CLASSES.items()}


def fit_impute_stats(train_df):
    """Median for numeric, mode for categorical - learned from TRAINING data only."""
    s = {c: float(train_df[c].median()) for c in NUMERIC_RAW}
    s.update({c: str(train_df[c].mode()[0]) for c in CATEGORICAL_RAW})
    return s


def prepare(df, stats, encoders):
    """Raw applicant columns -> 36-feature model matrix."""
    d = df[RAW_COLS].copy()
    for c in CATEGORICAL_RAW:
        d[c] = d[c].astype(object).where(d[c].notna(), stats[c]).astype(str).str.strip()
    for c in NUMERIC_RAW:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(stats[c])
    d["Dependents"] = d["Dependents"].replace({"3": "3+"})
    bad = d.loc[~d["Dependents"].isin(DEPENDENTS), "Dependents"].unique()
    if len(bad):
        raise ValueError(f"Unknown Dependents value(s): {list(bad)} (use 0, 1, 2 or 3+)")
    d["Dependents"] = d["Dependents"].map({"0": 0, "1": 1, "2": 2, "3+": 3}).astype(int)
    for c, enc in encoders.items():
        unk = set(d[c].unique()) - set(enc.classes_)
        if unk:
            raise ValueError(f"Unknown {c} value(s): {sorted(unk)}; allowed: {list(enc.classes_)}")
        d[c] = enc.transform(d[c])

    a, c_, L, t, ch = d["ApplicantIncome"], d["CoapplicantIncome"], d["LoanAmount"], d["Loan_Amount_Term"], d["Credit_History"]
    f = d.copy()
    f["total_income"] = a + c_
    f["income_diff"] = a - c_
    f["income_ratio"] = a / (c_ + 1)
    f["income_log"] = np.log1p(f["total_income"])
    f["applicant_income_log"] = np.log1p(a)
    f["coapplicant_income_log"] = np.log1p(c_)
    f["loan_log"] = np.log1p(L)
    f["loan_sqrt"] = np.sqrt(L)
    f["loan_to_income_ratio"] = L * 1000 / (a + 1)             # LoanAmount is in thousands
    f["loan_to_total_income"] = L * 1000 / (f["total_income"] + 1)
    f["credit_with_income"] = ch * f["income_log"]
    f["credit_with_loan"] = ch * f["loan_log"]
    f["loan_term_years"] = t / 12
    f["has_dependents"] = (d["Dependents"] > 0).astype(int)
    f["has_coapplicant"] = (c_ > 0).astype(int)
    f["emi"] = L * 1000 / t.clip(lower=1)                      # simple monthly instalment (no interest)
    f["emi_to_income"] = f["emi"] / (f["total_income"] + 1)
    f["balance_income"] = f["total_income"] - f["emi"]
    f["income_per_dependent"] = f["total_income"] / (d["Dependents"] + 1)
    f["family_size"] = d["Married"] + 1 + d["Dependents"]
    f["income_per_family_member"] = f["total_income"] / f["family_size"]
    f["credit_with_emi"] = ch * f["emi_to_income"]
    f["edu_credit"] = (d["Education"] == 0).astype(int) * ch   # graduate x credit history
    f["semiurban_flag"] = (d["Property_Area"] == 1).astype(int)
    f["long_term_loan"] = (t >= 360).astype(int)
    return f[FEATURES]
