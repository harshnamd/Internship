"""Shared preprocessing + feature engineering for CHIPGUARD (used by train.py and app.py)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

TARGET = "Personal_Loan"
FEATURE_COLUMNS = [
    "Age", "Experience", "Income", "Family", "CCAvg", "Education", "Mortgage",
    "Securities_Account", "CD_Account", "Online", "CreditCard",
    "Income_to_CC_ratio", "Income_to_Mortgage_ratio", "Exp_to_Age_ratio",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, drop non-predictive columns, add the 3 ratio features."""
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.drop(columns=[c for c in ("ID", "ZIP_Code") if c in df.columns])
    # Experience has a few negative (data-entry error) values in the Universal Bank data
    df["Experience"] = df["Experience"].clip(lower=0)
    df["Income_to_CC_ratio"] = df["Income"] / (df["CCAvg"] + 1)
    df["Income_to_Mortgage_ratio"] = df["Income"] / (df["Mortgage"] + 1)
    df["Exp_to_Age_ratio"] = df["Experience"] / (df["Age"] + 1)
    return df


class RegressionClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a regressor (Lasso / Ridge) so it behaves like a binary classifier.

    The regressor is fitted on the 0/1 target; its continuous output is the score
    (used for ROC-AUC) and a 0.5 threshold turns it into a class label.
    Needed because sklearn's roc_auc scorer requires decision_function / predict_proba.
    """

    def __init__(self, estimator=None, threshold=0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        p = np.clip(self.decision_function(X), 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= self.threshold).astype(int)

    @property
    def coef_(self):
        return self.estimator_.coef_
