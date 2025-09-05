# utils/custom_transformers.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, column="timestamp"):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=[self.column])

        ts = pd.to_datetime(df[self.column], errors="coerce")
        out = pd.DataFrame({
            "year":  ts.dt.year.fillna(0).astype(int),
            "month": ts.dt.month.fillna(0).astype(int),
            "day":   ts.dt.day.fillna(0).astype(int),
            "hour":  ts.dt.hour.fillna(0).astype(int),
        }, index=df.index)
        return out


class TextCleaner(BaseEstimator, TransformerMixin):
    """Column-aware text cleaning: returns 1D array of strings for TF-IDF."""
    def __init__(self, mode="text"):  # <-- this must exist
        assert mode in {"text", "mentions", "hashtags"}
        self.mode = mode
        self.regex_pattern = r"[^\w\s]|[\d]"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).squeeze(), dtype=str)
        s = s.fillna("").str.lower()

        if self.mode == "text":
            s = s.str.replace(r"@\w+", " ", regex=True)
            s = s.str.replace(r"#\w+", " ", regex=True)
            s = s.str.replace(self.regex_pattern, " ", regex=True)
        elif self.mode == "mentions":
            s = s.str.replace("@", " ", regex=False)
            s = s.str.replace(r"[^\w\s]", " ", regex=True)
        elif self.mode == "hashtags":
            s = s.str.replace("#", " ", regex=False)
            s = s.str.replace(r"[^\w\s]", " ", regex=True)

        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        s = s.replace("", "__empty__")
        return s.to_numpy()
