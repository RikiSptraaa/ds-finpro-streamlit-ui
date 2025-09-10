# utils/custom_transformers.py
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack

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

class Preproccess(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df = pd.read_csv("data/dataset.csv")  

        # Load fitted transformers only once
        self.tfidf_text = joblib.load("artifacts/tfidf_text.joblib")
        self.tfidf_mentions = joblib.load("artifacts/tfidf_mentions.joblib")
        self.tfidf_hashtags = joblib.load("artifacts/tfidf_hashtags.joblib")
        self.ordinal_encoder = joblib.load("artifacts/ordinal_encoder.joblib")
        self.ohe_columns = joblib.load("artifacts/ohe_columns.joblib")
        self.dfc_columns = joblib.load("artifacts/dfc_columns.joblib")
        self.scaler = joblib.load("artifacts/scaler.joblib")

        # Create separate cleaners for each type
        self.cleaner_text = TextCleaner(mode="text")
        self.cleaner_mentions = TextCleaner(mode="mentions")
        self.cleaner_hashtags = TextCleaner(mode="hashtags")

        self.text_columns = ["text_content", "mentions", "hashtags"]
        self.ordinal_columns = ["day_of_week", "sentiment_label", "emotion_type", "campaign_phase"]
        self.ordinal_categories = [
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            ["Negative", "Neutral", "Positive"],
            ["Confused", "Angry", "Sad", "Happy", "Excited"],
            ["Pre-Launch", "Launch", "Post-Launch"],
        ]

        # self.numeric_columns = [
        #     "toxicity_score", "likes_count", "shares_count",
        #     "comments_count", "impressions", "engagement_rate",
        #     "user_past_sentiment_avg", "user_engagement_growth", "buzz_change_rate"
        # ]

        self.decimal_cols = ['day_of_week', 'sentiment_label', 'emotion_type', 'likes_count',
            'shares_count', 'comments_count', 'impressions', 'campaign_phase',
            'year', 'month', 'day', 'hour']

        self.categorical_columns = [
            "platform", "location", "language", "topic_category",
            "brand_name", "product_name", "campaign_name"
        ]

        self.drop_columns = ["user_id", "post_id", "keywords"]

    def fit(self, X, y=None):
        return self
    
    def tfidf_transform(self, X):
         # Clean per column with the right cleaner
        X["text_content"] = self.cleaner_text.transform(X["text_content"])
        X["mentions"] = self.cleaner_mentions.transform(X["mentions"])
        X["hashtags"] = self.cleaner_hashtags.transform(X["hashtags"])

        # Transform with pre-trained TF-IDF vectorizers
        tfidf_text_result = self.tfidf_text.transform(X["text_content"])
        tfidf_mentions_result = self.tfidf_mentions.transform(X["mentions"])
        tfidf_hashtags_result = self.tfidf_hashtags.transform(X["hashtags"])

        # Combine all TF-IDF outputs
        X_tfidf = hstack([tfidf_text_result, tfidf_mentions_result, tfidf_hashtags_result])

        # Feature names
        feature_names = (
            [f"text_{w}" for w in self.tfidf_text.get_feature_names_out()] +
            [f"mentions_{w}" for w in self.tfidf_mentions.get_feature_names_out()] +
            [f"hashtag_{w}" for w in self.tfidf_hashtags.get_feature_names_out()]
        )

        # Return DataFrame
        df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=feature_names)
        return df_tfidf
    
    def oe_transform(self, X):
        # X_oe = X[self.ordinal_columns]
        X[self.ordinal_columns]= self.ordinal_encoder.transform(X[self.ordinal_columns])

        # df_oe = pd.DataFrame(X_oe, columns=self.ordinal_columns)
        X[self.ordinal_columns] = X[self.ordinal_columns].astype(int)
        return X

    def ohe_transform(self, X):

        # One-hot encode new data
        df_ohe = pd.get_dummies(self.df, columns=self.categorical_columns)

        inputed_ohe = pd.get_dummies(X, columns=self.categorical_columns)
        
        # Reindex to match training
        df_inputed_ohe = inputed_ohe.reindex(columns=df_ohe.columns, fill_value=False)

        return df_inputed_ohe
    
    def date_transform(self, X):
        X["timestamp"] = pd.to_datetime(X["timestamp"])

        X["year"] = X["timestamp"].dt.year.astype(int)
        X["month"] = X["timestamp"].dt.month.astype(int)
        X["day"] = X["timestamp"].dt.day.astype(int)
        X["hour"] = X["timestamp"].dt.hour.astype(int)

        X = X.drop(["timestamp"], axis=1)

        return X



    def transform(self, X):
        X = X.copy()

        df_tfidf = self.tfidf_transform(X)
        X = self.oe_transform(X)
        df_ohe = self.ohe_transform(X)
        
        # # X.drop(self.ordinal_columns, axis=1)
        X = X.drop(self.categorical_columns, axis=1)

        X = pd.merge(X, df_tfidf, left_index=True, right_index=True)
        # X = pd.concat([X, df_oe])
        X = pd.merge(X, df_ohe)
        X = self.date_transform(X)
        X = X.drop(self.text_columns, axis=1)
        X = X.drop(self.drop_columns, axis=1)
        X[self.decimal_cols] = X[self.decimal_cols].astype(int)
        X["sentiment_score"] = X["sentiment_score"].astype(float)

        X = X[self.dfc_columns]
        X = self.scaler.transform(X)
        X = pd.DataFrame(X, columns=self.dfc_columns)

        X = X.drop(["sentiment_score"], axis=1)
        return X
