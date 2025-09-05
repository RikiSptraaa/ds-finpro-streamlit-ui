# engagement_predictor.py

import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sentence_transformers import SentenceTransformer

# --------------------------
# Custom transformer for structured text features
# --------------------------
class TextStructuredFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, text_cols=None):
        if text_cols is None:
            text_cols = ['text_content', 'hashtags', 'keywords', 'mentions', 'product_name']
        self.text_cols = text_cols

    @staticmethod
    def split_and_clean(x):
        if pd.isna(x) or str(x).lower() == "none":
            return []
        return [token.strip().lower() for token in str(x).split(",")]

    @staticmethod
    def safe_text(x):
        if pd.isna(x) or str(x).lower() == "none":
            return ""
        return str(x)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Structured counts
        X["hashtag_count"] = X["hashtags"].apply(self.split_and_clean).apply(len)
        X["keyword_count"] = X["keywords"].apply(self.split_and_clean).apply(len)
        X["mention_count"] = X["mentions"].apply(self.split_and_clean).apply(len)
        X["text_length"] = X["text_content"].fillna("").apply(len)

        # Combine text columns for embeddings
        X["combined_text"] = (
            X["text_content"].apply(self.safe_text) + " " +
            X["hashtags"].apply(self.safe_text) + " " +
            X["keywords"].apply(self.safe_text) + " " +
            X["mentions"].apply(self.safe_text) + " " +
            X["product_name"].apply(self.safe_text)
        ).str.replace(r"\s+", " ", regex=True).str.strip()

        # Return structured numerical features + combined_text
        structured_features = X[["hashtag_count","keyword_count","mention_count","text_length"]].values
        return structured_features, X["combined_text"]

# --------------------------
# Custom transformer for embeddings
# --------------------------
class TextEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        embeddings = self.model.encode(X.tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

# --------------------------
# Full pipeline
# --------------------------
class FullTextPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols, numerical_cols, text_cols=None):
        self.text_structured = TextStructuredFeatures(text_cols)
        self.text_embedding = TextEmbeddingTransformer()
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.num_scaler = StandardScaler()
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X, y=None):
        structured_features, combined_text = self.text_structured.transform(X)
        self.num_scaler.fit(np.hstack([structured_features, X[self.numerical_cols].values]))
        self.cat_encoder.fit(X[self.categorical_cols])
        self.text_embedding.fit(combined_text)
        return self

    def transform(self, X):
        structured_features, combined_text = self.text_structured.transform(X)
        num_features = self.num_scaler.transform(np.hstack([structured_features, X[self.numerical_cols].values]))
        cat_features = self.cat_encoder.transform(X[self.categorical_cols])
        text_embeds = self.text_embedding.transform(combined_text)
        X_final = np.hstack([num_features, cat_features, text_embeds])
        return X_final

# --------------------------
# Wrapper for pipeline + model
# --------------------------
class EngagementPredictor:
    def __init__(self, pipeline, model):
        self.pipeline = pipeline
        self.model = model

    def predict(self, df):
        X_ready = self.pipeline.transform(df)
        return self.model.predict(X_ready)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        import sys
        from utils import engagement_predictor

        # Patch all custom classes so pickle can resolve them
        sys.modules['__main__'].EngagementPredictor = engagement_predictor.EngagementPredictor
        sys.modules['__main__'].FullTextPipeline = engagement_predictor.FullTextPipeline
        sys.modules['__main__'].TextStructuredFeatures = engagement_predictor.TextStructuredFeatures
        sys.modules['__main__'].TextEmbeddingTransformer = engagement_predictor.TextEmbeddingTransformer

        with open(filepath, "rb") as f:
            return pickle.load(f)

