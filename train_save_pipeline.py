# train_and_save_pipeline.py
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from utils.custom_transformers import DateTimeFeatures, TextCleaner

# --- adjust paths ---
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# === columns based on your raw schema ===
text_columns = ["text_content", "mentions", "hashtags"]
ordinal_columns = ["day_of_week", "sentiment_label", "emotion_type", "campaign_phase"]
ordinal_categories = [
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    ["Negative", "Neutral", "Positive"],
    ["Confused", "Angry", "Sad", "Happy", "Excited"],
    ["Pre-Launch", "Launch", "Post-Launch"],
]

numeric_columns = [
    "toxicity_score", "likes_count", "shares_count",
    "comments_count", "impressions", "engagement_rate",
    "user_past_sentiment_avg", "user_engagement_growth", "buzz_change_rate"
]

categorical_columns = [
    "platform", "location", "language", "topic_category",
    "brand_name", "product_name", "campaign_name"
]

drop_columns = ["user_id", "post_id", "keywords"]

# --- build text transformers for ColumnTransformer ---
text_transformers = [
    ("tfidf_text_content", Pipeline([
        ("clean_text", TextCleaner(mode="text")),
        ("tfidf", TfidfVectorizer(max_features=3000, stop_words="english", token_pattern=r"(?u)\b\w+\b"))
    ]), "text_content"),

    ("tfidf_mentions", Pipeline([
        ("clean_mentions", TextCleaner(mode="mentions")),
        ("tfidf", TfidfVectorizer(max_features=1000, stop_words=None, token_pattern=r"(?u)\b\w+\b"))
    ]), "mentions"),

    ("tfidf_hashtags", Pipeline([
        ("clean_hashtags", TextCleaner(mode="hashtags")),
        ("tfidf", TfidfVectorizer(max_features=1000, stop_words=None, token_pattern=r"(?u)\b\w+\b"))
    ]), "hashtags"),
]

preprocessor = ColumnTransformer(
    transformers=[
        ("datetime", Pipeline([("dt", DateTimeFeatures("timestamp")), ("scaler", StandardScaler())]), ["timestamp"]),
        ("ordinals", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1), ordinal_columns),
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        *text_transformers,
    ],
    remainder="drop"
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])

# === Load raw df (replace with your path or dataframe) ===
df = pd.read_csv("data/dataset.csv")   # or however you load df

# target
y = df["sentiment_score"]
X = df.drop(columns=["sentiment_score"] + drop_columns, errors="ignore")

# Fit & Save
final_pipeline.fit(X, y)
joblib.dump(final_pipeline, os.path.join(ARTIFACT_DIR, "final_pipeline.joblib"))
print("Saved pipeline ->", os.path.join(ARTIFACT_DIR, "final_pipeline.joblib"))
print("Training complete.")