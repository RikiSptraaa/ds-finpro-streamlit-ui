import joblib
import numpy as np

class EngagementPredictorWrapper:
    """
    Safe wrapper for inference: loads pipeline + model separately
    """
    def __init__(self, pipeline_path, model_path):
        self.pipeline = joblib.load(pipeline_path)
        self.model = joblib.load(model_path)

    def predict_proba(self, df):
        X_ready = self.pipeline.transform(df)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_ready)[:, 1]
        else:
            # fallback if model has no predict_proba
            preds = self.model.predict(X_ready)
            return preds.astype(float)

    def predict(self, df, threshold=0.5):
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(int)
