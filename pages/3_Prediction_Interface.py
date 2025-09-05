import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils.inference import EngagementPredictorWrapper


st.set_page_config(page_title="Prediction", layout="wide")
st.title("🔮 Prediction Interface")

# Specify the path to your saved .pkl file
model_path = 'engagement_predictor.pkl'
predictor = EngagementPredictorWrapper("artifacts/pipeline.pkl", "artifacts/model.pkl")

# Target and features
target = "sentiment_score"
features = ['platform', 'text_content', 'hashtags', 'mentions', 'keywords', 'topic_category', 
            'sentiment_score', 'emotion_type', 'toxicity_score', 'engagement_rate', 'brand_name', 
            'product_name', 'campaign_phase', 'user_past_sentiment_avg', 'user_engagement_growth', 
            'buzz_change_rate', 'time_of_day', 'day_type', 'language_country']

st.write(f"Model is set to predict **{target}**")
st.subheader("Enter input values for each feature")
user_input = {}

# For each feature, create a numeric input. If desired, user can paste a CSV row later for batch predictions.
for f in features:
    # default value is 0.0; user can adjust
    val = st.number_input(f, value=0.0, format="%.6f")
    user_input[f] = val

if st.button("Predict"):
    X_new = pd.DataFrame([user_input])
    try:
        pred = model.predict(X_new)[0]
        st.success(f"Predicted {target}: {pred:.6f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.write("---")
st.subheader("Batch prediction (optional)")
st.markdown("Paste CSV rows (header=feature names) to run batch predictions")
text = st.text_area("""CSV input (optional)
Example:
feature1,feature2,feature3\n1.0,2.0,3.0\n4.0,5.0,6.0""", height=120)

if st.button("Predict batch"):
    if not text.strip():
        st.warning("Please paste CSV rows in the text area.")
    else:
        try:
            from io import StringIO
            df_new = pd.read_csv(StringIO(text))
            missing = [c for c in features if c not in df_new.columns]
            if missing:
                st.error(f"Input CSV missing required features: {missing}")
            else:
                preds = model.predict(df_new[features])
                out = df_new.copy()
                out[f"pred_{target}"] = preds
                st.dataframe(out)
                st.markdown("You can copy the table or export it separately from here.")
        except Exception as e:
            st.error(f"Failed to parse CSV input: {e}")
            
st.write(f"Test Predict")

# Sample data provided by the user
data = {
    'platform': ['Instagram'],
    'text_content': ['Just tried the Chromebook from Google. Best pu...'],
    'hashtags': ['#Food'],
    'mentions': [None],
    'keywords': ['price, unique, traditional, efficient'],
    'topic_category': ['Pricing'],
    'sentiment_score': [0.9826],
    'emotion_type': ['Confused'],
    'toxicity_score': [0.0376],
    'engagement_rate': [0.19319],
    'brand_name': ['Google'],
    'product_name': ['Chromebook'],
    'campaign_phase': ['Launch'],
    'user_past_sentiment_avg': [0.0953],
    'user_engagement_growth': [-0.3672],
    'buzz_change_rate': [19.1],
    'time_of_day': ['Morning'],
    'day_type': ['Weekday'],
    'language_country': ['Brazil']
}

# Create the DataFrame
new_data = pd.DataFrame(data)
new_data['mentions'] = new_data['mentions'].replace('None', np.nan)
st.write("New data for prediction:")
st.dataframe(new_data)

predictions = predictor.predict(new_data)
st.write("Predictions for new data:")
st.dataframe(predictions)