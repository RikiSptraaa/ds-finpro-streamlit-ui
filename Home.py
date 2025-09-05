# Home.py
import streamlit as st

st.set_page_config(page_title="Social Media Sentiment Score Predictor", layout="wide")

# Title & Intro
st.title("📊 Digital Skola Bootcamp — Batch 50")
st.subheader("Final Project: Social Media Sentiment Score Predictor")

st.write(
    "Welcome! This app showcases our **end-to-end data science project** "
    "using the *Kaggle Social Media Engagement dataset*."
)

# Group Members
st.markdown("---")
st.subheader("👥 Group 2 — Data Fellas")

st.markdown(
"""
- **Riki Eprilion Saputra**  
- **Delima Nabila Cahyani**  
- **Fairuz Ramadhan**  
- **Fais Ermansyah**  
- **Yolanda Rizki Sinaga**  
- **Mohammad Erwinsyah Hartono**  
- **Rafif Imaduddin Yudono**  
"""
)
 
# Objective
st.markdown("---")
st.subheader("🎯 Project Objective")

st.write(
    "We aim to build a predictive model for **`sentiment_score`**, "
    "while practicing the full **data science lifecycle**:."
)

st.markdown(
"""
1. **Data Collection & Understanding**  
2. **Data Cleaning & Preprocessing**  
3. **Exploratory Data Analysis (EDA)**  
4. **Feature Engineering**  
5. **Model Training & Evaluation**  
6. **Deployment & Prediction**  
"""
)

# App Structure
st.markdown("---")
st.subheader("🛠️ App Structure")

st.markdown(
"""
- **📊 Data Overview** — dataset preview, summary stats, missing values  
- **⚙️ Model Training & Evaluation** — feature selection, Random Forest training, R² score, feature importance  
- **🎯 Prediction** — load saved model & predict sentiment for new inputs  
"""
)

# Dataset Overview
st.markdown("---")
st.subheader("📑 About the Dataset")

st.write(
    "This dataset simulates social media activity across metrics like "
    "**likes, shares, comments, impressions, sentiment, toxicity, and engagement growth.**"
)

st.markdown(
"""
**It is useful for:**
- 📈 Detecting spikes or drops in engagement  
- 💬 Tracking sentiment shifts over time  
- 📊 Building dashboards for digital trends  
- 🤖 Testing algorithms for sentiment prediction  

🔗 **Resources**  
- [Kaggle Dataset](https://www.kaggle.com/datasets/subashmaster0411/social-media-engagement-dataset)  
- [Project Notebook](https://colab.research.google.com/drive/19kl45Dbvse0dTwEN9dFpe5aypCbtxywU?usp=sharing)  
"""
)

# Get Started
st.markdown("---")
st.info("👉 Start by selecting **Data Overview** from the sidebar.")
