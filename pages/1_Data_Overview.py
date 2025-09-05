import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Data Overview", layout="wide")
st.title("📂 Data Overview")

DATA_PATH = Path("data/dataset.csv")

@st.cache_data
def load_data(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df

df = load_data(DATA_PATH)

if df is None:
    st.error(f"No dataset found at {DATA_PATH}. Please add your CSV and reload.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(150))

st.subheader("Basic Info")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.write("Column types:")
st.write(df.dtypes)

st.subheader("Summary Statistics (numeric)")
st.write(df.describe())

st.subheader("Missing Values")
missing = df.isna().sum()
st.dataframe(missing[missing > 0].sort_values(ascending=False))

# Quick visualizations
st.subheader("Quick Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.write("Histogram for a numeric column")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        sel = st.selectbox("Choose numeric column", numeric_cols, key="hist_col")
        st.bar_chart(df[sel].dropna().sample(min(5000, len(df))).reset_index(drop=True))
    else:
        st.info("No numeric columns found.")

with col2:
    st.write("Counts for a categorical column")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        sel2 = st.selectbox("Choose categorical column", cat_cols, key="cat_col")
        st.write(df[sel2].value_counts().head(50))
    else:
        st.info("No categorical columns found.")