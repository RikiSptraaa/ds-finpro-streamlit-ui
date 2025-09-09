import streamlit as st

st.set_page_config(page_title="Model Training & Evaluation", layout="wide")
st.title("📂 Model Training & Evaluation")

# st.header("We build this model with step by step")
st.write("We build this mode for predicting sentiment score from social media post.")

st.header("1. Exploratory Data Analysis & Data Preproccess")
st.write("We apply some basic knowledge about EDA & Data preproccess like removing & filing null datas, removing duplicates datas, removing data anomalies, encode data and applying TF-IDF from 'text_content' column.")
st.write("Applying TF-IDF to 'text_content', 'hashtag', and 'mentions' columns will incrasing model performance to predict sentiment score.")

st.header("2. Data Training")
st.subheader("Cross Validation")
st.write("Using Cross Validation with 5 iteration spliting data test to find the best model with some regression algorithm like Linear, Lasso, Ridge, ElasticNet, and XGBRegressor.")
st.markdown(":blue-badge[Linear] :blue-badge[Lasso] :blue-badge[Ridge] :blue-badge[Elastic Net] :blue-badge[XGBRegressor]")
st.write("Scoring method we used is R2, RMSE, and MAE Scoring")
st.markdown(":blue-badge[R2] :blue-badge[RMSE] :blue-badge[MAE]")
st.subheader("Cross Validation Score Average")
st.markdown(":blue-badge[Linear] -> R²: 0.859488161470313, RMSE: 0.14058558947119745, MAE: 0.31631483683876993")
st.markdown(":blue-badge[Lasso] -> R²: -0.00016444947496010442, RMSE: 1.000594112563496, MAE: 0.8696194420766259")
st.markdown(":green-badge[Ridge] -> R²: 0.8595111533491959, RMSE: 0.1405628660110705, MAE: 0.3163107026168273")
st.markdown(":blue-badge[ElasticNet] -> R²: 0.859488161470313, RMSE: 0.14058558947119745, MAE: 0.31631483683876993")
st.markdown(":blue-badge[XGBRegressor] -> R²: 0.848188098884823, RMSE: 0.15188176042952023, MAE: 0.3257178279570986")
st.markdown("From the score above we know the best method is :green-badge[Ridge Regression]")

st.subheader("Hyper Parameter Tuning (RidgeCV)")
st.markdown("""
RidgeCV is a regression model in scikit-learn that combines Ridge Regression with built-in cross-validation to automatically choose the best regularization parameter alpha.
Instead of manually testing different alpha values with Ridge + GridSearchCV, you can directly use RidgeCV, which is optimized and more convenient.

Result:   
* Best alpha (RidgeCV): 100.0
* Performance of the Best Ridge Model on the Test Set:
RidgeCV
* R²: 0.8610
* RMSE: 0.3724
* MAE: 0.3118
            
RidgeCV is best use when:

*   We want to focus about tuning α
*   We want efficiency (especially for large sparse TF-IDF data)
* less time consuming

**Rationalization for (TF-IDF + sentiment regression):**


Since the main hyperparameter for Ridge is α, and solvers don’t usually change performance much (only efficiency), RidgeCV is the right tool: faster, cleaner, and designed for exactly what you’re doing.
 """)