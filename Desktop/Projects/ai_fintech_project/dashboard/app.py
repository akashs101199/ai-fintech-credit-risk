import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# --- Simulated Dataset for Visuals ---
np.random.seed(42)
df = pd.DataFrame({
    "default_probability": np.clip(np.random.normal(0.5, 0.2, 100), 0, 1),
    "DebtRatio": np.random.beta(2, 5, 100),
    "MonthlyIncome": np.random.normal(5000, 1500, 100).astype(int),
    "NumberOfOpenCreditLinesAndLoans": np.random.randint(1, 15, 100),
    "RevolvingUtilizationOfUnsecuredLines": np.random.uniform(0.1, 1.5, 100),
    "NumberOfTime30_59DaysPastDueNotWorse": np.random.randint(0, 4, 100),
    "NumberOfTimes90DaysLate": np.random.randint(0, 3, 100),
    "NumberOfDependents": np.random.randint(0, 5, 100)
})
df["risk_category"] = pd.cut(
    df["default_probability"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)
df = df.dropna()

# --- Sidebar Form ---
st.sidebar.header("📋 Enter Customer Info")
with st.sidebar.form("user_input"):
    RevolvingUtilizationOfUnsecuredLines = st.slider("Revolving Utilization", 0.0, 2.0, 0.5)
    age = st.slider("Age", 18, 100, 45)
    NumberOfTime30_59DaysPastDueNotWorse = st.slider("30-59 Days Past Due", 0, 10, 1)
    DebtRatio = st.slider("Debt Ratio", 0.0, 5.0, 0.4)
    MonthlyIncome = st.number_input("Monthly Income", 0, 20000, 6000)
    NumberOfOpenCreditLinesAndLoans = st.slider("Open Credit Lines", 0, 20, 5)
    NumberOfTimes90DaysLate = st.slider("90 Days Late", 0, 10, 0)
    NumberRealEstateLoansOrLines = st.slider("Real Estate Loans", 0, 10, 1)
    NumberOfTime60_89DaysPastDueNotWorse = st.slider("60-89 Days Past Due", 0, 10, 0)
    NumberOfDependents = st.slider("Dependents", 0, 10, 2)
    submitted = st.form_submit_button("🧠 Predict Risk")

# --- Header ---
st.title("🔍 Credit Risk Scoring Dashboard")
st.markdown("A simple, intuitive dashboard that estimates customer default probability and explains the risk visually.")

# --- Prediction API Call ---
if submitted:
    input_data = {
        "RevolvingUtilizationOfUnsecuredLines": RevolvingUtilizationOfUnsecuredLines,
        "age": age,
        "NumberOfTime30_59DaysPastDueNotWorse": NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": DebtRatio,
        "MonthlyIncome": MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": NumberRealEstateLoansOrLines,
        "NumberOfTime60_89DaysPastDueNotWorse": NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": NumberOfDependents,
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data).json()
        st.success(f"🧾 Default Probability: **{response['default_probability']:.2f}**")

        # Gauge Chart
        st.markdown("### 🧮 Risk Score Gauge")
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=response['default_probability'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Risk Level", 'font': {'size': 22}},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0.0, 0.3], 'color': "green"},
                    {'range': [0.3, 0.6], 'color': "orange"},
                    {'range': [0.6, 1.0], 'color': "red"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': response['default_probability']
                }
            }
        ))
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Top Features
        st.markdown("### 🔍 Top Contributing Features")
        features_df = pd.DataFrame(response["top_features"])
        fig_feat = px.bar(features_df, x="impact", y="feature", orientation="h", title="Feature Impact")
        st.plotly_chart(fig_feat, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ API connection failed: {e}")

# --- Data Visualizations ---
st.markdown("---")
st.markdown("## 📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x="risk_category", color="risk_category", title="Risk Category Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="DebtRatio", y="default_probability", color="risk_category",
                      title="Debt Ratio vs Default Probability",
                      hover_data=["MonthlyIncome", "NumberOfDependents"])
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.box(df, x="risk_category", y="MonthlyIncome", color="risk_category",
                  title="Income Distribution by Risk")
    st.plotly_chart(fig3, use_container_width=True)

    corr = df.corr(numeric_only=True)
    fig4 = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmin=-1, zmax=1
    ))
    fig4.update_layout(title="Feature Correlation Heatmap")
    st.plotly_chart(fig4, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.caption("📈 Powered by FastAPI + LightGBM + Streamlit + Plotly")
