import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
df = df.dropna(subset=["risk_category"])

# 1. Risk Category Distribution
fig1 = px.histogram(df, x="risk_category", color="risk_category", title="📊 Risk Category Distribution")

# 2. Debt Ratio vs Default Probability
fig2 = px.scatter(df, x="DebtRatio", y="default_probability", color="risk_category",
                  hover_data=["MonthlyIncome", "NumberOfDependents"],
                  title="📈 Debt Ratio vs Default Probability")

# 3. Feature Impact
feature_impact_df = pd.DataFrame({
    "Feature": ["DebtRatio", "RevolvingUtilizationOfUnsecuredLines", "NumberOfTimes90DaysLate"],
    "Impact": [0.3, 0.25, 0.2]
})
fig3 = px.bar(feature_impact_df, x="Impact", y="Feature", orientation="h", title="💡 Top Contributing Features")

# 4. Correlation Heatmap
corr = df.corr(numeric_only=True)
fig4 = go.Figure(data=go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.columns,
    colorscale='RdBu', zmin=-1, zmax=1
))
fig4.update_layout(title="📌 Feature Correlation Heatmap")

# Show plots
fig1.show()
fig2.show()
fig3.show()
fig4.show()
