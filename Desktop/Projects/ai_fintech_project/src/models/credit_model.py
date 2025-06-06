import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import shap
import joblib
import os

# Load data
df = pd.read_csv("data/credit/cs-training.csv", index_col=0)
df = df.rename(columns=lambda x: x.strip())  # Remove whitespace from column names

# Handle missing values
df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
df["NumberOfDependents"].fillna(0, inplace=True)

# Split
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/credit_risk_model.pkl")

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap_df = pd.DataFrame(shap_values[1], columns=X.columns)
shap_df["prediction"] = y_prob
shap_df["actual"] = y_test.values

shap_df.to_csv("data/credit/shap_outputs.csv", index=False)
print("✅ Model trained and SHAP values saved.")
