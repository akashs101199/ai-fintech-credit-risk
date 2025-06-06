from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

# Initialize FastAPI app
app = FastAPI(title="Credit Risk Scoring API", version="0.1.0")

# Define input schema
class CustomerData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int

# Map underscore names to model's hyphenated feature names
FIELD_NAME_MAP = {
    "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse"
}

# Load model and explainer
model = joblib.load("models/credit_risk_model.pkl")
explainer = shap.TreeExplainer(model)

@app.post("/predict")
def predict_risk(data: CustomerData):
    try:
        # Convert to dict and remap column names
        input_dict = data.dict()
        for old_key, new_key in FIELD_NAME_MAP.items():
            input_dict[new_key] = input_dict.pop(old_key)

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[model.feature_name_]  # Ensure column order

        # Make prediction
        probability = model.predict_proba(input_df)[0][1]

        # Get SHAP values and top features
        shap_values = explainer.shap_values(input_df)
        top_features = sorted(
            zip(input_df.columns, shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        return {
            "default_probability": round(probability, 4),
            "top_features": [
                {"feature": name, "impact": round(impact, 4)} for name, impact in top_features
            ]
        }

    except Exception as e:
        return {"error": str(e)}
