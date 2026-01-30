from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("logistic_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

class ChurnRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: ChurnRequest):
    df = pd.DataFrame([data.model_dump()])
    X = pd.get_dummies(df, drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    return {
        "prediction": pred,
        "churn_probability": round(float(prob), 4)
    }
