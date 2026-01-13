from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ---------------- LOAD MODELS ----------------
model_m1 = joblib.load("COPD_M1_randomforest.joblib")
model_m2 = joblib.load("COPD_M2_gb_regressor.joblib")
model_m3 = joblib.load("COPD_M3_risk_model.joblib")

app = FastAPI(title="COPD Prediction API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- INPUT SCHEMA ----------------
class PatientInput(BaseModel):
    Age: int
    Gender: str
    SmokingStatus: str
    SmokingYears: int
    BMI: float
    FEV1_pct: float
    FVC_pct: float
    FEV1_FVC: float
    ChronicCough: int
    CRP_mg_L: float
    SpO2: float

# ---------------- API ENDPOINT ----------------
@app.post("/predict")
def predict_copd(data: PatientInput):

    df = pd.DataFrame([data.dict()])

    # MODEL 1
    copd_pred = int(model_m1.predict(df)[0])
    copd_prob = float(model_m1.predict_proba(df)[0][1])

    # MODEL 2
    severity = float(model_m2.predict(df)[0]) if copd_pred == 1 else None

    # MODEL 3
    future_risk = float(model_m3.predict_proba(df)[0][1])

    return {
    "copd_prediction": copd_pred,
    "copd_probability": round(copd_prob * 100, 2),
    "severity_percent": round(severity, 2) if severity is not None else None,
    "future_copd_risk": round(future_risk * 100, 2)
}

