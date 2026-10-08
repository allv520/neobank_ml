from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

try:
    model = joblib.load(os.path.join(MODELS_DIR, "logreg_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print("✅ Модель загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    raise

FEATURE_COLS = [
    "income",
    "expense",
    "debt_ratio",
    "balance",
    "active_cards",
    "days_on_platform",
    "closed_credits",
    "has_overdue",
]

app = FastAPI(title="NEOBANK ML Scoring Service", version="1.0.0")


class Features(BaseModel):
    income: float
    expense: float
    debt_ratio: float
    balance: float
    active_cards: int
    days_on_platform: int
    closed_credits: int
    has_overdue: int


@app.get("/")
def root():
    return {
        "service": "NEOBANK ML Scoring",
        "status": "ok",
        "model": "LogisticRegression",
    }


@app.post("/predict")
def predict(features: Features):
    try:
        X = np.array(
            [
                [
                    features.income,
                    features.expense,
                    features.debt_ratio,
                    features.balance,
                    features.active_cards,
                    features.days_on_platform,
                    features.closed_credits,
                    features.has_overdue,
                ]
            ]
        )
        X_scaled = scaler.transform(X)
        proba = float(model.predict_proba(X_scaled)[0, 1])
        rate = round(5 + proba * 20, 2)
        return {
            "default_proba": round(proba, 4),
            "rate": rate,
            "model": "LogisticRegression",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
