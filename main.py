import joblib
import logging 
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feedback import log_prediction, report_outcome, check_accuracy, check_for_drift

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.FileHandler("a_main.log", mode="a"),
                    logging.StreamHandler()])

model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

logger = logging.getLogger(__name__)
logger.info("Server started")
logger.info("Model loaded successfully")

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
    Time:  float
    V1:    float
    V2:    float
    V3:    float
    V4:    float
    V5:    float
    V6:    float
    V7:    float
    V8:    float
    V9:    float
    V10:   float
    V11:   float
    V12:   float
    V13:   float
    V14:   float
    V15:   float
    V16:   float
    V17:   float
    V18:   float
    V19:   float
    V20:   float
    V21:   float
    V22:   float
    V23:   float
    V24:   float
    V25:   float
    V26:   float
    V27:   float
    V28:   float
    Amount: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        data = pd.DataFrame([transaction.model_dump()])

        # same scaler fitted on training data — never refit here
        data[["Time", "Amount"]] = scaler.transform(data[["Time", "Amount"]])

        prediction  = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        record_id = log_prediction(
            input_data=transaction.model_dump(),
            predicted_risk=float(probability),
            model_used="rf",
        )

        return {
            "record_id": record_id,
            "fraud":       bool(prediction),
            "probability": round(float(probability), 4),
            "risk":        "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/report/{record_id}")
def report(record_id: str, occurred: bool):
    result = report_outcome(record_id, occurred)

    if result == "not_found":
        raise HTTPException(status_code=404, detail="Record not found")
    if result == "already_reported":
        raise HTTPException(status_code=409, detail="Already reported")

    return {"message": "Outcome recorded"}

@app.get("/accuracy")
def accuracy():
    result = check_accuracy()
    logger.info(f"Accuracy checked - Brier score: {result['brier_score']}")
    return result

@app.get("/drift")
def drift():
    result = check_for_drift()
    if result["drift_detected"]:
        logger.warning(f"DRIFT DETECTED - {result['percent_change']}% change")
    return result
