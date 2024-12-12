from fastapi import FastAPI, HTTPException
import joblib
import json
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load model and features
try:
    model = joblib.load("api/model.pkl")
    with open("features.json", "r") as f:
        feature_names = json.load(f)
except Exception as e:
    print("Error loading model or features:", e)
    raise

category_cols = [
    "customer_id", "merchant_category", "currency", "country",
    "city_size", "card_type", "device", "channel", "hour_category"
]

bool_cols = ["card_present", "weekend_transaction", "high_risk_merchant"]

numeric_cols = {
    "amount": "float64",
    "GDP": "float64",
    "GDP_per_capita": "float64",
    "channel_usage": "float64",
    "value_by_category": "float64",
    "transaction_month": "int32",
    "transaction_day": "int32",
    "payment_safety": "int64",
    "distance_from_home": "int64"
}

class PredictionInput(BaseModel):
    customer_id: str
    merchant_category: str
    amount: float
    currency: str
    country: str
    city_size: str
    card_type: str
    card_present: bool
    device: str
    channel: str
    weekend_transaction: bool
    GDP: float
    GDP_per_capita: float
    transaction_month: int
    transaction_day: int
    hour_category: str
    channel_usage: float
    value_by_category: float
    payment_safety: int
    high_risk_merchant: bool
    distance_from_home: int

@app.post("/predict")
def predict(input_data: PredictionInput):
    input_dict = input_data.dict()

    # Validate features
    missing_features = [f for f in feature_names if f not in input_dict]
    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Apply dtypes
    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    for col, dtype in numeric_cols.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Reorder columns to match model expectation
    df = df[feature_names]

    try:
        prediction = model.predict(df)
        
        # >>> INSERT THE CONVERSION CODE HERE <<<
        # Convert the prediction to a Python native type
        python_prediction = prediction[0]

        # If it's a NumPy bool, convert to Python bool
        if isinstance(python_prediction, np.bool_):
            python_prediction = bool(python_prediction)
        # If it's a NumPy integer (e.g. np.int64), convert to Python int
        elif isinstance(python_prediction, np.integer):
            python_prediction = int(python_prediction)
        # If it's a NumPy float (np.float64), convert to Python float
        elif isinstance(python_prediction, np.floating):
            python_prediction = float(python_prediction)

        return {"prediction": python_prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
