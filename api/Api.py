from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import json

# Load the saved model
model = joblib.load("api/model.pkl")
with open("features.json", "r") as f:
    feature_names = json.load(f)

# Initialize the FastAPI app
app = FastAPI()

# Define the input schema using BaseModel
class ModelInput(BaseModel):
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
    customer_id: str
    merchant_category: str
    amount: float
    currency: str
    country: str
    city_size: str
    card_type: str
    card_present: bool
    distance_from_home: int
    high_risk_merchant: bool

# Define the endpoint
@app.post("/predict")
async def predict(input_data: ModelInput):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data.dict()])  # Convert Pydantic model to DataFrame
    try:
        # Make the prediction
        prediction = model.predict(input_df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        # Handle any errors
        raise HTTPException(status_code=400, detail=str(e))
