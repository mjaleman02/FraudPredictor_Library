from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
import joblib
import pandas as pd
import json

# Load the saved model and feature names
model = joblib.load("api/model.pkl")
with open("features.json", "r") as f:
    feature_names = json.load(f)

# Initialize the FastAPI app
app = FastAPI()

# Enums for categorical fields
class DeviceEnum(str, Enum):
    android_app = "Android App"
    chip_reader = "Chip Reader"
    chrome = "Chrome"
    edge = "Edge"
    firefox = "Firefox"
    ios_app = "iOS App"
    magnetic_stripe = "Magnetic Stripe"
    nfc_payment = "NFC Payment"
    safari = "Safari"

class ChannelEnum(str, Enum):
    mobile = "mobile"
    pos = "pos"
    web = "web"

class HourCategoryEnum(str, Enum):
    morning = "morning"
    lunch = "lunch"
    afternoon = "afternoon"
    dinner = "dinner"
    evening = "evening"
    night = "night"

class MerchantCategoryEnum(str, Enum):
    education = "Education"
    entertainment = "Entertainment"
    gas = "Gas"
    grocery = "Grocery"
    healthcare = "Healthcare"
    restaurant = "Restaurant"
    retail = "Retail"
    travel = "Travel"

class CurrencyEnum(str, Enum):
    aud = "AUD"
    brl = "BRL"
    cad = "CAD"
    eur = "EUR"
    gbp = "GBP"
    jpy = "JPY"
    mxn = "MXN"
    ngn = "NGN"
    rub = "RUB"
    sgd = "SGD"
    usd = "USD"

class CountryEnum(str, Enum):
    australia = "Australia"
    brazil = "Brazil"
    canada = "Canada"
    france = "France"
    germany = "Germany"
    japan = "Japan"
    mexico = "Mexico"
    nigeria = "Nigeria"
    russia = "Russia"
    singapore = "Singapore"
    uk = "UK"
    usa = "USA"

class CitySizeEnum(str, Enum):
    medium = "medium"
    large = "large"
    metropolitan = "metropolitan"

class CardTypeEnum(str, Enum):
    basic_credit = "Basic Credit"
    basic_debit = "Basic Debit"
    gold_credit = "Gold Credit"
    platinum_credit = "Platinum Credit"
    premium_debit = "Premium Debit"

# Define the input schema using BaseModel
class ModelInput(BaseModel):
    device: DeviceEnum = Field(example="Android App")
    channel: ChannelEnum = Field(example="mobile")
    weekend_transaction: bool = Field(example=True)
    GDP: float = Field(example=217367000000.0)
    GDP_per_capita: float = Field(example=10043.62)
    transaction_month: int = Field(example=10)
    transaction_day: int = Field(example=9)
    hour_category: HourCategoryEnum = Field(example="night")
    channel_usage: float = Field(example=0.5)
    value_by_category: float = Field(example=0.00548)
    payment_safety: int = Field(example=4)
    customer_id: str = Field(example="CUST_29238")
    merchant_category: MerchantCategoryEnum = Field(example="Entertainment")
    amount: float = Field(example=156.3)
    currency: CurrencyEnum = Field(example="USD")
    country: CountryEnum = Field(example="USA")
    city_size: CitySizeEnum = Field(example="large")
    card_type: CardTypeEnum = Field(example="Platinum Credit")
    card_present: bool = Field(example=True)
    distance_from_home: int = Field(example=1)
    high_risk_merchant: bool = Field(example=False)

# Preprocessing functions
def preprocess_input(input_data):
    """
    Preprocess the input JSON data for prediction.
    """
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data.dict()])

    # Categorical columns list (from the trained model's expected features)
    categorical_columns = [
        'device', 'channel', 'hour_category', 'merchant_category',
        'currency', 'country', 'city_size', 'card_type'
    ]
    
    # Convert categorical columns to category dtype
    for col in categorical_columns:
        if col in df.columns:
            # Ensure that the column is categorized with the correct categories
            df[col] = pd.Categorical(df[col], categories=feature_names[col], ordered=True)
    
    # Ensure numeric fields remain as they are
    numeric_columns = ['GDP', 'GDP_per_capita', 'channel_usage', 'value_by_category', 'amount', 'distance_from_home']
    for col in numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def validate_features(input_df):
    """
    Validate that all features required by the model are present in the input.
    """
    missing_features = [feat for feat in feature_names if feat not in input_df.columns]
    if missing_features:
        raise ValueError(f"Missing features in input: {missing_features}")
    return input_df

# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: ModelInput):
    try:
        # Preprocess input
        input_df = preprocess_input(input_data)
        input_df = validate_features(input_df)

        # Predict with the loaded model
        prediction = model.predict(input_df)

        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    
def validate_features(input_df):
    """
    Validate that all features required by the model are present in the input.
    """
    print("Feature names from model:", feature_names)  # Debugging
    print("Input DataFrame columns:", input_df.columns)  # Debugging

    missing_features = [feat for feat in feature_names if feat not in input_df.columns]
    if missing_features:
        raise ValueError(f"Missing features in input: {missing_features}")
    return input_df

validate_features(input)