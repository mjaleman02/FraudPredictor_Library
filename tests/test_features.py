import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fraud_predictor.preprocessors.preprocessing import load_df, drop_unnecessary_columns 
from fraud_predictor.features.features_creation import create_time_columns, transform_to_datetime_type

def test_create_time_columns():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fraud_predictor/data/dropped_df.csv'))
    
    df = pd.read_csv(csv_path)
    df = drop_unnecessary_columns(df)
    df = transform_to_datetime_type(df)
    result_df = create_time_columns(df)
    
    expected_columns = [
        'customer_id', 'timestamp', 'merchant_category', 'amount', 
        'currency', 'country', 'city_size', 'card_type', 'card_present', 
        'device', 'channel', 'distance_from_home', 'high_risk_merchant', 
        'transaction_hour', 'weekend_transaction', 'is_fraud', 
        'transaction_month', 'transaction_day'
    ]
    
    assert set(result_df.columns) == set(expected_columns)
