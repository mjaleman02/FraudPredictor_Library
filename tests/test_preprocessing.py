import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fraud_predictor.preprocessors.preprocessing import load_df, drop_unnecessary_columns


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_drop_unnecessary_columns():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fraud_predictor/data/dropped_df.csv'))
    
    df = pd.read_csv(csv_path)

    result_df = drop_unnecessary_columns(df)

    expected_columns = [
        'customer_id', 'timestamp', 'merchant_category', 'amount', 
        'currency', 'country', 'city_size', 'card_type', 'card_present', 
        'device', 'channel', 'distance_from_home', 'high_risk_merchant', 
        'transaction_hour', 'weekend_transaction', 'is_fraud'
    ]

    assert list(result_df.columns) == expected_columns
