import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fraud_predictor.preprocessors.preprocessing import load_df, drop_unnecessary_columns 
from fraud_predictor.features.features_creation import create_time_columns, transform_to_datetime_type, create_channel_usage, create_interaction_by_category

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



def test_create_channel_usage():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fraud_predictor/data/dropped_df.csv'))
    
    df = pd.read_csv(csv_path)
    result_df = create_channel_usage(df)

    assert (result_df['channel_usage'] >= 0).all(), "channel_usage has values lower tham 0"
    assert (result_df['channel_usage'] <= 1).all(), "channel_usage has values greater than 1"


    

def test_create_interaction_by_category():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fraud_predictor/data/dropped_df.csv'))
    
    df = pd.read_csv(csv_path)

    result_df = create_interaction_by_category(df, 
                                               col1='amount', 
                                               col2='merchant_category', 
                                               new_col_name='value_by_category')
    
    assert (result_df['value_by_category'] > 0).all(), "value_by_category ha valori minori di 0"
