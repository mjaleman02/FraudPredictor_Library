import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fraud_predictor.merging.merging_df import rename_columns, gdp_capita_columns_keep, rename_values
from fraud_predictor.preprocessors.preprocessing import drop_unnecessary_columns

def test_rename_columns_df2():
    # read in csv file using path
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fraud_predictor/data/gdp_country.csv'))
    
    # include dictionary
    rename_mapping = {'Country Name': 'country', '2023': 'GDP'}
    
    # read file and rename locally
    df2 = pd.read_csv(csv_path)
    df2 = df2.rename(columns=rename_mapping)
    
    # rename columns using our created function
    result_df = rename_columns(df2, rename_mapping)
    
    assert set(result_df.columns) == set(df2)
    

def test_rename_columns_df3():
    # read in csv file using path
    csv_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fraud_predictor/data/gdp_per_capita.csv'))
    
    # include dictionary
    rename_mapping = {'Country Name': 'country', '2023': 'GDP_per_capita'}
    
    # read csv and columns to keep locally
    df3 = pd.read_csv(csv_path2, header=2)
    df3 = gdp_capita_columns_keep(df3)
    
    df3 = df3.rename(columns=rename_mapping)
    
    # rename columns using our created function
    result_df = rename_columns(df3, rename_mapping)
    
    assert set(result_df.columns) == set(df3)
    

def test_rename_values():
    # read in csv file
    csv_path3 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fraud_predictor/data/dropped_df.csv'))
    
    # include new renaming dictionary
    rename_mapping = {'country': {'usa': 'united states', 'uk': 'united kingdom', 'russia': 'russian federation'}}

    # read csv and drop columns 
    df = pd.read_csv(csv_path3)
    df = drop_unnecessary_columns(df)
    
    # for loop for renaming values locally 
    for col, mapping in rename_mapping.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)  # Map values within a column
        else:
            df = df.rename(columns={col: mapping})
    
    # rename values using our created function
    result_df = rename_values(df, rename_mapping)
    
    assert set(result_df.columns) == set(df)
    
    
def test_gdp_capita_columns_keep():
    # read in csv file
    csv_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fraud_predictor/data/gdp_per_capita.csv'))
    
    # read csv and columns to keep locally
    df3 = pd.read_csv(csv_path2, header=2)
    
    # list of wanted columns
    expected_columns = [
        "Country Name", 
        "Country Code",
        "Indicator Code",
        "2023"
    ]
    existing_columns_to_keep = [col for col in expected_columns if col in df3.columns]
    
    # create new df locally 
    df3 = df3[existing_columns_to_keep]
    
    # create result df using our created function 
    result_df = gdp_capita_columns_keep(df3)
    
    assert set(result_df.columns) == set(df3)