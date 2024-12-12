# FraudPredictor_Library - Computing for Data Science Final Project

## Group Members :
- Noemi Lucchi
- Marta Sala
- Nicolas Rauth
- Maria Aleman Hernandez 


## Dataset Used :
Website - https://www.kaggle.com/datasets/ismetsemedov/transactions/data 

Overview - This Kaggle dataset is composed of financial transactions across different categories such as dining, groceries, retail, etc. over a period of time and specific to each individual transaction. Our main goal was to create a model that could help distinguish between those transactions that are legitimate versus those that are fraudulant with the help of additional characteristics. We found this dataset to be interesting because financial fraud is an issue that happens everyday and being able to create a model that can help detect fraud gives us a glimpse of how detections like these can occur. It is important to note that although this dataset was inspired by real-world transaction data, the data was generated synthetically to abide privacy concerns.     

Columns in Original Dataset (24 columns) :
- transaction_id: Unique identifier for each transaction.
- customer_id: Unique identifier for each customer in the dataset.
- card_number: Masked card number associated with the transaction.
- timestamp: Date and time of the transaction.
- merchant_category: General category of the merchant (e.g., Retail, Grocery, Travel).
- merchant_type: Specific type within the merchant category (e.g., "online" for Retail).
- merchant: Name of the merchant where the transaction took place.
- amount: Transaction amount (currency based on the country).
- currency: Currency used for the transaction (e.g., USD, EUR, JPY).
- country: Country where the transaction occurred.
- city: City where the transaction took place.
- city_size: Size of the city (e.g., medium, large).
- card_type: Type of card used (e.g., Basic Credit, Gold Credit).
- card_present: Indicates if the card was physically present during the transaction (used in POS transactions).
- device: Device used for the transaction (e.g., Chrome, iOS App, NFC Payment).
- channel: Type of channel used for the transaction (web, mobile, POS).
- device_fingerprint: Unique fingerprint for the device used in the transaction.
- ip_address: IP address associated with the transaction.
- distance_from_home: Binary indicator showing if the transaction occurred outside the customer's home country.
- high_risk_merchant: Indicates if the merchant category is known for higher fraud risk (e.g., Travel, Entertainment).
- transaction_hour: Hour of the day when the transaction was made.
- weekend_transaction: Boolean indicating if the transaction took place on a weekend.
- velocity_last_hour: Dictionary containing metrics on the transaction velocity, including:
    - num_transactions: Number of transactions in the last hour for this customer.
    - total_amount: Total amount spent in the last hour.
    - unique_merchants: Count of unique merchants in the last hour.
    - unique_countries: Count of unique countries in the last hour.
    - max_single_amount: Maximum single transaction amount in the last hour.
- is_fraud: Binary indicator showing if the transaction is fraudulent (True for fraudulent transactions, False for legitimate ones).

Columns That We Created (5 columns) :
- hour_category : Time of day placed into a category of morning, lunch, afternoon, dinner, evening, or night.
- transaction_month : Column for month from the 'timestamp' column.
- transaction_day : Column for day from the 'timestamp' column.
- channel_usage : Calculate frequency with which each customer makes purchases based on individual channels.
- value_by_category : Interaction term that normalizes the amount within each merchant category.
- payment_safety : Column that indicates the level of safety of the payment (manually mapped from 1 to 4)

Target Variable : is_fraud

Disclaimer - Due to the large size of the synthetic_fraud_data.csv of 3GB from Kaggle, in our repository we have only included a reduced csv sample file with 10,000 observations, which is indeed the one that has been used for the whole analysis. The functions used with original dataset such as reading the csv file and creating the sample have been left in the code but have been commented out in order for the whole process to still be seen from start to finish. 


## Additional Datasets :
In order to improve our model and include more characteristics that could potentially help distinguish a transaction from being legitimate or fraudulant, we decided to include two more additional datasets that were later on merged within our code.

1. GDP
Website : https://data.worldbank.org/indicator/NY.GDP.MKTP.CD 

2. GDP per capita
Website : https://data.worldbank.org/indicator/NY.GDP.PCAP.CD?end=2023&start=1960&view=chart


## Github Repository Structure :
- README file
- setup.py file
- notebook_project_fp.ipynb - Our main jupyter notebook to run all functions and model
- fraud_predictor folder
    - data - holds the three datasets
          - dropped_df.csv - Sample of Kaggle's transactions dataset (10,000 random sample)
          - gdp_country.csv - GDP data
          - gdp_per_capita.csv - GDP per capita data
    - features folder
          - features_creation.py - python file of functions used for features
    - merging folder
          - merging_df.py - python file of functions used for merging
    - model folder
          - model_and_metrics.py - python file of functions used to make model
    - preprocessors folder
          - preprocessing.py - python file of functions used to preprocess 
- test folder
    - test_features.py - tests features functions 
    - test_merging.py - tests merging functions
    - test_preprocessing.py - tests preprocessing functions 
- api folder
    - model.pkl - pickle that contains the model trained 
    - main.py - python file that defines the API
    - features.json - json file that lists the features 
    - example_input.json - json file that inputs some values to the endpoint in order to yield a prediction 
    - test_predict_endpoint.py - python file to test the API


## Scalability of the Library :
To add new preprocessors, you can:
- Adjust list of columns to be dropped in the drop_unnecessary_columns function.
- Modify the sample size of the dataset by changing the number of observations.
- Include new preprocessor functions in preprocessors/preprocessing.py.
- Will need to update the respective test cases in tests/test_preprocessing.py to ensure the edits or adjustments work correctly.

To add new features, you can:
- Adjust categories used to for hours of the day in the categorize_hour_column function. 
- Create a new interaction column by updating the col1 and col2 chosen columns in the create_interaction_by_category function.
- Modify the device_mapping dictionary used for the create_payment_safety function in case new device groupings want to be used.
- Include new features functions in features/features_creation.py.
- Will need to update the respective test cases in tests/test_features.py to ensure the edits or adjustments work correctly.
     
To add new models and metrics, you can:
- Modify those values and columns that have been renamed within their respective dictionaries in the rename_columns and rename_values functions.
- Adjust the test size or lightgbm model metrics within the split_data and tune_lightgbm functions.
- Include new metrics that can be calculated from the model in model/model_and_metrics.py
- Include new model, metrics, or visualization functions in model/model_and_metrics.py.

Note : For any new columns or features added, the Api.py and json files will need to be updated.
