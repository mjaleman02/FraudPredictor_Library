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

Disclaimer - Due to the large size of the synthetic_fraud_data.csv of 3GB, in the repository we have only included a reduced .csv with 10,000 observations, that is indeed the one that has been used for the whole analysis.


## Github Repository Structure :
