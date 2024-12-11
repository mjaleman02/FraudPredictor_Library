import requests
import json

# Define the API endpoint
url = "http://127.0.0.1:8000/predict"

# Load the example input JSON
with open("example_input.json", "r") as file:
    input_data = json.load(file)

# Send the POST request
response = requests.post(url, json=input_data)

# Print the response
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)