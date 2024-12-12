import requests
import json

# Define the API endpoint
url = "http://127.0.0.1:8000/predict"

def test_predict_endpoint():
    """Test the /predict endpoint of the API."""
    # Load the example input JSON
    with open("api/example_input.json", "r") as file:
        input_data = json.load(file)

    # Send the POST request
    response = requests.post(url, json=input_data)

    # Assert that the response status code is 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Assert that the response contains a valid JSON body
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        assert False, "Response is not valid JSON"

    # Optional: Add assertions for specific keys or values in the response
    assert "prediction" in response_data, "'prediction' key not found in the response"
    
    # Log the full response for debugging
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
