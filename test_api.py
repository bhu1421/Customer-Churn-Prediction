import requests
import json

# Test data for API
test_customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 1024.00
}

def test_api():
    """Test the FastAPI prediction endpoint"""
    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health check: {health_response.status_code}")
        print(f"Health response: {health_response.json()}")

        # Test prediction endpoint
        predict_response = requests.post(
            "http://localhost:8000/predict",
            json=test_customer,
            timeout=10
        )
        print(f"\nPrediction status: {predict_response.status_code}")
        print(f"Prediction response: {json.dumps(predict_response.json(), indent=2)}")

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the FastAPI server is running with: uvicorn app:app --reload")

if __name__ == "__main__":
    test_api()