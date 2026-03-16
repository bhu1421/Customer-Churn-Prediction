from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent

# Load model and preprocessing components
try:
    with open(BASE_DIR / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(BASE_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(BASE_DIR / 'features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model files not found. Please run train_and_save_model.py first")

# Define input data structure
class CustomerData(BaseModel):
    gender: str  # "Male" or "Female"
    SeniorCitizen: int  # 0 or 1
    Partner: str  # "Yes" or "No"
    Dependents: str  # "Yes" or "No"
    tenure: int  # Number of months
    PhoneService: str  # "Yes" or "No"
    MultipleLines: str  # "Yes", "No", or "No phone service"
    InternetService: str  # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str  # "Yes", "No", or "No internet service"
    OnlineBackup: str  # "Yes", "No", or "No internet service"
    DeviceProtection: str  # "Yes", "No", or "No internet service"
    TechSupport: str  # "Yes", "No", or "No internet service"
    StreamingTV: str  # "Yes", "No", or "No internet service"
    StreamingMovies: str  # "Yes", "No", or "No internet service"
    Contract: str  # "Month-to-month", "One year", or "Two year"
    PaperlessBilling: str  # "Yes" or "No"
    PaymentMethod: str  # "Electronic check", "Mailed check", "Bank transfer (automatic)", or "Credit card (automatic)"
    MonthlyCharges: float
    TotalCharges: float

def preprocess_input(data: CustomerData) -> np.ndarray:
    """Preprocess input data to match training format"""

    # Create a dictionary from input
    input_dict = data.model_dump()

    # Convert to DataFrame for easier processing
    df = pd.DataFrame([input_dict])

    # Apply same preprocessing as training

    # Binary encoding for yes/no columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0})

    # Gender encoding
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

    # One-hot encoding for categorical columns
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])

    # Ensure all expected columns exist (add missing ones with 0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training data
    df = df[feature_names]

    # Scale numerical columns
    scaling_cols = ['MonthlyCharges', 'tenure', 'TotalCharges']
    df[scaling_cols] = scaler.transform(df[scaling_cols])

    return df.values

@app.post("/predict")
async def predict_churn(customer: CustomerData):
    """Predict customer churn probability"""
    try:
        # Preprocess input
        processed_data = preprocess_input(customer)

        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[0]
        prediction = model.predict(processed_data)[0]

        # Return results
        return {
            "churn_probability": float(prediction_proba[1]),
            "churn_prediction": bool(prediction),
            "prediction_text": "Will churn" if prediction else "Will not churn",
            "confidence": float(max(prediction_proba))
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    """API root endpoint"""
    return {"message": "Customer Churn Prediction API", "status": "active"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
