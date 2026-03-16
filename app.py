from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import pickle
import pandas as pd
from pathlib import Path
from enum import Enum
from typing import Literal

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent

# Load model and preprocessing components
try:
    with open(BASE_DIR / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(BASE_DIR / 'features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model files not found. Please run train_and_save_model.py first")


class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"


class MultipleLines(str, Enum):
    YES = "Yes"
    NO = "No"
    NO_PHONE_SERVICE = "No phone service"


class InternetService(str, Enum):
    DSL = "DSL"
    FIBER = "Fiber optic"
    NO = "No"


class InternetAddon(str, Enum):
    YES = "Yes"
    NO = "No"
    NO_INTERNET_SERVICE = "No internet service"


class ContractType(str, Enum):
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class PaymentMethod(str, Enum):
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"

# Define input data structure
class CustomerData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Gender
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    PhoneService: YesNo
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: ContractType
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=1000000)

def preprocess_input(data: CustomerData) -> pd.DataFrame:
    """Build model input row in training feature order."""
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])
    return df[feature_names]

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
