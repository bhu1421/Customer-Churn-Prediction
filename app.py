from enum import Enum
import pickle
from typing import Literal

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import FEATURES_PATH, MODEL_PATH


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


def _load_pickle(path):
    if path.exists():
        with open(path, "rb") as file:
            return pickle.load(file)
    raise FileNotFoundError


try:
    model = _load_pickle(MODEL_PATH)
    feature_names = _load_pickle(FEATURES_PATH)
except FileNotFoundError:
    raise Exception("Model files not found in the model_files folder. Please run train_and_save_model.py first")


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

    @model_validator(mode="after")
    def validate_service_consistency(self):
        internet_fields = [
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies,
        ]

        if self.InternetService == InternetService.NO:
            if any(value != InternetAddon.NO_INTERNET_SERVICE for value in internet_fields):
                raise ValueError(
                    "If InternetService is 'No', internet addon fields must be 'No internet service'."
                )
        else:
            if any(value == InternetAddon.NO_INTERNET_SERVICE for value in internet_fields):
                raise ValueError(
                    "If InternetService is not 'No', internet addon fields must be only 'Yes' or 'No'."
                )

        if self.PhoneService == YesNo.NO and self.MultipleLines != MultipleLines.NO_PHONE_SERVICE:
            raise ValueError("If PhoneService is 'No', MultipleLines must be 'No phone service'.")
        if self.PhoneService == YesNo.YES and self.MultipleLines == MultipleLines.NO_PHONE_SERVICE:
            raise ValueError("If PhoneService is 'Yes', MultipleLines must be only 'Yes' or 'No'.")

        return self


def preprocess_input(data: CustomerData) -> pd.DataFrame:
    input_dict = data.model_dump()
    dataframe = pd.DataFrame([input_dict])
    return dataframe[feature_names]


@app.post("/predict")
async def predict_churn(customer: CustomerData):
    try:
        processed_data = preprocess_input(customer)
        prediction_proba = model.predict_proba(processed_data)[0]
        prediction = model.predict(processed_data)[0]

        return {
            "churn_probability": float(prediction_proba[1]),
            "churn_prediction": bool(prediction),
            "prediction_text": "Will churn" if prediction else "Will not churn",
            "confidence": float(max(prediction_proba)),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(exc)}")


@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API", "status": "active"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}