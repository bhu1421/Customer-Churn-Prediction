import unittest

from pydantic import ValidationError

from app import CustomerData, health_check, predict_churn
from config import DATA_FILE
from train_and_save_model import build_training_frame, load_dataset


def make_valid_customer(**overrides) -> CustomerData:
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 1024.0,
    }
    payload.update(overrides)
    return CustomerData(**payload)


class PredictionApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_check(self):
        self.assertEqual(await health_check(), {"status": "healthy"})

    async def test_predict_churn_returns_expected_fields(self):
        result = await predict_churn(make_valid_customer())

        self.assertEqual(
            set(result),
            {"churn_probability", "churn_prediction", "prediction_text", "confidence"},
        )
        self.assertIsInstance(result["churn_prediction"], bool)
        self.assertGreaterEqual(result["churn_probability"], 0.0)
        self.assertLessEqual(result["churn_probability"], 1.0)

    def test_invalid_service_combination_is_rejected(self):
        with self.assertRaises(ValidationError):
            make_valid_customer(
                InternetService="No",
                OnlineSecurity="Yes",
                OnlineBackup="No internet service",
                DeviceProtection="No internet service",
                TechSupport="No internet service",
                StreamingTV="No internet service",
                StreamingMovies="No internet service",
            )


class TrainingDataTests(unittest.TestCase):
    def test_load_dataset_reads_the_current_source_file(self):
        dataframe = load_dataset(DATA_FILE)

        self.assertGreater(len(dataframe), 0)
        self.assertIn("customerID", dataframe.columns)
        self.assertIn("Churn", dataframe.columns)

    def test_build_training_frame_prepares_model_inputs(self):
        dataframe = build_training_frame()

        self.assertIn("TotalCharges", dataframe.columns)
        self.assertIn("Churn", dataframe.columns)
        self.assertNotIn("customerID", dataframe.columns)
        self.assertFalse(dataframe["TotalCharges"].isna().any())


if __name__ == "__main__":
    unittest.main()
