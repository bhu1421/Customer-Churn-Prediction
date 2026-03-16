# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn based on demographics, subscribed services, and billing behavior.

This project is designed to be easy to demo, easy to explain, and easy to extend. It combines:
- a `scikit-learn` training pipeline
- a `FastAPI` backend for inference
- a `Streamlit` frontend for interactive predictions

## Why This Project Matters

Customer churn is a major business problem because retaining an existing customer is often cheaper than acquiring a new one. This project helps identify high-risk customers early so teams can take retention actions before revenue is lost.

## Key Highlights

- Clean project structure with modular `app.py` (FastAPI), `streamlit_app.py` (UI), and `model_files` layers
- Leakage-safe training flow using `train_test_split(..., stratify=y)` before fitting
- Preprocessing embedded inside a `Pipeline`, so training and inference stay consistent
- Strict API validation using `Pydantic`
- Cross-field business-rule validation for realistic customer inputs
- Interactive web interface for quick manual testing and demonstrations

## Tech Stack

- Python
- pandas
- scikit-learn
- FastAPI
- Streamlit
- Uvicorn
- Pydantic

## Machine Learning Approach

### Dataset

The project uses the telecom churn dataset stored in `customer_churn.xls`.

### Target

- `Churn`
- Encoded as:
  - `1` for churn
  - `0` for no churn

### Features Used

The model learns from customer information such as:
- gender
- senior citizen status
- partner and dependents
- tenure
- phone and internet services
- online security, backup, device protection, tech support
- streaming services
- contract type
- paperless billing
- payment method
- monthly charges
- total charges

### Preprocessing

The preprocessing logic is built directly into the sklearn pipeline:
- invalid `TotalCharges` values are converted and cleaned
- numeric features are scaled using `MinMaxScaler`
- categorical features are encoded using `OneHotEncoder(handle_unknown="ignore")`

### Model

- Algorithm: `LogisticRegression`
- Why it was chosen:
  - strong baseline for tabular classification
  - interpretable
  - lightweight and fast
  - good fit for a clean API + UI demo

## System Architecture

```text
Streamlit UI
    ->
FastAPI Prediction API
    ->
Saved sklearn Pipeline
    ->
Prediction Response
```

### Flow

1. The user enters customer information in the Streamlit UI.
2. The frontend sends the data to the FastAPI `/predict` endpoint.
3. The backend validates the request using Pydantic schemas and business rules.
4. The saved sklearn pipeline preprocesses the data and generates predictions.
5. The API returns churn probability, predicted class, and confidence.

## Input Validation

The backend does more than just type checking.

It also enforces logical combinations such as:
- if `InternetService = No`, internet add-on fields must be `No internet service`
- if `PhoneService = No`, `MultipleLines` must be `No phone service`

This improves data quality and prevents unrealistic inputs from reaching the model.

## Project Structure

```text
Customer-Churn-Prediction/
|-- model_files/
|   |-- model.pkl
|   |-- features.pkl
|   `-- .gitkeep
|-- config.py
|-- app.py
|-- streamlit_app.py
|-- train_and_save_model.py
|-- run_deployment.py
|-- customer_churn.xls
|-- customer_churn_classification.ipynb
|-- requirements.txt
`-- README.md
```

## How To Run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train and save model files

```bash
python train_and_save_model.py
```

This generates:
- `model_files/model.pkl`
- `model_files/features.pkl`

### 4. Start the full application

```bash
python run_deployment.py
```

Open:
- Frontend: `http://localhost:8501`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Run Services Separately

### Backend

```bash
uvicorn app:app --reload
```

### Frontend

```bash
streamlit run streamlit_app.py
```

## API Endpoints

### `GET /health`

Returns API health status.

### `POST /predict`

Predicts churn probability and churn class.

Example request:

```json
{
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
  "TotalCharges": 1024.0
}
```

Example response:

```json
{
  "churn_probability": 0.638,
  "churn_prediction": true,
  "prediction_text": "Will churn",
  "confidence": 0.638
}
```

## What This Project Demonstrates

This project is a good showcase of:
- applied machine learning on tabular business data
- model deployment using FastAPI
- product-style frontend integration using Streamlit
- clean Python project organization
- validation-aware ML serving

## Possible Future Improvements

- add automated tests for API routes and preprocessing behavior
- containerize the project with Docker
- deploy the API and frontend to cloud platforms

## Author

Bhuvan Patil
