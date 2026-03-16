## Customer-Churn-Prediction

### Project Overview
This project predicts customer churn (whether a customer is likely to leave) using a Logistic Regression model. It includes:

- A training script to preprocess data and save artifacts
- A FastAPI backend for predictions
- A Streamlit frontend for interactive usage
- A deployment helper script to run both services

### Project Structure
```text
Customer-Churn-Prediction/
|-- customer_churn_classification.ipynb   # Notebook workflow
|-- train_and_save_model.py               # Training + artifact export
|-- app.py                                # FastAPI backend
|-- streamlit_app.py                      # Streamlit frontend
|-- run_deployment.py                     # Starts backend + frontend
|-- test_api.py                           # API smoke test
|-- requirements.txt                      # Dependencies
|-- customer_churn.xls                    # Dataset
|-- model.pkl                             # Trained model (generated)
|-- features.pkl                          # Feature list (generated)
`-- README.md
```

### Dataset
- File: `customer_churn.xls`
- Contains customer profile, service usage, billing details, and `Churn` target.

### Model and Preprocessing
- Algorithm: Logistic Regression
- Test accuracy: ~79.12%
- Preprocessing:
  - Missing/invalid `TotalCharges` handling
  - Binary encoding for yes/no fields
  - One-hot encoding for categorical fields
  - MinMax scaling for numeric columns

### Installation
```bash
pip install -r requirements.txt
```

### Run Steps
1. Train and save artifacts:
```bash
python train_and_save_model.py
```

2. Start full deployment (recommended):
```bash
python run_deployment.py
```

3. Open:
- Frontend: http://localhost:8501
- API: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Run Services Separately
Terminal 1:
```bash
uvicorn app:app --reload
```

Terminal 2:
```bash
streamlit run streamlit_app.py
```

### Test API
```bash
python test_api.py
```

### API Endpoints
- `POST /predict`: Predict churn probability and class
- `GET /health`: Health check

### Notes
- Keep `model.pkl` and `features.pkl` in the project root.
- If artifacts are missing, retrain using `train_and_save_model.py`.

### Author
Bhuvan Patil
