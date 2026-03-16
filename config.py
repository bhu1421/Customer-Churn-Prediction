from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "customer_churn.xls"

MODEL_FILES_DIR = BASE_DIR / "model_files"
MODEL_PATH = MODEL_FILES_DIR / "model.pkl"
FEATURES_PATH = MODEL_FILES_DIR / "features.pkl"
