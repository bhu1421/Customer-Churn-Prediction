import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "customer_churn.xls"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load dataset safely for either true Excel or CSV-formatted files."""
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_csv(path)


# Load and preprocess data (same as notebook)
df = load_dataset(DATA_FILE)
df = df.drop(columns=["customerID"])

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).copy()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

X = df.drop("Churn", axis="columns")
y = df["Churn"]

# Split first to prevent leakage from preprocessing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=5,
    stratify=y,
)

numeric_cols = ["MonthlyCharges", "tenure", "TotalCharges"]
categorical_cols = [col for col in X.columns if col not in numeric_cols and col != "SeniorCitizen"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=0, max_iter=1000)),
    ]
)

model.fit(X_train, y_train)

# Save model pipeline and source feature names
with open(BASE_DIR / 'model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open(BASE_DIR / 'features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("Model and preprocessing components saved successfully!")
print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
