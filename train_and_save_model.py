import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
df1 = df[df["TotalCharges"] != " "].copy()
df1["TotalCharges"] = pd.to_numeric(df1["TotalCharges"], errors="coerce")
df1 = df1.dropna(subset=["TotalCharges"])

# Replace categorical values
df1 = df1.replace({"No internet service": "No", "No phone service": "No"})

# Binary encoding
yes_no = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

for column in yes_no:
    df1[column] = df1[column].map({'Yes': 1, 'No': 0}).astype(int)

df1['gender'] = df1['gender'].map({'Female': 0, 'Male': 1}).astype(int)

# One-hot encoding
df2 = pd.get_dummies(data=df1, columns=['InternetService', 'Contract', 'PaymentMethod'])

# Convert boolean columns to int
for column in df2.columns:
    if df2[column].dtype == bool:
        df2[column] = df2[column].astype(int)

# Scaling
scaling_column = ['MonthlyCharges', 'tenure', 'TotalCharges']
scaler = MinMaxScaler()
df2[scaling_column] = scaler.fit_transform(df2[scaling_column])

# Split data
X = df2.drop('Churn', axis='columns')
y = df2['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Train Logistic Regression (best performing model)
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

# Save model, scaler, and feature names
with open(BASE_DIR / 'model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open(BASE_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(BASE_DIR / 'features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("Model and preprocessing components saved successfully!")
print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
