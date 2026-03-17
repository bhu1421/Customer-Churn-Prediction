import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from config import MODEL_FILES_DIR, DATA_FILE, FEATURES_PATH, MODEL_PATH


TEXT_DATA_EXTENSIONS = {".csv", ".txt"}
EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm", ".xlsb"}


def _looks_like_text_data(path: Path) -> bool:
    with open(path, "rb") as file:
        sample = file.read(2048)

    if b"\x00" in sample:
        return False

    try:
        header = sample.decode("utf-8").splitlines()[0]
    except UnicodeDecodeError:
        try:
            header = sample.decode("latin-1").splitlines()[0]
        except UnicodeDecodeError:
            return False
    except IndexError:
        return False

    return "," in header


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in TEXT_DATA_EXTENSIONS or _looks_like_text_data(path):
        return pd.read_csv(path)

    if path.suffix.lower() in EXCEL_EXTENSIONS:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported dataset format: {path.suffix or 'no extension'}")


def build_training_frame() -> pd.DataFrame:
    dataframe = load_dataset(DATA_FILE)
    dataframe = dataframe.drop(columns=["customerID"])
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")
    dataframe = dataframe.dropna(subset=["TotalCharges"]).copy()
    dataframe["Churn"] = dataframe["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return dataframe


def train() -> tuple[Pipeline, list[str], float, float]:
    dataframe = build_training_frame()
    x_values = dataframe.drop("Churn", axis="columns")
    y_values = dataframe["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=0.25,
        random_state=5,
        stratify=y_values,
    )

    numeric_cols = ["MonthlyCharges", "tenure", "TotalCharges"]
    categorical_cols = [column for column in x_values.columns if column not in numeric_cols and column != "SeniorCitizen"]

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
    model.fit(x_train, y_train)

    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    return model, list(x_values.columns), train_accuracy, test_accuracy


def save_model_files(model: Pipeline, feature_names: list[str]) -> None:
    MODEL_FILES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)
    with open(FEATURES_PATH, "wb") as file:
        pickle.dump(feature_names, file)


if __name__ == "__main__":
    model, feature_names, train_accuracy, test_accuracy = train()
    save_model_files(model, feature_names)
    print("Model files saved successfully!")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
