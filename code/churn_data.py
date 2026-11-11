"""Dataset schema, split, and preprocessing for the churn experiment."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "Exited"
FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    if data[required].isna().any().any():
        raise ValueError("The predictors and labels must be complete for this experiment.")
    if set(data[TARGET].unique()) != {0, 1}:
        raise ValueError("Exited must contain both binary labels, 0 and 1.")
    return data


def split_dataset(data: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    train, holdout = train_test_split(
        data, test_size=0.30, stratify=data[TARGET], random_state=seed,
    )
    validation, test = train_test_split(
        holdout, test_size=0.50, stratify=holdout[TARGET], random_state=seed,
    )
    return {"train": train, "validation": validation, "test": test}


def make_preprocessor() -> Pipeline:
    encoder = ColumnTransformer([
        ("geography", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Geography"]),
        ("credit_score", "passthrough", ["CreditScore"]),
        ("gender", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["Gender"]),
        ("other", "passthrough", FEATURES[3:]),
    ])
    return Pipeline([("encode", encoder), ("scale", StandardScaler())])
