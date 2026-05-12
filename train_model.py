"""Train and save the RandomForestRegressor pricing model."""

import pickle

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from generate_dataset import generate_dynamic_pricing_dataset
from utils import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    TARGET_COLUMN,
    handle_missing_values,
    load_dataset,
)


def train_model() -> dict:
    """Train the ML model and save model, preprocessing, and metrics together."""
    if not DATASET_PATH.exists():
        dataset = generate_dynamic_pricing_dataset(rows=5000)
        dataset.to_csv(DATASET_PATH, index=False)
        print("dataset.csv was missing, so a synthetic dataset was generated.")

    df = load_dataset(DATASET_PATH)
    df = handle_missing_values(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_columns = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]

    # Random forests do not require scaling, so only categorical features are encoded.
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", numeric_columns),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "MAE": round(mean_absolute_error(y_test, predictions), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, predictions)), 2),
        "R2 Score": round(r2_score(y_test, predictions), 4),
    }

    model_package = {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model_package, file)

    print("Model training completed successfully.")
    print(f"Saved model: {MODEL_PATH.resolve()}")
    print(f"MAE: {metrics['MAE']}")
    print(f"RMSE: {metrics['RMSE']}")
    print(f"R2 Score: {metrics['R2 Score']}")

    return metrics


if __name__ == "__main__":
    train_model()
