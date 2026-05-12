"""Small model wrapper used by the Streamlit app."""

import pickle
from pathlib import Path
from typing import Dict

import pandas as pd

from utils import FEATURE_COLUMNS, MODEL_PATH


def load_model(model_path: Path = MODEL_PATH) -> Dict:
    """Load the saved pickle model package."""
    if not model_path.exists():
        raise FileNotFoundError(
            "saved_model.pkl was not found. Run `python train_model.py` first."
        )

    with open(model_path, "rb") as file:
        return pickle.load(file)


def predict_price(
    category: str,
    demand: int,
    inventory: int,
    competitor_price: float,
    season: str,
    customer_rating: float,
    historical_price: float | None = None,
) -> float:
    """Predict the recommended optimal price for one product."""
    model_package = load_model()
    pipeline = model_package["pipeline"]

    # If historical price is unknown, competitor price is a reasonable quick-demo proxy.
    if historical_price is None:
        historical_price = competitor_price

    input_data = pd.DataFrame(
        [
            {
                "product_category": category,
                "demand": demand,
                "inventory": inventory,
                "competitor_price": competitor_price,
                "season": season,
                "customer_rating": customer_rating,
                "historical_price": historical_price,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prediction = pipeline.predict(input_data)[0]
    return round(float(prediction), 2)
