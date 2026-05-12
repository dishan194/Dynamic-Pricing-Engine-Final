"""Utility functions shared by training and the Streamlit dashboard."""

from pathlib import Path
from typing import Dict, List

import pandas as pd


DATASET_PATH = Path("dataset.csv")
MODEL_PATH = Path("saved_model.pkl")

CATEGORICAL_COLUMNS = ["product_category", "season"]
FEATURE_COLUMNS = [
    "product_category",
    "demand",
    "inventory",
    "competitor_price",
    "season",
    "customer_rating",
    "historical_price",
]
TARGET_COLUMN = "optimal_price"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the project dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(
            "dataset.csv was not found. Run `python generate_dataset.py` first."
        )
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with simple beginner-friendly rules."""
    clean_df = df.copy()

    for column in clean_df.select_dtypes(include=["number"]).columns:
        clean_df[column] = clean_df[column].fillna(clean_df[column].median())

    for column in clean_df.select_dtypes(include=["object"]).columns:
        clean_df[column] = clean_df[column].fillna(clean_df[column].mode()[0])

    return clean_df


def business_recommendation(
    predicted_price: float,
    competitor_price: float,
    demand: int,
    inventory: int,
    rating: float,
) -> Dict[str, str]:
    """Create simple business recommendations from the predicted price."""
    if predicted_price > competitor_price * 1.05:
        pricing_suggestion = "Premium pricing is possible because the model recommends a price above competitor level."
    elif predicted_price < competitor_price * 0.95:
        pricing_suggestion = "Use a competitive discount to attract customers and protect conversion rate."
    else:
        pricing_suggestion = "Keep the price close to the competitor price for balanced competitiveness."

    if predicted_price > competitor_price:
        profit_insight = "Expected margin can improve if demand remains stable."
    else:
        profit_insight = "Margin may be lower, but the price can help increase sales volume."

    if demand > 700 and inventory < 300:
        inventory_recommendation = "High demand with low stock: avoid deep discounts and restock soon."
    elif demand < 250 and inventory > 900:
        inventory_recommendation = "Low demand with high stock: consider offers or bundles."
    else:
        inventory_recommendation = "Inventory looks balanced for the current demand level."

    if rating >= 4.5:
        rating_note = "Strong customer rating supports confident pricing."
    elif rating < 3.2:
        rating_note = "Lower rating may require a price incentive or quality improvement."
    else:
        rating_note = "Customer rating is moderate, so monitor demand after price changes."

    return {
        "pricing_suggestion": pricing_suggestion,
        "profit_insight": profit_insight,
        "inventory_recommendation": inventory_recommendation,
        "rating_note": rating_note,
    }


def available_categories() -> List[str]:
    """Return category choices used in the generated dataset."""
    return ["Electronics", "Clothing", "Grocery", "Footwear", "Home Decor"]


def available_seasons() -> List[str]:
    """Return season choices used in the generated dataset."""
    return ["Summer", "Winter", "Festival", "Normal"]
