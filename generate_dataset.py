"""Generate a realistic synthetic dataset for the dynamic pricing project.

Run this file when dataset.csv is missing or when you want fresh sample data:
    python generate_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PATH = Path("dataset.csv")
RANDOM_SEED = 42


def generate_dynamic_pricing_dataset(rows: int = 5000) -> pd.DataFrame:
    """Create synthetic e-commerce pricing data with practical relationships."""
    rng = np.random.default_rng(RANDOM_SEED)

    categories = ["Electronics", "Clothing", "Grocery", "Footwear", "Home Decor"]
    seasons = ["Summer", "Winter", "Festival", "Normal"]

    # Base prices keep each category in a realistic price range.
    category_base_price = {
        "Electronics": 650,
        "Clothing": 80,
        "Grocery": 25,
        "Footwear": 120,
        "Home Decor": 180,
    }

    category = rng.choice(categories, size=rows, p=[0.22, 0.24, 0.20, 0.17, 0.17])
    season = rng.choice(seasons, size=rows, p=[0.24, 0.24, 0.18, 0.34])
    demand = rng.integers(20, 1000, size=rows)
    inventory = rng.integers(5, 1500, size=rows)
    customer_rating = np.round(rng.uniform(2.5, 5.0, size=rows), 1)

    base_price = np.array([category_base_price[item] for item in category], dtype=float)
    season_multiplier = np.array(
        [
            {"Summer": 1.03, "Winter": 1.06, "Festival": 1.18, "Normal": 1.00}[item]
            for item in season
        ]
    )

    # Competitor price fluctuates around the category base price.
    competitor_price = base_price * rng.uniform(0.82, 1.25, size=rows) * season_multiplier

    # Historical price is what the store charged earlier.
    historical_price = competitor_price * rng.uniform(0.88, 1.15, size=rows)

    demand_factor = 1 + (demand - 500) / 5000
    inventory_factor = 1 - np.minimum(inventory / 3000, 0.25)
    rating_factor = 1 + (customer_rating - 3.5) * 0.035
    noise = rng.normal(0, base_price * 0.025, size=rows)

    optimal_price = (
        (0.48 * competitor_price)
        + (0.34 * historical_price)
        + (0.18 * base_price)
    )
    optimal_price = optimal_price * demand_factor * inventory_factor * rating_factor + noise
    optimal_price = np.maximum(optimal_price, base_price * 0.45)

    df = pd.DataFrame(
        {
            "product_id": [f"P{100001 + i}" for i in range(rows)],
            "product_category": category,
            "demand": demand,
            "inventory": inventory,
            "competitor_price": np.round(competitor_price, 2),
            "season": season,
            "customer_rating": customer_rating,
            "historical_price": np.round(historical_price, 2),
            "optimal_price": np.round(optimal_price, 2),
        }
    )

    return df


if __name__ == "__main__":
    dataset = generate_dynamic_pricing_dataset(rows=5000)
    dataset.to_csv(DATASET_PATH, index=False)
    print(f"Dataset generated successfully: {DATASET_PATH.resolve()}")
    print(f"Rows: {len(dataset)}")
