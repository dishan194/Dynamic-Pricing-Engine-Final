"""Streamlit dashboard for the AI-Powered Dynamic Pricing Engine."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from model import load_model, predict_price
from train_model import train_model
from utils import (
    DATASET_PATH,
    available_categories,
    available_seasons,
    business_recommendation,
    handle_missing_values,
    load_dataset,
)


st.set_page_config(
    page_title="AI Dynamic Pricing Engine",
    page_icon="USD",
    layout="wide",
)


def inject_custom_css() -> None:
    """Add light styling for a more professional dashboard."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #e6e8ef;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 1px 4px rgba(18, 38, 63, 0.06);
        }
        .recommendation-box {
            background: #f7fbff;
            border-left: 5px solid #2f80ed;
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .small-muted {
            color: #667085;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Load and clean the dataset once for faster dashboard refreshes."""
    return handle_missing_values(load_dataset(DATASET_PATH))


@st.cache_resource
def get_model_package() -> dict:
    """Load the saved model package once."""
    return load_model()


def ensure_project_files() -> None:
    """Create dataset/model automatically if the user starts the app first."""
    if not DATASET_PATH.exists():
        train_model()

    if not Path("saved_model.pkl").exists():
        train_model()


def show_header(df: pd.DataFrame, metrics: dict) -> None:
    """Render title and top-level KPI cards."""
    st.title("AI-Powered Dynamic Pricing Engine for E-commerce")
    st.markdown(
        "<span class='small-muted'>Predict optimal product prices using demand, inventory, season, competitor price, category, and customer rating.</span>",
        unsafe_allow_html=True,
    )

    avg_price = df["optimal_price"].mean()
    avg_demand = df["demand"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Rows", f"{len(df):,}")
    col2.metric("Average Optimal Price", f"${avg_price:,.2f}")
    col3.metric("Average Demand", f"{avg_demand:,.0f}")
    col4.metric("Model R2 Score", metrics.get("R2 Score", "N/A"))


def show_dataset_preview(df: pd.DataFrame) -> None:
    """Render dataset preview and quick filters."""
    st.subheader("Dataset Preview")

    selected_category = st.selectbox(
        "Filter preview by category",
        ["All Categories"] + available_categories(),
        key="preview_category",
    )

    preview_df = df.copy()
    if selected_category != "All Categories":
        preview_df = preview_df[preview_df["product_category"] == selected_category]

    st.dataframe(preview_df.head(20), use_container_width=True)


def show_eda_charts(df: pd.DataFrame) -> None:
    """Render required EDA charts with Plotly."""
    st.subheader("EDA Charts")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        demand_fig = px.scatter(
            df.sample(min(1000, len(df)), random_state=42),
            x="demand",
            y="optimal_price",
            color="product_category",
            title="Demand vs Optimal Price",
            labels={"demand": "Demand", "optimal_price": "Optimal Price"},
            opacity=0.75,
        )
        st.plotly_chart(demand_fig, use_container_width=True)

    with chart_col2:
        competitor_fig = px.scatter(
            df.sample(min(1000, len(df)), random_state=7),
            x="competitor_price",
            y="optimal_price",
            color="season",
            title="Competitor Price Comparison",
            labels={
                "competitor_price": "Competitor Price",
                "optimal_price": "Optimal Price",
            },
            opacity=0.75,
        )
        st.plotly_chart(competitor_fig, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        seasonal_df = (
            df.groupby("season", as_index=False)["optimal_price"]
            .mean()
            .sort_values("optimal_price", ascending=False)
        )
        seasonal_fig = px.bar(
            seasonal_df,
            x="season",
            y="optimal_price",
            color="season",
            title="Seasonal Price Trends",
            labels={"season": "Season", "optimal_price": "Average Optimal Price"},
        )
        st.plotly_chart(seasonal_fig, use_container_width=True)

    with chart_col4:
        category_df = (
            df.groupby("product_category", as_index=False)["optimal_price"]
            .mean()
            .sort_values("optimal_price", ascending=False)
        )
        category_fig = px.bar(
            category_df,
            x="product_category",
            y="optimal_price",
            color="product_category",
            title="Category-wise Average Pricing",
            labels={
                "product_category": "Product Category",
                "optimal_price": "Average Optimal Price",
            },
        )
        st.plotly_chart(category_fig, use_container_width=True)

    inventory_fig = px.scatter(
        df.sample(min(1200, len(df)), random_state=21),
        x="inventory",
        y="optimal_price",
        color="product_category",
        title="Inventory vs Optimal Price",
        labels={"inventory": "Inventory", "optimal_price": "Optimal Price"},
        opacity=0.7,
    )
    st.plotly_chart(inventory_fig, use_container_width=True)


def show_prediction_form(df: pd.DataFrame) -> None:
    """Render sidebar prediction form and result section."""
    st.sidebar.header("Price Prediction Form")

    category = st.sidebar.selectbox("Product Category", available_categories())
    demand = st.sidebar.slider("Demand", min_value=20, max_value=1000, value=500, step=10)
    inventory = st.sidebar.slider(
        "Inventory", min_value=5, max_value=1500, value=400, step=10
    )
    competitor_price = st.sidebar.number_input(
        "Competitor Price", min_value=1.0, max_value=5000.0, value=120.0, step=5.0
    )
    season = st.sidebar.selectbox("Season", available_seasons())
    customer_rating = st.sidebar.slider(
        "Customer Rating", min_value=2.5, max_value=5.0, value=4.2, step=0.1
    )
    historical_price = st.sidebar.number_input(
        "Historical Price", min_value=1.0, max_value=5000.0, value=125.0, step=5.0
    )

    st.subheader("Recommended Price Prediction")

    if st.sidebar.button("Predict Optimal Price", type="primary"):
        predicted_price = predict_price(
            category=category,
            demand=demand,
            inventory=inventory,
            competitor_price=competitor_price,
            season=season,
            customer_rating=customer_rating,
            historical_price=historical_price,
        )

        recommendations = business_recommendation(
            predicted_price=predicted_price,
            competitor_price=competitor_price,
            demand=demand,
            inventory=inventory,
            rating=customer_rating,
        )

        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Recommended Optimal Price", f"${predicted_price:,.2f}")
        result_col2.metric(
            "Competitor Difference", f"${predicted_price - competitor_price:,.2f}"
        )
        result_col3.metric(
            "Estimated Margin Signal",
            "Positive" if predicted_price >= historical_price else "Watch",
        )

        st.markdown("#### Business Recommendation")
        for text in recommendations.values():
            st.markdown(
                f"<div class='recommendation-box'>{text}</div>",
                unsafe_allow_html=True,
            )

        comparison_data = pd.DataFrame(
            {
                "Price Type": ["Competitor Price", "Historical Price", "Recommended Price"],
                "Price": [competitor_price, historical_price, predicted_price],
            }
        )
        comparison_fig = px.bar(
            comparison_data,
            x="Price Type",
            y="Price",
            color="Price Type",
            title="Input Price Comparison",
        )
        st.plotly_chart(comparison_fig, use_container_width=True)
    else:
        st.info("Enter product details in the sidebar and click Predict Optimal Price.")


def show_model_metrics(metrics: dict) -> None:
    """Render model performance metrics."""
    st.subheader("Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error", metrics.get("MAE", "N/A"))
    col2.metric("Root Mean Squared Error", metrics.get("RMSE", "N/A"))
    col3.metric("R2 Score", metrics.get("R2 Score", "N/A"))

    st.caption(
        "The model uses RandomForestRegressor with one-hot encoding for categorical fields. Scaling is not required for random forest models."
    )


def main() -> None:
    """Main Streamlit dashboard."""
    inject_custom_css()
    ensure_project_files()

    df = get_dataset()
    model_package = get_model_package()
    metrics = model_package.get("metrics", {})

    show_header(df, metrics)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Dataset", "EDA Charts", "Prediction", "Model Metrics"]
    )

    with tab1:
        show_dataset_preview(df)

    with tab2:
        show_eda_charts(df)

    with tab3:
        show_prediction_form(df)

    with tab4:
        show_model_metrics(metrics)


if __name__ == "__main__":
    main()
