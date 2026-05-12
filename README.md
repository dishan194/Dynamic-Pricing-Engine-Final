# AI-Powered Dynamic Pricing Engine for E-commerce

This is an internship-level machine learning project that predicts an optimal selling price for e-commerce products. It uses product demand, inventory, season, competitor price, product category, customer rating, and historical price to recommend a practical selling price.

The project is intentionally simple, clean, and presentation-ready. It avoids enterprise architecture and focuses on a working ML flow with a professional Streamlit dashboard.

## Features

- Synthetic dataset generation with 5000 realistic e-commerce records
- Data preprocessing and missing value handling
- Categorical feature encoding using OneHotEncoder
- RandomForestRegressor model training
- Model evaluation using MAE, RMSE, and R2 Score
- Pickle-based model saving and loading
- Streamlit dashboard with sidebar prediction form
- Plotly charts for EDA and pricing analysis
- Business recommendations based on prediction results

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- RandomForestRegressor
- Streamlit
- Plotly
- Matplotlib
- Pickle

## Folder Structure

```text
dynamic_pricing_project/
|
├── app.py
├── model.py
├── train_model.py
├── dataset.csv
├── generate_dataset.py
├── requirements.txt
├── saved_model.pkl
├── utils.py
├── README.md
|
├── charts/
|
└── assets/
```

## Dataset Columns

- product_id
- product_category
- demand
- inventory
- competitor_price
- season
- customer_rating
- historical_price
- optimal_price

## Installation Steps

Open a terminal in the `dynamic_pricing_project` folder and run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the virtual environment with:

```bash
source venv/bin/activate
```

## Generate Dataset

If `dataset.csv` is missing or you want to regenerate it:

```bash
python generate_dataset.py
```

## Train Model

Train the RandomForestRegressor and save it as `saved_model.pkl`:

```bash
python train_model.py
```

The script prints:

- MAE
- RMSE
- R2 Score

## Launch Streamlit App

Run:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Dashboard Sections

- Project title and overview metrics
- Dataset preview
- EDA charts
- User input form
- Predicted optimal price
- Business recommendation
- Model performance metrics

## Charts Included

- Demand vs optimal price
- Competitor price comparison
- Seasonal price trends
- Category-wise average pricing
- Inventory vs optimal price

## Prediction Inputs

The user can enter:

- Product category
- Demand
- Inventory
- Competitor price
- Season
- Customer rating
- Historical price

The dashboard returns:

- Recommended optimal price
- Pricing suggestion
- Profit insight
- Inventory recommendation
- Rating-based note

## Screenshot Placeholders

Add your screenshots here after running the app:

```text
assets/dashboard_home.png
assets/eda_charts.png
assets/prediction_result.png
```

## Business Use Cases

- Adjust pricing during seasonal demand changes
- Compare prices against competitors
- Reduce overstock by suggesting discounts
- Protect profit margins during high demand
- Support category-level pricing analysis

## Future Improvements

- Add real e-commerce transaction data
- Add date-based sales trends
- Compare multiple ML models
- Add downloadable prediction reports
- Improve recommendation rules with business constraints

## Troubleshooting

If Streamlit cannot find the model:

```bash
python train_model.py
```

If Streamlit cannot find the dataset:

```bash
python generate_dataset.py
```

If packages are missing:

```bash
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.
