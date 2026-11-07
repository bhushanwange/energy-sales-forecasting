# Energy Sales Forecasting Project

A machine learning project that loads historical energy sales data, engineers time-based lag features, and trains a `RandomForestRegressor` model to forecast future sales performance compared against a baseline.

## Project Structure
```text
sales_forecasting_project/
│
├── data/
│   └── energy_sales.csv       # Raw dataset containing years and historical sales
├── src/
│   └── train_model.py         # Main Python pipeline script
└── README.md