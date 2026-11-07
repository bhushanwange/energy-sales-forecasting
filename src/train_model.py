import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- Step 1: Load Data ---
file_path = r'C:\Users\admin\Desktop\Skillpilot\sales_forecasting_project\data\energy_sales.csv'
df = pd.read_csv(file_path)

# Clean dollar signs and convert columns to numeric
df['Historical Data'] = df['Historical Data'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
df['Target'] = df['Target'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
df['Years'] = pd.to_numeric(df['Years'])

# --- Step 2: Create Time-Based Features (Lag) ---
# Create a lag feature from Historical Data
df['sales_lag_1'] = df['Historical Data'].shift(1)

# Fill initial NaN caused by shifting
df['sales_lag_1'] = df['sales_lag_1'].fillna(0)

print("--- ENGINEERED DATA ---")
print(df)

# --- Step 4: Train RandomForestRegressor ---
features = ['Years', 'sales_lag_1']
X = df[features]
y = df['Target']

# Since this dataset is very small (6 rows), we train and test on the full set or split carefully
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- Step 5: Compare MAE with Baseline ---
# Baseline model (predicting the mean of training targets)
baseline_preds = np.full_like(y_test, fill_value=y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_preds)

# Random Forest predictions
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)

print("\n--- MODEL EVALUATION RESULTS ---")
print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Random Forest MAE: {rf_mae:.2f}")