import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Create directories for saving models and data
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Set the style for our plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Generate synthetic data (in a real scenario, you would load actual data)
np.random.seed(42)

# Time period (years)
years = np.arange(2000, 2023)
n_years = len(years)

# Create synthetic data for multiple reservoirs
n_reservoirs = 5
reservoir_names = [f"Reservoir_{i+1}" for i in range(n_reservoirs)]

# Population data with growth trend
population = 1000000 * (1 + 0.02) ** np.arange(n_years) + np.random.normal(0, 50000, n_years)

# Rainfall data with seasonal variations and random fluctuations
rainfall = 1000 + 200 * np.sin(np.arange(n_years) * 2 * np.pi / 5) + np.random.normal(0, 100, n_years)

# Temperature data with increasing trend (climate change)
temperature = 25 + 0.05 * np.arange(n_years) + np.random.normal(0, 1, n_years)

# Water demand sectors
domestic_demand = 0.1 * population + 0.05 * temperature * population / 1000000 + np.random.normal(0, 10000, n_years)
agricultural_demand = 500000 + 20000 * temperature - 100 * rainfall + np.random.normal(0, 50000, n_years)
industrial_demand = 300000 * (1 + 0.03) ** np.arange(n_years) + np.random.normal(0, 20000, n_years)

# Total water demand
total_demand = domestic_demand + agricultural_demand + industrial_demand

# Reservoir storage capacities (in million cubic meters)
max_capacities = np.array([500, 750, 1000, 1200, 800])
current_capacities = {}

for i, reservoir in enumerate(reservoir_names):
    # Create a time series with some trend and seasonal variations
    base_capacity = max_capacities[i] * (0.7 + 0.2 * np.sin(np.arange(n_years) * 2 * np.pi / 5))
    # Add rainfall effect
    rainfall_effect = 0.0002 * rainfall * max_capacities[i]
    # Add random variations
    random_variations = np.random.normal(0, max_capacities[i] * 0.05, n_years)
    # Calculate current capacity (with constraints)
    capacity = np.minimum(base_capacity + rainfall_effect + random_variations, max_capacities[i])
    capacity = np.maximum(capacity, max_capacities[i] * 0.2)  # Minimum 20% capacity
    current_capacities[reservoir] = capacity

# Create a DataFrame with all the data
data = pd.DataFrame({
    'Year': years,
    'Population': population,
    'Rainfall': rainfall,
    'Temperature': temperature,
    'Domestic_Demand': domestic_demand,
    'Agricultural_Demand': agricultural_demand,
    'Industrial_Demand': industrial_demand,
    'Total_Demand': total_demand
})

# Add reservoir capacities
for reservoir in reservoir_names:
    data[reservoir] = current_capacities[reservoir]

# Save the dataset
data.to_csv("data/water_data.csv", index=False)
print("Dataset saved to data/water_data.csv")

# Save the reservoir information
reservoir_info = pd.DataFrame({
    'Reservoir': reservoir_names,
    'Max_Capacity': max_capacities
})
reservoir_info.to_csv("data/reservoir_info.csv", index=False)
print("Reservoir info saved to data/reservoir_info.csv")

# Train the water demand forecasting model
print("\nTraining water demand forecasting model...")

# Prepare features for the model
features = ['Population', 'Rainfall', 'Temperature', 'Year']
X = data[features]
y = data['Total_Demand']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the model
joblib.dump(rf_model, "models/water_demand_model.joblib")
print("Model saved to models/water_demand_model.joblib")

# Save feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

feature_importance.to_csv("data/feature_importance.csv", index=False)
print("Feature importance saved to data/feature_importance.csv")

# Generate and save some example scenarios for the web app
future_years = np.arange(2023, 2033)
n_future = len(future_years)

# Create scenarios
scenarios = {
    'Normal': {
        'Population': population[-1] * (1 + 0.02) ** np.arange(1, n_future + 1),
        'Rainfall': np.mean(rainfall) + np.random.normal(0, np.std(rainfall), n_future),
        'Temperature': temperature[-1] + 0.05 * np.arange(1, n_future + 1)
    },
    'Drought': {
        'Population': population[-1] * (1 + 0.02) ** np.arange(1, n_future + 1),
        'Rainfall': np.mean(rainfall) * 0.7 + np.random.normal(0, np.std(rainfall) * 0.5, n_future),
        'Temperature': temperature[-1] + 0.1 * np.arange(1, n_future + 1)
    },
    'Rapid Growth': {
        'Population': population[-1] * (1 + 0.04) ** np.arange(1, n_future + 1),
        'Rainfall': np.mean(rainfall) + np.random.normal(0, np.std(rainfall), n_future),
        'Temperature': temperature[-1] + 0.05 * np.arange(1, n_future + 1)
    }
}

# Save scenario data
for scenario_name, scenario_data in scenarios.items():
    scenario_df = pd.DataFrame({
        'Year': future_years,
        'Population': scenario_data['Population'],
        'Rainfall': scenario_data['Rainfall'],
        'Temperature': scenario_data['Temperature']
    })
    scenario_df.to_csv(f"data/scenario_{scenario_name.lower().replace(' ', '_')}.csv", index=False)
    print(f"Scenario '{scenario_name}' saved to data/scenario_{scenario_name.lower().replace(' ', '_')}.csv")

print("\nAll data and models have been saved successfully!")