# Cleaned app.py for Hugging Face deployment (token-free, student-friendly)

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data generation (2010–2024)
years = list(range(2010, 2025))
water_usage = np.random.randint(50, 100, len(years))
population = [100 + i * 2 for i in range(len(years))]
rainfall = np.random.randint(700, 1200, len(years))

# Create DataFrame
df = pd.DataFrame({
    'Year': years,
    'WaterUsage': water_usage,
    'Population': population,
    'Rainfall': rainfall
})

# Train models
X = np.array(years).reshape(-1, 1)
pop_model = LinearRegression().fit(X, np.array(population))
rain_model = LinearRegression().fit(X, np.array(rainfall))
water_model = LinearRegression().fit(X, np.array(water_usage))

# Predict for future years
future_years = list(range(2010, 2035))
future_X = np.array(future_years).reshape(-1, 1)
predicted_population = pop_model.predict(future_X)
predicted_rainfall = rain_model.predict(future_X)
predicted_water = water_model.predict(future_X)

# Gap calculation
storage_capacity = 90
gap = predicted_water - storage_capacity

# Forecast table
def show_forecast():
    forecast_df = pd.DataFrame({
        'Year': future_years,
        'Predicted Water Usage': predicted_water.round(2),
        'Storage Capacity': storage_capacity,
        'Gap': gap.round(2)
    })
    return forecast_df.tail(10)

# Plot population and rainfall
def plot_trends():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(future_years, predicted_population, label="Population (millions)", color='blue')
    ax.plot(future_years, predicted_rainfall, label="Rainfall (mm)", color='green')
    ax.set_xlabel("Year")
    ax.set_ylabel("Values")
    ax.set_title("Predicted Population and Rainfall")
    ax.legend()
    ax.grid(True)
    return fig

# Plot water usage vs storage
def plot_water():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(future_years, predicted_water, label="Predicted Water Usage", color='orange')
    ax.axhline(y=storage_capacity, color='red', linestyle='--', label="Storage Capacity")
    ax.set_xlabel("Year")
    ax.set_ylabel("Water (billion cubic meters)")
    ax.set_title("Water Usage vs Storage Capacity")
    ax.legend()
    ax.grid(True)
    return fig

# Gradio Interface
demo = gr.Interface(
    fn=show_forecast,
    inputs=[],
    outputs=gr.Dataframe(headers=["Year", "Predicted Water Usage", "Storage Capacity", "Gap"]),
    title="Water Forecast Dashboard",
    description="Forecast of water usage, population, and rainfall from 2010 to 2034."
)

demo.launch()
