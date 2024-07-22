
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load saved data and models
@st.cache_resource
def load_saved_data():
    test_data = pd.read_pickle('test_data.pkl')
    with open('models.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    with open('target_cols.pkl', 'rb') as f:
        target_cols = pickle.load(f)
    return test_data, models, feature_cols, target_cols

# Function to simulate changes in injection well rates
def simulate_injection_change(data, injection_well, rate_change):
    modified_data = data.copy()
    injection_rate_col = f'{injection_well}_WI Rate, b/d'
    if injection_rate_col in modified_data.columns:
        modified_data[injection_rate_col] *= (1 + rate_change/100)
    return modified_data

# Forecast function
def forecast(models, data, feature_cols):
    forecasts = {}
    for col, model in models.items():
        forecasts[col] = model.predict(data[feature_cols])
    return pd.DataFrame(forecasts)

# Main Streamlit app
def main():
    st.title('Oil Field Management Forecasting Tool')

    test_data, models, feature_cols, target_cols = load_saved_data()

    # Sidebar for user inputs
    st.sidebar.header('Injection Well Parameters')
    injection_wells = [col.split('_')[0] for col in test_data.columns if '_WI Rate, b/d' in col]
    selected_injection_well = st.sidebar.selectbox('Select Injection Well', injection_wells)
    injection_rate_change = st.sidebar.slider('Injection Rate Change (%)', -50, 50, 0)

    # Run baseline and modified forecasts
    baseline_forecast = forecast(models, test_data, feature_cols)
    modified_data = simulate_injection_change(test_data, selected_injection_well, injection_rate_change)
    modified_forecast = forecast(models, modified_data, feature_cols)

    # Visualize results
    st.header('Forecasting Results')
    producing_wells = [col.split('_')[0] for col in target_cols]
    selected_producing_well = st.selectbox('Select Producing Well to Visualize', producing_wells)
    selected_col = f'{selected_producing_well}_WaterCut'

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_data['Date'], test_data[selected_col], label='Actual')
    ax.plot(test_data['Date'], baseline_forecast[selected_col], label='Baseline Forecast')
    ax.plot(test_data['Date'], modified_forecast[selected_col], label='Modified Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Water Cut (%) - {selected_producing_well}')
    ax.legend()
    st.pyplot(fig)

    # Display summary statistics
    st.header('Summary Statistics')
    st.write(f"Average Water Cut for {selected_producing_well} (Actual): {test_data[selected_col].mean():.2f}%")
    st.write(f"Average Water Cut for {selected_producing_well} (Baseline Forecast): {baseline_forecast[selected_col].mean():.2f}%")
    st.write(f"Average Water Cut for {selected_producing_well} (Modified Forecast): {modified_forecast[selected_col].mean():.2f}%")

    # Display injection well information
    st.header('Injection Well Information')
    injection_rate_col = f'{selected_injection_well}_WI Rate, b/d'
    if injection_rate_col in test_data.columns:
        st.write(f"Average Injection Rate for {selected_injection_well} (Baseline): {test_data[injection_rate_col].mean():.2f} b/d")
        st.write(f"Average Injection Rate for {selected_injection_well} (Modified): {modified_data[injection_rate_col].mean():.2f} b/d")
    else:
        st.write(f"No injection rate data available for {selected_injection_well}")

if __name__ == "__main__":
    main()
