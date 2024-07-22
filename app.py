
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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

def simulate_injection_change(data, injection_well, rate_change):
    modified_data = data.copy()
    injection_rate_cols = [col for col in data.columns if col.startswith(f'{injection_well}_') and ('WI_Rate' in col or 'GI_Rate' in col)]
    for col in injection_rate_cols:
        modified_data[col] *= (1 + rate_change/100)
    return modified_data

def forecast(models, data, feature_cols):
    forecasts = {}
    for col, model in models.items():
        forecasts[col] = model.predict(data[feature_cols])
    return pd.DataFrame(forecasts)

def main():
    st.title('Oil Field Management Forecasting Tool')

    test_data, models, feature_cols, target_cols = load_saved_data()

    # Debug information
    st.sidebar.write("Debug Information:")
    st.sidebar.write(f"Number of columns in test_data: {len(test_data.columns)}")
    st.sidebar.write(f"Number of models: {len(models)}")
    st.sidebar.write(f"Number of feature columns: {len(feature_cols)}")
    st.sidebar.write(f"Number of target columns: {len(target_cols)}")

    # Identify injection wells
    injection_wells = sorted(set([col.split('_')[0] for col in test_data.columns if 'WI_Rate' in col or 'GI_Rate' in col]))

    # Sidebar for user inputs
    st.sidebar.header('Injection Well Parameters')
    selected_injection_well = st.sidebar.selectbox('Select Injection Well', injection_wells)
    injection_rate_change = st.sidebar.slider('Injection Rate Change (%)', -50, 50, 0)

    # Split the test_data into actual test set (last 30%)
    test_size = int(0.3 * len(test_data))
    actual_test_data = test_data.iloc[-test_size:]

    # Run baseline and modified forecasts only on the actual test set
    baseline_forecast = forecast(models, actual_test_data, feature_cols)
    modified_data = simulate_injection_change(actual_test_data, selected_injection_well, injection_rate_change)
    modified_forecast = forecast(models, modified_data, feature_cols)

    # Visualize results
    st.header('Forecasting Results')
    producing_wells = sorted(set([col.split('_')[0] for col in actual_test_data.columns if '_WaterCut' in col]))
    if not producing_wells:
        st.error("No producing wells found in the data.")
    else:
        selected_producing_well = st.selectbox('Select Producing Well to Visualize', producing_wells)
        watercut_cols = [col for col in actual_test_data.columns if col.startswith(f'{selected_producing_well}_') and col.endswith('_WaterCut')]
        
        if watercut_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in watercut_cols:
                ax.plot(actual_test_data['Date'], actual_test_data[col], label=f'Actual {col}')
                ax.plot(actual_test_data['Date'], baseline_forecast[col], label=f'Baseline Forecast {col}')
                ax.plot(actual_test_data['Date'], modified_forecast[col], label=f'Modified Forecast {col}')
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Water Cut (%) - {selected_producing_well}')
            ax.legend()
            st.pyplot(fig)

            # Display summary statistics
            st.header('Summary Statistics')
            for col in watercut_cols:
                st.write(f"Average Water Cut for {col}:")
                st.write(f"  Actual: {actual_test_data[col].mean():.2f}%")
                st.write(f"  Baseline Forecast: {baseline_forecast[col].mean():.2f}%")
                st.write(f"  Modified Forecast: {modified_forecast[col].mean():.2f}%")

            # Calculate and display RMSE
            baseline_rmse = np.sqrt(mean_squared_error(actual_test_data[col], baseline_forecast[col]))
            modified_rmse = np.sqrt(mean_squared_error(actual_test_data[col], modified_forecast[col]))

            st.write(f"RMSE for {col}:")
            st.write(f"  Baseline Forecast: {baseline_rmse:.2f}")
            st.write(f"  Modified Forecast: {modified_rmse:.2f}")

        else:
            st.error(f"No water cut data available for {selected_producing_well}")

    # Display injection well information
    st.header('Injection Well Information')
    injection_rate_cols = [col for col in actual_test_data.columns if col.startswith(f'{selected_injection_well}_') and ('WI_Rate' in col or 'GI_Rate' in col)]
    if injection_rate_cols:
        for col in injection_rate_cols:
            rate_type = 'b/d' if 'WI_Rate' in col else 'MMscf/d'
            st.write(f"Average Injection Rate for {col}:")
            st.write(f"  Baseline: {actual_test_data[col].mean():.2f} {rate_type}")
            st.write(f"  Modified: {modified_data[col].mean():.2f} {rate_type}")
    else:
        st.error(f"No injection rate data available for {selected_injection_well}")

if __name__ == "__main__":
    main()
