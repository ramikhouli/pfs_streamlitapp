
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="Oil Field Management Forecasting Tool", layout="wide")

# Title
st.title('Oil Field Management Forecasting Tool')

# Load data and models
@st.cache_resource
def load_data_and_models():
    data = pd.read_excel('data/all_data_cleaned_final.xlsx')
    global_model_oil = joblib.load('models/global_model_oil_checkpoint.pkl')
    global_model_water = joblib.load('models/global_model_water_checkpoint.pkl')
    global_model_gas = joblib.load('models/global_model_gas_checkpoint.pkl')
    
    cluster_models_oil = {}
    cluster_models_water = {}
    cluster_models_gas = {}
    for cluster in range(3):
        cluster_models_oil[cluster] = joblib.load(f'models/cluster_model_oil_checkpoint_{cluster}.pkl')
        cluster_models_water[cluster] = joblib.load(f'models/cluster_model_water_checkpoint_{cluster}.pkl')
        cluster_models_gas[cluster] = joblib.load(f'models/cluster_model_gas_checkpoint_{cluster}.pkl')

    well_models_oil = {}
    well_models_water = {}
    well_models_gas = {}
    production_wells = [f'J{num:02d}-P' for num in range(1, 69) if num != 68]
    for well in production_wells:
        well_models_oil[well] = joblib.load(f'models/well_model_oil_{well}.pkl')
        well_models_water[well] = joblib.load(f'models/well_model_water_{well}.pkl')
        well_models_gas[well] = joblib.load(f'models/well_model_gas_{well}.pkl')

    return data, global_model_oil, global_model_water, global_model_gas, cluster_models_oil, cluster_models_water, cluster_models_gas, well_models_oil, well_models_water, well_models_gas

data, global_model_oil, global_model_water, global_model_gas, cluster_models_oil, cluster_models_water, cluster_models_gas, well_models_oil, well_models_water, well_models_gas = load_data_and_models()

# Sidebar
st.sidebar.header('Well Selection')
selected_well = st.sidebar.selectbox('Select a well', data['WellName'].unique())

st.sidebar.header('Date Range')
min_date = data['Date'].min()
max_date = data['Date'].max()
start_date = st.sidebar.date_input('Start date', min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input('End date', max_date, min_value=min_date, max_value=max_date)

# Filter data
filtered_data = data[(data['WellName'] == selected_well) & 
                     (data['Date'] >= pd.Timestamp(start_date)) & 
                     (data['Date'] <= pd.Timestamp(end_date))]

# Prepare features
def prepare_features(data):
    features = data.drop(['Oil, stb/d', 'Water, b/d', 'Gas, MMscf/d', 'Date', 'Actual Water Cut', 'Actual GOR', 'Actual WOR'], axis=1)
    return features

features = prepare_features(filtered_data)

# Make predictions
well_cluster = filtered_data['WellCluster'].iloc[0]

global_pred_oil = global_model_oil.predict(features)
global_pred_water = global_model_water.predict(features)
global_pred_gas = global_model_gas.predict(features)

cluster_pred_oil = cluster_models_oil[well_cluster].predict(features)
cluster_pred_water = cluster_models_water[well_cluster].predict(features)
cluster_pred_gas = cluster_models_gas[well_cluster].predict(features)

well_pred_oil = well_models_oil[selected_well].predict(features)
well_pred_water = well_models_water[selected_well].predict(features)
well_pred_gas = well_models_gas[selected_well].predict(features)

oil_predictions = (global_pred_oil + cluster_pred_oil + well_pred_oil) / 3
water_predictions = (global_pred_water + cluster_pred_water + well_pred_water) / 3
gas_predictions = (global_pred_gas + cluster_pred_gas + well_pred_gas) / 3

# Visualizations
st.header(f'Production Forecasts for Well {selected_well}')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

# Oil production plot
ax1.scatter(filtered_data['Date'], filtered_data['Oil, stb/d'], label='Actual', alpha=0.7)
ax1.plot(filtered_data['Date'], oil_predictions, label='Predicted', color='red')
ax1.set_title('Oil Production')
ax1.set_xlabel('Date')
ax1.set_ylabel('Oil Production (stb/d)')
ax1.legend()

# Water production plot
ax2.scatter(filtered_data['Date'], filtered_data['Water, b/d'], label='Actual', alpha=0.7)
ax2.plot(filtered_data['Date'], water_predictions, label='Predicted', color='red')
ax2.set_title('Water Production')
ax2.set_xlabel('Date')
ax2.set_ylabel('Water Production (b/d)')
ax2.legend()

# Gas production plot
ax3.scatter(filtered_data['Date'], filtered_data['Gas, MMscf/d'], label='Actual', alpha=0.7)
ax3.plot(filtered_data['Date'], gas_predictions, label='Predicted', color='red')
ax3.set_title('Gas Production')
ax3.set_xlabel('Date')
ax3.set_ylabel('Gas Production (MMscf/d)')
ax3.legend()

plt.tight_layout()
st.pyplot(fig)

# Summary statistics
st.header('Summary Statistics')
st.write(f"Average Oil Production (Actual): {filtered_data['Oil, stb/d'].mean():.2f} stb/d")
st.write(f"Average Oil Production (Predicted): {oil_predictions.mean():.2f} stb/d")
st.write(f"Average Water Production (Actual): {filtered_data['Water, b/d'].mean():.2f} b/d")
st.write(f"Average Water Production (Predicted): {water_predictions.mean():.2f} b/d")
st.write(f"Average Gas Production (Actual): {filtered_data['Gas, MMscf/d'].mean():.2f} MMscf/d")
st.write(f"Average Gas Production (Predicted): {gas_predictions.mean():.2f} MMscf/d")

# Performance metrics
st.header('Model Performance Metrics')
oil_rmse = np.sqrt(mean_squared_error(filtered_data['Oil, stb/d'], oil_predictions))
water_rmse = np.sqrt(mean_squared_error(filtered_data['Water, b/d'], water_predictions))
gas_rmse = np.sqrt(mean_squared_error(filtered_data['Gas, MMscf/d'], gas_predictions))

oil_r2 = r2_score(filtered_data['Oil, stb/d'], oil_predictions)
water_r2 = r2_score(filtered_data['Water, b/d'], water_predictions)
gas_r2 = r2_score(filtered_data['Gas, MMscf/d'], gas_predictions)

st.write(f"Oil RMSE: {oil_rmse:.2f}")
st.write(f"Water RMSE: {water_rmse:.2f}")
st.write(f"Gas RMSE: {gas_rmse:.2f}")
st.write(f"Oil R-squared: {oil_r2:.2f}")
st.write(f"Water R-squared: {water_r2:.2f}")
st.write(f"Gas R-squared: {gas_r2:.2f}")
