
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

@st.cache_resource
def load_data():
    data = pd.read_excel('/content/drive/MyDrive/Jubilee Project/jubilee data/mywork/clean excel sheets/all_data_cleaned_final.xlsx')
    
    columns_to_convert = ['BHP, psia', 'THP, psia']
    for column in columns_to_convert:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    data = data[data['WellName'] != 'J68-P']
    data = data.drop(['WellId', 'DepthReference, mTVDSS', 'FluidDensity', 'ReservoirPressure', 
                      'Top perf DepthReference, mTVDSS', 'Tubing head DepthReference, mTVDSS', 
                      'Easting', 'Northing', 'Category'], axis=1)

    columns_to_fill = ['Oil, stb/d', 'BHP, psia', 'THP, psia', 'Choke, %', 'WI Rate, b/d', 
                       'GI Rate, MMscf/d', 'Water, b/d', 'Gas, MMscf/d']
    data[columns_to_fill] = data[columns_to_fill].fillna(0)
    data.dropna(inplace=True)

    data['Gas, MMscf/d'] = pd.to_numeric(data['Gas, MMscf/d'], errors='coerce')
    data['Gas, MMscf/d'] = data['Gas, MMscf/d'].fillna(0)

    data['Actual Water Cut'] = np.where(
        (data['Oil, stb/d'] + data['Water, b/d']) > 0,
        data['Water, b/d'] / (data['Oil, stb/d'] + data['Water, b/d']),
        0
    )

    data['Actual GOR'] = np.where(
        data['Oil, stb/d'] > 0,
        data['Gas, MMscf/d'] * 1e6 / data['Oil, stb/d'],
        0
    )

    data['Actual WOR'] = np.where(
        data['Oil, stb/d'] > 0,
        data['Water, b/d'] / data['Oil, stb/d'],
        0
    )

    data['Date'] = pd.to_datetime(data['Date'])
    data['Date_Ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    data['Day_of_Week'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['DayOfYear'] = data['Date'].dt.dayofyear

    data['Month_sin'] = np.sin(2 * np.pi * data['Month']/12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month']/12)
    data['DayOfYear_sin'] = np.sin(2 * np.pi * data['DayOfYear']/365)
    data['DayOfYear_cos'] = np.cos(2 * np.pi * data['DayOfYear']/365)

    for lag in [1, 7, 30, 60, 90]:
        data[f'Oil_lag_{lag}'] = data.groupby('WellName')['Oil, stb/d'].shift(lag)
        data[f'Water_lag_{lag}'] = data.groupby('WellName')['Water, b/d'].shift(lag)
        data[f'Gas_lag_{lag}'] = data.groupby('WellName')['Gas, MMscf/d'].shift(lag)

    for window in [7, 30, 90]:
        data[f'Oil_rolling_mean_{window}'] = data.groupby('WellName')['Oil, stb/d'].rolling(window=window).mean().reset_index(0, drop=True)
        data[f'Oil_rolling_std_{window}'] = data.groupby('WellName')['Oil, stb/d'].rolling(window=window).std().reset_index(0, drop=True)
        data[f'Water_rolling_mean_{window}'] = data.groupby('WellName')['Water, b/d'].rolling(window=window).mean().reset_index(0, drop=True)
        data[f'Water_rolling_std_{window}'] = data.groupby('WellName')['Water, b/d'].rolling(window=window).std().reset_index(0, drop=True)
        data[f'Gas_rolling_mean_{window}'] = data.groupby('WellName')['Gas, MMscf/d'].rolling(window=window).mean().reset_index(0, drop=True)
        data[f'Gas_rolling_std_{window}'] = data.groupby('WellName')['Gas, MMscf/d'].rolling(window=window).std().reset_index(0, drop=True)

    data['Days_Since_Start'] = data.groupby('WellName')['Date'].transform(lambda x: (x - x.min()).dt.days)
    data['Cumulative_Oil'] = data.groupby('WellName')['Oil, stb/d'].cumsum()
    data['Cumulative_Water'] = data.groupby('WellName')['Water, b/d'].cumsum()
    data['Cumulative_Gas'] = data.groupby('WellName')['Gas, MMscf/d'].cumsum()

    return data

@st.cache_resource
def load_models():
    global_model_oil = joblib.load('/content/drive/MyDrive/Jubilee Project/test/global_model_oil_checkpoint.pkl')
    global_model_water = joblib.load('/content/drive/MyDrive/Jubilee Project/test/global_model_water_checkpoint.pkl')
    global_model_gas = joblib.load('/content/drive/MyDrive/Jubilee Project/test/global_model_gas_checkpoint.pkl')

    cluster_models_oil = {}
    cluster_models_water = {}
    cluster_models_gas = {}
    for cluster in range(3):
        cluster_models_oil[cluster] = joblib.load(f'/content/drive/MyDrive/Jubilee Project/test/cluster_model_oil_checkpoint_{cluster}.pkl')
        cluster_models_water[cluster] = joblib.load(f'/content/drive/MyDrive/Jubilee Project/test/cluster_model_water_checkpoint_{cluster}.pkl')
        cluster_models_gas[cluster] = joblib.load(f'/content/drive/MyDrive/Jubilee Project/test/cluster_model_gas_checkpoint_{cluster}.pkl')

    well_models_oil = {}
    well_models_water = {}
    well_models_gas = {}
    production_wells = [f'J{num:02d}-P' for num in range(1, 69) if num != 68]
    for well in production_wells:
        well_models_oil[well] = joblib.load(f'/content/drive/MyDrive/Jubilee Project/test/well_model_oil_{well}.pkl')
        well_models_water[well] = joblib.load(f'/content/drive/MyDrive/Jubilee Project/test/well_model_water_{well}.pkl')
        well_models_gas[well] = joblib.load(f'/content/drive/MyDrive/Jubilee Project/test/well_model_gas_{well}.pkl')

    return global_model_oil, global_model_water, global_model_gas, cluster_models_oil, cluster_models_water, cluster_models_gas, well_models_oil, well_models_water, well_models_gas

def simulate_injection_change(data, injection_well, rate_change):
    modified_data = data.copy()
    injection_rate_cols = [col for col in data.columns if col.startswith(f'{injection_well}_') and ('WI Rate' in col or 'GI Rate' in col)]
    for col in injection_rate_cols:
        modified_data[col] *= (1 + rate_change / 100)
    return modified_data

def forecast(models, data, features):
    global_model_oil, global_model_water, global_model_gas, cluster_models_oil, cluster_models_water, cluster_models_gas, well_models_oil, well_models_water, well_models_gas = models
    forecasts = {}
    for well in data['WellName'].unique():
        well_data = data[data['WellName'] == well].sort_values('Date')
        well_features = features[features['WellName'] == well]
        well_cluster = well_data['WellCluster'].iloc[0]

        global_pred_oil = global_model_oil.predict(well_features)
        global_pred_water = global_model_water.predict(well_features)
        global_pred_gas = global_model_gas.predict(well_features)

        cluster_pred_oil = cluster_models_oil[well_cluster].predict(well_features)
        cluster_pred_water = cluster_models_water[well_cluster].predict(well_features)
        cluster_pred_gas = cluster_models_gas[well_cluster].predict(well_features)

        well_pred_oil = well_models_oil[well].predict(well_features)
        well_pred_water = well_models_water[well].predict(well_features)
        well_pred_gas = well_models_gas[well].predict(well_features)

        forecasts[well] = {
            'Oil': (global_pred_oil + cluster_pred_oil + well_pred_oil) / 3,
            'Water': (global_pred_water + cluster_pred_water + well_pred_water) / 3,
            'Gas': (global_pred_gas + cluster_pred_gas + well_pred_gas) / 3
        }
    return forecasts

def main():
    st.title('Oil Field Management Forecasting Tool')

    data = load_data()
    models = load_models()

    st.sidebar.header('Injection Well Parameters')
    injection_wells = sorted(set([col.split('_')[0] for col in data.columns if 'WI Rate' in col or 'GI Rate' in col]))
    selected_injection_well = st.sidebar.selectbox('Select Injection Well', injection_wells)
    injection_rate_change = st.sidebar.slider('Injection Rate Change (%)', -50, 50, 0)

    features = data.drop(['Oil, stb/d', 'Water, b/d', 'Gas, MMscf/d', 'Date', 'Actual Water Cut', 'Actual GOR', 'Actual WOR'], axis=1)
    test_size = int(0.3 * len(data))
    train_data = data.iloc[:-test_size]
    test_data = data.iloc[-test_size:]

    baseline_forecast = forecast(models, data, features)
    modified_data = simulate_injection_change(data, selected_injection_well, injection_rate_change)
    modified_forecast = forecast(models, modified_data, features)

    st.header('Forecasting Results')
    producing_wells = sorted(set([col.split('_')[0] for col in data.columns if '_WaterCut' in col]))
    selected_producing_well = st.selectbox('Select Producing Well to Visualize', producing_wells)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18))
    ax1.plot(data['Date'], data['Oil, stb/d'], label='Actual Oil Production', alpha=0.5)
    ax1.plot(data['Date'], baseline_forecast[selected_producing_well]['Oil'], label='Baseline Forecast Oil', linestyle='--', alpha=0.5)
    ax1.plot(data['Date'], modified_forecast[selected_producing_well]['Oil'], label='Modified Forecast Oil', linestyle=':', alpha=0.5)
    ax1.set_title('Oil Production')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Oil (stb/d)')
    ax1.legend()

    ax2.plot(data['Date'], data['Water, b/d'], label='Actual Water Production', alpha=0.5)
    ax2.plot(data['Date'], baseline_forecast[selected_producing_well]['Water'], label='Baseline Forecast Water', linestyle='--', alpha=0.5)
    ax2.plot(data['Date'], modified_forecast[selected_producing_well]['Water'], label='Modified Forecast Water', linestyle=':', alpha=0.5)
    ax2.set_title('Water Production')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Water (b/d)')
    ax2.legend()

    ax3.plot(data['Date'], data['Gas, MMscf/d'], label='Actual Gas Production', alpha=0.5)
    ax3.plot(data['Date'], baseline_forecast[selected_producing_well]['Gas'], label='Baseline Forecast Gas', linestyle='--', alpha=0.5)
    ax3.plot(data['Date'], modified_forecast[selected_producing_well]['Gas'], label='Modified Forecast Gas', linestyle=':', alpha=0.5)
    ax3.set_title('Gas Production')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Gas (MMscf/d)')
    ax3.legend()

    st.pyplot(fig)

    st.header('Summary Statistics (Test Set)')
    st.write(f"Selected Producing Well: {selected_producing_well}")

    test_data = data.iloc[-test_size:]
    for prod_type in ['Oil', 'Water', 'Gas']:
        baseline_rmse = mean_squared_error(test_data[f'{prod_type}, stb/d' if prod_type == 'Oil' else f'{prod_type}, b/d' if prod_type == 'Water' else f'{prod_type}, MMscf/d'], baseline_forecast[selected_producing_well][prod_type], squared=False)
        modified_rmse = mean_squared_error(test_data[f'{prod_type}, stb/d' if prod_type == 'Oil' else f'{prod_type}, b/d' if prod_type == 'Water' else f'{prod_type}, MMscf/d'], modified_forecast[selected_producing_well][prod_type], squared=False)
        st.write(f"{prod_type} Production:")
        st.write(f"  Baseline RMSE: {baseline_rmse:.2f}")
        st.write(f"  Modified RMSE: {modified_rmse:.2f}")

if __name__ == "__main__":
    main()
