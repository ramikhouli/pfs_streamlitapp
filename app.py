
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import re

# Function to clean column names
def clean_column_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel('/content/drive/MyDrive/Jubilee Project/jubilee data/mywork/clean excel sheets/all_data_cleaned_final.xlsx')
    
    # Specify the columns to convert
    columns_to_convert = ['BHP, psia', 'THP, psia']

    # Replace non-numeric values with NaN
    for column in columns_to_convert:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Delete the row where WellName is 'J68-P'
    data = data[data['WellName'] != 'J68-P']

    data.drop(['WellId', 'DepthReference, mTVDSS', 'FluidDensity', 'ReservoirPressure',
                'Top perf DepthReference, mTVDSS', 'Tubing head DepthReference, mTVDSS'], axis=1, inplace=True)

    # Ensure the columns are numeric, coercing errors to NaN
    data['Gas, MMscf/d'] = pd.to_numeric(data['Gas, MMscf/d'], errors='coerce')

    # Calculate water cut and handle division by zero
    data['WaterCut'] = np.where(data['Oil, stb/d'] + data['Water, b/d'] != 0, 
                                (data['Water, b/d'] / (data['Oil, stb/d'] + data['Water, b/d'])) * 100, 0)

    # Calculate GOR (Gas-Oil Ratio) and handle division by zero
    data['GOR'] = data.apply(lambda row: (row['Gas, MMscf/d'] / row['Oil, stb/d']) * 1000000 if row['Oil, stb/d'] != 0 else 0, axis=1)

    # Determine the minimum and maximum dates across all wells
    min_date = data['Date'].min()
    max_date = data['Date'].max()

    # Create a date range from the minimum to the maximum date
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    # Initialize an empty DataFrame with the date range as index
    aligned_data = pd.DataFrame(index=date_range)

    # Specify features to include for producing and injection wells
    producing_features = ['Oil, stb/d', 'Water, b/d', 'Gas, MMscf/d', 'BHP, psia', 'THP, psia', 'Choke, %', 'WaterCut', 'GOR']
    water_injection_features = ['BHP, psia', 'THP, psia', 'Choke, %', 'WI Rate, b/d']
    gas_injection_features = ['BHP, psia', 'THP, psia', 'Choke, %', 'GI Rate, MMscf/d']

    # Pivot the data to wide format, resampling to ensure daily data
    for well in data['WellName'].unique():
        well_data = data[data['WellName'] == well].set_index('Date')
        if data[data['WellName'] == well]['Category'].iloc[0] == 'P':
            well_data = well_data[producing_features]
        elif data[data['WellName'] == well]['Category'].iloc[0] == 'WI':
            well_data = well_data[water_injection_features]
        else:
            well_data = well_data[gas_injection_features]
        well_data = well_data.reindex(date_range)
        well_data.columns = [f'{well}_{col}' for col in well_data.columns]
        aligned_data = aligned_data.join(well_data)

    # Reset index to make 'Date' a column again
    aligned_data = aligned_data.reset_index().rename(columns={'index': 'Date'})

    # Fill NaN values with zero
    aligned_data = aligned_data.fillna(0)

    return aligned_data

# Prepare features and targets
def prepare_data(data):
    # Print all column names for debugging
    print("All column names:")
    print(data.columns)
    
    # Select BHP and injection rate features
    feature_cols = [col for col in data.columns if '_BHP' in col or '_WI Rate' in col or '_GI Rate' in col]
    
    # Select water cut targets for producing wells
    target_cols = [col for col in data.columns if '_WaterCut' in col]
    
    print("Feature columns:", feature_cols)
    print("Target columns:", target_cols)
    
    X = data[feature_cols].copy()
    Y = data[target_cols].copy()
    
    # Clean column names
    X.columns = [clean_column_name(col) for col in X.columns]
    Y.columns = [clean_column_name(col) for col in Y.columns]
    
    feature_cols = list(X.columns)
    target_cols = list(Y.columns)
    
    return X, Y, feature_cols, target_cols

# Train the model
def train_model(X, Y):
    models = {}
    for col in Y.columns:
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X, Y[col])
        models[col] = model
    return models

# Function to simulate changes in injection well rates
def simulate_injection_change(data, injection_well, rate_change):
    modified_data = data.copy()
    injection_rate_col = clean_column_name(f'{injection_well}_WI Rate, b/d')
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

    # Load data
    data = load_data()

    # Split the data
    train_data = data.iloc[:int(0.7*len(data))]
    test_data = data.iloc[int(0.7*len(data)):]

    # Prepare data
    X_train, Y_train, feature_cols, target_cols = prepare_data(train_data)
    X_test, Y_test, _, _ = prepare_data(test_data)

    # Print shapes for debugging
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    # Train models
    models = train_model(X_train, Y_train)

    # Sidebar for user inputs
    st.sidebar.header('Injection Well Parameters')
    injection_wells = [col.split('_')[0] for col in X_train.columns if '_WI_Rate' in col]
    selected_injection_well = st.sidebar.selectbox('Select Injection Well', injection_wells)
    injection_rate_change = st.sidebar.slider('Injection Rate Change (%)', -50, 50, 0)

    # Run baseline and modified forecasts
    baseline_forecast = forecast(models, X_test, feature_cols)
    modified_data = simulate_injection_change(X_test, selected_injection_well, injection_rate_change)
    modified_forecast = forecast(models, modified_data, feature_cols)

    # Visualize results
    st.header('Forecasting Results')
    producing_wells = [col.split('_')[0] for col in Y_test.columns]
    selected_producing_well = st.selectbox('Select Producing Well to Visualize', producing_wells)
    selected_col = clean_column_name(f'{selected_producing_well}_WaterCut')

    if selected_col in Y_test.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_data['Date'], Y_test[selected_col], label='Actual')
        ax.plot(test_data['Date'], baseline_forecast[selected_col], label='Baseline Forecast')
        ax.plot(test_data['Date'], modified_forecast[selected_col], label='Modified Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Water Cut (%) - {selected_producing_well}')
        ax.legend()
        st.pyplot(fig)

        # Display summary statistics
        st.header('Summary Statistics')
        st.write(f"Average Water Cut for {selected_producing_well} (Actual): {Y_test[selected_col].mean():.2f}%")
        st.write(f"Average Water Cut for {selected_producing_well} (Baseline Forecast): {baseline_forecast[selected_col].mean():.2f}%")
        st.write(f"Average Water Cut for {selected_producing_well} (Modified Forecast): {modified_forecast[selected_col].mean():.2f}%")

        # Calculate and display RMSE
        baseline_rmse = np.sqrt(mean_squared_error(Y_test[selected_col], baseline_forecast[selected_col]))
        modified_rmse = np.sqrt(mean_squared_error(Y_test[selected_col], modified_forecast[selected_col]))
        st.write(f"RMSE for {selected_producing_well} (Baseline Forecast): {baseline_rmse:.2f}")
        st.write(f"RMSE for {selected_producing_well} (Modified Forecast): {modified_rmse:.2f}")
    else:
        st.write(f"No data available for {selected_producing_well}")

    # Display injection well information
    st.header('Injection Well Information')
    injection_rate_col = clean_column_name(f'{selected_injection_well}_WI_Rate')
    if injection_rate_col in X_test.columns:
        st.write(f"Average Injection Rate for {selected_injection_well} (Baseline): {X_test[injection_rate_col].mean():.2f} b/d")
        st.write(f"Average Injection Rate for {selected_injection_well} (Modified): {modified_data[injection_rate_col].mean():.2f} b/d")
    else:
        st.write(f"No injection rate data available for {selected_injection_well}")

if __name__ == "__main__":
    main()
