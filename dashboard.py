"""
Dashboard Module

Web-based visualization of live air quality predictions.
Uses Streamlit for simple deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import time
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Dashboard:
    """Web dashboard for air quality predictions."""
    
    def __init__(self, api_key: str, model_dir: str = "models"):
        """
        Initialize dashboard.
        
        Args:
            api_key: OpenWeatherMap API key
            model_dir: Directory containing trained models
        """
        self.api_key = api_key
        self.model_dir = model_dir
        
        # Load models and encoders
        self.load_models()
        
        # Load governorate data
        self.governorates_df = None
        
        # Calibration parameters (will be computed from live data)
        self.calibration_factor = None
        self.training_aqi_mean = 155.1  # From training data statistics
        
        logger.info("Dashboard initialized")
    
    def load_models(self) -> None:
        """Load trained models and preprocessing artifacts."""
        try:
            self.aqi_model = joblib.load(f"{self.model_dir}/aqi_model.pkl")
            self.gas_model = joblib.load(f"{self.model_dir}/gas_model.pkl")
            self.health_model = joblib.load(f"{self.model_dir}/health_model.pkl")
            
            self.gov_encoder = joblib.load(f"{self.model_dir}/gov_encoder.pkl")
            self.gas_encoder = joblib.load(f"{self.model_dir}/gas_encoder.pkl")
            self.health_encoder = joblib.load(f"{self.model_dir}/health_encoder.pkl")
            
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def fetch_live_data(self, governorate_file: str = "Egypt Governorates for API.xlsx") -> pd.DataFrame:
        """
        Fetch current data from API for all governorates.
        
        Args:
            governorate_file: Path to governorate coordinates file
            
        Returns:
            DataFrame with live data including API AQI
        """
        logger.info("Fetching live data from API...")
        
        # Load governorate coordinates
        gov_df = pd.read_excel(governorate_file)
        gov_df = gov_df.rename(columns={
            "Governorate Name (API Search)": "governorate",
            "Latitude (lat)": "lat",
            "Longitude (lon)": "lon"
        })
        
        results = []
        
        for _, row in gov_df.iterrows():
            try:
                # Fetch pollution data
                pollution_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={row['lat']}&lon={row['lon']}&appid={self.api_key}"
                pollution_response = requests.get(pollution_url, timeout=10)
                
                if pollution_response.status_code != 200:
                    continue
                
                pollution = pollution_response.json()
                comp = pollution["list"][0]["components"]
                
                # Get API AQI (1-5 scale from API)
                api_aqi_index = pollution["list"][0]["main"]["aqi"]
                
                # Fetch weather data
                weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={row['lat']}&lon={row['lon']}&appid={self.api_key}&units=metric"
                weather_response = requests.get(weather_url, timeout=10)
                
                if weather_response.status_code != 200:
                    continue
                
                weather = weather_response.json()
                
                # Combine data
                data = {
                    'governorate': row['governorate'],
                    'api_aqi_index': api_aqi_index,  # API AQI (1-5 scale)
                    'co': comp.get('co', 0),
                    'no': comp.get('no', 0),
                    'no2': comp.get('no2', 0),
                    'o3': comp.get('o3', 0),
                    'so2': comp.get('so2', 0),
                    'pm2_5': comp.get('pm2_5', 0),
                    'pm10': comp.get('pm10', 0),
                    'nh3': comp.get('nh3', 0),
                    'temperature': weather['main']['temp'],
                    'humidity': weather['main']['humidity'],
                    'pressure': weather['main']['pressure'],
                    'wind_speed': weather['wind'].get('speed', 0),
                    'wind_direction': weather['wind'].get('deg', 0),
                    'precipitation': weather.get('rain', {}).get('1h', 0)
                }
                
                results.append(data)
                
            except Exception as e:
                logger.warning(f"Error fetching data for {row['governorate']}: {e}")
                continue
        
        df = pd.DataFrame(results)
        logger.info(f"Fetched data for {len(df)} governorates")
        
        return df
    
    def calculate_aqi_from_pollutants(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate AQI from pollutant concentrations using US EPA formula.
        Returns the maximum AQI across all pollutants.
        
        Args:
            df: DataFrame with pollutant columns
            
        Returns:
            Series with calculated AQI values
        """
        def calculate_pm25_aqi(pm25):
            """Calculate AQI for PM2.5"""
            if pm25 <= 12.0:
                return (50 / 12.0) * pm25
            elif pm25 <= 35.4:
                return 50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1)
            elif pm25 <= 55.4:
                return 100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5)
            elif pm25 <= 150.4:
                return 150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5)
            elif pm25 <= 250.4:
                return 200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5)
            else:
                return 300 + ((500 - 300) / (500.4 - 250.5)) * (pm25 - 250.5)
        
        def calculate_pm10_aqi(pm10):
            """Calculate AQI for PM10"""
            if pm10 <= 54:
                return (50 / 54) * pm10
            elif pm10 <= 154:
                return 50 + ((100 - 50) / (154 - 55)) * (pm10 - 55)
            elif pm10 <= 254:
                return 100 + ((150 - 100) / (254 - 155)) * (pm10 - 155)
            elif pm10 <= 354:
                return 150 + ((200 - 150) / (354 - 255)) * (pm10 - 255)
            elif pm10 <= 424:
                return 200 + ((300 - 200) / (424 - 355)) * (pm10 - 355)
            else:
                return 300 + ((500 - 300) / (604 - 425)) * (pm10 - 425)
        
        def calculate_o3_aqi(o3):
            """Calculate AQI for O3 (μg/m³)"""
            # Convert μg/m³ to ppm: ppm = μg/m³ / 2000
            o3_ppm = o3 / 2000
            if o3_ppm <= 0.054:
                return (50 / 0.054) * o3_ppm
            elif o3_ppm <= 0.070:
                return 50 + ((100 - 50) / (0.070 - 0.055)) * (o3_ppm - 0.055)
            elif o3_ppm <= 0.085:
                return 100 + ((150 - 100) / (0.085 - 0.071)) * (o3_ppm - 0.071)
            elif o3_ppm <= 0.105:
                return 150 + ((200 - 150) / (0.105 - 0.086)) * (o3_ppm - 0.086)
            elif o3_ppm <= 0.200:
                return 200 + ((300 - 200) / (0.200 - 0.106)) * (o3_ppm - 0.106)
            else:
                return 300
        
        # Calculate AQI for each pollutant
        aqi_values = pd.DataFrame({
            'pm25_aqi': df['pm2_5'].apply(calculate_pm25_aqi),
            'pm10_aqi': df['pm10'].apply(calculate_pm10_aqi),
            'o3_aqi': df['o3'].apply(calculate_o3_aqi)
        })
        
        # Return maximum AQI (worst pollutant determines overall AQI)
        return aqi_values.max(axis=1)
    
    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for all governorates and calculate responsible gas from pollutants.
        
        Args:
            data: DataFrame with live data
            
        Returns:
            DataFrame with predictions and API values added
        """
        logger.info("Making predictions...")
        
        df = data.copy()
        
        # Calculate actual responsible gas from pollutant concentrations
        pollutants = {
            'co': df['co'],
            'no2': df['no2'],
            'o3': df['o3'],
            'so2': df['so2'],
            'pm2_5': df['pm2_5']
        }
        
        # Find which pollutant has highest concentration (normalized)
        # Using WHO guidelines for normalization
        normalized_pollutants = pd.DataFrame({
            'co': df['co'] / 10000,  # WHO: 10 mg/m³ = 10000 μg/m³
            'no2': df['no2'] / 200,   # WHO: 200 μg/m³
            'o3': df['o3'] / 100,     # WHO: 100 μg/m³
            'so2': df['so2'] / 500,   # WHO: 500 μg/m³
            'pm2_5': df['pm2_5'] / 25 # WHO: 25 μg/m³
        })
        
        df['api_responsible_gas'] = normalized_pollutants.idxmax(axis=1)
        
        # Calculate actual AQI from pollutant concentrations (not from API index)
        df['api_aqi'] = self.calculate_aqi_from_pollutants(df)
        
        # Map governorate names using the mapper
        from governorate_mapper import GovernorateMapper
        mapper = GovernorateMapper()
        df['governorate'] = df['governorate'].apply(mapper.map_governorate)
        
        # Save original governorate names for display
        df['governorate_name'] = df['governorate'].copy()
        
        # Filter out governorates not in the encoder
        valid_governorates = set(self.gov_encoder.classes_)
        df = df[df['governorate'].isin(valid_governorates)]
        
        if len(df) == 0:
            logger.warning("No valid governorates found after filtering")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(df)} governorates")
        
        # Add temporal features
        now = datetime.now()
        df['year'] = now.year
        df['month'] = now.month
        df['hour'] = now.hour
        df['day_of_week'] = now.weekday()
        
        # Add engineered features
        df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 1)
        df['no2_o3'] = df['no2'] * df['o3']
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['total_pollutants'] = df[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']].sum(axis=1)
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['wind_vector_x'] = df['wind_speed'] * np.cos(np.deg2rad(df['wind_direction']))
        df['wind_vector_y'] = df['wind_speed'] * np.sin(np.deg2rad(df['wind_direction']))
        
        # Encode governorate - keep the original column name
        df['governorate'] = self.gov_encoder.transform(df['governorate'])
        df['gov_id'] = df['governorate']
        
        # Prepare features in the EXACT order the model was trained with
        feature_order = [
            'governorate', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3',
            'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'precipitation',
            'gov_id', 'year', 'month', 'hour', 'day_of_week',
            'pm_ratio', 'no2_o3', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'total_pollutants', 'temp_humidity', 'wind_vector_x', 'wind_vector_y'
        ]
        
        # Select features in correct order
        X = df[feature_order]
        
        # Make predictions
        df['predicted_aqi'] = self.aqi_model.predict(X)
        gas_pred = self.gas_model.predict(X)
        health_pred = self.health_model.predict(X)
        
        df['predicted_gas'] = self.gas_encoder.inverse_transform(gas_pred)
        df['predicted_health'] = self.health_encoder.inverse_transform(health_pred)
        
        # Apply calibration to AQI predictions
        df = self.calibrate_predictions(df)
        
        # Calculate differences
        df['aqi_difference'] = df['calibrated_aqi'] - df['api_aqi']
        df['gas_match'] = (df['predicted_gas'] == df['api_responsible_gas']).astype(str)
        df['gas_match'] = df['gas_match'].map({'True': '✓', 'False': '✗'})
        
        # Restore governorate names for display
        df['governorate'] = df['governorate_name']
        
        logger.info("Predictions complete")
        
        return df
    
    def calibrate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrate model predictions to match current pollution levels.
        Uses adaptive scaling based on the difference between API AQI and raw predictions.
        
        Args:
            df: DataFrame with predictions and API AQI
            
        Returns:
            DataFrame with calibrated predictions
        """
        # Calculate calibration factor if not already computed
        if self.calibration_factor is None:
            # Use median ratio to avoid outlier influence
            ratio = df['api_aqi'] / df['predicted_aqi']
            self.calibration_factor = ratio.median()
            logger.info(f"Computed calibration factor: {self.calibration_factor:.3f}")
        
        # Apply calibration
        df['calibrated_aqi'] = df['predicted_aqi'] * self.calibration_factor
        
        # Ensure AQI stays in valid range [0, 500]
        df['calibrated_aqi'] = df['calibrated_aqi'].clip(0, 500)
        
        # Add confidence interval (±15% based on calibration uncertainty)
        df['aqi_lower'] = (df['calibrated_aqi'] * 0.85).clip(0, 500)
        df['aqi_upper'] = (df['calibrated_aqi'] * 1.15).clip(0, 500)
        
        return df
    
    def get_aqi_color(self, aqi: float) -> str:
        """Get color code for AQI value."""
        if aqi <= 50:
            return '#00E400'  # Green
        elif aqi <= 100:
            return '#FFFF00'  # Yellow
        elif aqi <= 150:
            return '#FF7E00'  # Orange
        elif aqi <= 200:
            return '#FF0000'  # Red
        elif aqi <= 300:
            return '#8F3F97'  # Purple
        else:
            return '#7E0023'  # Maroon


def run_dashboard(api_key: str):
    """
    Run the Streamlit dashboard.
    
    Args:
        api_key: OpenWeatherMap API key
    """
    st.set_page_config(page_title="Air Quality Dashboard - Egypt", layout="wide")
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = Dashboard(api_key)
    
    dashboard = st.session_state.dashboard
    
    # Title
    st.title("🌍 Air Quality Dashboard - Egypt")
    st.markdown("Real-time air quality predictions for Egyptian governorates")
    
    # Sidebar
    st.sidebar.header("Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)
    refresh_button = st.sidebar.button("🔄 Refresh Data")
    
    # Fetch and predict
    if refresh_button or 'predictions' not in st.session_state:
        with st.spinner("Fetching live data..."):
            try:
                live_data = dashboard.fetch_live_data()
                predictions = dashboard.make_predictions(live_data)
                st.session_state.predictions = predictions
                st.session_state.last_update = datetime.now()
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return
    
    predictions = st.session_state.get('predictions')
    
    if predictions is None or len(predictions) == 0:
        st.warning("No data available. Click 'Refresh Data' to fetch live data.")
        return
    
    # Display last update time
    last_update = st.session_state.get('last_update', datetime.now())
    st.sidebar.info(f"Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_aqi = predictions['calibrated_aqi'].mean()
        st.metric("Average Calibrated AQI", f"{avg_aqi:.1f}")
    
    with col2:
        api_avg_aqi = predictions['api_aqi'].mean()
        st.metric("Average API AQI", f"{api_avg_aqi:.1f}")
    
    with col3:
        avg_diff = predictions['aqi_difference'].abs().mean()
        st.metric("Avg AQI Difference", f"{avg_diff:.1f}")
    
    with col4:
        match_rate = (predictions['gas_match'] == '✓').sum() / len(predictions) * 100
        st.metric("Gas Match Rate", f"{match_rate:.1f}%")
    
    # Filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_gov = st.multiselect(
            "Select Governorates",
            options=predictions['governorate'].unique().tolist(),
            default=predictions['governorate'].unique().tolist()[:5]
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            options=['predicted_aqi', 'pm2_5', 'pm10', 'temperature'],
            index=0
        )
    
    # Filter data
    if selected_gov:
        filtered_df = predictions[predictions['governorate'].isin(selected_gov)]
    else:
        filtered_df = predictions
    
    filtered_df = filtered_df.sort_values(sort_by, ascending=False)
    
    # Main data table
    st.subheader("Governorate Data - API vs Calibrated Model Predictions")
    
    display_df = filtered_df[[
        'governorate', 
        'api_aqi', 'calibrated_aqi', 'aqi_difference',
        'aqi_lower', 'aqi_upper',
        'api_responsible_gas', 'predicted_gas', 'gas_match',
        'predicted_health',
        'pm2_5', 'pm10', 'o3', 'no2', 
        'temperature', 'humidity', 'wind_speed'
    ]].copy()
    
    display_df.columns = [
        'Governorate', 
        'API AQI', 'Model AQI', 'Diff',
        'AQI Min', 'AQI Max',
        'API Gas', 'Model Gas', 'Match',
        'Health',
        'PM2.5', 'PM10', 'O3', 'NO2', 
        'Temp (°C)', 'Humidity (%)', 'Wind (m/s)'
    ]
    
    # Format numbers
    for col in ['API AQI', 'Model AQI', 'Diff', 'AQI Min', 'AQI Max']:
        display_df[col] = display_df[col].round(1)
    for col in ['PM2.5', 'PM10', 'O3', 'NO2', 'Temp (°C)', 'Wind (m/s)']:
        display_df[col] = display_df[col].round(1)
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Calibration info
    if st.session_state.dashboard.calibration_factor:
        st.info(f"📊 Calibration Factor: {st.session_state.dashboard.calibration_factor:.3f} - Model predictions are automatically adjusted to match current pollution levels")
    
    # Charts
    st.subheader("Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # AQI Comparison
        comparison_df = pd.DataFrame({
            'API AQI': filtered_df.set_index('governorate')['api_aqi'],
            'Calibrated Model': filtered_df.set_index('governorate')['calibrated_aqi']
        })
        st.bar_chart(comparison_df)
        st.caption("AQI Comparison: API vs Calibrated Model")
    
    with col2:
        health_counts = filtered_df['predicted_health'].value_counts()
        st.bar_chart(health_counts)
        st.caption("Health Status Distribution")
    
    # Gas comparison
    st.subheader("Responsible Gas Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        api_gas_counts = filtered_df['api_responsible_gas'].value_counts()
        st.bar_chart(api_gas_counts)
        st.caption("API Responsible Gas Distribution")
    
    with col2:
        model_gas_counts = filtered_df['predicted_gas'].value_counts()
        st.bar_chart(model_gas_counts)
        st.caption("Model Predicted Gas Distribution")
    
    # Match statistics
    match_rate = (filtered_df['gas_match'] == '✓').sum() / len(filtered_df) * 100
    avg_diff = filtered_df['aqi_difference'].abs().mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gas Prediction Match Rate", f"{match_rate:.1f}%")
    with col2:
        st.metric("Average AQI Difference", f"±{avg_diff:.1f}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run dashboard
    API_KEY = "b59ca229e07baa0f2a5035c8c512ddc3"  # Replace with your API key
    run_dashboard(API_KEY)
