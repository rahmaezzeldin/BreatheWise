"""
Air Quality Prediction Inference Script

Loads trained models and makes predictions on new data.
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path


class AirQualityPredictor:
    """Inference class for air quality predictions."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize predictor with trained models.
        
        Args:
            model_dir: Directory containing model artifacts
        """
        self.model_dir = Path(model_dir)
        
        # Load models
        self.aqi_model = joblib.load(self.model_dir / "aqi_model.pkl")
        self.gas_model = joblib.load(self.model_dir / "gas_model.pkl")
        self.health_model = joblib.load(self.model_dir / "health_model.pkl")
        
        # Load encoders
        self.gov_encoder = joblib.load(self.model_dir / "gov_encoder.pkl")
        self.gas_encoder = joblib.load(self.model_dir / "gas_encoder.pkl")
        self.health_encoder = joblib.load(self.model_dir / "health_encoder.pkl")
        
        # Load scaler
        self.scaler = joblib.load(self.model_dir / "scaler.pkl")
        
        print("Models loaded successfully")
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data.
        
        Args:
            df: Input DataFrame with raw features
            
        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        # Encode governorate if present
        if 'governorate' in df_processed.columns:
            df_processed['governorate'] = self.gov_encoder.transform(df_processed['governorate'])
        
        # Scale numerical features
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['governorate', 'gov_id', 'year', 'month', 'hour', 'day_of_week']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if numerical_cols:
            df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])
        
        return df_processed
    
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Make predictions on input data.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess
        df_processed = self.preprocess(df)
        
        # Make predictions
        aqi_pred = self.aqi_model.predict(df_processed)
        gas_pred = self.gas_model.predict(df_processed)
        health_pred = self.health_model.predict(df_processed)
        
        # Decode predictions
        gas_names = self.gas_encoder.inverse_transform(gas_pred)
        health_names = self.health_encoder.inverse_transform(health_pred)
        
        return {
            'aqi': aqi_pred.tolist(),
            'responsible_gas': gas_names.tolist(),
            'health_status': health_names.tolist()
        }


if __name__ == "__main__":
    # Example usage
    predictor = AirQualityPredictor("models")
    
    # Create sample input
    sample_data = pd.DataFrame({
        'governorate': ['Cairo'],
        'co': [500.0],
        'no': [10.0],
        'no2': [30.0],
        'o3': [60.0],
        'so2': [20.0],
        'pm2_5': [45.0],
        'pm10': [80.0],
        'nh3': [5.0],
        'temperature': [25.0],
        'humidity': [60.0],
        'pressure': [1013.0],
        'wind_speed': [5.0],
        'wind_direction': [90.0],
        'precipitation': [0.0],
        'gov_id': [0],
        'year': [2024],
        'month': [1],
        'hour': [12],
        'day_of_week': [0],
        'pm_ratio': [0.56],
        'no2_o3': [1800.0]
    })
    
    # Make predictions
    predictions = predictor.predict(sample_data)
    
    print("\nPredictions:")
    print(f"AQI: {predictions['aqi'][0]:.2f}")
    print(f"Responsible Gas: {predictions['responsible_gas'][0]}")
    print(f"Health Status: {predictions['health_status'][0]}")
