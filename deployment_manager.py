"""
Deployment Manager Module

Packages trained models and creates deployment-ready artifacts.
"""

import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DeploymentManager:
    """Manages model deployment artifacts and metadata."""
    
    def __init__(self, models: Dict, encoders: Dict, scaler, output_dir: str = "models"):
        """
        Initialize deployment manager.
        
        Args:
            models: Dictionary with trained models
            encoders: Dictionary with label encoders
            scaler: Fitted scaler object
            output_dir: Directory for saving deployment artifacts
        """
        self.models = models
        self.encoders = encoders
        self.scaler = scaler
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Deployment Manager initialized with output directory: {self.output_dir}")
    
    def save_models(self) -> None:
        """Save all models using joblib."""
        logger.info("Saving trained models...")
        
        for model_name, model in self.models.items():
            if model is not None:
                model_path = self.output_dir / f"{model_name}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"  Saved {model_name} to {model_path}")
    
    def save_encoders(self) -> None:
        """Save all encoder objects."""
        logger.info("Saving encoders...")
        
        for encoder_name, encoder in self.encoders.items():
            if encoder is not None:
                encoder_path = self.output_dir / f"{encoder_name}.pkl"
                joblib.dump(encoder, encoder_path)
                logger.info(f"  Saved {encoder_name} to {encoder_path}")
    
    def save_scaler(self) -> None:
        """Save scaler object."""
        if self.scaler is not None:
            scaler_path = self.output_dir / "scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
    
    def create_metadata(self, metrics: Dict, feature_list: list, 
                       hyperparameters: Dict = None) -> None:
        """
        Create model_metadata.json with training information.
        
        Args:
            metrics: Dictionary with model performance metrics
            feature_list: List of feature names
            hyperparameters: Dictionary with model hyperparameters (optional)
        """
        logger.info("Creating model metadata...")
        
        metadata = {
            "version": "1.0.0",
            "training_date": datetime.now().isoformat(),
            "models": {
                "aqi": {
                    "algorithm": "XGBRegressor",
                    "metrics": metrics.get('aqi', {}),
                    "hyperparameters": hyperparameters.get('aqi', {}) if hyperparameters else {}
                },
                "gas": {
                    "algorithm": "XGBClassifier",
                    "metrics": metrics.get('gas', {}),
                    "hyperparameters": hyperparameters.get('gas', {}) if hyperparameters else {}
                },
                "health": {
                    "algorithm": "XGBClassifier",
                    "metrics": metrics.get('health', {}),
                    "hyperparameters": hyperparameters.get('health', {}) if hyperparameters else {}
                }
            },
            "features": feature_list,
            "encoders": {
                "governorate_classes": self.encoders['gov_encoder'].classes_.tolist() if hasattr(self.encoders['gov_encoder'], 'classes_') else [],
                "gas_classes": self.encoders['gas_encoder'].classes_.tolist() if hasattr(self.encoders['gas_encoder'], 'classes_') else [],
                "health_classes": self.encoders['health_encoder'].classes_.tolist() if hasattr(self.encoders['health_encoder'], 'classes_') else []
            }
        }
        
        metadata_path = self.output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
    
    def create_inference_script(self) -> None:
        """Generate standalone inference script."""
        logger.info("Creating inference script...")
        
        inference_code = '''"""
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
    
    print("\\nPredictions:")
    print(f"AQI: {predictions['aqi'][0]:.2f}")
    print(f"Responsible Gas: {predictions['responsible_gas'][0]}")
    print(f"Health Status: {predictions['health_status'][0]}")
'''
        
        inference_path = self.output_dir / "inference.py"
        with open(inference_path, 'w') as f:
            f.write(inference_code)
        
        logger.info(f"Inference script saved to {inference_path}")
    
    def validate_artifacts(self) -> bool:
        """
        Load and test all saved artifacts.
        
        Returns:
            True if all artifacts are valid, False otherwise
        """
        logger.info("Validating saved artifacts...")
        
        try:
            # Check model files exist
            required_files = [
                "aqi_model.pkl",
                "gas_model.pkl",
                "health_model.pkl",
                "gov_encoder.pkl",
                "gas_encoder.pkl",
                "health_encoder.pkl",
                "scaler.pkl",
                "model_metadata.json",
                "inference.py"
            ]
            
            for filename in required_files:
                filepath = self.output_dir / filename
                if not filepath.exists():
                    logger.error(f"Missing file: {filename}")
                    return False
                logger.info(f"  ✓ {filename} exists")
            
            # Try loading models
            aqi_model = joblib.load(self.output_dir / "aqi_model.pkl")
            gas_model = joblib.load(self.output_dir / "gas_model.pkl")
            health_model = joblib.load(self.output_dir / "health_model.pkl")
            logger.info("  ✓ All models loaded successfully")
            
            # Try loading encoders
            gov_encoder = joblib.load(self.output_dir / "gov_encoder.pkl")
            gas_encoder = joblib.load(self.output_dir / "gas_encoder.pkl")
            health_encoder = joblib.load(self.output_dir / "health_encoder.pkl")
            logger.info("  ✓ All encoders loaded successfully")
            
            # Try loading scaler
            scaler = joblib.load(self.output_dir / "scaler.pkl")
            logger.info("  ✓ Scaler loaded successfully")
            
            # Validate metadata
            with open(self.output_dir / "model_metadata.json", 'r') as f:
                metadata = json.load(f)
            logger.info("  ✓ Metadata loaded successfully")
            
            logger.info("All artifacts validated successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def deploy_all(self, metrics: Dict, feature_list: list, 
                   hyperparameters: Dict = None) -> bool:
        """
        Execute complete deployment workflow.
        
        Args:
            metrics: Model performance metrics
            feature_list: List of feature names
            hyperparameters: Model hyperparameters (optional)
            
        Returns:
            True if deployment successful, False otherwise
        """
        logger.info("Starting deployment workflow...")
        
        try:
            # Save all artifacts
            self.save_models()
            self.save_encoders()
            self.save_scaler()
            self.create_metadata(metrics, feature_list, hyperparameters)
            self.create_inference_script()
            
            # Validate
            if self.validate_artifacts():
                logger.info("Deployment completed successfully!")
                return True
            else:
                logger.error("Deployment validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Deployment Manager module loaded")
