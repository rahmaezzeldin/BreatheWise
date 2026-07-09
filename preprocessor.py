"""
Preprocessor Module

Transforms features into model-ready format through scaling and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """Handles preprocessing operations: scaling and encoding."""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        """
        self.scaler_type = scaler_type
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Initialize encoders
        self.gov_encoder = LabelEncoder()
        self.gas_encoder = LabelEncoder()
        self.health_encoder = LabelEncoder()
        
        self.is_fitted = False
        
        logger.info(f"Preprocessor initialized with {scaler_type} scaler")
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoders/scalers and transform training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Fitting and transforming data...")
        
        df_transformed = df.copy()
        
        # Encode categorical features
        if 'governorate' in df_transformed.columns:
            df_transformed['governorate'] = self.gov_encoder.fit_transform(df_transformed['governorate'])
            logger.info(f"Encoded governorate: {len(self.gov_encoder.classes_)} classes")
        
        if 'responsible_gas' in df_transformed.columns:
            df_transformed['responsible_gas'] = self.gas_encoder.fit_transform(df_transformed['responsible_gas'])
            logger.info(f"Encoded responsible_gas: {len(self.gas_encoder.classes_)} classes")
        
        if 'health_status' in df_transformed.columns:
            df_transformed['health_status'] = self.health_encoder.fit_transform(df_transformed['health_status'])
            logger.info(f"Encoded health_status: {len(self.health_encoder.classes_)} classes")
        
        # Scale numerical features
        numerical_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude already encoded categorical columns and target variables
        exclude_cols = ['governorate', 'responsible_gas', 'health_status', 
                       'final_aqi_value', 'gov_id', 'year', 'month', 'hour', 'day_of_week']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if numerical_cols:
            df_transformed[numerical_cols] = self.scaler.fit_transform(df_transformed[numerical_cols])
            logger.info(f"Scaled {len(numerical_cols)} numerical features")
        
        self.is_fitted = True
        logger.info("Fit and transform complete")
        
        return df_transformed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform validation/test/inference data using fitted objects.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        logger.info("Transforming data...")
        
        df_transformed = df.copy()
        
        # Encode categorical features
        if 'governorate' in df_transformed.columns:
            df_transformed['governorate'] = self.gov_encoder.transform(df_transformed['governorate'])
        
        if 'responsible_gas' in df_transformed.columns:
            df_transformed['responsible_gas'] = self.gas_encoder.transform(df_transformed['responsible_gas'])
        
        if 'health_status' in df_transformed.columns:
            df_transformed['health_status'] = self.health_encoder.transform(df_transformed['health_status'])
        
        # Scale numerical features
        numerical_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['governorate', 'responsible_gas', 'health_status', 
                       'final_aqi_value', 'gov_id', 'year', 'month', 'hour', 'day_of_week']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if numerical_cols:
            df_transformed[numerical_cols] = self.scaler.transform(df_transformed[numerical_cols])
        
        logger.info("Transform complete")
        
        return df_transformed
    
    def save_artifacts(self, output_dir: str = "models") -> None:
        """
        Save scaler and encoder objects.
        
        Args:
            output_dir: Directory to save artifacts
        """
        if not self.is_fitted:
            logger.warning("Preprocessor not fitted. Nothing to save.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        scaler_path = output_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save encoders
        gov_encoder_path = output_path / "gov_encoder.pkl"
        joblib.dump(self.gov_encoder, gov_encoder_path)
        logger.info(f"Saved gov_encoder to {gov_encoder_path}")
        
        gas_encoder_path = output_path / "gas_encoder.pkl"
        joblib.dump(self.gas_encoder, gas_encoder_path)
        logger.info(f"Saved gas_encoder to {gas_encoder_path}")
        
        health_encoder_path = output_path / "health_encoder.pkl"
        joblib.dump(self.health_encoder, health_encoder_path)
        logger.info(f"Saved health_encoder to {health_encoder_path}")
        
        logger.info(f"All preprocessing artifacts saved to {output_dir}")
    
    def load_artifacts(self, artifact_dir: str = "models") -> None:
        """
        Load saved scaler and encoders.
        
        Args:
            artifact_dir: Directory containing saved artifacts
        """
        artifact_path = Path(artifact_dir)
        
        try:
            # Load scaler
            scaler_path = artifact_path / "scaler.pkl"
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
            
            # Load encoders
            gov_encoder_path = artifact_path / "gov_encoder.pkl"
            self.gov_encoder = joblib.load(gov_encoder_path)
            logger.info(f"Loaded gov_encoder from {gov_encoder_path}")
            
            gas_encoder_path = artifact_path / "gas_encoder.pkl"
            self.gas_encoder = joblib.load(gas_encoder_path)
            logger.info(f"Loaded gas_encoder from {gas_encoder_path}")
            
            health_encoder_path = artifact_path / "health_encoder.pkl"
            self.health_encoder = joblib.load(health_encoder_path)
            logger.info(f"Loaded health_encoder from {health_encoder_path}")
            
            self.is_fitted = True
            logger.info("All preprocessing artifacts loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Artifact file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def validate_encoded_values(self, df: pd.DataFrame) -> bool:
        """
        Ensure encoded values are within expected ranges.
        
        Args:
            df: DataFrame with encoded values
            
        Returns:
            True if all values are valid, False otherwise
        """
        logger.info("Validating encoded values...")
        
        valid = True
        
        # Check governorate encoding
        if 'governorate' in df.columns:
            max_gov = len(self.gov_encoder.classes_) - 1
            if df['governorate'].min() < 0 or df['governorate'].max() > max_gov:
                logger.error(f"Governorate encoding out of range: [{df['governorate'].min()}, {df['governorate'].max()}]")
                valid = False
            else:
                logger.info(f"Governorate encoding valid: [0, {max_gov}]")
        
        # Check gas encoding
        if 'responsible_gas' in df.columns:
            max_gas = len(self.gas_encoder.classes_) - 1
            if df['responsible_gas'].min() < 0 or df['responsible_gas'].max() > max_gas:
                logger.error(f"Gas encoding out of range: [{df['responsible_gas'].min()}, {df['responsible_gas'].max()}]")
                valid = False
            else:
                logger.info(f"Gas encoding valid: [0, {max_gas}]")
        
        # Check health encoding
        if 'health_status' in df.columns:
            max_health = len(self.health_encoder.classes_) - 1
            if df['health_status'].min() < 0 or df['health_status'].max() > max_health:
                logger.error(f"Health encoding out of range: [{df['health_status'].min()}, {df['health_status'].max()}]")
                valid = False
            else:
                logger.info(f"Health encoding valid: [0, {max_health}]")
        
        if valid:
            logger.info("All encoded values are valid")
        
        return valid
    
    def get_encoder_classes(self) -> Dict[str, list]:
        """
        Get the classes for each encoder.
        
        Returns:
            Dictionary with encoder classes
        """
        return {
            'governorate': self.gov_encoder.classes_.tolist() if hasattr(self.gov_encoder, 'classes_') else [],
            'responsible_gas': self.gas_encoder.classes_.tolist() if hasattr(self.gas_encoder, 'classes_') else [],
            'health_status': self.health_encoder.classes_.tolist() if hasattr(self.health_encoder, 'classes_') else []
        }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage
    logger.info("Preprocessor module loaded")
