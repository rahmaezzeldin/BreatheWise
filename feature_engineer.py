"""
Feature Engineer Module

Creates derived features that capture domain knowledge and non-linear relationships.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates engineered features for air quality prediction."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_documentation = {}
        logger.info("Feature Engineer initialized")
    
    def create_pm_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        PM2.5 / (PM10 + 1) - fine to coarse particle ratio.
        
        Args:
            df: DataFrame with pm2_5 and pm10 columns
            
        Returns:
            Series with pm_ratio values
        """
        pm_ratio = df['pm2_5'] / (df['pm10'] + 1)
        self.feature_documentation['pm_ratio'] = "pm2_5 / (pm10 + 1) - Fine to coarse particle ratio"
        return pm_ratio
    
    def create_no2_o3_interaction(self, df: pd.DataFrame) -> pd.Series:
        """
        NO2 * O3 - photochemical smog indicator.
        
        Args:
            df: DataFrame with no2 and o3 columns
            
        Returns:
            Series with no2_o3 interaction values
        """
        no2_o3 = df['no2'] * df['o3']
        self.feature_documentation['no2_o3'] = "no2 * o3 - Photochemical smog indicator"
        return no2_o3
    
    def create_cyclical_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create hour_sin, hour_cos, month_sin, month_cos.
        Preserves cyclical nature of time.
        
        Args:
            df: DataFrame with hour and month columns
            
        Returns:
            DataFrame with cyclical time features
        """
        result = pd.DataFrame(index=df.index)
        
        if 'hour' in df.columns:
            result['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            result['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            self.feature_documentation['hour_sin'] = "sin(2π * hour / 24) - Cyclical hour encoding"
            self.feature_documentation['hour_cos'] = "cos(2π * hour / 24) - Cyclical hour encoding"
        
        if 'month' in df.columns:
            result['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            result['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            self.feature_documentation['month_sin'] = "sin(2π * month / 12) - Cyclical month encoding"
            self.feature_documentation['month_cos'] = "cos(2π * month / 12) - Cyclical month encoding"
        
        return result
    
    def create_total_pollutants(self, df: pd.DataFrame) -> pd.Series:
        """
        Sum of all pollutant concentrations.
        
        Args:
            df: DataFrame with pollutant columns
            
        Returns:
            Series with total pollutant values
        """
        pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        available_cols = [col for col in pollutant_cols if col in df.columns]
        
        total_pollutants = df[available_cols].sum(axis=1)
        self.feature_documentation['total_pollutants'] = f"sum({', '.join(available_cols)}) - Overall pollution burden"
        
        return total_pollutants
    
    def create_temp_humidity_interaction(self, df: pd.DataFrame) -> pd.Series:
        """
        Temperature * Humidity - comfort index proxy.
        
        Args:
            df: DataFrame with temperature and humidity columns
            
        Returns:
            Series with temp_humidity interaction values
        """
        temp_humidity = df['temperature'] * df['humidity']
        self.feature_documentation['temp_humidity'] = "temperature * humidity - Comfort index proxy"
        return temp_humidity
    
    def create_wind_vectors(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        wind_vector_x = wind_speed * cos(wind_direction)
        wind_vector_y = wind_speed * sin(wind_direction)
        Decomposes wind into directional components.
        
        Args:
            df: DataFrame with wind_speed and wind_direction columns
            
        Returns:
            Tuple of (wind_vector_x, wind_vector_y) Series
        """
        # Convert wind direction from degrees to radians
        wind_direction_rad = np.deg2rad(df['wind_direction'])
        
        wind_vector_x = df['wind_speed'] * np.cos(wind_direction_rad)
        wind_vector_y = df['wind_speed'] * np.sin(wind_direction_rad)
        
        self.feature_documentation['wind_vector_x'] = "wind_speed * cos(wind_direction) - East-west wind component"
        self.feature_documentation['wind_vector_y'] = "wind_speed * sin(wind_direction) - North-south wind component"
        
        return wind_vector_x, wind_vector_y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features added
        """
        logger.info("Applying feature engineering transformations...")
        
        df_transformed = df.copy()
        
        # Create all engineered features
        try:
            # PM ratio
            if 'pm2_5' in df.columns and 'pm10' in df.columns:
                df_transformed['pm_ratio'] = self.create_pm_ratio(df)
                logger.info("Created pm_ratio feature")
            
            # NO2-O3 interaction
            if 'no2' in df.columns and 'o3' in df.columns:
                df_transformed['no2_o3'] = self.create_no2_o3_interaction(df)
                logger.info("Created no2_o3 feature")
            
            # Cyclical time features
            if 'hour' in df.columns or 'month' in df.columns:
                cyclical_features = self.create_cyclical_time_features(df)
                for col in cyclical_features.columns:
                    df_transformed[col] = cyclical_features[col]
                logger.info(f"Created {len(cyclical_features.columns)} cyclical time features")
            
            # Total pollutants
            pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
            if any(col in df.columns for col in pollutant_cols):
                df_transformed['total_pollutants'] = self.create_total_pollutants(df)
                logger.info("Created total_pollutants feature")
            
            # Temperature-humidity interaction
            if 'temperature' in df.columns and 'humidity' in df.columns:
                df_transformed['temp_humidity'] = self.create_temp_humidity_interaction(df)
                logger.info("Created temp_humidity feature")
            
            # Wind vectors
            if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
                wind_x, wind_y = self.create_wind_vectors(df)
                df_transformed['wind_vector_x'] = wind_x
                df_transformed['wind_vector_y'] = wind_y
                logger.info("Created wind_vector_x and wind_vector_y features")
            
            logger.info(f"Feature engineering complete. Added {len(df_transformed.columns) - len(df.columns)} new features")
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            raise
        
        return df_transformed
    
    def get_feature_documentation(self) -> Dict[str, str]:
        """
        Return dictionary of feature names and their formulas.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return self.feature_documentation
    
    def print_feature_documentation(self) -> None:
        """Print feature documentation in a readable format."""
        logger.info("\n" + "="*60)
        logger.info("ENGINEERED FEATURES DOCUMENTATION")
        logger.info("="*60)
        
        for feature_name, description in self.feature_documentation.items():
            logger.info(f"\n{feature_name}:")
            logger.info(f"  {description}")
        
        logger.info("\n" + "="*60)
    
    def save_feature_documentation(self, output_path: str = "feature_documentation.txt") -> None:
        """
        Save feature documentation to a file.
        
        Args:
            output_path: Path to save documentation
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("ENGINEERED FEATURES DOCUMENTATION\n")
            f.write("=" * 60 + "\n\n")
            
            for feature_name, description in self.feature_documentation.items():
                f.write(f"{feature_name}:\n")
                f.write(f"  {description}\n\n")
        
        logger.info(f"Feature documentation saved to {output_path}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage
    logger.info("Feature Engineer module loaded")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'pm2_5': [25.5, 45.2, 12.3],
        'pm10': [50.1, 80.5, 30.2],
        'no2': [30.0, 45.0, 20.0],
        'o3': [60.0, 40.0, 80.0],
        'co': [500, 700, 400],
        'no': [10, 15, 8],
        'so2': [20, 25, 15],
        'nh3': [5, 8, 3],
        'temperature': [25.0, 30.0, 20.0],
        'humidity': [60.0, 70.0, 50.0],
        'wind_speed': [5.0, 8.0, 3.0],
        'wind_direction': [90, 180, 270],
        'hour': [10, 14, 18],
        'month': [1, 6, 12]
    })
    
    # Apply feature engineering
    fe = FeatureEngineer()
    transformed_data = fe.transform(sample_data)
    
    logger.info(f"\nOriginal features: {len(sample_data.columns)}")
    logger.info(f"After engineering: {len(transformed_data.columns)}")
    logger.info(f"New features added: {len(transformed_data.columns) - len(sample_data.columns)}")
    
    # Print documentation
    fe.print_feature_documentation()
