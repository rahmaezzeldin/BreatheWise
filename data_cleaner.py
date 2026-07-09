"""
Data Cleaner Module

Ensures data quality by handling missing values, duplicates, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List
import logging
import json

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning operations for air quality dataset."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize data cleaner.
        
        Args:
            dataset_path: Path to raw dataset CSV file
        """
        self.dataset_path = dataset_path
        self.df = None
        self.cleaning_log = {
            'missing_values': {},
            'duplicates_removed': 0,
            'invalid_governorates': 0,
            'out_of_range_values': {},
            'rows_before': 0,
            'rows_after': 0
        }
        
        # Valid governorates in Egypt
        self.valid_governorates = [
            'Cairo', 'Alexandria', 'Port Said', 'Suez', 'Damietta',
            'Dakahlia', 'Sharqia', 'Gharbia', 'Monufia', 'Qalyubia',
            'Kafr El Sheikh', 'Beheira', 'Ismailia', 'Giza', 'Beni Suef',
            'Fayoum', 'Minya', 'Asyut', 'Sohag', 'Qena', 'Luxor', 'Aswan',
            'Red Sea', 'New Valley', 'Matrouh', 'North Sinai', 'South Sinai'
        ]
        
        # Realistic ranges for numerical features
        self.numerical_ranges = {
            'pm2_5': (0, 500),
            'pm10': (0, 500),
            'co': (0, 50000),
            'no': (0, 1000),
            'no2': (0, 1000),
            'o3': (0, 500),
            'so2': (0, 1000),
            'nh3': (0, 500),
            'temperature': (-20, 50),
            'humidity': (0, 100),
            'pressure': (900, 1100),
            'wind_speed': (0, 50),
            'wind_direction': (0, 360),
            'precipitation': (0, 500),
            'final_aqi_value': (0, 500)
        }
        
        logger.info("Data Cleaner initialized")
    
    def load_data(self) -> None:
        """Load dataset from CSV file."""
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.cleaning_log['rows_before'] = len(self.df)
            logger.info(f"Loaded dataset: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def detect_missing_values(self) -> pd.DataFrame:
        """
        Return count and percentage of missing values per column.
        
        Returns:
            DataFrame with missing value statistics
        """
        logger.info("Detecting missing values...")
        
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'missing_count': missing_count,
            'missing_percentage': missing_pct
        })
        
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(
            'missing_count', ascending=False
        )
        
        # Log to cleaning log
        self.cleaning_log['missing_values'] = missing_df.to_dict('index')
        
        if len(missing_df) > 0:
            logger.info(f"Found missing values in {len(missing_df)} columns")
            for col, row in missing_df.iterrows():
                logger.info(f"  {col}: {row['missing_count']} ({row['missing_percentage']:.2f}%)")
        else:
            logger.info("No missing values found")
        
        return missing_df
    
    def impute_numerical(self, strategy: str = 'median') -> None:
        """
        Impute missing numerical values using median or mean.
        
        Args:
            strategy: 'median' or 'mean'
        """
        logger.info(f"Imputing numerical values using {strategy}...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                if strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mean':
                    fill_value = self.df[col].mean()
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                self.df[col].fillna(fill_value, inplace=True)
                logger.info(f"  Imputed {col} with {strategy}: {fill_value:.2f}")
    
    def impute_categorical(self, strategy: str = 'mode') -> None:
        """
        Impute missing categorical values using mode.
        
        Args:
            strategy: 'mode' (most frequent value)
        """
        logger.info(f"Imputing categorical values using {strategy}...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                if strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                self.df[col].fillna(fill_value, inplace=True)
                logger.info(f"  Imputed {col} with {strategy}: {fill_value}")
    
    def remove_duplicates(self) -> int:
        """
        Remove duplicate rows.
        
        Returns:
            Count of duplicates removed
        """
        logger.info("Removing duplicate rows...")
        
        before_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after_count = len(self.df)
        
        duplicates_removed = before_count - after_count
        self.cleaning_log['duplicates_removed'] = duplicates_removed
        
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return duplicates_removed
    
    def validate_governorates(self, valid_governorates: List[str] = None) -> None:
        """
        Ensure all governorate values are valid.
        
        Args:
            valid_governorates: List of valid governorate names (optional)
        """
        if valid_governorates is None:
            valid_governorates = self.valid_governorates
        
        logger.info("Validating governorate values...")
        
        if 'governorate' not in self.df.columns:
            logger.warning("No 'governorate' column found")
            return
        
        # Import and use the governorate mapper
        from governorate_mapper import GovernorateMapper
        mapper = GovernorateMapper()
        
        # Map all governorate names to standardized format
        self.df['governorate'] = self.df['governorate'].apply(mapper.map_governorate)
        
        # Find invalid governorates (those that couldn't be mapped)
        invalid_mask = ~self.df['governorate'].isin(valid_governorates)
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            invalid_govs = self.df[invalid_mask]['governorate'].unique()
            logger.warning(f"Found {invalid_count} rows with invalid governorates: {invalid_govs}")
            
            # Remove invalid governorates
            self.df = self.df[~invalid_mask]
            self.cleaning_log['invalid_governorates'] = invalid_count
            
            logger.info(f"Removed {invalid_count} rows with invalid governorates")
        else:
            logger.info("All governorate values are valid")
    
    def validate_numerical_ranges(self, ranges: Dict[str, Tuple[float, float]] = None) -> None:
        """
        Check numerical features are within realistic bounds.
        
        Args:
            ranges: Dictionary mapping feature names to (min, max) tuples
        """
        if ranges is None:
            ranges = self.numerical_ranges
        
        logger.info("Validating numerical ranges...")
        
        for col, (min_val, max_val) in ranges.items():
            if col not in self.df.columns:
                continue
            
            # Find out-of-range values
            out_of_range = (self.df[col] < min_val) | (self.df[col] > max_val)
            out_of_range_count = out_of_range.sum()
            
            if out_of_range_count > 0:
                logger.warning(f"  {col}: {out_of_range_count} values out of range [{min_val}, {max_val}]")
                
                # Clip to valid range
                self.df[col] = self.df[col].clip(lower=min_val, upper=max_val)
                
                self.cleaning_log['out_of_range_values'][col] = out_of_range_count
                
                logger.info(f"  Clipped {col} to range [{min_val}, {max_val}]")
        
        logger.info("Numerical range validation complete")
    
    def get_cleaning_log(self) -> Dict[str, Any]:
        """
        Return summary of all cleaning operations.
        
        Returns:
            Dictionary with cleaning statistics
        """
        self.cleaning_log['rows_after'] = len(self.df) if self.df is not None else 0
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            else:
                return obj
        
        return convert_to_native(self.cleaning_log)
    
    def save_cleaned_dataset(self, output_path: str = "cleaned_air_dataset.csv") -> None:
        """
        Save cleaned dataset to file.
        
        Args:
            output_path: Path to save cleaned dataset
        """
        if self.df is None:
            logger.error("No data to save. Load and clean data first.")
            return
        
        try:
            self.df.to_csv(output_path, index=False)
            logger.info(f"Cleaned dataset saved to {output_path}")
            logger.info(f"Final shape: {self.df.shape}")
            
            # Save cleaning log
            log_path = output_path.replace('.csv', '_cleaning_log.json')
            with open(log_path, 'w') as f:
                json.dump(self.get_cleaning_log(), f, indent=2)
            logger.info(f"Cleaning log saved to {log_path}")
            
        except Exception as e:
            logger.error(f"Error saving cleaned dataset: {e}")
            raise
    
    def clean_all(self, output_path: str = "cleaned_air_dataset.csv") -> pd.DataFrame:
        """
        Execute all cleaning operations.
        
        Args:
            output_path: Path to save cleaned dataset
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting complete data cleaning pipeline...")
        
        if self.df is None:
            self.load_data()
        
        # Execute cleaning steps
        self.detect_missing_values()
        self.impute_numerical(strategy='median')
        self.impute_categorical(strategy='mode')
        self.remove_duplicates()
        self.validate_governorates()
        self.validate_numerical_ranges()
        
        # Save cleaned data
        self.save_cleaned_dataset(output_path)
        
        logger.info("Data cleaning complete!")
        logger.info(f"Rows before: {self.cleaning_log['rows_before']}")
        logger.info(f"Rows after: {self.cleaning_log['rows_after']}")
        logger.info(f"Rows removed: {self.cleaning_log['rows_before'] - self.cleaning_log['rows_after']}")
        
        return self.df


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run data cleaning
    cleaner = DataCleaner("clean_air_dataset.csv")
    cleaned_df = cleaner.clean_all("cleaned_air_dataset.csv")
