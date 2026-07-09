"""
Quick Start Script

Runs a simplified version of the pipeline on a sample of the data
for quick testing and validation.
"""

import logging
import sys
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_start():
    """Run a quick test of the pipeline on sample data."""
    
    logger.info("="*60)
    logger.info("QUICK START - AIR QUALITY ML PIPELINE")
    logger.info("="*60)
    
    try:
        # Check if dataset exists
        if not Path("clean_air_dataset.csv").exists():
            logger.error("Dataset 'clean_air_dataset.csv' not found!")
            logger.info("Please place your dataset in the root directory.")
            return False
        
        # Load sample data (10,000 rows for quick testing)
        logger.info("\nLoading sample data (10,000 rows)...")
        df = pd.read_csv("clean_air_dataset.csv", nrows=10000)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Quick data check
        logger.info("\nDataset Info:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing values: {df.isnull().sum().sum()}")
        logger.info(f"  Duplicates: {df.duplicated().sum()}")
        
        # Show column names
        logger.info(f"\nColumns: {df.columns.tolist()}")
        
        # Show target distributions
        if 'responsible_gas' in df.columns:
            logger.info("\nResponsible Gas Distribution:")
            logger.info(df['responsible_gas'].value_counts())
        
        if 'health_status' in df.columns:
            logger.info("\nHealth Status Distribution:")
            logger.info(df['health_status'].value_counts())
        
        if 'final_aqi_value' in df.columns:
            logger.info(f"\nAQI Statistics:")
            logger.info(f"  Mean: {df['final_aqi_value'].mean():.2f}")
            logger.info(f"  Min: {df['final_aqi_value'].min():.2f}")
            logger.info(f"  Max: {df['final_aqi_value'].max():.2f}")
        
        # Import and test key modules
        logger.info("\n" + "="*60)
        logger.info("Testing Pipeline Components")
        logger.info("="*60)
        
        # Test 1: Governorate Mapper
        logger.info("\n1. Testing Governorate Mapper...")
        from governorate_mapper import GovernorateMapper
        mapper = GovernorateMapper()
        test_gov = mapper.map_governorate("al qahirah")
        logger.info(f"   ✓ Mapped 'al qahirah' to '{test_gov}'")
        
        # Test 2: Data Cleaner
        logger.info("\n2. Testing Data Cleaner...")
        from data_cleaner import DataCleaner
        cleaner = DataCleaner("clean_air_dataset.csv")
        cleaner.load_data()
        missing = cleaner.detect_missing_values()
        logger.info(f"   ✓ Detected missing values in {len(missing)} columns")
        
        # Test 3: Preprocessor
        logger.info("\n3. Testing Preprocessor...")
        from preprocessor import Preprocessor
        preprocessor = Preprocessor(scaler_type='standard')
        logger.info(f"   ✓ Preprocessor initialized with standard scaler")
        
        # Test 4: Feature Engineer
        logger.info("\n4. Testing Feature Engineer...")
        from feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        sample_df = df.head(100).copy()
        engineered = fe.transform(sample_df)
        new_features = len(engineered.columns) - len(sample_df.columns)
        logger.info(f"   ✓ Created {new_features} new features")
        
        # Test 5: Model Trainer
        logger.info("\n5. Testing Model Trainer...")
        from model_trainer import ModelTrainer
        trainer = ModelTrainer(random_state=42)
        logger.info(f"   ✓ Model Trainer initialized")
        
        # Success!
        logger.info("\n" + "="*60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("="*60)
        
        logger.info("\nYour environment is ready! Next steps:")
        logger.info("\n1. Run full pipeline:")
        logger.info("   python src/main_pipeline.py")
        logger.info("\n2. Or run individual components:")
        logger.info("   - EDA: python -c 'from src.eda_module import EDAModule; eda = EDAModule(\"clean_air_dataset.csv\"); eda.save_all_outputs()'")
        logger.info("   - Clean: python -c 'from src.data_cleaner import DataCleaner; cleaner = DataCleaner(\"clean_air_dataset.csv\"); cleaner.clean_all()'")
        logger.info("\n3. After training, run dashboard:")
        logger.info("   streamlit run src/dashboard.py")
        
        return True
        
    except ImportError as e:
        logger.error(f"\nImport Error: {e}")
        logger.info("\nPlease install required packages:")
        logger.info("  pip install -r requirements.txt")
        return False
        
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = quick_start()
    sys.exit(0 if success else 1)
