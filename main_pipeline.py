"""
Main Pipeline Script

Executes the complete ML pipeline from EDA to deployment.
"""

import logging
import sys
from pathlib import Path

# Import all modules
from governorate_mapper import GovernorateMapper
from eda_module import EDAModule
from data_cleaner import DataCleaner
from preprocessor import Preprocessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from deployment_manager import DeploymentManager
from monitor import Monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def run_full_pipeline(dataset_path: str = "clean_air_dataset.csv",
                     run_eda: bool = True,
                     run_cleaning: bool = True):
    """
    Execute the complete ML pipeline.
    
    Args:
        dataset_path: Path to the dataset
        run_eda: Whether to run EDA (can skip if already done)
        run_cleaning: Whether to run data cleaning (can skip if already done)
    """
    logger.info("="*60)
    logger.info("STARTING AIR QUALITY ML PIPELINE")
    logger.info("="*60)
    
    try:
        # ========================================
        # STEP 1: EDA (Optional)
        # ========================================
        if run_eda:
            logger.info("\n" + "="*60)
            logger.info("STEP 1: EXPLORATORY DATA ANALYSIS")
            logger.info("="*60)
            
            eda = EDAModule(dataset_path, "eda_outputs")
            eda.save_all_outputs()
            
            logger.info("EDA complete. Review outputs in 'eda_outputs/' directory")
            input("\nPress Enter to continue to data cleaning...")
        
        # ========================================
        # STEP 2: Data Cleaning
        # ========================================
        if run_cleaning:
            logger.info("\n" + "="*60)
            logger.info("STEP 2: DATA CLEANING")
            logger.info("="*60)
            
            cleaner = DataCleaner(dataset_path)
            cleaned_df = cleaner.clean_all("cleaned_air_dataset.csv")
            
            logger.info("Data cleaning complete")
            dataset_path = "cleaned_air_dataset.csv"
        
        # ========================================
        # STEP 3: Governorate Mapping
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("STEP 3: GOVERNORATE MAPPING")
        logger.info("="*60)
        
        gov_mapper = GovernorateMapper()
        gov_mapper.save_mapping_reference()
        
        # ========================================
        # STEP 4: Preprocessing
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("STEP 4: PREPROCESSING")
        logger.info("="*60)
        
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        # Apply governorate mapping
        df['governorate'] = df['governorate'].apply(gov_mapper.map_governorate)
        
        preprocessor = Preprocessor(scaler_type='standard')
        df_preprocessed = preprocessor.fit_transform(df)
        
        logger.info("Preprocessing complete")
        
        # ========================================
        # STEP 5: Feature Engineering
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("STEP 5: FEATURE ENGINEERING")
        logger.info("="*60)
        
        feature_engineer = FeatureEngineer()
        df_engineered = feature_engineer.transform(df_preprocessed)
        feature_engineer.save_feature_documentation()
        
        logger.info("Feature engineering complete")
        
        # ========================================
        # STEP 6: Model Training
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("STEP 6: MODEL TRAINING")
        logger.info("="*60)
        
        # Prepare features and targets
        X = df_engineered.drop(columns=['final_aqi_value', 'responsible_gas', 'health_status'])
        y_aqi = df_engineered['final_aqi_value']
        y_gas = df_engineered['responsible_gas']
        y_health = df_engineered['health_status']
        
        # Create trainer and split data
        trainer = ModelTrainer(random_state=42)
        data_splits = trainer.create_data_splits(X, y_aqi, y_gas, y_health)
        
        # Train models
        logger.info("\nTraining AQI model...")
        trainer.train_aqi_model(
            data_splits['X_train'], data_splits['y_aqi_train'],
            data_splits['X_val'], data_splits['y_aqi_val']
        )
        
        logger.info("\nTraining Gas model...")
        trainer.train_gas_model(
            data_splits['X_train'], data_splits['y_gas_train'],
            data_splits['X_val'], data_splits['y_gas_val']
        )
        
        logger.info("\nTraining Health model...")
        trainer.train_health_model(
            data_splits['X_train'], data_splits['y_health_train'],
            data_splits['X_val'], data_splits['y_health_val']
        )
        
        logger.info("Model training complete")
        
        # ========================================
        # STEP 7: Model Evaluation
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("STEP 7: MODEL EVALUATION")
        logger.info("="*60)
        
        models = trainer.get_models()
        encoders = {
            'gov_encoder': preprocessor.gov_encoder,
            'gas_encoder': preprocessor.gas_encoder,
            'health_encoder': preprocessor.health_encoder
        }
        
        evaluator = ModelEvaluator(models, encoders, "evaluation_outputs")
        evaluation_results = evaluator.evaluate_all_splits(data_splits)
        evaluator.save_evaluation_results()
        
        logger.info("Model evaluation complete")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        
        for model_name in ['aqi', 'gas', 'health']:
            logger.info(f"\n{model_name.upper()} MODEL:")
            for split in ['train', 'val', 'test']:
                logger.info(f"  {split.capitalize()}: {evaluation_results[model_name][split]}")
        
        # ========================================
        # STEP 8: Deployment
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("STEP 8: DEPLOYMENT")
        logger.info("="*60)
        
        deployment_manager = DeploymentManager(
            models, encoders, preprocessor.scaler, "models"
        )
        
        feature_list = X.columns.tolist()
        
        success = deployment_manager.deploy_all(
            evaluation_results, feature_list
        )
        
        if success:
            logger.info("Deployment successful!")
        else:
            logger.error("Deployment failed!")
            return False
        
        # ========================================
        # STEP 9: Setup Monitoring
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("STEP 9: SETUP MONITORING")
        logger.info("="*60)
        
        monitor = Monitor("monitoring_logs")
        monitor.set_baseline_metrics(evaluation_results)
        
        logger.info("Monitoring setup complete")
        
        # ========================================
        # PIPELINE COMPLETE
        # ========================================
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        logger.info("\nNext steps:")
        logger.info("1. Review evaluation outputs in 'evaluation_outputs/' directory")
        logger.info("2. Check deployed models in 'models/' directory")
        logger.info("3. Run dashboard: streamlit run dashboard.py")
        logger.info("4. Test inference: python models/inference.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Air Quality ML Pipeline")
    parser.add_argument("--dataset", type=str, default="clean_air_dataset.csv",
                       help="Path to dataset CSV file")
    parser.add_argument("--skip-eda", action="store_true",
                       help="Skip EDA step")
    parser.add_argument("--skip-cleaning", action="store_true",
                       help="Skip data cleaning step")
    
    args = parser.parse_args()
    
    success = run_full_pipeline(
        dataset_path=args.dataset,
        run_eda=not args.skip_eda,
        run_cleaning=not args.skip_cleaning
    )
    
    sys.exit(0 if success else 1)
