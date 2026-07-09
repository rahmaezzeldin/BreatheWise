"""
Model Trainer Module

Trains three ML models with proper data splits and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains AQI, Gas, and Health models with proper validation."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.aqi_model = None
        self.gas_model = None
        self.health_model = None
        self.training_history = {}
        
        logger.info(f"Model Trainer initialized with random_state={random_state}")
    
    def create_data_splits(self, X: pd.DataFrame, y_aqi: pd.Series, 
                          y_gas: pd.Series, y_health: pd.Series) -> Dict:
        """
        Create 70/20/10 train/val/test splits with stratification.
        
        Args:
            X: Feature DataFrame
            y_aqi: AQI target
            y_gas: Gas target
            y_health: Health target
            
        Returns:
            Dictionary containing all splits
        """
        logger.info("Creating 70/20/10 train/validation/test splits...")
        
        # First split: 70% train, 30% temp (for val+test)
        X_train, X_temp, y_aqi_train, y_aqi_temp, y_gas_train, y_gas_temp, y_health_train, y_health_temp = train_test_split(
            X, y_aqi, y_gas, y_health, 
            test_size=0.30, 
            random_state=self.random_state,
            stratify=y_health  # Stratify on health status for balanced splits
        )
        
        # Second split: Split temp into 20% validation, 10% test (2/3 and 1/3 of temp)
        X_val, X_test, y_aqi_val, y_aqi_test, y_gas_val, y_gas_test, y_health_val, y_health_test = train_test_split(
            X_temp, y_aqi_temp, y_gas_temp, y_health_temp,
            test_size=1/3,  # 1/3 of 30% = 10% of total
            random_state=self.random_state,
            stratify=y_health_temp
        )
        
        splits = {
            'X_train': X_train, 'y_aqi_train': y_aqi_train, 'y_gas_train': y_gas_train, 'y_health_train': y_health_train,
            'X_val': X_val, 'y_aqi_val': y_aqi_val, 'y_gas_val': y_gas_val, 'y_health_val': y_health_val,
            'X_test': X_test, 'y_aqi_test': y_aqi_test, 'y_gas_test': y_gas_test, 'y_health_test': y_health_test
        }
        
        logger.info(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return splits
    
    def handle_class_imbalance(self, y: pd.Series, method: str = 'class_weight') -> Dict:
        """
        Calculate class weights or apply SMOTE for imbalanced classes.
        
        Args:
            y: Target variable
            method: 'class_weight' or 'smote'
            
        Returns:
            Dictionary with class weights or resampled data
        """
        logger.info(f"Handling class imbalance using {method}...")
        
        # Check class distribution
        class_counts = y.value_counts()
        logger.info(f"Class distribution:\n{class_counts}")
        
        # Calculate imbalance ratio
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        if method == 'class_weight':
            # Compute class weights
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            weight_dict = dict(zip(classes, class_weights))
            
            logger.info(f"Computed class weights: {weight_dict}")
            return {'sample_weight': weight_dict}
        
        elif method == 'smote':
            logger.info("SMOTE will be applied during training")
            return {'use_smote': True}
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def train_aqi_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Train XGBoost regressor for AQI prediction.
        
        Args:
            X_train: Training features
            y_train: Training AQI values
            X_val: Validation features
            y_val: Validation AQI values
        """
        logger.info("Training AQI model (XGBoost Regressor)...")
        
        # Model configuration
        aqi_config = {
            'n_estimators': 600,
            'learning_rate': 0.03,
            'max_depth': 8,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'tree_method': 'hist',
            'objective': 'reg:squarederror',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        self.aqi_model = xgb.XGBRegressor(**aqi_config)
        
        # Train with early stopping
        self.aqi_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Store training history
        evals_result = self.aqi_model.evals_result()
        self.training_history['aqi'] = {
            'train_rmse': evals_result['validation_0']['rmse'],
            'val_rmse': evals_result['validation_1']['rmse']
        }
        
        logger.info(f"AQI model trained. Total iterations: {len(self.training_history['aqi']['train_rmse'])}")
        logger.info(f"Final train RMSE: {self.training_history['aqi']['train_rmse'][-1]:.4f}")
        logger.info(f"Final val RMSE: {self.training_history['aqi']['val_rmse'][-1]:.4f}")
    
    def train_gas_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       handle_imbalance: bool = True) -> None:
        """
        Train XGBoost classifier for gas identification with intelligent imbalance handling.
        
        Args:
            X_train: Training features
            y_train: Training gas labels
            X_val: Validation features
            y_val: Validation gas labels
            handle_imbalance: Whether to handle class imbalance
        """
        logger.info("Training Gas model (XGBoost Classifier)...")
        
        # Handle severe class imbalance intelligently
        X_train_resampled = X_train
        y_train_resampled = y_train
        
        if handle_imbalance:
            # Check imbalance ratio
            imbalance_info = self.handle_class_imbalance(y_train, method='class_weight')
            
            # Check class sizes
            class_counts = y_train.value_counts()
            min_samples = class_counts.min()
            
            logger.info(f"Minimum class size: {min_samples} samples")
            
            # Only apply SMOTE if minimum class has enough samples (>= 50)
            if min_samples >= 50:
                try:
                    from imblearn.combine import SMOTETomek
                    from imblearn.over_sampling import SMOTE
                    logger.info("Applying SMOTETomek to balance classes...")
                    logger.info(f"Before resampling: {len(X_train)} samples")
                    
                    # Use SMOTETomek for better boundary cleaning
                    smotetomek = SMOTETomek(
                        smote=SMOTE(random_state=self.random_state, k_neighbors=min(3, min_samples-1)),
                        random_state=self.random_state
                    )
                    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train, y_train)
                    
                    logger.info(f"After SMOTETomek: {len(X_train_resampled)} samples")
                    logger.info(f"Class distribution after resampling:")
                    unique, counts = np.unique(y_train_resampled, return_counts=True)
                    for cls, cnt in zip(unique, counts):
                        logger.info(f"  Class {cls}: {cnt} samples")
                        
                except Exception as e:
                    logger.warning(f"SMOTETomek failed: {e}. Using class weights only.")
                    X_train_resampled = X_train
                    y_train_resampled = y_train
            else:
                logger.warning(f"Minimum class size ({min_samples}) too small for SMOTE.")
                logger.info("Using class weights only (no oversampling).")
                X_train_resampled = X_train
                y_train_resampled = y_train
        
        # Model configuration optimized for imbalanced data
        num_classes = len(np.unique(y_train))
        
        # Calculate sample weights for training
        sample_weights = None
        smote_applied = X_train_resampled is not X_train
        if not smote_applied:  # If SMOTE wasn't applied
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train_resampled)
            logger.info("Using sample weights for training")
        else:
            logger.info("SMOTE applied — skipping sample weights")
        
        gas_config = {
            'n_estimators': 500,
            'learning_rate': 0.03,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        self.gas_model = xgb.XGBClassifier(**gas_config)
        
        # Train with resampled data and optional sample weights
        self.gas_model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=False
        )
        
        # Store training history
        evals_result = self.gas_model.evals_result()
        self.training_history['gas'] = {
            'val_mlogloss': evals_result['validation_0']['mlogloss']
        }
        
        logger.info(f"Gas model trained. Total iterations: {len(self.training_history['gas']['val_mlogloss'])}")
        logger.info(f"Final val mlogloss: {self.training_history['gas']['val_mlogloss'][-1]:.4f}")
    
    def train_health_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          handle_imbalance: bool = True) -> None:
        """
        Train XGBoost classifier for health status with intelligent imbalance handling.
        
        Args:
            X_train: Training features
            y_train: Training health labels
            X_val: Validation features
            y_val: Validation health labels
            handle_imbalance: Whether to handle class imbalance
        """
        logger.info("Training Health model (XGBoost Classifier)...")
        
        # Handle class imbalance intelligently
        X_train_resampled = X_train
        y_train_resampled = y_train
        
        if handle_imbalance:
            # Check imbalance ratio
            imbalance_info = self.handle_class_imbalance(y_train, method='class_weight')
            
            # Check class sizes
            class_counts = y_train.value_counts()
            min_samples = class_counts.min()
            
            logger.info(f"Minimum class size: {min_samples} samples")
            
            # Only apply SMOTE if minimum class has enough samples (>= 50)
            if min_samples >= 50:
                try:
                    from imblearn.combine import SMOTETomek
                    from imblearn.over_sampling import SMOTE
                    logger.info("Applying SMOTETomek to balance classes...")
                    logger.info(f"Before resampling: {len(X_train)} samples")
                    
                    # Use SMOTETomek for better boundary cleaning
                    smotetomek = SMOTETomek(
                        smote=SMOTE(random_state=self.random_state, k_neighbors=min(3, min_samples-1)),
                        random_state=self.random_state
                    )
                    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train, y_train)
                    
                    logger.info(f"After SMOTETomek: {len(X_train_resampled)} samples")
                    logger.info(f"Class distribution after resampling:")
                    unique, counts = np.unique(y_train_resampled, return_counts=True)
                    for cls, cnt in zip(unique, counts):
                        logger.info(f"  Class {cls}: {cnt} samples")
                        
                except Exception as e:
                    logger.warning(f"SMOTETomek failed: {e}. Using class weights only.")
                    X_train_resampled = X_train
                    y_train_resampled = y_train
            else:
                logger.warning(f"Minimum class size ({min_samples}) too small for SMOTE.")
                logger.info("Using class weights only (no oversampling).")
                X_train_resampled = X_train
                y_train_resampled = y_train
        
        # Model configuration optimized for imbalanced data
        num_classes = len(np.unique(y_train))
        
        # Calculate sample weights for training
        sample_weights = None
        smote_applied = X_train_resampled is not X_train
        if not smote_applied:  # If SMOTE wasn't applied
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train_resampled)
            logger.info("Using sample weights for training")
        else:
            logger.info("SMOTE applied — skipping sample weights")
        
        health_config = {
            'n_estimators': 500,
            'learning_rate': 0.03,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        self.health_model = xgb.XGBClassifier(**health_config)
        
        # Train with resampled data and optional sample weights
        self.health_model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=False
        )
        
        # Store training history
        evals_result = self.health_model.evals_result()
        self.training_history['health'] = {
            'val_mlogloss': evals_result['validation_0']['mlogloss']
        }
        
        logger.info(f"Health model trained. Total iterations: {len(self.training_history['health']['val_mlogloss'])}")
        logger.info(f"Final val mlogloss: {self.training_history['health']['val_mlogloss'][-1]:.4f}")
    
    def get_training_history(self) -> Dict:
        """
        Return training loss curves.
        
        Returns:
            Dictionary with training history for all models
        """
        return self.training_history
    
    def get_models(self) -> Dict:
        """
        Return trained models.
        
        Returns:
            Dictionary with all trained models
        """
        return {
            'aqi_model': self.aqi_model,
            'gas_model': self.gas_model,
            'health_model': self.health_model
        }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Model Trainer module loaded")
