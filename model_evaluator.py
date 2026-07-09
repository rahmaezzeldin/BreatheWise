"""
Model Evaluator Module

Comprehensive evaluation of model performance across all data splits.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from pathlib import Path
from typing import Dict, List, Any
import json
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates model performance with comprehensive metrics."""
    
    def __init__(self, models: Dict, encoders: Dict, output_dir: str = "evaluation_outputs"):
        """
        Initialize model evaluator.
        
        Args:
            models: Dictionary with trained models
            encoders: Dictionary with label encoders
            output_dir: Directory for saving evaluation results
        """
        self.models = models
        self.encoders = encoders
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_results = {}
        
        logger.info(f"Model Evaluator initialized with output directory: {self.output_dir}")
    
    def evaluate_regression(self, model, X: pd.DataFrame, y: pd.Series,
                           split_name: str) -> Dict[str, float]:
        """
        Calculate MAE, RMSE, R2 for AQI model.
        
        Args:
            model: Trained regression model
            X: Features
            y: True values
            split_name: Name of the split (train/val/test)
            
        Returns:
            Dictionary with regression metrics
        """
        logger.info(f"Evaluating regression model on {split_name} set...")
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        
        return metrics
    
    def evaluate_classification(self, model, X: pd.DataFrame, y: pd.Series,
                               split_name: str, encoder) -> Dict[str, Any]:
        """
        Calculate accuracy, precision, recall, F1 for classifiers.
        
        Args:
            model: Trained classification model
            X: Features
            y: True labels
            split_name: Name of the split (train/val/test)
            encoder: Label encoder for this target
            
        Returns:
            Dictionary with classification metrics
        """
        logger.info(f"Evaluating classification model on {split_name} set...")
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Calculate per-class F1 scores
        f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_per_class': f1_per_class.tolist()
        }
        
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision (weighted): {precision:.4f}")
        logger.info(f"  Recall (weighted): {recall:.4f}")
        logger.info(f"  F1 (weighted): {f1:.4f}")
        logger.info(f"  Per-class F1 scores:")
        for i, f1_val in enumerate(f1_per_class):
            logger.info(f"    Class {i}: {f1_val:.4f}")
        
        return metrics
    
    def generate_confusion_matrix(self, y_true, y_pred, labels: List[str],
                                 output_path: str, title: str = "Confusion Matrix") -> None:
        """
        Create and save confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            output_path: Path to save the plot
            title: Plot title
        """
        logger.info(f"Generating confusion matrix: {title}...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        plt.title(title, fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def generate_classification_report(self, y_true, y_pred,
                                      labels: List[str], output_path: str) -> str:
        """
        Generate per-class metrics report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            output_path: Path to save the report
            
        Returns:
            Classification report as string
        """
        logger.info("Generating classification report...")
        
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        
        # Log the report to console for immediate visibility
        logger.info("Classification Report:")
        logger.info("\n" + report)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Classification report saved to {output_path}")
        
        return report
    
    def plot_residuals(self, y_true, y_pred, output_path: str) -> None:
        """
        Create residual plot for AQI model.
        
        Args:
            y_true: True AQI values
            y_pred: Predicted AQI values
            output_path: Path to save the plot
        """
        logger.info("Creating residual plot...")
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted AQI', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title('Residuals vs Predicted Values', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residuals Distribution', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Residual plot saved to {output_path}")
    
    def calculate_feature_importance(self, model, feature_names: List[str],
                                    output_path: str, title: str = "Feature Importance") -> pd.DataFrame:
        """
        Extract and visualize feature importance scores.
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            output_path: Path to save the plot
            title: Plot title
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Calculating feature importance: {title}...")
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        top_n = min(20, len(importance_df))
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importance_df['importance'].head(top_n))
        plt.yticks(range(top_n), importance_df['feature'].head(top_n))
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'{title} (Top {top_n})', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
        
        return importance_df
    
    def evaluate_all_splits(self, data_splits: Dict) -> Dict[str, Dict]:
        """
        Evaluate all models on train/val/test splits.
        
        Args:
            data_splits: Dictionary with all data splits
            
        Returns:
            Dictionary with evaluation results for all models and splits
        """
        logger.info("Evaluating all models on all splits...")
        
        results = {
            'aqi': {},
            'gas': {},
            'health': {}
        }
        
        # Get feature names (exclude target columns)
        feature_names = [col for col in data_splits['X_train'].columns 
                        if col not in ['final_aqi_value', 'responsible_gas', 'health_status']]
        
        # Evaluate on each split
        for split in ['train', 'val', 'test']:
            X = data_splits[f'X_{split}']
            y_aqi = data_splits[f'y_aqi_{split}']
            y_gas = data_splits[f'y_gas_{split}']
            y_health = data_splits[f'y_health_{split}']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating on {split.upper()} set")
            logger.info(f"{'='*60}")
            
            # AQI Model
            if self.models.get('aqi_model'):
                results['aqi'][split] = self.evaluate_regression(
                    self.models['aqi_model'], X, y_aqi, split
                )
                
                # Generate residual plot for test set
                if split == 'test':
                    y_pred = self.models['aqi_model'].predict(X)
                    self.plot_residuals(
                        y_aqi, y_pred,
                        self.output_dir / "aqi_residuals_test.png"
                    )
            
            # Gas Model
            if self.models.get('gas_model'):
                results['gas'][split] = self.evaluate_classification(
                    self.models['gas_model'], X, y_gas, split,
                    self.encoders['gas_encoder']
                )
                
                # Generate confusion matrix and report for test set
                if split == 'test':
                    y_pred = self.models['gas_model'].predict(X)
                    gas_labels = self.encoders['gas_encoder'].classes_
                    
                    self.generate_confusion_matrix(
                        y_gas, y_pred, gas_labels,
                        self.output_dir / "gas_confusion_matrix_test.png",
                        "Gas Model - Confusion Matrix (Test Set)"
                    )
                    
                    self.generate_classification_report(
                        y_gas, y_pred, gas_labels,
                        self.output_dir / "gas_classification_report_test.txt"
                    )
            
            # Health Model
            if self.models.get('health_model'):
                results['health'][split] = self.evaluate_classification(
                    self.models['health_model'], X, y_health, split,
                    self.encoders['health_encoder']
                )
                
                # Generate confusion matrix and report for test set
                if split == 'test':
                    y_pred = self.models['health_model'].predict(X)
                    health_labels = self.encoders['health_encoder'].classes_
                    
                    self.generate_confusion_matrix(
                        y_health, y_pred, health_labels,
                        self.output_dir / "health_confusion_matrix_test.png",
                        "Health Model - Confusion Matrix (Test Set)"
                    )
                    
                    self.generate_classification_report(
                        y_health, y_pred, health_labels,
                        self.output_dir / "health_classification_report_test.txt"
                    )
        
        # Calculate feature importance for all models
        if self.models.get('aqi_model'):
            self.calculate_feature_importance(
                self.models['aqi_model'], feature_names,
                self.output_dir / "aqi_feature_importance.png",
                "AQI Model - Feature Importance"
            )
        
        if self.models.get('gas_model'):
            self.calculate_feature_importance(
                self.models['gas_model'], feature_names,
                self.output_dir / "gas_feature_importance.png",
                "Gas Model - Feature Importance"
            )
        
        if self.models.get('health_model'):
            self.calculate_feature_importance(
                self.models['health_model'], feature_names,
                self.output_dir / "health_feature_importance.png",
                "Health Model - Feature Importance"
            )
        
        self.evaluation_results = results
        
        return results
    
    def save_evaluation_results(self) -> None:
        """Save all metrics and visualizations."""
        logger.info("Saving evaluation results...")
        
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Save metrics to JSON
        metrics_path = self.output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(convert(self.evaluation_results), f, indent=2)
        
        logger.info(f"Evaluation metrics saved to {metrics_path}")
        logger.info(f"All evaluation outputs saved to {self.output_dir}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Model Evaluator module loaded")
