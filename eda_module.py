"""
EDA (Exploratory Data Analysis) Module

Comprehensive exploratory data analysis to understand data characteristics,
identify outliers, and visualize patterns before modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class EDAModule:
    """Performs comprehensive exploratory data analysis on air quality dataset."""
    
    def __init__(self, dataset_path: str, output_dir: str = "eda_outputs"):
        """
        Initialize EDA module.
        
        Args:
            dataset_path: Path to clean_air_dataset.csv
            output_dir: Directory for saving visualizations and reports
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.numerical_features = []
        self.categorical_features = []
        self.outliers = {}
        
        logger.info(f"EDA Module initialized with output directory: {self.output_dir}")
    
    def load_data(self) -> None:
        """Load dataset from CSV file."""
        try:
            self.df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Identify feature types
            self.numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
            
            logger.info(f"Numerical features: {len(self.numerical_features)}")
            logger.info(f"Categorical features: {len(self.categorical_features)}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def generate_statistical_summary(self) -> pd.DataFrame:
        """
        Generate descriptive statistics for all 23 columns.
        
        Returns:
            DataFrame with statistical summary
        """
        logger.info("Generating statistical summary...")
        
        summary = self.df.describe(include='all').T
        summary['missing'] = self.df.isnull().sum()
        summary['missing_pct'] = (self.df.isnull().sum() / len(self.df)) * 100
        
        # Save to CSV
        output_path = self.output_dir / "statistical_summary.csv"
        summary.to_csv(output_path)
        logger.info(f"Statistical summary saved to {output_path}")
        
        return summary
    
    def plot_distributions(self, features: List[str] = None) -> None:
        """
        Create histograms and KDE plots for numerical features.
        
        Args:
            features: List of features to plot (default: all numerical features)
        """
        if features is None:
            features = [f for f in self.numerical_features 
                       if f not in ['gov_id', 'year', 'month', 'hour', 'day_of_week']]
        
        logger.info(f"Plotting distributions for {len(features)} features...")
        
        # Create distributions directory
        dist_dir = self.output_dir / "distributions"
        dist_dir.mkdir(exist_ok=True)
        
        for feature in features:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram
            axes[0].hist(self.df[feature].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[0].set_title(f'{feature} - Histogram')
            axes[0].set_xlabel(feature)
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # KDE plot
            self.df[feature].dropna().plot(kind='kde', ax=axes[1])
            axes[1].set_title(f'{feature} - KDE Plot')
            axes[1].set_xlabel(feature)
            axes[1].set_ylabel('Density')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(dist_dir / f"{feature}_distribution.png", dpi=100, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Distribution plots saved to {dist_dir}")
    
    def generate_correlation_matrix(self) -> None:
        """Create heatmap showing feature correlations."""
        logger.info("Generating correlation matrix...")
        
        # Select numerical features for correlation
        corr_features = [f for f in self.numerical_features 
                        if f not in ['gov_id']]
        
        corr_matrix = self.df[corr_features].corr()
        
        # Create heatmap
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        
        output_path = self.output_dir / "correlation_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlation matrix saved to {output_path}")
    
    def detect_outliers(self, method: str = 'iqr') -> Dict[str, List[int]]:
        """
        Identify outliers using IQR or Z-score methods.
        
        Args:
            method: 'iqr' or 'zscore'
            
        Returns:
            Dictionary mapping feature names to outlier indices
        """
        logger.info(f"Detecting outliers using {method} method...")
        
        outliers = {}
        
        numerical_cols = [f for f in self.numerical_features 
                         if f not in ['gov_id', 'year', 'month', 'hour', 'day_of_week']]
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = self.df[
                    (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                ].index.tolist()
                
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outlier_indices = self.df[z_scores > 3].index.tolist()
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if outlier_indices:
                outliers[col] = outlier_indices
                logger.info(f"  {col}: {len(outlier_indices)} outliers ({len(outlier_indices)/len(self.df)*100:.2f}%)")
        
        self.outliers = outliers
        
        # Save outliers report
        outliers_summary = {col: len(indices) for col, indices in outliers.items()}
        output_path = self.output_dir / "outliers_report.json"
        with open(output_path, 'w') as f:
            json.dump(outliers_summary, f, indent=2)
        
        logger.info(f"Outliers report saved to {output_path}")
        
        return outliers
    
    def plot_outliers(self) -> None:
        """Visualize outliers using box plots."""
        logger.info("Creating outlier visualizations...")
        
        numerical_cols = [f for f in self.numerical_features 
                         if f not in ['gov_id', 'year', 'month', 'hour', 'day_of_week']]
        
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(numerical_cols):
            axes[idx].boxplot(self.df[col].dropna(), vert=True)
            axes[idx].set_title(f'{col}')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / "outliers_boxplots.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Outlier box plots saved to {output_path}")
    
    def analyze_temporal_patterns(self) -> None:
        """Visualize patterns across year, month, hour, day_of_week."""
        logger.info("Analyzing temporal patterns...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Year distribution
        if 'year' in self.df.columns:
            self.df['year'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Distribution by Year')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Month distribution
        if 'month' in self.df.columns:
            self.df['month'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Distribution by Month')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Hour distribution
        if 'hour' in self.df.columns:
            self.df['hour'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Distribution by Hour')
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Day of week distribution
        if 'day_of_week' in self.df.columns:
            self.df['day_of_week'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Distribution by Day of Week')
            axes[1, 1].set_xlabel('Day of Week (0=Monday)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "temporal_patterns.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Temporal patterns saved to {output_path}")
    
    def plot_categorical_distributions(self) -> None:
        """Bar plots for governorate, responsible_gas, health_status."""
        logger.info("Creating categorical distribution plots...")
        
        categorical_cols = [col for col in ['governorate', 'responsible_gas', 'health_status'] 
                           if col in self.df.columns]
        
        if not categorical_cols:
            logger.warning("No categorical columns found")
            return
        
        fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(12, 5 * len(categorical_cols)))
        if len(categorical_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(categorical_cols):
            value_counts = self.df[col].value_counts()
            value_counts.plot(kind='bar', ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Count')
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / "categorical_distributions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Categorical distributions saved to {output_path}")
    
    def generate_pairwise_plots(self, features: List[str] = None) -> None:
        """
        Scatter plot matrix for pollutant relationships.
        
        Args:
            features: List of features for pairwise plots (default: key pollutants)
        """
        if features is None:
            features = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
        
        features = [f for f in features if f in self.df.columns]
        
        if len(features) < 2:
            logger.warning("Not enough features for pairwise plots")
            return
        
        logger.info(f"Generating pairwise plots for {len(features)} features...")
        
        # Sample data if dataset is too large
        sample_size = min(10000, len(self.df))
        df_sample = self.df[features].sample(n=sample_size, random_state=42)
        
        pairplot = sns.pairplot(df_sample, diag_kind='kde', plot_kws={'alpha': 0.6})
        pairplot.fig.suptitle('Pairwise Pollutant Relationships', y=1.02, fontsize=16)
        
        output_path = self.output_dir / "pairwise_pollutants.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Pairwise plots saved to {output_path}")
    
    def save_all_outputs(self) -> None:
        """Execute all EDA analyses and save outputs."""
        logger.info("Running complete EDA analysis...")
        
        if self.df is None:
            self.load_data()
        
        # Generate all analyses
        self.generate_statistical_summary()
        self.plot_distributions()
        self.generate_correlation_matrix()
        self.detect_outliers(method='iqr')
        self.plot_outliers()
        self.analyze_temporal_patterns()
        self.plot_categorical_distributions()
        self.generate_pairwise_plots()
        
        logger.info(f"EDA analysis complete. All outputs saved to {self.output_dir}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run EDA
    eda = EDAModule("clean_air_dataset.csv", "eda_outputs")
    eda.save_all_outputs()
