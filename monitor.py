"""
Monitor Module

Tracks model performance in production and detects drift.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class Monitor:
    """Monitors model performance and detects data drift."""
    
    def __init__(self, log_path: str = "monitoring_logs", 
                 alert_thresholds: Dict = None):
        """
        Initialize monitor.
        
        Args:
            log_path: Path to monitoring log file/directory
            alert_thresholds: Dictionary with performance thresholds
        """
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Default alert thresholds
        self.thresholds = alert_thresholds or {
            'aqi_mae_increase_pct': 20,  # Alert if MAE increases by 20%
            'gas_accuracy_drop_pct': 10,  # Alert if accuracy drops by 10%
            'health_accuracy_drop_pct': 10,
            'drift_ks_statistic': 0.3  # KS test threshold for drift detection
        }
        
        self.prediction_log_file = self.log_path / "predictions.jsonl"
        self.performance_log_file = self.log_path / "performance.json"
        self.alerts_file = self.log_path / "alerts.jsonl"
        
        # Baseline metrics (set during initial deployment)
        self.baseline_metrics = {}
        
        logger.info(f"Monitor initialized with log path: {self.log_path}")
    
    def set_baseline_metrics(self, metrics: Dict) -> None:
        """
        Set baseline performance metrics.
        
        Args:
            metrics: Dictionary with baseline metrics
        """
        self.baseline_metrics = metrics
        
        # Save baseline
        baseline_path = self.log_path / "baseline_metrics.json"
        with open(baseline_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Baseline metrics set and saved to {baseline_path}")
    
    def log_prediction(self, input_data: Dict, predictions: Dict,
                      timestamp: Optional[datetime] = None,
                      model_version: str = "1.0.0") -> None:
        """
        Log prediction inputs and outputs.
        
        Args:
            input_data: Dictionary with input features
            predictions: Dictionary with model predictions
            timestamp: Prediction timestamp (default: now)
            model_version: Model version identifier
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'model_version': model_version,
            'input': input_data,
            'predictions': predictions
        }
        
        # Append to log file
        with open(self.prediction_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def calculate_rolling_metrics(self, window: str = '7d') -> Dict:
        """
        Calculate metrics over time windows.
        
        Args:
            window: Time window ('1d', '7d', '30d')
            
        Returns:
            Dictionary with rolling metrics
        """
        logger.info(f"Calculating rolling metrics for window: {window}")
        
        # Parse window
        window_days = int(window.replace('d', ''))
        cutoff_date = datetime.now() - pd.Timedelta(days=window_days)
        
        # Load prediction logs
        if not self.prediction_log_file.exists():
            logger.warning("No prediction logs found")
            return {}
        
        predictions = []
        with open(self.prediction_log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff_date:
                    predictions.append(entry)
        
        if not predictions:
            logger.warning(f"No predictions found in {window} window")
            return {}
        
        logger.info(f"Found {len(predictions)} predictions in {window} window")
        
        # Calculate metrics (if actual values are available)
        rolling_metrics = {
            'window': window,
            'prediction_count': len(predictions),
            'start_date': predictions[0]['timestamp'],
            'end_date': predictions[-1]['timestamp']
        }
        
        return rolling_metrics
    
    def check_performance_degradation(self, current_metrics: Dict) -> bool:
        """
        Compare current metrics to baseline thresholds.
        
        Args:
            current_metrics: Dictionary with current performance metrics
            
        Returns:
            True if degradation detected, False otherwise
        """
        logger.info("Checking for performance degradation...")
        
        if not self.baseline_metrics:
            logger.warning("No baseline metrics set. Cannot check degradation.")
            return False
        
        degradation_detected = False
        
        # Check AQI MAE
        if 'aqi' in current_metrics and 'aqi' in self.baseline_metrics:
            baseline_mae = self.baseline_metrics['aqi'].get('test', {}).get('mae', 0)
            current_mae = current_metrics['aqi'].get('test', {}).get('mae', 0)
            
            if baseline_mae > 0:
                mae_increase_pct = ((current_mae - baseline_mae) / baseline_mae) * 100
                
                if mae_increase_pct > self.thresholds['aqi_mae_increase_pct']:
                    logger.warning(f"AQI MAE increased by {mae_increase_pct:.2f}% (threshold: {self.thresholds['aqi_mae_increase_pct']}%)")
                    self.generate_alert('performance_degradation', {
                        'model': 'aqi',
                        'metric': 'mae',
                        'baseline': baseline_mae,
                        'current': current_mae,
                        'increase_pct': mae_increase_pct
                    })
                    degradation_detected = True
        
        # Check Gas accuracy
        if 'gas' in current_metrics and 'gas' in self.baseline_metrics:
            baseline_acc = self.baseline_metrics['gas'].get('test', {}).get('accuracy', 0)
            current_acc = current_metrics['gas'].get('test', {}).get('accuracy', 0)
            
            if baseline_acc > 0:
                acc_drop_pct = ((baseline_acc - current_acc) / baseline_acc) * 100
                
                if acc_drop_pct > self.thresholds['gas_accuracy_drop_pct']:
                    logger.warning(f"Gas accuracy dropped by {acc_drop_pct:.2f}% (threshold: {self.thresholds['gas_accuracy_drop_pct']}%)")
                    self.generate_alert('performance_degradation', {
                        'model': 'gas',
                        'metric': 'accuracy',
                        'baseline': baseline_acc,
                        'current': current_acc,
                        'drop_pct': acc_drop_pct
                    })
                    degradation_detected = True
        
        # Check Health accuracy
        if 'health' in current_metrics and 'health' in self.baseline_metrics:
            baseline_acc = self.baseline_metrics['health'].get('test', {}).get('accuracy', 0)
            current_acc = current_metrics['health'].get('test', {}).get('accuracy', 0)
            
            if baseline_acc > 0:
                acc_drop_pct = ((baseline_acc - current_acc) / baseline_acc) * 100
                
                if acc_drop_pct > self.thresholds['health_accuracy_drop_pct']:
                    logger.warning(f"Health accuracy dropped by {acc_drop_pct:.2f}% (threshold: {self.thresholds['health_accuracy_drop_pct']}%)")
                    self.generate_alert('performance_degradation', {
                        'model': 'health',
                        'metric': 'accuracy',
                        'baseline': baseline_acc,
                        'current': current_acc,
                        'drop_pct': acc_drop_pct
                    })
                    degradation_detected = True
        
        if not degradation_detected:
            logger.info("No performance degradation detected")
        
        return degradation_detected
    
    def detect_data_drift(self, production_data: pd.DataFrame,
                         training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Compare feature distributions using statistical tests.
        
        Args:
            production_data: Recent production data
            training_data: Original training data
            
        Returns:
            Dictionary with drift scores per feature
        """
        logger.info("Detecting data drift...")
        
        drift_scores = {}
        
        # Get numerical columns
        numerical_cols = production_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col not in training_data.columns:
                continue
            
            # Kolmogorov-Smirnov test for numerical features
            ks_statistic, p_value = stats.ks_2samp(
                training_data[col].dropna(),
                production_data[col].dropna()
            )
            
            drift_scores[col] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'drift_detected': ks_statistic > self.thresholds['drift_ks_statistic']
            }
            
            if drift_scores[col]['drift_detected']:
                logger.warning(f"Drift detected in {col}: KS={ks_statistic:.4f} (threshold: {self.thresholds['drift_ks_statistic']})")
                self.generate_alert('data_drift', {
                    'feature': col,
                    'ks_statistic': ks_statistic,
                    'p_value': p_value
                })
        
        # Save drift report
        drift_report_path = self.log_path / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(drift_report_path, 'w') as f:
            json.dump(drift_scores, f, indent=2)
        
        logger.info(f"Drift report saved to {drift_report_path}")
        
        return drift_scores
    
    def generate_alert(self, alert_type: str, details: Dict) -> None:
        """
        Generate alert for performance degradation or drift.
        
        Args:
            alert_type: Type of alert ('performance_degradation', 'data_drift')
            details: Dictionary with alert details
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'details': details
        }
        
        # Append to alerts file
        with open(self.alerts_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
        
        logger.warning(f"ALERT: {alert_type} - {details}")
    
    def trigger_retraining(self, current_metrics: Dict = None) -> bool:
        """
        Determine if retraining is needed based on thresholds.
        
        Args:
            current_metrics: Current performance metrics (optional)
            
        Returns:
            True if retraining recommended, False otherwise
        """
        logger.info("Checking if retraining is needed...")
        
        # Check if performance degradation detected
        if current_metrics:
            if self.check_performance_degradation(current_metrics):
                logger.warning("Retraining recommended due to performance degradation")
                return True
        
        # Check recent alerts
        if self.alerts_file.exists():
            with open(self.alerts_file, 'r') as f:
                recent_alerts = [json.loads(line) for line in f.readlines()[-10:]]
            
            # Count drift alerts in recent history
            drift_alerts = sum(1 for alert in recent_alerts if alert['type'] == 'data_drift')
            
            if drift_alerts >= 3:
                logger.warning(f"Retraining recommended due to {drift_alerts} recent drift alerts")
                return True
        
        logger.info("No retraining needed at this time")
        return False
    
    def generate_performance_report(self, period: str = '30d') -> str:
        """
        Create periodic performance report.
        
        Args:
            period: Reporting period ('7d', '30d', '90d')
            
        Returns:
            Report as formatted string
        """
        logger.info(f"Generating performance report for {period}...")
        
        rolling_metrics = self.calculate_rolling_metrics(period)
        
        report = f"""
{'='*60}
PERFORMANCE MONITORING REPORT
Period: {period}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

Prediction Statistics:
  Total Predictions: {rolling_metrics.get('prediction_count', 0)}
  Period Start: {rolling_metrics.get('start_date', 'N/A')}
  Period End: {rolling_metrics.get('end_date', 'N/A')}

Baseline Metrics:
  AQI MAE: {self.baseline_metrics.get('aqi', {}).get('test', {}).get('mae', 'N/A')}
  Gas Accuracy: {self.baseline_metrics.get('gas', {}).get('test', {}).get('accuracy', 'N/A')}
  Health Accuracy: {self.baseline_metrics.get('health', {}).get('test', {}).get('accuracy', 'N/A')}

Alert Thresholds:
  AQI MAE Increase: {self.thresholds['aqi_mae_increase_pct']}%
  Gas Accuracy Drop: {self.thresholds['gas_accuracy_drop_pct']}%
  Health Accuracy Drop: {self.thresholds['health_accuracy_drop_pct']}%
  Drift KS Statistic: {self.thresholds['drift_ks_statistic']}

{'='*60}
"""
        
        # Save report
        report_path = self.log_path / f"performance_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Performance report saved to {report_path}")
        
        return report


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Monitor module loaded")
