import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger
from src.utils.helpers import load_config, get_project_root

logger = setup_logger('evaluate')


class ModelEvaluator:
    
    def __init__(self, config_path: str = "config.yaml"):
        
        self.config = load_config(config_path)
        
        # Output directory for plots
        project_root = get_project_root()
        self.output_dir = os.path.join(project_root, 'models', 'evaluation')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.evaluation_results = {}
    
    def compute_point_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        
        metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred) * 100),
            'max_error': float(np.max(np.abs(y_true - y_pred))),
            'median_ae': float(np.median(np.abs(y_true - y_pred)))
        }
        
        logger.info("Point Prediction Metrics:")
        logger.info(f"  MAE:  {metrics['mae']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  R²:   {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        confidence_levels: List[float] = [0.5, 0.68, 0.9, 0.95, 0.99]
    ) -> Dict[str, Any]:
        
        z_scores = {
            0.50: 0.674,
            0.68: 1.0,
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
        calibration = {
            'expected_coverage': [],
            'actual_coverage': [],
            'coverage_gap': []
        }
        
        for level in confidence_levels:
            z = z_scores.get(level, 1.96)
            
            # Compute interval bounds
            lower = y_pred - z * y_std
            upper = y_pred + z * y_std
            
            # Count how many true values fall within interval
            within_interval = np.sum((y_true >= lower) & (y_true <= upper))
            actual_coverage = within_interval / len(y_true)
            
            calibration['expected_coverage'].append(level)
            calibration['actual_coverage'].append(float(actual_coverage))
            calibration['coverage_gap'].append(float(actual_coverage - level))
        
        # Compute overall calibration error
        calibration['mean_calibration_error'] = float(
            np.mean(np.abs(np.array(calibration['coverage_gap'])))
        )
        
        # Average interval width (for 95% CI)
        z_95 = 1.96
        interval_widths = 2 * z_95 * y_std
        calibration['mean_interval_width_95'] = float(np.mean(interval_widths))
        calibration['std_interval_width_95'] = float(np.std(interval_widths))
        
        logger.info("\nCalibration Metrics:")
        for i, level in enumerate(confidence_levels):
            expected = calibration['expected_coverage'][i]
            actual = calibration['actual_coverage'][i]
            logger.info(f"  {int(level*100)}% CI: Expected {expected:.0%}, Actual {actual:.1%}")
        
        logger.info(f"  Mean Calibration Error: {calibration['mean_calibration_error']:.4f}")
        
        return calibration
    
    def compute_residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        
        residuals = y_true - y_pred
        
        analysis = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'skewness': float(self._compute_skewness(residuals)),
            'kurtosis': float(self._compute_kurtosis(residuals)),
            'percentiles': {
                '5': float(np.percentile(residuals, 5)),
                '25': float(np.percentile(residuals, 25)),
                '50': float(np.percentile(residuals, 50)),
                '75': float(np.percentile(residuals, 75)),
                '95': float(np.percentile(residuals, 95))
            }
        }
        
        logger.info("\nResidual Analysis:")
        logger.info(f"  Mean: {analysis['mean']:.4f} (ideally 0)")
        logger.info(f"  Std:  {analysis['std']:.4f}")
        logger.info(f"  Skewness: {analysis['skewness']:.4f} (ideally 0)")
        
        return analysis
    
    def _compute_skewness(self, x: np.ndarray) -> float:
        """Compute skewness of distribution."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.sum(((x - mean) / std) ** 3) / n)
    
    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Compute excess kurtosis of distribution."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.sum(((x - mean) / std) ** 4) / n - 3)
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray
    ) -> Dict[str, Any]:
        
        logger.info("=" * 60)
        logger.info("Model Evaluation")
        logger.info("=" * 60)
        
        self.evaluation_results = {
            'n_samples': len(y_true),
            'point_metrics': self.compute_point_metrics(y_true, y_pred),
            'calibration': self.compute_calibration_metrics(y_true, y_pred, y_std),
            'residuals': self.compute_residual_analysis(y_true, y_pred),
            'target_stats': {
                'min': float(np.min(y_true)),
                'max': float(np.max(y_true)),
                'mean': float(np.mean(y_true)),
                'std': float(np.std(y_true))
            },
            'prediction_stats': {
                'mean_std': float(np.mean(y_std)),
                'min_std': float(np.min(y_std)),
                'max_std': float(np.max(y_std))
            },
            'evaluated_at': datetime.now().isoformat()
        }
        
        return self.evaluation_results
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        save: bool = True
    ) -> Optional[str]:
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Predicted vs Actual
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        ax1.set_xlabel('Actual Magnitude', fontsize=12)
        ax1.set_ylabel('Predicted Magnitude', fontsize=12)
        ax1.set_title('Predicted vs Actual Magnitude', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics = self.evaluation_results.get('point_metrics', {})
        text = f"MAE: {metrics.get('mae', 0):.3f}\nRMSE: {metrics.get('rmse', 0):.3f}\nR²: {metrics.get('r2', 0):.3f}"
        ax1.text(0.05, 0.95, text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Residuals
        ax2 = axes[1]
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Magnitude', fontsize=12)
        ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
        ax2.set_title('Residual Plot', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = os.path.join(self.output_dir, 'predictions_plot.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {path}")
            plt.close()
            return path
        else:
            plt.show()
            return None
    
    def plot_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        save: bool = True
    ) -> Optional[str]:
        
        calibration = self.evaluation_results.get('calibration', {})
        
        if not calibration:
            calibration = self.compute_calibration_metrics(y_true, y_pred, y_std)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Calibration curve
        ax1 = axes[0]
        expected = calibration['expected_coverage']
        actual = calibration['actual_coverage']
        
        ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
        ax1.plot(expected, actual, 'bo-', lw=2, markersize=8, label='Model')
        ax1.fill_between(expected, expected, actual, alpha=0.3)
        
        ax1.set_xlabel('Expected Coverage', fontsize=12)
        ax1.set_ylabel('Actual Coverage', fontsize=12)
        ax1.set_title('Calibration Curve', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Uncertainty distribution
        ax2 = axes[1]
        ax2.hist(y_std, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(y_std), color='r', linestyle='--', lw=2, 
                   label=f'Mean: {np.mean(y_std):.3f}')
        ax2.set_xlabel('Prediction Uncertainty (Std Dev)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Prediction Uncertainty', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = os.path.join(self.output_dir, 'calibration_plot.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Calibration plot saved to {path}")
            plt.close()
            return path
        else:
            plt.show()
            return None
    
    def plot_uncertainty_vs_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        save: bool = True
    ) -> Optional[str]:
        
        errors = np.abs(y_true - y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_std, errors, alpha=0.5, s=20)
        
        # Add trend line
        z = np.polyfit(y_std, errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(y_std.min(), y_std.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', lw=2, label='Trend')
        
        # Perfect uncertainty line (error = std)
        ax.plot(x_line, x_line, 'g--', lw=2, label='Error = Std')
        
        ax.set_xlabel('Predicted Uncertainty (Std Dev)', fontsize=12)
        ax.set_ylabel('Absolute Error', fontsize=12)
        ax.set_title('Uncertainty vs Actual Error', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Correlation
        corr = np.corrcoef(y_std, errors)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            path = os.path.join(self.output_dir, 'uncertainty_vs_error.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Uncertainty vs Error plot saved to {path}")
            plt.close()
            return path
        else:
            plt.show()
            return None
    
    def generate_all_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray
    ) -> List[str]:
        
        paths = []
        
        path = self.plot_predictions(y_true, y_pred, y_std)
        if path:
            paths.append(path)
        
        path = self.plot_calibration(y_true, y_pred, y_std)
        if path:
            paths.append(path)
        
        path = self.plot_uncertainty_vs_error(y_true, y_pred, y_std)
        if path:
            paths.append(path)
        
        return paths
    
    def save_results(self):
        """Save evaluation results to JSON."""
        if not self.evaluation_results:
            logger.warning("No evaluation results to save")
            return
        
        path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {path}")
    
    def print_summary(self):
        """Print evaluation summary."""
        if not self.evaluation_results:
            print("No evaluation results available")
            return
        
        print("\n" + "=" * 60)
        print("  MODEL EVALUATION SUMMARY")
        print("=" * 60)
        
        pm = self.evaluation_results['point_metrics']
        print("\nPoint Prediction Metrics:")
        print(f"  MAE:        {pm['mae']:.4f}")
        print(f"  RMSE:       {pm['rmse']:.4f}")
        print(f"  R²:         {pm['r2']:.4f}")
        print(f"  MAPE:       {pm['mape']:.2f}%")
        print(f"  Median AE:  {pm['median_ae']:.4f}")
        
        cal = self.evaluation_results['calibration']
        print("\nCalibration (Expected vs Actual Coverage):")
        for i, exp in enumerate(cal['expected_coverage']):
            act = cal['actual_coverage'][i]
            print(f"  {int(exp*100):2d}% CI: {act:.1%}")
        print(f"  Mean Calibration Error: {cal['mean_calibration_error']:.4f}")
        
        res = self.evaluation_results['residuals']
        print("\nResidual Analysis:")
        print(f"  Mean:     {res['mean']:.4f}")
        print(f"  Std:      {res['std']:.4f}")
        print(f"  Skewness: {res['skewness']:.4f}")
        
        print("\n" + "=" * 60)


def evaluate_model():
    """Main function to evaluate the trained model."""
    from src.model.train import train_model
    
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60 + "\n")
    
    # Train model and get predictions
    print("Training model for evaluation...")
    results = train_model(kernel_type="simple")
    
    y_test = results['predictions']['y_test']
    y_pred = results['predictions']['y_pred']
    y_std = results['predictions']['y_std']
    
    # Evaluate
    evaluator = ModelEvaluator()
    evaluation = evaluator.evaluate(y_test, y_pred, y_std)
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    evaluator.generate_all_plots(y_test, y_pred, y_std)
    
    # Save results
    evaluator.save_results()
    
    # Print summary
    evaluator.print_summary()
    
    return evaluation


if __name__ == "__main__":
    evaluate_model()