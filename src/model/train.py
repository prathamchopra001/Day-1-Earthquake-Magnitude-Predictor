import os
import json
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.kernel import get_kernel
from src.model.data_prep import DataPreparation
from src.utils.logger import setup_logger
from src.utils.helpers import load_config, get_project_root

logger = setup_logger('train')


class GPTrainer:
    
    def __init__(self, config_path: str = "config.yaml"):
        
        self.config = load_config(config_path)
        self.config_path = config_path
        
        # Model settings
        self.n_restarts = self.config['model']['n_restarts_optimizer']
        self.random_state = self.config['model']['random_state']
        
        # Paths
        project_root = get_project_root()
        self.model_path = os.path.join(project_root, self.config['paths']['model'])
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        self.model = None
        self.training_info = {}
    
    def create_model(self, kernel_type: str = "simple") -> GaussianProcessRegressor:
        
        kernel = get_kernel(kernel_type, self.config_path)
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
            normalize_y=True,  # Important for magnitude data
            alpha=1e-10  # Small regularization for numerical stability
        )
        
        logger.info(f"Created GP model with {kernel_type} kernel")
        logger.info(f"  n_restarts_optimizer: {self.n_restarts}")
        logger.info(f"  normalize_y: True")
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel_type: str = "simple"
    ) -> GaussianProcessRegressor:
        
        logger.info("=" * 60)
        logger.info("Starting GP Training")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Create model
        self.model = self.create_model(kernel_type)
        
        logger.info(f"Training on {len(X_train)} samples with {X_train.shape[1]} features")
        logger.info("This may take a while...")
        
        # Train
        self.model.fit(X_train, y_train)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log results
        logger.info("Training complete!")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Log Marginal Likelihood: {self.model.log_marginal_likelihood_value_:.4f}")
        logger.info(f"  Optimized kernel: {self.model.kernel_}")
        
        # Store training info
        self.training_info = {
            'kernel_type': kernel_type,
            'n_train_samples': len(X_train),
            'n_features': X_train.shape[1],
            'duration_seconds': duration,
            'log_marginal_likelihood': float(self.model.log_marginal_likelihood_value_),
            'initial_kernel': str(get_kernel(kernel_type, self.config_path)),
            'optimized_kernel': str(self.model.kernel_),
            'trained_at': datetime.now().isoformat()
        }
        
        return self.model
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if return_std:
            y_pred, y_std = self.model.predict(X, return_std=True)
            return y_pred, y_std
        else:
            y_pred = self.model.predict(X, return_std=False)
            return y_pred, None
    
    def save_model(self):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        # Save model
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        # Save training info
        info_path = self.model_path.replace('.pkl', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(self.training_info, f, indent=2)
        logger.info(f"Training info saved to {info_path}")
    
    def load_model(self):
        """Load model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
        
        # Load training info if available
        info_path = self.model_path.replace('.pkl', '_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.training_info = json.load(f)
        
        return self.model


def train_model(
    kernel_type: str = "simple",
    min_magnitude: Optional[float] = None,
    max_samples: int = 2000
) -> Dict[str, Any]:
    
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from src.utils.progress import ProgressTracker, SpinnerProgress
    
    # Define pipeline steps
    steps = [
        "Loading data from database",
        "Preparing feature matrix", 
        "Splitting train/test sets",
        "Scaling features",
        "Training Gaussian Process model",
        "Generating predictions",
        "Computing evaluation metrics",
        "Saving model artifacts"
    ]
    
    tracker = ProgressTracker(steps, title="GAUSSIAN PROCESS MODEL TRAINING")
    tracker.start()
    
    # Step 1: Load data
    tracker.next_step()
    data_prep = DataPreparation()
    
    try:
        df = data_prep.load_data(min_magnitude)
        tracker.update_status(f"Loaded {len(df)} records from database")
        
        # Step 2: Prepare features
        tracker.next_step()
        X, y = data_prep.prepare_features(df)
        
        # Limit samples
        if max_samples is not None and len(X) > max_samples:
            tracker.update_status(f"Limiting to {max_samples} most recent samples (was {len(X)})")
            X = X[-max_samples:]
            y = y[-max_samples:]
        else:
            tracker.update_status(f"Using all {len(X)} samples")
        
        # Step 3: Split data
        tracker.next_step()
        X_train, X_test, y_train, y_test = data_prep.chronological_split(X, y)
        tracker.update_status(f"Training: {len(X_train)} samples | Test: {len(X_test)} samples")
        
        # Step 4: Scale features
        tracker.next_step()
        X_train_scaled = data_prep.fit_scaler(X_train)
        X_test_scaled = data_prep.transform_features(X_test)
        data_prep.save_scaler()
        tracker.update_status(f"Scaled {X_train.shape[1]} features")
        
    finally:
        data_prep.close()
    
    # Step 5: Train model
    tracker.next_step()
    tracker.update_status(f"Kernel type: {kernel_type}")
    tracker.update_status(f"Optimizing hyperparameters...")
    
    trainer = GPTrainer()
    
    # Use spinner during training since we can't track internal progress
    print()
    with SpinnerProgress(f"Training GP model ({len(X_train)} samples)") as spinner:
        model = trainer.train(X_train_scaled, y_train, kernel_type)
    
    print(f"    Log Marginal Likelihood: {trainer.model.log_marginal_likelihood_value_:.4f}")
    print(f"    Optimized kernel: {str(trainer.model.kernel_)[:60]}...")
    
    # Step 6: Generate predictions
    tracker.next_step()
    y_pred, y_std = trainer.predict(X_test_scaled)
    tracker.update_status(f"Generated {len(y_pred)} predictions with uncertainty estimates")
    tracker.update_status(f"Mean uncertainty (std): {np.mean(y_std):.4f}")
    
    # Step 7: Compute metrics
    tracker.next_step()
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Pretty print metrics
    print()
    print("    ┌─────────────────────────────────┐")
    print("    │      MODEL PERFORMANCE          │")
    print("    ├─────────────────────────────────┤")
    print(f"    │  MAE:   {mae:>8.4f}  (lower=better) │")
    print(f"    │  RMSE:  {rmse:>8.4f}  (lower=better) │")
    print(f"    │  R²:    {r2:>8.4f}  (higher=better)│")
    print("    └─────────────────────────────────┘")
    print()
    
    # Interpret results
    if r2 > 0.3:
        tracker.update_status("✓ Model shows good predictive power")
    elif r2 > 0:
        tracker.update_status("⚠ Model shows weak predictive power - consider more data")
    else:
        tracker.update_status("✗ Model performs worse than baseline - check data quality")
    
    # Step 8: Save model
    tracker.next_step()
    trainer.save_model()
    tracker.update_status("Saved: models/gp_model.pkl")
    tracker.update_status("Saved: models/scaler.pkl")
    tracker.update_status("Saved: models/gp_model_info.json")
    
    # Complete
    tracker.complete()
    
    return {
        'model': model,
        'trainer': trainer,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'data': {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        },
        'predictions': {
            'y_pred': y_pred,
            'y_std': y_std,
            'y_test': y_test
        }
    }


if __name__ == "__main__":
    train_model(kernel_type="simple")