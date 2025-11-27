import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.database import EarthquakeDatabase
from src.features.feature_pipeline import FeaturePipeline
from src.utils.logger import setup_logger
from src.utils.helpers import load_config, get_project_root

logger = setup_logger('data_prep')


class DataPreparation:
    
    # Feature columns used by the model
    FEATURE_COLUMNS = [
        # Temporal
        'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos',
        
        # Spatial
        'lat_norm', 'lon_norm', 'depth_norm',
        
        # Rolling 7-day
        'rolling_count_7d', 'rolling_mean_mag_7d',
        'rolling_max_mag_7d', 'rolling_std_mag_7d',
        
        # Rolling 30-day
        'rolling_count_30d', 'rolling_mean_mag_30d',
        'rolling_max_mag_30d', 'rolling_std_mag_30d',
        
        # Derived
        'days_since_last_significant', 'local_seismic_density',
        
        # Quality
        'nst_norm', 'gap_norm', 'dmin_norm', 'rms_norm'
    ]
    
    TARGET_COLUMN = 'magnitude'
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.database = EarthquakeDatabase()
        
        self.test_size = self.config['model']['test_size']
        self.random_state = self.config['model']['random_state']
        
        # Paths for saving artifacts
        project_root = get_project_root()
        self.scaler_path = os.path.join(project_root, self.config['paths']['scaler'])
        self.feature_config_path = os.path.join(project_root, self.config['paths']['feature_config'])
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        
        self.scaler = None
        self.feature_names = self.FEATURE_COLUMNS.copy()
    
    def load_data(self, min_magnitude: Optional[float] = None) -> pd.DataFrame:
    
        logger.info("Loading data from database...")
        
        # Get events with features
        data = self.database.get_events_with_features()
        
        if not data:
            raise ValueError("No data found in database. Run ETL and feature pipelines first.")
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records")
        
        # Apply magnitude filter if specified
        if min_magnitude is not None:
            df = df[df['magnitude'] >= min_magnitude]
            logger.info(f"After magnitude filter (>= {min_magnitude}): {len(df)} records")
        
        # Sort by time for chronological split
        df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        
        logger.info("Preparing feature matrix...")
        
        # Check for missing columns
        missing_cols = [col for col in self.FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        # Extract features
        X = df[self.FEATURE_COLUMNS].values
        y = df[self.TARGET_COLUMN].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        
        return X, y
    
    def chronological_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if test_size is None:
            test_size = self.test_size
        
        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"Chronological split:")
        logger.info(f"  Training set: {len(X_train)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def fit_scaler(self, X_train: np.ndarray) -> np.ndarray:

        logger.info("Fitting scaler on training data...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        logger.info(f"Feature means: {self.scaler.mean_[:5]}... (showing first 5)")
        logger.info(f"Feature stds: {self.scaler.scale_[:5]}... (showing first 5)")
        
        return X_train_scaled
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        return self.scaler.transform(X)
    
    def save_scaler(self):
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Nothing to save.")
        
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler saved to {self.scaler_path}")
        
        # Also save feature configuration
        feature_config = {
            'feature_columns': self.FEATURE_COLUMNS,
            'target_column': self.TARGET_COLUMN,
            'n_features': len(self.FEATURE_COLUMNS),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.feature_config_path, 'w') as f:
            json.dump(feature_config, f, indent=2)
        
        logger.info(f"Feature config saved to {self.feature_config_path}")
    
    def load_scaler(self):
        """Load scaler from disk."""
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
        
        self.scaler = joblib.load(self.scaler_path)
        logger.info(f"Scaler loaded from {self.scaler_path}")
    
    def prepare_training_data(
        self,
        min_magnitude: Optional[float] = None,
        max_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        logger.info("=" * 60)
        logger.info("Data Preparation Pipeline Started")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_data(min_magnitude)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Limit samples if needed (GP is O(n³))
        if max_samples is not None and len(X) > max_samples:
            logger.info(f"Limiting to {max_samples} most recent samples for GP training")
            X = X[-max_samples:]
            y = y[-max_samples:]
        
        # Chronological split
        X_train, X_test, y_train, y_test = self.chronological_split(X, y)
        
        # Fit and apply scaler
        X_train_scaled = self.fit_scaler(X_train)
        X_test_scaled = self.transform_features(X_test)
        
        # Save scaler
        self.save_scaler()
        
        logger.info("=" * 60)
        logger.info("Data Preparation Complete")
        logger.info("=" * 60)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_unscaled': X_train,
            'X_test_unscaled': X_test,
            'feature_names': self.FEATURE_COLUMNS
        }
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
    
        summary = {
            'n_samples': len(df),
            'n_features': len(self.FEATURE_COLUMNS),
            'magnitude_stats': {
                'min': float(df['magnitude'].min()),
                'max': float(df['magnitude'].max()),
                'mean': float(df['magnitude'].mean()),
                'std': float(df['magnitude'].std())
            },
            'feature_columns': self.FEATURE_COLUMNS
        }
        
        return summary
    
    def close(self):
        self.database.close()


def prepare_data():
    print("\n" + "=" * 60)
    print("  DATA PREPARATION")
    print("=" * 60 + "\n")
    
    prep = DataPreparation()
    
    try:
        # Prepare data (limit samples for GP scalability)
        data = prep.prepare_training_data(
            min_magnitude=None,
            max_samples=2000  # GP works well with up to ~2000 samples
        )
        
        print("\n" + "-" * 40)
        print("DATA SUMMARY")
        print("-" * 40)
        print(f"Training samples: {len(data['X_train'])}")
        print(f"Test samples: {len(data['X_test'])}")
        print(f"Features: {len(data['feature_names'])}")
        print(f"Target (y_train) range: [{data['y_train'].min():.2f}, {data['y_train'].max():.2f}]")
        print("-" * 40 + "\n")
        
        return data
        
    finally:
        prep.close()


if __name__ == "__main__":
    prepare_data()