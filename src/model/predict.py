import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.feature_pipeline import FeaturePipeline
from src.model.data_prep import DataPreparation
from src.utils.logger import setup_logger
from src.utils.helpers import load_config, get_project_root

logger = setup_logger('predict')


class EarthquakePredictor:
    
    def __init__(self, config_path: str = "config.yaml"):
        
        self.config = load_config(config_path)
        self.config_path = config_path
        
        # Paths
        project_root = get_project_root()
        self.model_path = os.path.join(project_root, self.config['paths']['model'])
        self.scaler_path = os.path.join(project_root, self.config['paths']['scaler'])
        
        self.model = None
        self.scaler = None
        self.feature_pipeline = None
    
    def load(self):
        
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Train the model first."
            )
        
        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
        
        # Load scaler
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Scaler not found at {self.scaler_path}. Train the model first."
            )
        
        self.scaler = joblib.load(self.scaler_path)
        logger.info(f"Scaler loaded from {self.scaler_path}")
        
        # Initialize feature pipeline
        self.feature_pipeline = FeaturePipeline(self.config_path)
        logger.info("Feature pipeline initialized")
    
    def predict_from_features(
        self,
        features: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Extract features in correct order
        feature_names = DataPreparation.FEATURE_COLUMNS
        X = np.array([[features.get(name, 0.0) for name in feature_names]])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict with uncertainty
        y_pred, y_std = self.model.predict(X_scaled, return_std=True)
        
        # Compute confidence interval
        # For 95% CI: z = 1.96
        z_score = {
            0.50: 0.674,
            0.68: 1.0,
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }.get(confidence_level, 1.96)
        
        ci_lower = y_pred[0] - z_score * y_std[0]
        ci_upper = y_pred[0] + z_score * y_std[0]
        
        return {
            'magnitude': float(y_pred[0]),
            'std': float(y_std[0]),
            'confidence_level': confidence_level,
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'prediction_time': datetime.now().isoformat()
        }
    
    def predict(
        self,
        latitude: float,
        longitude: float,
        depth: float,
        timestamp_ms: Optional[int] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        
        if self.feature_pipeline is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Default to current time
        if timestamp_ms is None:
            timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
        
        logger.info(f"Predicting for location: ({latitude}, {longitude}), depth: {depth} km")
        
        # Compute features
        features = self.feature_pipeline.compute_features_for_prediction(
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            timestamp_ms=timestamp_ms
        )
        
        # Make prediction
        result = self.predict_from_features(features, confidence_level)
        
        # Add input info
        result['input'] = {
            'latitude': latitude,
            'longitude': longitude,
            'depth': depth,
            'timestamp_ms': timestamp_ms
        }
        
        # Add context features
        result['context'] = {
            'rolling_count_7d': features.get('rolling_count_7d', 0),
            'rolling_count_30d': features.get('rolling_count_30d', 0),
            'rolling_mean_mag_7d': features.get('rolling_mean_mag_7d', 0),
            'days_since_last_significant': features.get('days_since_last_significant', 365)
        }
        
        logger.info(f"Prediction: {result['magnitude']:.2f} ± {result['std']:.2f}")
        
        return result
    
    def predict_batch(
        self,
        locations: List[Dict[str, float]],
        confidence_level: float = 0.95
    ) -> List[Dict[str, Any]]:
        
        results = []
        
        for loc in locations:
            result = self.predict(
                latitude=loc['latitude'],
                longitude=loc['longitude'],
                depth=loc.get('depth', 10.0),
                timestamp_ms=loc.get('timestamp_ms'),
                confidence_level=confidence_level
            )
            results.append(result)
        
        return results
    
    def get_uncertainty_interpretation(self, std: float) -> str:
        
        if std < 0.3:
            return "High confidence - Well-characterized seismic region"
        elif std < 0.5:
            return "Moderate confidence - Typical uncertainty"
        elif std < 0.8:
            return "Lower confidence - Limited historical data"
        else:
            return "Low confidence - Sparse data or unusual location"
    
    def close(self):
        """Close resources."""
        if self.feature_pipeline:
            self.feature_pipeline.close()


def test_prediction():
    """Test prediction functionality."""
    print("\n" + "=" * 60)
    print("  PREDICTION TEST")
    print("=" * 60 + "\n")
    
    predictor = EarthquakePredictor()
    
    try:
        predictor.load()
        
        # Test prediction for Fiji region (common earthquake zone)
        result = predictor.predict(
            latitude=-17.9244,
            longitude=-178.371,
            depth=100.0
        )
        
        print("Input:")
        print(f"  Location: ({result['input']['latitude']}, {result['input']['longitude']})")
        print(f"  Depth: {result['input']['depth']} km")
        print()
        print("Prediction:")
        print(f"  Magnitude: {result['magnitude']:.2f}")
        print(f"  Uncertainty (std): ±{result['std']:.2f}")
        print(f"  95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
        print()
        print("Regional Context:")
        print(f"  Events in past 7 days: {result['context']['rolling_count_7d']}")
        print(f"  Events in past 30 days: {result['context']['rolling_count_30d']}")
        print(f"  Mean magnitude (7d): {result['context']['rolling_mean_mag_7d']:.2f}")
        print()
        print(f"Interpretation: {predictor.get_uncertainty_interpretation(result['std'])}")
        
        return result
        
    finally:
        predictor.close()


if __name__ == "__main__":
    test_prediction()