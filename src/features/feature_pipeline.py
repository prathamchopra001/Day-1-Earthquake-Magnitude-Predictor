from datetime import datetime
from typing import Dict, List, Any, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.temporal_features import extract_temporal_features
from src.features.spatial_features import extract_spatial_features
from src.features.rolling_features import (
    compute_rolling_features,
    compute_quality_features,
    compute_all_rolling_features
)
from src.data.database import EarthquakeDatabase
from src.utils.logger import setup_logger
from src.utils.helpers import load_config

logger = setup_logger('feature_pipeline')


class FeaturePipeline:
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.database = EarthquakeDatabase()
        
        # Feature engineering parameters from config
        self.radius_km = self.config['features']['spatial_radius_km']
        self.windows_days = self.config['features']['rolling_windows']
        self.significant_threshold = self.config['features']['significant_magnitude_threshold']
    
    def compute_features_for_event(
        self,
        event: Dict[str, Any],
        historical_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        # Temporal features
        temporal = extract_temporal_features(event['time'])
        
        # Spatial features
        spatial = extract_spatial_features(
            event['latitude'],
            event['longitude'],
            event['depth']
        )
        
        # Rolling features
        rolling = compute_rolling_features(
            event,
            historical_events,
            radius_km=self.radius_km,
            windows_days=self.windows_days,
            significant_threshold=self.significant_threshold
        )
        
        # Quality features
        quality = compute_quality_features(event)
        
        # Combine all features
        features = {
            'event_id': event['id'],
            
            # Temporal
            'hour': temporal['hour'],
            'day_of_week': temporal['day_of_week'],
            'day_of_year': temporal['day_of_year'],
            'hour_sin': temporal['hour_sin'],
            'hour_cos': temporal['hour_cos'],
            'dow_sin': temporal['dow_sin'],
            'dow_cos': temporal['dow_cos'],
            
            # Spatial
            'lat_norm': spatial['lat_norm'],
            'lon_norm': spatial['lon_norm'],
            'depth_norm': spatial['depth_norm'],
            
            # Rolling (7-day)
            'rolling_count_7d': rolling['rolling_count_7d'],
            'rolling_mean_mag_7d': rolling['rolling_mean_mag_7d'],
            'rolling_max_mag_7d': rolling['rolling_max_mag_7d'],
            'rolling_std_mag_7d': rolling['rolling_std_mag_7d'],
            
            # Rolling (30-day)
            'rolling_count_30d': rolling['rolling_count_30d'],
            'rolling_mean_mag_30d': rolling['rolling_mean_mag_30d'],
            'rolling_max_mag_30d': rolling['rolling_max_mag_30d'],
            'rolling_std_mag_30d': rolling['rolling_std_mag_30d'],
            
            # Other derived
            'days_since_last_significant': rolling['days_since_last_significant'],
            'local_seismic_density': rolling['local_seismic_density'],
            
            # Quality
            'nst_norm': quality['nst_norm'],
            'gap_norm': quality['gap_norm'],
            'dmin_norm': quality['dmin_norm'],
            'rms_norm': quality['rms_norm'],
        }
        
        return features
    
    def run(
        self,
        min_magnitude: Optional[float] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        
        from tqdm import tqdm
        
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Feature Engineering Pipeline Started")
        logger.info("=" * 60)
        
        # Get all events from database
        events = self.database.get_all_events(min_magnitude=min_magnitude)
        
        if not events:
            logger.warning("No events found in database")
            return {'events_processed': 0, 'features_stored': 0}
        
        logger.info(f"Processing {len(events)} events")
        logger.info(f"Parameters: radius={self.radius_km}km, windows={self.windows_days}")
        
        # Sort events by time
        events_sorted = sorted(events, key=lambda x: x['time'])
        
        all_features = []
        total = len(events_sorted)
        
        # Progress bar for feature computation
        print("\n")
        with tqdm(total=total, desc="Computing features", unit="event", 
                  bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}") as pbar:
            
            for i, event in enumerate(events_sorted):
                # Historical events are all events before current
                historical = events_sorted[:i]
                
                # Compute features
                features = self.compute_features_for_event(event, historical)
                all_features.append(features)
                
                # Update progress bar
                pbar.update(1)
                
                # Batch insert
                if len(all_features) >= batch_size:
                    self.database.insert_features(all_features)
                    all_features = []
        
        # Insert remaining features
        if all_features:
            self.database.insert_features(all_features)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Summary
        stats = {
            'events_processed': total,
            'features_stored': self.database.get_feature_count(),
            'duration_seconds': duration
        }
        
        logger.info("=" * 60)
        logger.info("Feature Engineering Pipeline Completed")
        logger.info(f"  Events Processed: {stats['events_processed']}")
        logger.info(f"  Features Stored:  {stats['features_stored']}")
        logger.info(f"  Duration:         {stats['duration_seconds']:.2f} seconds")
        logger.info("=" * 60)
        
        return stats
    
    def compute_features_for_prediction(
        self,
        latitude: float,
        longitude: float,
        depth: float,
        timestamp_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        
        if timestamp_ms is None:
            timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
        
        # Create synthetic event for feature computation
        synthetic_event = {
            'id': 'prediction',
            'time': timestamp_ms,
            'latitude': latitude,
            'longitude': longitude,
            'depth': depth,
            'magnitude': None,  # Unknown, this is what we're predicting
            'nst': None,
            'gap': None,
            'dmin': None,
            'rms': None
        }
        
        # Get historical events from database
        historical_events = self.database.get_all_events()
        
        # Compute features
        features = self.compute_features_for_event(synthetic_event, historical_events)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        
        return [
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
    
    def close(self):
        self.database.close()


def run_feature_pipeline():
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 60 + "\n")
    
    pipeline = FeaturePipeline()
    
    try:
        stats = pipeline.run()
        
        print("\n" + "-" * 40)
        print("SUMMARY")
        print("-" * 40)
        print(f"Events Processed: {stats['events_processed']}")
        print(f"Features Stored:  {stats['features_stored']}")
        print(f"Duration:         {stats['duration_seconds']:.2f}s")
        print("-" * 40 + "\n")
        
        return stats
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    run_feature_pipeline()