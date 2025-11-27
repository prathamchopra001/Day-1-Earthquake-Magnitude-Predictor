import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.api_client import USGSApiClient
from src.data.database import EarthquakeDatabase
from src.utils.logger import setup_logger
from src.utils.helpers import load_config, unix_to_datetime

logger = setup_logger('etl_pipeline')


class ETLPipeline:
    
    def __init__(self, config_path: str = "config.yaml"):
        
        self.config = load_config(config_path)
        self.api_client = USGSApiClient(config_path)
        self.database = EarthquakeDatabase()
        
        # Initialize database tables
        self.database.create_tables()
    
    def extract(
        self,
        days_back: Optional[int] = None,
        min_magnitude: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        
        if days_back is None:
            days_back = self.config['data']['days_to_fetch']
        
        if min_magnitude is None:
            min_magnitude = self.config['data']['min_magnitude_for_training']
        
        logger.info(f"EXTRACT: Fetching {days_back} days of data (min mag: {min_magnitude})")
        
        events = self.api_client.get_earthquakes(
            days_back=days_back,
            min_magnitude=min_magnitude,
            **kwargs
        )
        
        logger.info(f"EXTRACT: Retrieved {len(events)} events from API")
        
        return events
    
    def transform(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        from tqdm import tqdm
        
        logger.info(f"TRANSFORM: Processing {len(events)} events")
        
        transformed = []
        
        stats = {
            'total': len(events),
            'filtered_type': 0,
            'filtered_null_mag': 0,
            'filtered_null_coords': 0,
            'depth_corrected': 0,
            'passed': 0
        }
        
        # Progress bar for transformation
        print("\n")
        for event in tqdm(events, desc="Transforming data", unit="event",
                         bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}"):
            # Filter: Keep only earthquakes
            if event.get('type') != 'earthquake':
                stats['filtered_type'] += 1
                continue
            
            # Filter: Must have magnitude
            if event.get('magnitude') is None:
                stats['filtered_null_mag'] += 1
                continue
            
            # Filter: Must have coordinates
            if event.get('latitude') is None or event.get('longitude') is None:
                stats['filtered_null_coords'] += 1
                continue
            
            # Transform: Correct negative depths
            if event.get('depth') is not None and event['depth'] < 0:
                event['depth'] = 0.0
                stats['depth_corrected'] += 1
            
            # Transform: Handle missing depth (set to 0 for very shallow)
            if event.get('depth') is None:
                event['depth'] = 0.0
            
            # Transform: Ensure numeric types
            event['magnitude'] = float(event['magnitude'])
            event['latitude'] = float(event['latitude'])
            event['longitude'] = float(event['longitude'])
            event['depth'] = float(event['depth'])
            
            # Validate: Coordinate ranges
            if not (-90 <= event['latitude'] <= 90):
                logger.warning(f"Invalid latitude for event {event['id']}: {event['latitude']}")
                continue
            
            if not (-180 <= event['longitude'] <= 180):
                logger.warning(f"Invalid longitude for event {event['id']}: {event['longitude']}")
                continue
            
            # Validate: Reasonable depth (0-700 km)
            if event['depth'] > 700:
                logger.warning(f"Unusual depth for event {event['id']}: {event['depth']} km")
                # Still keep it, just log warning
            
            transformed.append(event)
            stats['passed'] += 1
        
        # Log transformation statistics
        print(f"\n   Transform Statistics:")
        print(f"   ├── Total input:        {stats['total']}")
        print(f"   ├── Non-earthquake:     {stats['filtered_type']} (removed)")
        print(f"   ├── Null magnitude:     {stats['filtered_null_mag']} (removed)")
        print(f"   ├── Null coordinates:   {stats['filtered_null_coords']} (removed)")
        print(f"   ├── Depth corrected:    {stats['depth_corrected']}")
        print(f"   └── Output:             {stats['passed']}")
        
        return transformed
    
    def load(self, events: List[Dict[str, Any]]) -> int:
        
        logger.info(f"LOAD: Inserting {len(events)} events into database")
        
        count = self.database.insert_events(events)
        
        logger.info(f"LOAD: Successfully loaded {count} events")
        
        return count
    
    def run(
        self,
        days_back: Optional[int] = None,
        min_magnitude: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("ETL Pipeline Started")
        logger.info("=" * 60)
        
        # Extract
        raw_events = self.extract(days_back, min_magnitude, **kwargs)
        
        # Transform
        clean_events = self.transform(raw_events)
        
        # Load
        loaded_count = self.load(clean_events)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Summary
        stats = {
            'extracted': len(raw_events),
            'transformed': len(clean_events),
            'loaded': loaded_count,
            'duration_seconds': duration,
            'total_in_database': self.database.get_event_count()
        }
        
        logger.info("=" * 60)
        logger.info("ETL Pipeline Completed")
        logger.info(f"  Extracted:  {stats['extracted']} events")
        logger.info(f"  Transformed: {stats['transformed']} events")
        logger.info(f"  Loaded:     {stats['loaded']} events")
        logger.info(f"  Duration:   {stats['duration_seconds']:.2f} seconds")
        logger.info(f"  Total in DB: {stats['total_in_database']} events")
        logger.info("=" * 60)
        
        return stats
    
    def get_database_summary(self) -> Dict[str, Any]:
        
        event_count = self.database.get_event_count()
        feature_count = self.database.get_feature_count()
        min_time, max_time = self.database.get_date_range()
        
        summary = {
            'event_count': event_count,
            'feature_count': feature_count,
            'date_range': {
                'min': unix_to_datetime(min_time).isoformat() if min_time else None,
                'max': unix_to_datetime(max_time).isoformat() if max_time else None
            }
        }
        
        return summary
    
    def close(self):
        """Close database connection."""
        self.database.close()


def run_etl():
    """Main function to run the ETL pipeline."""
    print("\n" + "=" * 60)
    print("  EARTHQUAKE DATA ETL PIPELINE")
    print("=" * 60 + "\n")
    
    pipeline = ETLPipeline()
    
    try:
        # Run pipeline with default settings from config
        stats = pipeline.run()
        
        # Print summary
        print("\n" + "-" * 40)
        print("DATABASE SUMMARY")
        print("-" * 40)
        
        summary = pipeline.get_database_summary()
        print(f"Total Events: {summary['event_count']}")
        print(f"Features Computed: {summary['feature_count']}")
        
        if summary['date_range']['min']:
            print(f"Date Range: {summary['date_range']['min'][:10]} to {summary['date_range']['max'][:10]}")
        
        print("-" * 40 + "\n")
        
        return stats
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    run_etl()