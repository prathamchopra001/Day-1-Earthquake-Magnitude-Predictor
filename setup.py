import argparse
import sys
import os
import random
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def create_demo_data():
    from src.data.database import EarthquakeDatabase
    
    print("\n📊 Creating demo earthquake data...")
    
    db = EarthquakeDatabase()
    db.create_tables()
    db.clear_tables()
    
    base_time = int(datetime.utcnow().timestamp() * 1000)
    day_ms = 24 * 60 * 60 * 1000
    
    # Create realistic-looking sample data
    regions = [
        {'lat_center': -17.9, 'lon_center': -178.4, 'name': 'Fiji'},
        {'lat_center': 35.7, 'lon_center': 139.7, 'name': 'Japan'},
        {'lat_center': -33.4, 'lon_center': -70.6, 'name': 'Chile'},
        {'lat_center': -6.2, 'lon_center': 106.8, 'name': 'Indonesia'},
        {'lat_center': 36.8, 'lon_center': -121.9, 'name': 'California'},
        {'lat_center': 61.2, 'lon_center': -149.9, 'name': 'Alaska'},
    ]
    
    sample_events = []
    
    for i in range(200):
        region = random.choice(regions)
        
        # Add randomness to location
        lat = region['lat_center'] + random.uniform(-5, 5)
        lon = region['lon_center'] + random.uniform(-5, 5)
        
        # Time spread over 60 days
        time_offset = random.randint(0, 60 * day_ms)
        
        # Magnitude distribution (more small quakes than large)
        mag = 2.5 + random.expovariate(0.8)
        mag = min(mag, 7.5)  # Cap at 7.5
        
        event = {
            'id': f'demo_{i:04d}',
            'code': f'demo_{i:04d}',
            'time': base_time - time_offset,
            'latitude': lat,
            'longitude': lon,
            'depth': random.uniform(5, 500),
            'magnitude': round(mag, 1),
            'mag_type': 'mb',
            'place': f'{random.randint(10, 300)} km from {region["name"]}',
            'type': 'earthquake',
            'status': 'reviewed',
            'tsunami': 0,
            'sig': int(mag * 100),
            'nst': random.randint(10, 150),
            'gap': random.uniform(20, 200),
            'dmin': random.uniform(0.1, 5),
            'rms': random.uniform(0.1, 1.5)
        }
        sample_events.append(event)
    
    db.insert_events(sample_events)
    print(f"   ✅ Created {len(sample_events)} demo events")
    
    db.close()
    return len(sample_events)


def run_etl():
    """Run ETL pipeline with real API data."""
    from src.data.etl_pipeline import ETLPipeline
    
    print("\n🔄 Running ETL pipeline...")
    
    pipeline = ETLPipeline()
    
    try:
        stats = pipeline.run(days_back=30, min_magnitude=2.5)
        print(f"   ✅ Loaded {stats['loaded']} events from USGS API")
        return stats['loaded']
    except Exception as e:
        print(f"   ⚠️ ETL failed: {e}")
        print("   ℹ️ Using demo data instead...")
        return 0
    finally:
        pipeline.close()


def compute_features():
    from src.features.feature_pipeline import FeaturePipeline
    
    print("\n⚙️ Computing features...")
    
    pipeline = FeaturePipeline()
    
    try:
        stats = pipeline.run()
        print(f"   ✅ Computed features for {stats['events_processed']} events")
        return stats['events_processed']
    finally:
        pipeline.close()


def train_model():
    from src.model.train import train_model as _train_model
    
    print("\n🧠 Training model...")
    
    results = _train_model(kernel_type='simple', max_samples=1500)
    
    metrics = results['metrics']
    print(f"   ✅ Model trained successfully!")
    print(f"      MAE:  {metrics['mae']:.4f}")
    print(f"      RMSE: {metrics['rmse']:.4f}")
    print(f"      R²:   {metrics['r2']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Setup earthquake predictor project')
    parser.add_argument('--demo', action='store_true', 
                       help='Use demo data instead of API')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🌍 EARTHQUAKE MAGNITUDE PREDICTOR SETUP")
    print("=" * 60)
    
    # Step 1: Load data
    if args.demo:
        count = create_demo_data()
    else:
        count = run_etl()
        if count == 0:
            count = create_demo_data()
    
    if count == 0:
        print("\n❌ Failed to load any data. Exiting.")
        return 1
    
    # Step 2: Compute features
    compute_features()
    
    # Step 3: Train model
    train_model()
    
    print("\n" + "=" * 60)
    print("  ✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("   1. Run the Streamlit app:")
    print("      streamlit run app/streamlit_app.py")
    print("")
    print("   2. Open http://localhost:8501 in your browser")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())