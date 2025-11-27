import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.features.feature_pipeline import FeaturePipeline


def main():
    parser = argparse.ArgumentParser(description='Run feature engineering pipeline')
    
    parser.add_argument(
        '--min-mag',
        type=float,
        default=None,
        help='Minimum magnitude filter (default: process all)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for database inserts (default: 1000)'
    )
    
    args = parser.parse_args()
    
    print(f"\nRunning Feature Engineering Pipeline")
    print(f"  Min Magnitude Filter: {args.min_mag or 'None (all events)'}")
    print(f"  Batch Size: {args.batch_size}")
    print()
    
    pipeline = FeaturePipeline()
    
    try:
        stats = pipeline.run(
            min_magnitude=args.min_mag,
            batch_size=args.batch_size
        )
        
        print("\nFeature engineering completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nFeature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    sys.exit(main())