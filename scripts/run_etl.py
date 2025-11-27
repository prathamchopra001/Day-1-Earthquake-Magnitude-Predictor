import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.etl_pipeline import ETLPipeline


def main():
    parser = argparse.ArgumentParser(description='Run earthquake data ETL pipeline')
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to fetch (default: 30)'
    )
    
    parser.add_argument(
        '--min-mag',
        type=float,
        default=2.5,
        help='Minimum magnitude (default: 2.5)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of events to fetch (default: no limit)'
    )
    
    args = parser.parse_args()
    
    print(f"\nRunning ETL with parameters:")
    print(f"  Days: {args.days}")
    print(f"  Min Magnitude: {args.min_mag}")
    print(f"  Limit: {args.limit or 'None'}")
    print()
    
    pipeline = ETLPipeline()
    
    try:
        kwargs = {}
        if args.limit:
            kwargs['limit'] = args.limit
        
        stats = pipeline.run(
            days_back=args.days,
            min_magnitude=args.min_mag,
            **kwargs
        )
        
        print("\nPipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        return 1
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    sys.exit(main())