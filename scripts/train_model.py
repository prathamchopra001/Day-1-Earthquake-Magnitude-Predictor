import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model.train import train_model
from src.model.evaluate import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description='Train earthquake magnitude predictor')
    
    parser.add_argument(
        '--kernel',
        type=str,
        choices=['simple', 'composite', 'advanced'],
        default='simple',
        help='Kernel type (default: simple)'
    )
    
    parser.add_argument(
        '--min-mag',
        type=float,
        default=None,
        help='Minimum magnitude filter (default: None)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=4000,
        help='Maximum training samples'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating evaluation plots'
    )
    
    args = parser.parse_args()
    
    print(f"\nTraining Configuration:")
    print(f"  Kernel Type:   {args.kernel}")
    print(f"  Min Magnitude: {args.min_mag or 'None (all)'}")
    print(f"  Max Samples:   {args.max_samples}")
    print()
    
    try:
        # Train model
        results = train_model(
            kernel_type=args.kernel,
            min_magnitude=args.min_mag,
            max_samples=args.max_samples
        )
        
        # Generate evaluation plots
        if not args.no_plots:
            print("\nGenerating evaluation plots...")
            evaluator = ModelEvaluator()
            
            y_test = results['predictions']['y_test']
            y_pred = results['predictions']['y_pred']
            y_std = results['predictions']['y_std']
            
            evaluator.evaluate(y_test, y_pred, y_std)
            evaluator.generate_all_plots(y_test, y_pred, y_std)
            evaluator.save_results()
            evaluator.print_summary()
        
        print("\nTraining completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())