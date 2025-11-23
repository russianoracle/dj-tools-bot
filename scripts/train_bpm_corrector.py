#!/usr/bin/env python3
"""
CLI tool for training BPM correction models.

Usage:
    python scripts/train_bpm_corrector.py data.txt --algorithms xgboost neural_network
    python scripts/train_bpm_corrector.py data.txt --algorithms ensemble --grid-search
    python scripts/train_bpm_corrector.py data.txt --output models/custom --epochs 300
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import BPMTrainer
from src.utils import get_logger

logger = get_logger(__name__)


def progress_callback(current: int, total: int, message: str):
    """Print progress."""
    percent = (current / total) * 100
    print(f"[{current}/{total}] ({percent:.1f}%) {message}")


def log_callback(level: str, message: str):
    """Print log message."""
    print(f"[{level}] {message}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train BPM correction models using labeled data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'data_file',
        type=str,
        help='Path to training data file (TSV with BPM labels)'
    )

    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=['xgboost', 'neural_network', 'ensemble'],
        default=['xgboost', 'neural_network', 'ensemble'],
        help='Algorithms to train (default: all)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='models/bpm_correctors',
        help='Output directory for trained models (default: models/bpm_correctors)'
    )

    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Perform hyperparameter grid search for XGBoost (slower)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of epochs for neural network training (default: 200)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for neural network (default: 32)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Validate data file
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    print("="*70)
    print("BPM Corrector Training")
    print("="*70)
    print(f"Data file: {data_path}")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    print(f"Output directory: {args.output}")
    print()

    try:
        # Initialize trainer
        trainer = BPMTrainer(test_data_path=str(data_path))

        # Prepare kwargs
        kwargs = {
            'grid_search': args.grid_search,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        }

        # Run training
        progress_cb = None if args.quiet else progress_callback
        log_cb = None if args.quiet else log_callback

        results = trainer.run_full_training_pipeline(
            algorithms=args.algorithms,
            save_dir=args.output,
            progress_callback=progress_cb,
            log_callback=log_cb,
            **kwargs
        )

        # Print summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)

        print(f"\nBaseline MAE (librosa): {results['baseline_mae']:.2f} BPM")

        for algo in args.algorithms:
            if algo in results:
                print(f"\n{algo.replace('_', ' ').title()}:")
                print(f"  Test MAE: {results[algo]['test_mae']:.2f} BPM")
                print(f"  Accuracy (±2 BPM): {results[algo]['accuracy_within_2bpm']:.1f}%")
                print(f"  Model saved: {results[algo]['model_path']}")

        print(f"\nBest model: {results['best_model']}")
        print("\n" + "="*70)

        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Training failed")
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
