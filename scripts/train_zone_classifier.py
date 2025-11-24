#!/usr/bin/env python3
"""
CLI tool for training energy zone classification models.

Usage:
    python scripts/train_zone_classifier.py data.txt --algorithms xgboost neural_network
    python scripts/train_zone_classifier.py data.txt --algorithms ensemble --grid-search
    python scripts/train_zone_classifier.py data.txt --output models/custom --epochs 300
    python scripts/train_zone_classifier.py data.txt --no-embeddings  # Use only librosa features
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import ZoneTrainer, CheckpointManager
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
        description='Train energy zone classification models using labeled data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models with default settings
  python scripts/train_zone_classifier.py tests/test_data.txt

  # Train only XGBoost with grid search
  python scripts/train_zone_classifier.py data.txt --algorithms xgboost --grid-search

  # Train with more epochs for neural network
  python scripts/train_zone_classifier.py data.txt --epochs 300

  # Use only librosa features (no torchaudio embeddings)
  python scripts/train_zone_classifier.py data.txt --no-embeddings

  # Enable checkpoints for long training runs
  python scripts/train_zone_classifier.py data.txt --checkpoint-dir models/checkpoints
        """
    )

    parser.add_argument(
        'data_file',
        type=str,
        help='Path to training data file (TSV with Zone labels: yellow/green/purple)'
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
        default='models/zone_classifiers',
        help='Output directory for trained models (default: models/zone_classifiers)'
    )

    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Perform hyperparameter grid search for XGBoost (slower but better)'
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
        '--no-embeddings',
        action='store_true',
        help='Use only librosa features (skip torchaudio embeddings for faster training)'
    )

    parser.add_argument(
        '--use-music-emotion',
        action='store_true',
        help='Extract arousal/valence using music emotion model (VERY slow: ~15 sec/track)'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration (use CPU only)'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Enable checkpointing and specify checkpoint directory'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from latest checkpoint (requires --checkpoint-dir)'
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

    print("=" * 70)
    print("ZONE CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"Data file: {data_path}")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    print(f"Output directory: {args.output}")
    print(f"Use embeddings: {not args.no_embeddings}")
    print(f"Use music emotion: {args.use_music_emotion}")
    print(f"Use GPU: {not args.no_gpu}")
    if args.checkpoint_dir:
        print(f"Checkpoint directory: {args.checkpoint_dir}")
        print(f"Resume: {args.resume}")
    print()

    try:
        # Initialize trainer
        trainer = ZoneTrainer(
            test_data_path=str(data_path),
            use_gpu=not args.no_gpu,
            use_music_emotion=args.use_music_emotion
        )

        # Setup checkpoint manager
        checkpoint_manager = None
        if args.checkpoint_dir:
            checkpoint_manager = CheckpointManager(args.checkpoint_dir)
            print(f"Checkpointing enabled: {args.checkpoint_dir}")

        # Prepare kwargs (checkpoint_manager is passed separately in run_full_training_pipeline)
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
            checkpoint_dir=args.checkpoint_dir,
            include_embeddings=not args.no_embeddings,
            progress_callback=progress_cb,
            log_callback=log_cb,
            **kwargs
        )

        # Print summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)

        print(f"\nTotal tracks: {results.get('num_tracks', 0)}")

        for algo in args.algorithms:
            if algo in results and isinstance(results[algo], dict):
                algo_results = results[algo]
                print(f"\n{algo.replace('_', ' ').title()}:")
                print(f"  Overall Test Accuracy: {algo_results['test_accuracy']:.1%}")
                print(f"  Yellow Zone Accuracy: {algo_results['test_accuracy_yellow']:.1%}")
                print(f"  Green Zone Accuracy: {algo_results['test_accuracy_green']:.1%}")
                print(f"  Purple Zone Accuracy: {algo_results['test_accuracy_purple']:.1%}")
                print(f"  Model saved: {algo_results['model_path']}")

                # Show confusion matrix
                if 'confusion_matrix' in algo_results:
                    print(f"  Confusion Matrix:")
                    cm = algo_results['confusion_matrix']
                    print(f"    [Yellow] [Green] [Purple]")
                    for i, zone in enumerate(['Yellow', 'Green', 'Purple']):
                        print(f"    {zone:7s} {cm[i]}")

        print(f"\nBest model: {results['best_model']}")
        print(f"Best test accuracy: {results['best_test_accuracy']:.1%}")
        print("\n" + "=" * 70)

        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        if args.checkpoint_dir:
            print(f"Progress saved in checkpoints. Use --resume to continue.")
        sys.exit(130)
    except Exception as e:
        logger.exception("Training failed")
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
