#!/usr/bin/env python3
"""
Train Production Model - Single Entry Point

This script is the ONLY place to train production models.
All optimizations are consolidated in ProductionPipeline.

Usage:
    # Standard training with best practices
    python scripts/train_production.py \
        --user results/user_50_frames.pkl \
        --deam results/deam_50_frames.pkl \
        --annotations data/deam/annotations/annotations_averaged.csv

    # Custom output path
    python scripts/train_production.py \
        --user results/user_50_frames.pkl \
        --deam results/deam_50_frames.pkl \
        -o models/my_model.pkl

    # Override settings (for experiments only)
    python scripts/train_production.py \
        --user results/user_200_frames.pkl \
        --deam results/deam_full_frames.pkl \
        --top-n 80 \
        --algorithm randomforest
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.production_pipeline import ProductionPipeline, PipelineConfig


def interactive_menu() -> dict:
    """Интерактивный выбор параметров обучения."""
    print("\n" + "=" * 60)
    print("НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ")
    print("=" * 60)

    # Значения по умолчанию (лучшая модель: 65.27% CV accuracy)
    defaults = {
        'top_n_features': 100,
        'yellow_threshold': 4.0,
        'purple_threshold': 5.7
    }

    params = {}

    # top_n_features
    print(f"\n1. top_n_features [модель обучена на: {defaults['top_n_features']}]")
    print("   Количество топ-фичей для отбора (рекомендуется 15-50)")
    val = input(f"   Введите значение (Enter = {defaults['top_n_features']}): ").strip()
    params['top_n_features'] = int(val) if val else defaults['top_n_features']

    # yellow_threshold
    print(f"\n2. yellow_threshold [модель обучена на: {defaults['yellow_threshold']}]")
    print("   Порог arousal для YELLOW зоны (низкая энергия)")
    val = input(f"   Введите значение (Enter = {defaults['yellow_threshold']}): ").strip()
    params['yellow_threshold'] = float(val) if val else defaults['yellow_threshold']

    # purple_threshold
    print(f"\n3. purple_threshold [модель обучена на: {defaults['purple_threshold']}]")
    print("   Порог arousal для PURPLE зоны (высокая энергия)")
    val = input(f"   Введите значение (Enter = {defaults['purple_threshold']}): ").strip()
    params['purple_threshold'] = float(val) if val else defaults['purple_threshold']

    # Показать выбранные параметры
    print("\n" + "=" * 60)
    print("Выбранные параметры:")
    print(f"  top_n_features: {params['top_n_features']}")
    print(f"  yellow_threshold: {params['yellow_threshold']}")
    print(f"  purple_threshold: {params['purple_threshold']}")
    print("=" * 60)

    return params


def main():
    parser = argparse.ArgumentParser(
        description='Train Production Zone Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Best Practices (automatically applied):
  - XGBoost classifier (65% accuracy vs 62% RandomForest)
  - Top-50 feature selection (noise reduction)
  - Normalized drop features (fixes YELLOW paradox)
  - Source normalization (DEAM -> User distribution)

These settings are the result of extensive optimization.
Only change them if you have a specific reason.
        """
    )

    parser.add_argument(
        '--user', '-u',
        type=Path,
        required=True,
        help='Path to user frame features (.pkl)'
    )

    parser.add_argument(
        '--deam', '-d',
        type=Path,
        required=True,
        help='Path to DEAM frame features (.pkl)'
    )

    parser.add_argument(
        '--annotations', '-a',
        type=Path,
        default=Path('data/deam/annotations/annotations_averaged.csv'),
        help='Path to DEAM annotations CSV'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output model path (default: models/production/zone_classifier_YYYYMMDD.pkl)'
    )

    # Advanced options (use with caution)
    parser.add_argument(
        '--algorithm',
        choices=['xgboost', 'randomforest'],
        default='xgboost',
        help='Algorithm (default: xgboost - best performer)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=100,
        help='Number of top features (default: 100 - best result)'
    )

    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable source normalization'
    )

    parser.add_argument(
        '--no-drop-features',
        action='store_true',
        help='Disable normalized drop features'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Интерактивный выбор параметров перед обучением'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not args.user.exists():
        print(f"Error: User features not found: {args.user}")
        print("\nExtract features first:")
        print("  python scripts/extract_unified_features.py \\")
        print("      --input tests/test_sample_50.txt \\")
        print("      --output results/user_50_frames.pkl \\")
        print("      --source audio --method frames")
        return 1

    if not args.deam.exists():
        print(f"Error: DEAM features not found: {args.deam}")
        print("\nExtract DEAM features first:")
        print("  python scripts/extract_unified_features.py \\")
        print("      --input data/deam/annotations/annotations_averaged.csv \\")
        print("      --output results/deam_50_frames.pkl \\")
        print("      --source deam --method frames")
        return 1

    # Default output path with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = Path(f'models/production/zone_classifier_{timestamp}.pkl')

    # Create config - интерактивно или из defaults
    if args.interactive:
        params = interactive_menu()
        config = PipelineConfig(
            top_n_features=params['top_n_features'],
            yellow_threshold=params['yellow_threshold'],
            purple_threshold=params['purple_threshold']
        )
    else:
        config = PipelineConfig()

    print("\n" + "=" * 60)
    print("PRODUCTION MODEL TRAINING")
    print("=" * 60)
    print(f"\nInputs:")
    print(f"  User frames: {args.user}")
    print(f"  DEAM frames: {args.deam}")
    print(f"  Annotations: {args.annotations}")
    print(f"\nSettings (from PipelineConfig):")
    print(f"  algorithm: {config.algorithm}")
    print(f"  top_n_features: {config.top_n_features}")
    print(f"  yellow_threshold: {config.yellow_threshold}")
    print(f"  purple_threshold: {config.purple_threshold}")
    print(f"  class_balance: {config.class_balance}")
    print(f"  normalize_sources: {config.normalize_sources}")
    print(f"  add_drop_features: {config.add_drop_features}")
    print(f"\nOutput: {args.output}")
    print("=" * 60 + "\n")

    # Train
    pipeline = ProductionPipeline(config)
    results = pipeline.train(args.user, args.deam, args.annotations)
    pipeline.save(args.output)

    # Also save as 'latest' symlink
    latest_path = args.output.parent / 'zone_classifier_latest.pkl'
    if latest_path.exists():
        latest_path.unlink()

    # Copy instead of symlink for better portability
    import shutil
    shutil.copy(args.output, latest_path)
    print(f"\nAlso saved as: {latest_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTo use this model in production:")
    print(f"  python main.py --model {args.output} --file track.mp3")
    print(f"\nOr load in code:")
    print(f"  from src.training.production_pipeline import ProductionPipeline")
    print(f"  pipeline = ProductionPipeline.load('{args.output}')")
    print(f"  zone, conf = pipeline.predict(features_dict)")

    return 0


if __name__ == '__main__':
    sys.exit(main())