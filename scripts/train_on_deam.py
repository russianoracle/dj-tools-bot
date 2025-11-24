#!/usr/bin/env python3
"""
Train Zone Classifier on DEAM Dataset

Обучает классификатор зон на DEAM датасете с предрассчитанными фичами.

Требует:
  1. dataset/deam_tempo.csv (создаётся extract_deam_tempo.py)
  2. dataset/deam_processed/ (создаётся prepare_deam_dataset.py)
"""

import sys
import argparse
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.deam_trainer import DEAMZoneTrainer
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train zone classifier on DEAM dataset')
    parser.add_argument('--deam-dir', type=str, default='dataset/deam_processed',
                       help='DEAM processed dataset directory')
    parser.add_argument('--algorithm', type=str, default='ensemble',
                       choices=['xgboost', 'neural', 'ensemble'],
                       help='Training algorithm (default: ensemble)')
    parser.add_argument('--model-dir', type=str, default='models/deam_trained',
                       help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs for neural network (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for neural network (default: 32)')

    args = parser.parse_args()

    try:
        logger.info("=" * 80)
        logger.info("DEAM Zone Classifier Training")
        logger.info("=" * 80)

        # Шаг 1: Инициализация тренера
        logger.info("\n[1/3] Initializing DEAM trainer...")
        trainer = DEAMZoneTrainer(deam_dir=args.deam_dir)

        # Шаг 2: Загрузка датасета
        logger.info("\n[2/3] Loading DEAM dataset...")
        train_size, val_size, test_size = trainer.load_deam_dataset(
            use_precomputed_splits=True
        )

        logger.info(f"\nDataset loaded successfully:")
        logger.info(f"  Train: {train_size} samples")
        logger.info(f"  Val:   {val_size} samples")
        logger.info(f"  Test:  {test_size} samples")

        # Шаг 3: Обучение модели
        logger.info(f"\n[3/3] Training {args.algorithm} model...")

        # Параметры обучения
        train_kwargs = {}
        if args.algorithm in ['neural', 'ensemble']:
            train_kwargs['epochs'] = args.epochs
            train_kwargs['batch_size'] = args.batch_size

        # Обучение
        model, metrics = trainer.train_model(
            algorithm=args.algorithm,
            save_path=args.model_dir,
            **train_kwargs
        )

        # Результаты
        logger.info("\n" + "=" * 80)
        logger.info("✅ Training completed successfully!")
        logger.info("=" * 80)

        logger.info(f"\nTest set metrics:")
        logger.info(f"  Accuracy: {metrics.get('test_accuracy', 0):.1f}%")

        if 'test_report' in metrics:
            logger.info(f"\nClassification Report:")
            logger.info(f"{metrics['test_report']}")

        if 'test_confusion_matrix' in metrics:
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"{metrics['test_confusion_matrix']}")

        logger.info(f"\nModel saved to: {args.model_dir}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"\n❌ Dataset not found: {e}")
        logger.error(f"\nPlease run:")
        logger.error(f"  1. python scripts/extract_deam_tempo.py")
        logger.error(f"  2. python scripts/prepare_deam_dataset.py")
        return 1

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
