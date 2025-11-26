#!/usr/bin/env python3
"""
Extract Features from User Tracks

Извлекает аудио фичи из пользовательских треков для последующего
применения arousal-valence регрессора.

Использует ZoneTrainer с автоматическим кешированием фичей.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.zone_trainer import ZoneTrainer


def extract_features(test_data_path: str = "tests/test_data.txt",
                    use_cache: bool = True,
                    checkpoint_dir: str = "models/checkpoints",
                    output_path: str = "results/user_tracks_features.csv"):
    """
    Извлекает фичи из пользовательских треков.

    Args:
        test_data_path: Путь к файлу со списком треков
        use_cache: Использовать кеш (если есть)
        checkpoint_dir: Директория для кеша
        output_path: Путь для сохранения CSV с фичами

    Returns:
        DataFrame с фичами
    """
    print("=" * 80)
    print("🎵 USER TRACK FEATURE EXTRACTION")
    print("=" * 80)

    # Проверка входного файла
    test_data_path = Path(test_data_path)
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    print(f"\n📂 Input file: {test_data_path}")

    # Подсчет треков
    with open(test_data_path, 'r', encoding='utf-16') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"📊 Total tracks to process: {len(lines)}")

    # Создаем ZoneTrainer (checkpoint_manager создаётся автоматически)
    print(f"\n🔧 Creating ZoneTrainer with automatic caching...")
    print(f"   Cache directory: {checkpoint_dir}")

    trainer = ZoneTrainer(
        test_data_path=str(test_data_path),
        use_gpu=False,  # Не нужен GPU для feature extraction
        use_embeddings=False,
        use_music_emotion=False,
        use_fast_mode=False  # Полный набор фичей
    )

    # Загружаем данные
    print(f"\n📥 Loading training data...")

    def log_callback(level, message):
        """Callback для логов."""
        print(f"  {message}")

    def progress_callback(current, total, message):
        """Callback для прогресса."""
        percent = (current / total) * 100 if total > 0 else 0
        print(f"  [{current}/{total}] ({percent:.1f}%) - {message}")

    trainer.load_training_data(
        progress_callback=progress_callback,
        log_callback=log_callback
    )

    # Извлекаем фичи (с кешированием)
    print(f"\n🎯 Extracting features...")
    print(f"   Use cache: {use_cache}")
    print(f"   Checkpoint interval: every 5 tracks")

    features_df = trainer.extract_features(
        use_cache=use_cache,
        progress_callback=progress_callback,
        log_callback=log_callback,
        checkpoint_interval=5
    )

    print(f"\n✅ Feature extraction completed!")
    print(f"   Extracted features: {len(features_df)}")
    print(f"   Feature columns: {features_df.shape[1]}")

    # Сохраняем в CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(output_path, index=False)
    print(f"\n💾 Features saved to: {output_path}")

    # Статистика
    print(f"\n📈 Feature statistics:")
    print(f"   Columns: {', '.join(features_df.columns[:5].tolist())}...")
    print(f"\n   Sample (first 3 tracks):")
    print(features_df.head(3).to_string())

    # Проверка на NaN
    nan_count = features_df.isna().sum().sum()
    if nan_count > 0:
        print(f"\n⚠️  Warning: Found {nan_count} NaN values in features")
        print(f"   Columns with NaNs:")
        nan_cols = features_df.columns[features_df.isna().any()].tolist()
        for col in nan_cols:
            nan_count_col = features_df[col].isna().sum()
            print(f"     - {col}: {nan_count_col} NaNs")
    else:
        print(f"\n✅ No NaN values found")

    print(f"\n" + "=" * 80)
    print(f"✅ FEATURE EXTRACTION COMPLETED!")
    print(f"=" * 80)
    print(f"\nFeatures cached to: {checkpoint_dir}/features.pkl")
    print(f"Features CSV saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Apply arousal-valence regressor to predict arousal/valence")
    print(f"  2. Visualize distribution compared to DEAM dataset")

    return features_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from user tracks for arousal-valence prediction'
    )
    parser.add_argument(
        '--test-data', type=str, default='tests/test_data.txt',
        help='Path to test data file (default: tests/test_data.txt)'
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Force re-extraction (ignore cache)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='models/checkpoints',
        help='Checkpoint directory for caching (default: models/checkpoints)'
    )
    parser.add_argument(
        '--output', type=str, default='results/user_tracks_features.csv',
        help='Output CSV path (default: results/user_tracks_features.csv)'
    )

    args = parser.parse_args()

    try:
        extract_features(
            test_data_path=args.test_data,
            use_cache=not args.no_cache,
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output
        )
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
