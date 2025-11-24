#!/usr/bin/env python3
"""
DEAM Dataset Preparation

Объединяет DEAM фичи + tempo + arousal-valence аннотации
в готовый датасет для обучения классификатора зон.

Требует: dataset/deam_tempo.csv (создаётся extract_deam_tempo.py)
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import logging

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.deam_loader import DEAMLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_deam_dataset(
    tempo_csv: str = "dataset/deam_tempo.csv",
    yellow_threshold: float = 4.0,
    purple_threshold: float = 6.0,
    output_dir: str = "dataset/deam_processed",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
):
    """
    Подготавливает DEAM датасет для обучения

    Args:
        tempo_csv: Путь к CSV с извлечённым tempo
        yellow_threshold: Порог arousal для Yellow зоны (arousal < threshold)
        purple_threshold: Порог arousal для Purple зоны (arousal > threshold)
        output_dir: Директория для сохранения результатов
        test_size: Размер test set (0.1 = 10%)
        val_size: Размер validation set (0.1 = 10%)
        random_state: Random seed для воспроизводимости
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("DEAM Dataset Preparation")
    logger.info("=" * 80)

    # Шаг 1: Загружаем DEAM фичи + аннотации
    logger.info("\n[1/6] Loading DEAM features and annotations...")
    loader = DEAMLoader()

    deam_features = loader.load_all_features()
    logger.info(f"  Loaded features for {len(deam_features)} tracks")
    logger.info(f"  Features: {list(deam_features.columns)}")

    annotations = loader.load_annotations()
    logger.info(f"  Loaded annotations for {len(annotations)} tracks")

    # Шаг 2: Загружаем tempo
    logger.info("\n[2/6] Loading extracted tempo...")
    tempo_path = Path(tempo_csv)

    if not tempo_path.exists():
        raise FileNotFoundError(
            f"Tempo file not found: {tempo_path}\n"
            f"Please run: python scripts/extract_deam_tempo.py"
        )

    tempo_df = pd.read_csv(tempo_path)
    tempo_df = tempo_df[tempo_df['success'] == True][['track_id', 'tempo']]
    logger.info(f"  Loaded tempo for {len(tempo_df)} tracks")
    logger.info(f"  Tempo range: [{tempo_df['tempo'].min():.1f}, {tempo_df['tempo'].max():.1f}] BPM")

    # Шаг 3: Объединяем все данные
    logger.info("\n[3/6] Merging features + tempo + annotations...")

    # Объединяем фичи с tempo
    dataset = deam_features.merge(tempo_df, on='track_id', how='inner')
    logger.info(f"  After merging with tempo: {len(dataset)} tracks")

    # Объединяем с аннотациями
    dataset = dataset.merge(annotations, on='track_id', how='inner')
    logger.info(f"  After merging with annotations: {len(dataset)} tracks")

    # Шаг 4: Конвертируем arousal → зоны
    logger.info("\n[4/6] Converting arousal to zones...")
    logger.info(f"  Yellow threshold: arousal < {yellow_threshold}")
    logger.info(f"  Purple threshold: arousal > {purple_threshold}")

    dataset['zone'] = dataset['arousal'].apply(
        lambda a: loader.convert_arousal_to_zone(a, yellow_threshold, purple_threshold)
    )

    # Статистика зон
    zone_counts = dataset['zone'].value_counts()
    logger.info(f"  Zone distribution:")
    for zone, count in zone_counts.items():
        pct = count / len(dataset) * 100
        logger.info(f"    {zone}: {count} ({pct:.1f}%)")

    # Проверка на дисбаланс
    min_zone_count = zone_counts.min()
    max_zone_count = zone_counts.max()
    imbalance_ratio = max_zone_count / min_zone_count
    if imbalance_ratio > 2.0:
        logger.warning(f"  ⚠️  Class imbalance detected: {imbalance_ratio:.1f}x ratio")
        logger.warning(f"  Consider adjusting thresholds or using class weights")

    # Шаг 5: Создаём train/val/test splits
    logger.info("\n[5/6] Creating train/val/test splits...")

    # Сначала отделяем test set
    train_val, test = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset['zone']  # Сохраняем пропорции зон
    )

    # Затем из оставшегося отделяем validation
    val_size_adjusted = val_size / (1 - test_size)  # Корректируем размер
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val['zone']
    )

    logger.info(f"  Train: {len(train)} tracks ({len(train)/len(dataset)*100:.1f}%)")
    logger.info(f"  Val:   {len(val)} tracks ({len(val)/len(dataset)*100:.1f}%)")
    logger.info(f"  Test:  {len(test)} tracks ({len(test)/len(dataset)*100:.1f}%)")

    # Проверка стратификации
    logger.info(f"\n  Zone distribution per split:")
    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        zone_dist = split_df['zone'].value_counts(normalize=True) * 100
        logger.info(f"    {split_name}: " + ", ".join([f"{z}:{p:.1f}%" for z, p in zone_dist.items()]))

    # Шаг 6: Сохраняем результаты
    logger.info("\n[6/6] Saving processed dataset...")

    # Сохраняем splits
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    # Сохраняем полный датасет
    dataset.to_csv(output_dir / "deam_complete.csv", index=False)

    # Сохраняем метаданные
    metadata = {
        'total_tracks': len(dataset),
        'train_tracks': len(train),
        'val_tracks': len(val),
        'test_tracks': len(test),
        'yellow_threshold': yellow_threshold,
        'purple_threshold': purple_threshold,
        'zone_distribution': zone_counts.to_dict(),
        'features': list(deam_features.columns),
        'arousal_range': [float(dataset['arousal'].min()), float(dataset['arousal'].max())],
        'valence_range': [float(dataset['valence'].min()), float(dataset['valence'].max())],
        'tempo_range': [float(dataset['tempo'].min()), float(dataset['tempo'].max())],
        'random_state': random_state
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Saved to: {output_dir}")
    logger.info(f"    - train.csv ({len(train)} tracks)")
    logger.info(f"    - val.csv ({len(val)} tracks)")
    logger.info(f"    - test.csv ({len(test)} tracks)")
    logger.info(f"    - deam_complete.csv ({len(dataset)} tracks)")
    logger.info(f"    - metadata.json")

    logger.info("\n" + "=" * 80)
    logger.info("✅ DEAM dataset preparation complete!")
    logger.info("=" * 80)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review zone distribution (check for imbalance)")
    logger.info(f"  2. Adjust thresholds if needed")
    logger.info(f"  3. Train model: python scripts/train_zone_classifier.py --dataset deam")

    return dataset, train, val, test


def main():
    parser = argparse.ArgumentParser(description='Prepare DEAM dataset for training')
    parser.add_argument('--tempo-csv', type=str, default='dataset/deam_tempo.csv',
                       help='Path to extracted tempo CSV')
    parser.add_argument('--yellow-threshold', type=float, default=4.0,
                       help='Arousal threshold for Yellow zone (default: 4.0)')
    parser.add_argument('--purple-threshold', type=float, default=6.0,
                       help='Arousal threshold for Purple zone (default: 6.0)')
    parser.add_argument('--output-dir', type=str, default='dataset/deam_processed',
                       help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.1,
                       help='Test set size (default: 0.1 = 10%%)')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Validation set size (default: 0.1 = 10%%)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    try:
        prepare_deam_dataset(
            tempo_csv=args.tempo_csv,
            yellow_threshold=args.yellow_threshold,
            purple_threshold=args.purple_threshold,
            output_dir=args.output_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state
        )
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
