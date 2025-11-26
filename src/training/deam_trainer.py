"""
DEAM Dataset Trainer Extension

Расширение ZoneTrainer для работы с DEAM датасетом.
Загружает предрассчитанные фичи вместо извлечения из аудио.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple
from sklearn.preprocessing import StandardScaler

from .zone_trainer import ZoneTrainer, ZONE_LABELS
from .zone_features import ZoneFeatures
from .checkpoint_manager import CheckpointManager
from ..utils import get_logger

logger = get_logger(__name__)


class DEAMZoneTrainer(ZoneTrainer):
    """
    Тренер для DEAM датасета с предрассчитанными фичами.

    Отличия от базового ZoneTrainer:
    - Загружает готовые фичи из CSV (не извлекает из аудио)
    - Работает с arousal-valence метками
    - Использует готовые train/val/test splits
    """

    def __init__(self, deam_dir: str = "dataset/deam_processed"):
        """
        Initialize DEAM trainer.

        Args:
            deam_dir: Директория с подготовленными DEAM данными
                     (создаётся scripts/prepare_deam_dataset.py)
        """
        # Не вызываем super().__init__() так как не нужен test_data_path
        self.deam_dir = Path(deam_dir)
        self.use_gpu = False  # Фичи уже извлечены

        # Checkpoint manager для совместимости с базовым ZoneTrainer
        self.checkpoint_manager = CheckpointManager(checkpoint_dir="models/checkpoints")
        self.use_embeddings = False
        self.use_music_emotion = False
        self.use_fast_mode = False
        self._should_stop = False

        # Feature extractor не нужен (фичи уже готовы)
        self.feature_extractor = None

        # Data storage
        self.audio_paths = []
        self.zone_labels = []
        self.features_list = []

        # Training data (будут заполнены в load_deam_dataset)
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # Scaler для нормализации
        self.scaler = StandardScaler()

        # Feature names (для совместимости с базовым ZoneTrainer)
        self.feature_names = None

    def load_deam_dataset(self,
                         use_precomputed_splits: bool = True,
                         test_size: float = 0.15,
                         val_size: float = 0.15,
                         random_state: int = 42,
                         progress_callback: Optional[Callable] = None,
                         log_callback: Optional[Callable] = None) -> Tuple[int, int, int]:
        """
        Загружает DEAM датасет с предрассчитанными фичами.

        Args:
            use_precomputed_splits: Использовать готовые splits из prepare_deam_dataset.py
            test_size: Размер test set (если не использовать готовые splits)
            val_size: Размер val set
            random_state: Random seed
            progress_callback: Callback для прогресса
            log_callback: Callback для логов

        Returns:
            Tuple (train_size, val_size, test_size)
        """
        self._log(log_callback, "INFO", f"📂 Loading DEAM dataset from: {self.deam_dir}")

        if not self.deam_dir.exists():
            raise FileNotFoundError(
                f"DEAM directory not found: {self.deam_dir}\n"
                f"Please run: python scripts/prepare_deam_dataset.py"
            )

        if use_precomputed_splits:
            # Загружаем готовые splits
            self._log(log_callback, "INFO", "📊 Loading precomputed train/val/test splits...")

            train_df = pd.read_csv(self.deam_dir / "train.csv")
            val_df = pd.read_csv(self.deam_dir / "val.csv")
            test_df = pd.read_csv(self.deam_dir / "test.csv")

            self._log(log_callback, "INFO", f"  Train: {len(train_df)} tracks")
            self._log(log_callback, "INFO", f"  Val:   {len(val_df)} tracks")
            self._log(log_callback, "INFO", f"  Test:  {len(test_df)} tracks")

        else:
            # Загружаем полный датасет и делаем split
            self._log(log_callback, "INFO", "📊 Loading complete dataset and creating splits...")
            complete_df = pd.read_csv(self.deam_dir / "deam_complete.csv")

            from sklearn.model_selection import train_test_split

            train_val, test_df = train_test_split(
                complete_df, test_size=test_size, random_state=random_state,
                stratify=complete_df['zone']
            )

            val_size_adjusted = val_size / (1 - test_size)
            train_df, val_df = train_test_split(
                train_val, test_size=val_size_adjusted, random_state=random_state,
                stratify=train_val['zone']
            )

        # Извлекаем фичи и метки
        self._log(log_callback, "INFO", "🔧 Extracting features and labels...")

        # Определяем колонки фичей (все кроме служебных)
        meta_columns = ['track_id', 'audio_path', 'arousal', 'valence', 'zone',
                       'arousal_std', 'valence_std', 'success', 'error']

        # Все остальные колонки - это фичи
        all_columns = train_df.columns.tolist()
        feature_columns = [col for col in all_columns if col not in meta_columns]

        self.feature_names = feature_columns
        self._log(log_callback, "INFO", f"  Features: {len(feature_columns)}")
        self._log(log_callback, "INFO", f"  Feature list: {', '.join(feature_columns[:5])}...")

        # Извлекаем X (фичи) и y (зоны)
        X_train = train_df[feature_columns].values
        X_val = val_df[feature_columns].values
        X_test = test_df[feature_columns].values

        # Конвертируем зоны в числовые метки
        y_train = train_df['zone'].str.lower().map(ZONE_LABELS).values
        y_val = val_df['zone'].str.lower().map(ZONE_LABELS).values
        y_test = test_df['zone'].str.lower().map(ZONE_LABELS).values

        # Нормализуем фичи
        self._log(log_callback, "INFO", "📏 Normalizing features...")
        self.scaler.fit(X_train)

        self.X_train = self.scaler.transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        # Сохраняем audio paths для совместимости
        self.audio_paths = train_df['audio_path'].tolist() if 'audio_path' in train_df.columns else []
        self.zone_labels = train_df['zone'].tolist()

        # Статистика
        self._log(log_callback, "INFO", "✅ Dataset loaded successfully!")
        self._log(log_callback, "INFO", f"")
        self._log(log_callback, "INFO", f"📈 Dataset statistics:")
        self._log(log_callback, "INFO", f"  Train: {len(X_train)} samples, {X_train.shape[1]} features")
        self._log(log_callback, "INFO", f"  Val:   {len(X_val)} samples")
        self._log(log_callback, "INFO", f"  Test:  {len(X_test)} samples")

        # Распределение зон
        from collections import Counter
        train_dist = Counter(train_df['zone'])
        self._log(log_callback, "INFO", f"")
        self._log(log_callback, "INFO", f"🎯 Zone distribution (train):")
        for zone, count in sorted(train_dist.items()):
            pct = count / len(train_df) * 100
            self._log(log_callback, "INFO", f"  {zone}: {count} ({pct:.1f}%)")

        return len(X_train), len(X_val), len(X_test)

    def load_training_data(self, *args, **kwargs):
        """
        Override базового метода - для DEAM используйте load_deam_dataset()
        """
        raise NotImplementedError(
            "For DEAM dataset, use load_deam_dataset() instead of load_training_data()"
        )

    def extract_features(self, *args, **kwargs):
        """
        Override базового метода - фичи уже извлечены в DEAM
        """
        raise NotImplementedError(
            "For DEAM dataset, features are pre-extracted. Use load_deam_dataset()"
        )

    def prepare_datasets(self, *args, **kwargs):
        """
        Override базового метода - splits уже готовы в DEAM
        """
        raise NotImplementedError(
            "For DEAM dataset, splits are prepared in load_deam_dataset()"
        )

    def _log(self, callback: Optional[Callable], level: str, message: str):
        """Helper для логирования"""
        if callback:
            callback(level, message)
        else:
            log_func = getattr(logger, level.lower(), logger.info)
            log_func(message)


def main():
    """Тестовая функция"""
    import sys

    logging.basicConfig(level=logging.INFO)

    try:
        trainer = DEAMZoneTrainer()
        train_size, val_size, test_size = trainer.load_deam_dataset()

        print(f"\n✅ Successfully loaded DEAM dataset!")
        print(f"  Train: {train_size}")
        print(f"  Val:   {val_size}")
        print(f"  Test:  {test_size}")
        print(f"  Features: {trainer.X_train.shape[1]}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import logging
    sys.exit(main())
