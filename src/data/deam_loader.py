"""
DEAM Dataset Loader

Загружает предрассчитанные фичи из DEAM датасета и конвертирует их
в формат ZoneFeatures для обучения классификатора зон.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import sys

from src.training.zone_features import ZoneFeatures

logger = logging.getLogger(__name__)


class DEAMLoader:
    """Загрузчик DEAM датасета с маппингом фичей"""

    def __init__(self, dataset_root: str = "dataset"):
        """
        Args:
            dataset_root: Путь к корневой директории DEAM датасета
        """
        self.dataset_root = Path(dataset_root)
        self.audio_dir = self.dataset_root / "MEMD_audio"
        self.features_dir = self.dataset_root / "features"
        self.annotations_dir = self.dataset_root / "annotations" / "annotations averaged per song" / "song_level"
        self.metadata_dir = self.dataset_root / "metadata"

        # Проверка существования директорий
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")

    def load_annotations(self) -> pd.DataFrame:
        """
        Загружает arousal-valence аннотации из DEAM

        Returns:
            DataFrame с колонками: song_id, valence_mean, valence_std, arousal_mean, arousal_std
        """
        # Загружаем оба файла с аннотациями
        ann_files = [
            self.annotations_dir / "static_annotations_averaged_songs_1_2000.csv",
            self.annotations_dir / "static_annotations_averaged_songs_2000_2058.csv"
        ]

        dfs = []
        for ann_file in ann_files:
            if ann_file.exists():
                # skipinitialspace=True to handle spaces after commas in CSV
                df = pd.read_csv(ann_file, skipinitialspace=True)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} annotations from {ann_file.name}")
            else:
                logger.warning(f"Annotation file not found: {ann_file}")

        if not dfs:
            raise FileNotFoundError("No annotation files found")

        # Объединяем
        annotations = pd.concat(dfs, ignore_index=True)

        # Переименовываем колонки для единообразия
        annotations = annotations.rename(columns={
            'song_id': 'track_id',
            'valence_mean': 'valence',
            'valence_std': 'valence_std',
            'arousal_mean': 'arousal',
            'arousal_std': 'arousal_std'
        })

        logger.info(f"Total annotations loaded: {len(annotations)}")
        logger.info(f"Arousal range: [{annotations['arousal'].min():.2f}, {annotations['arousal'].max():.2f}]")
        logger.info(f"Valence range: [{annotations['valence'].min():.2f}, {annotations['valence'].max():.2f}]")

        return annotations

    def load_single_feature_file(self, track_id: int) -> Optional[Dict[str, float]]:
        """
        Загружает предрассчитанные фичи для одного трека из DEAM
        и конвертирует их в базовые 15 фичей (без tempo)

        Args:
            track_id: ID трека в DEAM

        Returns:
            Словарь с фичами или None если файл не найден
        """
        feature_file = self.features_dir / f"{track_id}.csv"

        if not feature_file.exists():
            logger.warning(f"Feature file not found: {feature_file}")
            return None

        try:
            # Загружаем CSV с semicolon delimiter
            df = pd.read_csv(feature_file, sep=';')

            # Извлекаем фичи путём усреднения по времени
            features = {}

            # === EXACT MATCHES ===

            # 1. Zero Crossing Rate
            if 'pcm_zcr_sma_amean' in df.columns:
                features['zero_crossing_rate'] = float(df['pcm_zcr_sma_amean'].mean())

            # 2. RMS Energy
            if 'pcm_RMSenergy_sma_amean' in df.columns:
                features['rms_energy'] = float(df['pcm_RMSenergy_sma_amean'].mean())

            # 3. Spectral Centroid
            if 'pcm_fftMag_spectralCentroid_sma_amean' in df.columns:
                features['spectral_centroid'] = float(df['pcm_fftMag_spectralCentroid_sma_amean'].mean())

            # 4. Spectral Rolloff (90% вместо 85%, но близко)
            if 'pcm_fftMag_spectralRollOff90.0_sma_amean' in df.columns:
                features['spectral_rolloff'] = float(df['pcm_fftMag_spectralRollOff90.0_sma_amean'].mean())

            # 5. Energy Variance (используем stddev как proxy)
            if 'pcm_RMSenergy_sma_stddev' in df.columns:
                features['energy_variance'] = float(df['pcm_RMSenergy_sma_stddev'].mean())

            # 6-15. MFCCs (coefficients 1-5, mean and std)
            for i in range(1, 6):
                mean_col = f'pcm_fftMag_mfcc_sma[{i}]_amean'
                std_col = f'pcm_fftMag_mfcc_sma[{i}]_stddev'

                if mean_col in df.columns:
                    features[f'mfcc_{i}_mean'] = float(df[mean_col].mean())

                if std_col in df.columns:
                    features[f'mfcc_{i}_std'] = float(df[std_col].mean())

            # === CALCULATED FEATURES ===

            # 16. Low Energy (процент кадров ниже среднего RMS)
            if 'pcm_RMSenergy_sma_amean' in df.columns:
                rms_values = df['pcm_RMSenergy_sma_amean'].values
                mean_rms = rms_values.mean()
                features['low_energy'] = float((rms_values < mean_rms).sum() / len(rms_values))

            # 17. Brightness (высокочастотная энергия >3kHz)
            # audSpec_Rfilt_sma имеет 26 band, band 20-25 примерно соответствуют высоким частотам
            high_bands = [f'audSpec_Rfilt_sma[{i}]_amean' for i in range(20, 26)]
            all_bands = [f'audSpec_Rfilt_sma[{i}]_amean' for i in range(0, 26)]

            if all(col in df.columns for col in high_bands + all_bands):
                high_energy = df[high_bands].sum(axis=1)
                total_energy = df[all_bands].sum(axis=1)
                brightness_values = high_energy / (total_energy + 1e-6)
                features['brightness'] = float(brightness_values.mean())

            # 18. Drop Intensity (90-й процентиль производной энергии)
            if 'pcm_RMSenergy_sma_de_amean' in df.columns:
                energy_derivative = df['pcm_RMSenergy_sma_de_amean'].values
                features['drop_intensity'] = float(np.percentile(np.abs(energy_derivative), 90))

            # Tempo НЕ включаем - он будет извлечён отдельно
            # features['tempo'] = None

            return features

        except Exception as e:
            logger.error(f"Error loading features for track {track_id}: {e}")
            return None

    def load_all_features(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Загружает фичи для всех треков DEAM

        Args:
            limit: Максимальное количество треков для загрузки (для тестирования)

        Returns:
            DataFrame с колонками: track_id, feature_1, feature_2, ..., feature_15
        """
        # Получаем список всех feature files
        feature_files = sorted(self.features_dir.glob("*.csv"))

        if limit:
            feature_files = feature_files[:limit]

        logger.info(f"Loading features from {len(feature_files)} tracks...")

        records = []
        failed = 0

        for feature_file in feature_files:
            track_id = int(feature_file.stem)  # filename без расширения
            features = self.load_single_feature_file(track_id)

            if features:
                features['track_id'] = track_id
                features['audio_path'] = str(self.audio_dir / f"{track_id}.mp3")
                records.append(features)
            else:
                failed += 1

        logger.info(f"Successfully loaded {len(records)} tracks, failed: {failed}")

        df = pd.DataFrame(records)
        return df

    def load_metadata(self) -> pd.DataFrame:
        """
        Загружает метаданные треков (artist, album, genre, etc.)

        Returns:
            DataFrame с метаданными
        """
        metadata_files = [
            self.metadata_dir / "metadata_2013.csv",
            self.metadata_dir / "metadata_2014.csv",
            self.metadata_dir / "metadata_2015.csv"
        ]

        dfs = []
        for meta_file in metadata_files:
            if meta_file.exists():
                df = pd.read_csv(meta_file)
                dfs.append(df)

        if not dfs:
            raise FileNotFoundError("No metadata files found")

        metadata = pd.concat(dfs, ignore_index=True)

        # Переименовываем для единообразия
        if 'Id' in metadata.columns:
            metadata = metadata.rename(columns={'Id': 'track_id'})

        logger.info(f"Loaded metadata for {len(metadata)} tracks")

        return metadata

    def convert_arousal_to_zone(self, arousal: float,
                                yellow_threshold: float = 4.0,
                                purple_threshold: float = 6.0) -> str:
        """
        Конвертирует arousal значение в зону (Yellow/Green/Purple)

        Args:
            arousal: значение arousal (обычно 1-9 шкала)
            yellow_threshold: порог для Yellow зоны (arousal < threshold)
            purple_threshold: порог для Purple зоны (arousal > threshold)

        Returns:
            "YELLOW", "GREEN", или "PURPLE"
        """
        if arousal < yellow_threshold:
            return "YELLOW"
        elif arousal > purple_threshold:
            return "PURPLE"
        else:
            return "GREEN"

    def load_complete_dataset(self,
                             yellow_threshold: float = 4.0,
                             purple_threshold: float = 6.0,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """
        Загружает полный датасет: фичи + аннотации + зоны

        Args:
            yellow_threshold: Порог arousal для Yellow зоны
            purple_threshold: Порог arousal для Purple зоны
            limit: Ограничение количества треков (для тестирования)

        Returns:
            DataFrame с колонками: track_id, audio_path, features..., arousal, valence, zone
        """
        logger.info("Loading DEAM complete dataset...")

        # 1. Загружаем фичи
        features_df = self.load_all_features(limit=limit)
        logger.info(f"Loaded features for {len(features_df)} tracks")

        # 2. Загружаем аннотации
        annotations_df = self.load_annotations()

        # 3. Объединяем по track_id
        dataset = features_df.merge(annotations_df, on='track_id', how='inner')
        logger.info(f"Merged dataset size: {len(dataset)} tracks")

        # 4. Конвертируем arousal → zone
        dataset['zone'] = dataset['arousal'].apply(
            lambda a: self.convert_arousal_to_zone(a, yellow_threshold, purple_threshold)
        )

        # Статистика зон
        zone_counts = dataset['zone'].value_counts()
        logger.info(f"Zone distribution:")
        for zone, count in zone_counts.items():
            pct = count / len(dataset) * 100
            logger.info(f"  {zone}: {count} ({pct:.1f}%)")

        return dataset


def main():
    """Тестовая функция для проверки загрузчика"""
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        loader = DEAMLoader()

        # Тест 1: Загрузка аннотаций
        print("\n=== Test 1: Loading annotations ===")
        annotations = loader.load_annotations()
        print(f"Loaded {len(annotations)} annotations")
        print(annotations.head())

        # Тест 2: Загрузка одного трека
        print("\n=== Test 2: Loading single track features ===")
        track_id = 2  # первый трек
        features = loader.load_single_feature_file(track_id)
        if features:
            print(f"Loaded {len(features)} features for track {track_id}")
            for key, value in features.items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        # Тест 3: Загрузка всех фичей (только первые 10 для теста)
        print("\n=== Test 3: Loading all features (limit=10) ===")
        all_features = loader.load_all_features(limit=10)
        print(f"Loaded features DataFrame shape: {all_features.shape}")
        print(f"Columns: {list(all_features.columns)}")

        # Тест 4: Полный датасет (первые 10 треков)
        print("\n=== Test 4: Complete dataset (limit=10) ===")
        dataset = loader.load_complete_dataset(limit=10)
        print(f"Complete dataset shape: {dataset.shape}")
        print(f"\nSample rows:")
        print(dataset[['track_id', 'arousal', 'valence', 'zone', 'rms_energy', 'spectral_centroid']].head())

        print("\n✅ All tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
