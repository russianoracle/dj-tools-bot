#!/usr/bin/env python3
"""
Production Training Pipeline - Best Practices Consolidated

This module consolidates all optimization findings from experiments:
- XGBoost classifier (best performer: 65.48% CV accuracy)
- Top-50 feature selection
- Normalized drop features
- Source-aware normalization

Usage:
    from src.training.production_pipeline import ProductionPipeline

    pipeline = ProductionPipeline()
    pipeline.train(user_frames_path, deam_frames_path, annotations_path)
    pipeline.save('models/production/zone_classifier.pkl')
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Конфигурация пайплайна обучения с лучшими практиками.

    Значения по умолчанию - результат экспериментов:
    - XGBoost показал лучшую точность (65.48% CV vs 63% RandomForest)
    - Топ-50 фичей - оптимальный баланс сигнала и шума
    - Нормализованные drop-фичи решают проблему YELLOW-парадокса
    - Балансировка классов улучшает точность на несбалансированных данных
    """

    # ─────────────────────────────────────────────────────────────
    # АЛГОРИТМ
    # Тестировались: RandomForest, GradientBoosting, XGBoost
    # XGBoost показал лучший результат (+3% к accuracy)
    # ─────────────────────────────────────────────────────────────
    algorithm: str = 'xgboost'
    #algorithm: str = 'GradientBoosting'
    #algorithm: str = 'RandomForest'

    # ─────────────────────────────────────────────────────────────
    # ОТБОР ФИЧЕЙ
    # Лучший результат: 100 фичей -> 65.27% CV accuracy
    # ─────────────────────────────────────────────────────────────
    top_n_features: int = 100

    # ─────────────────────────────────────────────────────────────
    # ГИПЕРПАРАМЕТРЫ XGBOOST
    # Подобраны через grid search на DEAM + user данных
    # ─────────────────────────────────────────────────────────────
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 200,      # Количество деревьев
        'max_depth': 5,           # Глубина дерева (защита от переобучения)
        'learning_rate': 0.1,     # Скорость обучения
        'min_child_weight': 3,    # Мин. вес листа (регуляризация)
        'subsample': 0.8,         # Доля сэмплов для каждого дерева
        'colsample_bytree': 0.8,  # Доля фичей для каждого дерева
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    })

    # ─────────────────────────────────────────────────────────────
    # КРОСС-ВАЛИДАЦИЯ
    # 5 фолдов - стандарт для средних датасетов
    # ─────────────────────────────────────────────────────────────
    cv_folds: int = 5

    # ─────────────────────────────────────────────────────────────
    # НОРМАЛИЗАЦИЯ ИСТОЧНИКОВ
    # DEAM и user-аудио имеют разные распределения фичей
    # Z-score нормализация DEAM -> user распределение
    # ─────────────────────────────────────────────────────────────
    normalize_sources: bool = True

    # ─────────────────────────────────────────────────────────────
    # НОРМАЛИЗОВАННЫЕ DROP-ФИЧИ
    # Решают YELLOW-парадокс: тихие треки показывали высокий
    # buildup_score просто потому что любое изменение энергии
    # на фоне низкого baseline выглядит драматично.
    # Нормализуем на уровень энергии трека.
    # ─────────────────────────────────────────────────────────────
    add_drop_features: bool = True

    # ─────────────────────────────────────────────────────────────
    # БАЛАНСИРОВКА КЛАССОВ
    # DEAM имеет дисбаланс: PURPLE только ~18% треков
    # Веса обратно пропорциональны частоте класса
    # weight[class] = total / (n_classes * count[class])
    # ─────────────────────────────────────────────────────────────
    class_balance: bool = True

    # ─────────────────────────────────────────────────────────────
    # ПОРОГИ AROUSAL ДЛЯ DEAM
    # Маппинг arousal_mean из DEAM в зоны:
    #   YELLOW: arousal < yellow_threshold (низкая энергия)
    #   PURPLE: arousal > purple_threshold (высокая энергия)
    #   GREEN:  остальное (переходная зона)
    # ─────────────────────────────────────────────────────────────
    yellow_threshold: float = 4.0
    purple_threshold: float = 5.7


class ProductionPipeline:
    """
    Production-ready training pipeline consolidating best practices.

    Key optimizations included:
    1. XGBoost classifier (65.48% CV vs 63% RandomForest)
    2. Top-50 feature selection
    3. Normalized drop features (fixes YELLOW > PURPLE paradox)
    4. Source-aware normalization
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.selected_features: List[str] = []
        self.all_features: List[str] = []
        self.training_stats: Dict = {}

    def train(
        self,
        user_frames_path: Path,
        deam_frames_path: Path,
        annotations_path: Optional[Path] = None
    ) -> Dict:
        """
        Train the production model.

        Args:
            user_frames_path: Path to user frame features (.pkl)
            deam_frames_path: Path to DEAM frame features (.pkl)
            annotations_path: Path to DEAM annotations CSV (optional)

        Returns:
            Dictionary with training results
        """
        logger.info("=" * 60)
        logger.info("PRODUCTION PIPELINE - Training")
        logger.info("=" * 60)
        logger.info(f"Config: {self.config}")

        # 1. Load and merge data
        df = self._load_and_merge(user_frames_path, deam_frames_path, annotations_path)

        # 2. Aggregate frames to track level
        df_agg = self._aggregate_frames(df)

        # 3. Add normalized drop features
        if self.config.add_drop_features:
            df_agg = self._add_normalized_drop_features(df_agg)

        # 4. Normalize sources (DEAM to user distribution)
        if self.config.normalize_sources:
            df_agg = self._normalize_sources(df_agg)

        # 5. Prepare features
        X, y, feature_names = self._prepare_features(df_agg)
        self.all_features = feature_names

        # 6. Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 7. Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # 8. Feature selection (train preliminary model for importance)
        X_selected, selected_features = self._select_top_features(
            X_scaled, y_encoded, feature_names, self.config.top_n_features
        )
        self.selected_features = selected_features

        # 9. Train final model
        results = self._train_model(X_selected, y_encoded)

        # 10. Store training stats
        self.training_stats = {
            'cv_accuracy': results['cv_mean'],
            'cv_std': results['cv_std'],
            'n_features': len(self.selected_features),
            'n_samples': len(y),
            'class_distribution': dict(pd.Series(y).value_counts()),
            'config': {
                'algorithm': self.config.algorithm,
                'top_n_features': self.config.top_n_features,
                'normalize_sources': self.config.normalize_sources,
                'add_drop_features': self.config.add_drop_features,
                'class_balance': self.config.class_balance
            }
        }

        self._print_results(results)
        return results

    def _load_and_merge(
        self,
        user_path: Path,
        deam_path: Path,
        annotations_path: Optional[Path]
    ) -> pd.DataFrame:
        """Load and merge user and DEAM datasets."""
        logger.info(f"Loading user frames: {user_path}")
        user_df = pd.read_pickle(user_path)
        logger.info(f"  {len(user_df)} frames, {user_df['track_id'].nunique()} tracks")

        logger.info(f"Loading DEAM frames: {deam_path}")
        deam_df = pd.read_pickle(deam_path)
        logger.info(f"  {len(deam_df)} frames, {deam_df['track_id'].nunique()} tracks")

        # Add zones to DEAM if needed
        if 'zone' not in deam_df.columns or deam_df['zone'].isna().all():
            if annotations_path and annotations_path.exists():
                logger.info(f"Loading annotations: {annotations_path}")
                annotations = pd.read_csv(annotations_path)
                annotations.columns = [c.strip() for c in annotations.columns]

                # Используем пороги из конфига
                yellow_th = self.config.yellow_threshold
                purple_th = self.config.purple_threshold
                logger.info(f"Arousal thresholds: YELLOW < {yellow_th}, PURPLE > {purple_th}")

                def arousal_to_zone(arousal):
                    if pd.isna(arousal):
                        return 'GREEN'
                    if arousal < yellow_th:
                        return 'YELLOW'
                    elif arousal > purple_th:
                        return 'PURPLE'
                    return 'GREEN'

                track_zones = {}
                for _, row in annotations.iterrows():
                    track_id = int(row['song_id'])
                    arousal = row.get('arousal_mean', 5.0)
                    track_zones[track_id] = arousal_to_zone(arousal)

                deam_df['zone'] = deam_df['track_id'].map(track_zones)

        # Combine
        combined = pd.concat([user_df, deam_df], ignore_index=True)

        # Filter valid zones
        valid_zones = ['GREEN', 'PURPLE', 'YELLOW']
        combined = combined[combined['zone'].isin(valid_zones)]

        logger.info(f"Combined: {len(combined)} frames, {combined['track_id'].nunique()} tracks")
        return combined

    def _aggregate_frames(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate frame features to track level."""
        logger.info("Aggregating frames per track...")

        # Feature columns (exclude metadata)
        meta_cols = ['track_id', 'source', 'zone', 'path', 'frameTime']
        feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in ['float64', 'float32', 'int64']]

        grouped = df.groupby(['track_id', 'source', 'zone'])
        records = []

        for (track_id, source, zone), group in grouped:
            record = {
                'track_id': track_id,
                'source': source,
                'zone': zone,
                'n_frames': len(group)
            }

            for feat in feature_cols:
                if feat not in group.columns:
                    continue

                values = group[feat].dropna().values
                if len(values) == 0:
                    continue

                # Statistics
                record[f'{feat}_mean'] = np.mean(values)
                record[f'{feat}_std'] = np.std(values)
                record[f'{feat}_min'] = np.min(values)
                record[f'{feat}_max'] = np.max(values)
                record[f'{feat}_p10'] = np.percentile(values, 10)
                record[f'{feat}_p50'] = np.percentile(values, 50)
                record[f'{feat}_p90'] = np.percentile(values, 90)
                record[f'{feat}_range'] = np.max(values) - np.min(values)

                # Binary features: sum and rate
                if feat in ['drop_candidate', 'energy_peak', 'energy_valley', 'beat_sync', 'low_energy_flag']:
                    record[f'{feat}_sum'] = np.sum(values)
                    record[f'{feat}_rate'] = np.mean(values)

            records.append(record)

        agg_df = pd.DataFrame(records)
        logger.info(f"Aggregated to {len(agg_df)} tracks")
        return agg_df

    def _add_normalized_drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add normalized drop features.

        Key insight from experiments: absolute drop features are misleading
        because quiet tracks (YELLOW) show dramatic relative changes.
        Normalized features fix this.
        """
        logger.info("Adding normalized drop features...")
        df = df.copy()
        n_added = 0

        # 1. Normalized buildup max
        if 'energy_buildup_score_max' in df.columns and 'rms_energy_mean' in df.columns:
            df['normalized_buildup_max'] = df['energy_buildup_score_max'] / (df['rms_energy_mean'] + 1e-6)
            n_added += 1

        # 2. Normalized buildup mean
        if 'energy_buildup_score_mean' in df.columns and 'rms_energy_mean' in df.columns:
            df['normalized_buildup_mean'] = df['energy_buildup_score_mean'] / (df['rms_energy_mean'] + 1e-6)
            n_added += 1

        # 3. Relative drop intensity
        if all(c in df.columns for c in ['energy_buildup_score_max', 'energy_buildup_score_mean', 'rms_energy_std']):
            drop_intensity = df['energy_buildup_score_max'] - df['energy_buildup_score_mean']
            df['relative_drop_intensity'] = drop_intensity / (df['rms_energy_std'] + 1e-6)
            n_added += 1

        # 4. Drop prominence
        if all(c in df.columns for c in ['energy_buildup_score_p90', 'energy_buildup_score_p10', 'rms_energy_range']):
            buildup_range = df['energy_buildup_score_p90'] - df['energy_buildup_score_p10']
            df['drop_prominence'] = buildup_range / (df['rms_energy_range'] + 1e-6)
            n_added += 1

        # 5. Energy dynamics score
        if all(c in df.columns for c in ['drop_candidate_rate', 'rms_energy_std', 'energy_buildup_score_std']):
            df['energy_dynamics_score'] = (
                df['drop_candidate_rate'] * 10 +
                df['rms_energy_std'] * 5 +
                df['energy_buildup_score_std']
            )
            n_added += 1

        # 6. Drop frequency
        if 'drop_candidate_sum' in df.columns and 'n_frames' in df.columns:
            df['drop_frequency'] = df['drop_candidate_sum'] / df['n_frames']
            n_added += 1

        # ═══════════════════════════════════════════════════════════════
        # NEW: Dynamic features based on clustering analysis
        # These features capture temporal dynamics that distinguish zones
        # ═══════════════════════════════════════════════════════════════

        # 7. Energy delta ratio (how much energy changes relative to mean)
        if 'rms_energy_delta_std' in df.columns and 'rms_energy_mean' in df.columns:
            df['energy_delta_ratio'] = df['rms_energy_delta_std'] / (df['rms_energy_mean'] + 1e-6)
            n_added += 1

        # 8. Energy acceleration (second derivative magnitude)
        if 'rms_energy_delta2_mean' in df.columns:
            df['energy_acceleration'] = df['rms_energy_delta2_mean'].abs()
            n_added += 1

        # 9. Energy volatility (std of delta / mean of delta)
        if 'rms_energy_delta_std' in df.columns and 'rms_energy_delta_mean' in df.columns:
            df['energy_volatility'] = df['rms_energy_delta_std'] / (df['rms_energy_delta_mean'].abs() + 1e-6)
            n_added += 1

        # 10. MFCC dynamics score (combination of MFCC deltas)
        mfcc_delta_cols = [f'mfcc_{i}_delta_std' for i in range(1, 6) if f'mfcc_{i}_delta_std' in df.columns]
        if len(mfcc_delta_cols) >= 3:
            df['mfcc_dynamics_score'] = df[mfcc_delta_cols].mean(axis=1)
            n_added += 1

        # 11. Brightness dynamics (brightness delta relative to mean)
        if 'brightness_delta_std' in df.columns and 'brightness_mean' in df.columns:
            df['brightness_dynamics'] = df['brightness_delta_std'] / (df['brightness_mean'] + 1e-6)
            n_added += 1

        # 12. Spectral flux ratio (flux std / flux mean - spectral change consistency)
        if 'spectral_flux_std' in df.columns and 'spectral_flux_mean' in df.columns:
            df['spectral_flux_ratio'] = df['spectral_flux_std'] / (df['spectral_flux_mean'] + 1e-6)
            n_added += 1

        # 13. Energy peak density (peaks per frame)
        if 'energy_peak_sum' in df.columns and 'n_frames' in df.columns:
            df['energy_peak_density'] = df['energy_peak_sum'] / df['n_frames']
            n_added += 1

        # 14. Energy valley density (valleys per frame)
        if 'energy_valley_sum' in df.columns and 'n_frames' in df.columns:
            df['energy_valley_density'] = df['energy_valley_sum'] / df['n_frames']
            n_added += 1

        # 15. Peak-valley ratio (more peaks = more energetic structure)
        if 'energy_peak_sum' in df.columns and 'energy_valley_sum' in df.columns:
            df['peak_valley_ratio'] = df['energy_peak_sum'] / (df['energy_valley_sum'] + 1)
            n_added += 1

        # 16. Onset dynamics (onset strength variation)
        if 'onset_strength_std' in df.columns and 'onset_strength_mean' in df.columns:
            df['onset_dynamics'] = df['onset_strength_std'] / (df['onset_strength_mean'] + 1e-6)
            n_added += 1

        # 17. Combined dynamics score (weighted sum of key dynamic features)
        dynamic_components = []
        if 'rms_energy_std' in df.columns:
            dynamic_components.append(df['rms_energy_std'] * 2)
        if 'brightness_delta_std' in df.columns:
            dynamic_components.append(df['brightness_delta_std'])
        if 'spectral_flux_std' in df.columns:
            dynamic_components.append(df['spectral_flux_std'])
        if len(dynamic_components) >= 2:
            df['combined_dynamics'] = sum(dynamic_components) / len(dynamic_components)
            n_added += 1

        logger.info(f"Added {n_added} normalized/dynamic features")
        return df

    def _normalize_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DEAM features to user distribution."""
        logger.info("Normalizing sources (DEAM -> user distribution)...")

        meta_cols = ['track_id', 'source', 'zone', 'n_frames']
        feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in ['float64', 'float32', 'int64']]

        user_df = df[df['source'] == 'user']
        deam_df = df[df['source'] == 'deam'].copy()

        if len(user_df) == 0 or len(deam_df) == 0:
            logger.warning("Cannot normalize - missing user or DEAM data")
            return df

        for col in feature_cols:
            if col not in user_df.columns or col not in deam_df.columns:
                continue

            user_mean = user_df[col].mean()
            user_std = user_df[col].std()
            deam_mean = deam_df[col].mean()
            deam_std = deam_df[col].std()

            if deam_std > 0 and user_std > 0:
                deam_df[col] = (deam_df[col] - deam_mean) / deam_std * user_std + user_mean

        result = pd.concat([user_df, deam_df], ignore_index=True)
        logger.info(f"Normalized {len(feature_cols)} features")
        return result

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix."""
        meta_cols = ['track_id', 'source', 'zone', 'n_frames', 'path']
        feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in ['float64', 'float32', 'int64']]

        X = df[feature_cols].values
        y = df['zone'].values

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"Feature matrix: {X.shape}")
        return X, y, feature_cols

    def _select_top_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        top_n: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Select top N features by importance."""
        logger.info(f"Selecting top {top_n} features...")

        # Train preliminary model for feature importance
        try:
            from xgboost import XGBClassifier
            temp_model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            logger.warning("XGBoost not available, using RandomForest for feature selection")
            temp_model = RandomForestClassifier(n_estimators=100, random_state=42)

        temp_model.fit(X, y)

        # Get feature importance
        importance = temp_model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]

        selected_features = [feature_names[i] for i in indices]
        X_selected = X[:, indices]

        # Store feature selector indices
        self.feature_selector = indices

        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Top 5: {selected_features[:5]}")

        return X_selected, selected_features

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute sample weights for class balancing.

        Weights are inversely proportional to class frequency:
        weight[class] = total_samples / (n_classes * class_count)

        This gives higher weight to underrepresented classes (like PURPLE in DEAM).
        """
        from collections import Counter

        class_counts = Counter(y)
        total = len(y)
        n_classes = len(class_counts)

        # Compute weight for each class
        class_weights = {cls: total / (n_classes * count) for cls, count in class_counts.items()}

        # Create sample weight array
        sample_weights = np.array([class_weights[label] for label in y])

        logger.info(f"Class weights: {class_weights}")
        return sample_weights

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the final model with optional class balancing."""
        logger.info(f"Training {self.config.algorithm} model...")

        # Compute sample weights if class balancing is enabled
        sample_weights = None
        if self.config.class_balance:
            sample_weights = self._compute_sample_weights(y)
            logger.info("Class balancing enabled")

        if self.config.algorithm == 'xgboost':
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(**self.config.xgb_params)
            except ImportError:
                logger.warning("XGBoost not available, falling back to GradientBoosting")
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

        # Cross-validation with sample weights
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)

        if sample_weights is not None:
            # Manual CV with sample weights
            from sklearn.metrics import accuracy_score
            cv_scores = []
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                w_train = sample_weights[train_idx]

                self.model.fit(X_train, y_train, sample_weight=w_train)
                y_pred_fold = self.model.predict(X_test)
                cv_scores.append(accuracy_score(y_test, y_pred_fold))
            cv_scores = np.array(cv_scores)
        else:
            cv_scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy')

        logger.info(f"CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

        # Fit on full data
        if sample_weights is not None:
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            self.model.fit(X, y)

        # Get predictions for report
        y_pred = self.model.predict(X)

        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y, y_pred, target_names=self.label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classes': self.label_encoder.classes_
        }

    def _print_results(self, results: Dict):
        """Print training results."""
        print("\n" + "=" * 60)
        print("PRODUCTION PIPELINE - Training Results")
        print("=" * 60)
        print(f"\nAlgorithm: {self.config.algorithm.upper()}")
        print(f"Features: {len(self.selected_features)} (top-{self.config.top_n_features})")
        print(f"CV Accuracy: {results['cv_mean']:.2%} (+/- {results['cv_std']:.2%})")
        print("\nTop 10 features:")
        for i, feat in enumerate(self.selected_features[:10], 1):
            print(f"  {i:2d}. {feat}")
        print("\nClassification Report:")
        print(results['classification_report'])
        print("=" * 60)

    def save(self, path: Path):
        """Save the trained pipeline."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'all_features': self.all_features,
            'training_stats': self.training_stats,
            'config': {
                'algorithm': self.config.algorithm,
                'top_n_features': self.config.top_n_features,
                'normalize_sources': self.config.normalize_sources,
                'add_drop_features': self.config.add_drop_features,
                'xgb_params': self.config.xgb_params
            },
            'version': '1.0'
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved pipeline to: {path}")
        print(f"\nModel saved: {path}")

    @classmethod
    def load(cls, path: Path) -> 'ProductionPipeline':
        """Load a trained pipeline."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        config = PipelineConfig(
            algorithm=data['config']['algorithm'],
            top_n_features=data['config']['top_n_features'],
            normalize_sources=data['config']['normalize_sources'],
            add_drop_features=data['config']['add_drop_features']
        )

        pipeline = cls(config)
        pipeline.model = data['model']
        pipeline.scaler = data['scaler']
        pipeline.label_encoder = data['label_encoder']
        pipeline.feature_selector = data['feature_selector']
        pipeline.selected_features = data['selected_features']
        pipeline.all_features = data['all_features']
        pipeline.training_stats = data.get('training_stats', {})

        return pipeline

    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Predict zone for a single track.

        Args:
            features: Dictionary of feature_name -> value

        Returns:
            Tuple of (zone_name, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # Build feature vector from ALL features (scaler expects all_features)
        X_all = np.array([features.get(f, 0.0) for f in self.all_features]).reshape(1, -1)

        # Scale ALL features first
        X_scaled = self.scaler.transform(X_all)

        # Then select top features using feature_selector indices
        X_selected = X_scaled[:, self.feature_selector]

        # Predict
        pred = self.model.predict(X_selected)[0]
        proba = self.model.predict_proba(X_selected)[0]

        zone = self.label_encoder.inverse_transform([pred])[0]
        confidence = float(np.max(proba))

        return zone, confidence


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Production Training Pipeline')
    parser.add_argument('--user', '-u', type=Path, required=True, help='User frames pickle')
    parser.add_argument('--deam', '-d', type=Path, required=True, help='DEAM frames pickle')
    parser.add_argument('--annotations', '-a', type=Path, help='DEAM annotations CSV')
    parser.add_argument('--output', '-o', type=Path, default=Path('models/production/zone_classifier.pkl'))
    parser.add_argument('--algorithm', choices=['xgboost', 'randomforest'], default='xgboost')
    parser.add_argument('--top-n', type=int, default=100, help='Top N features to select')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = PipelineConfig(
        algorithm=args.algorithm,
        top_n_features=args.top_n
    )

    pipeline = ProductionPipeline(config)
    pipeline.train(args.user, args.deam, args.annotations)
    pipeline.save(args.output)


if __name__ == '__main__':
    main()