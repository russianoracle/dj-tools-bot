"""
Training Pipeline - Universal ML model training following project architecture.

Architecture (Primitives -> Tasks -> Pipelines):
- Uses AudioLoader for audio loading (from src/audio/loader.py)
- Uses CacheRepository for feature caching (from src/core/cache/)
- Uses Task classes for feature extraction (e.g., DropDetectorML)
- Provides abstract base classes for extensibility

Single Responsibility Principle:
- BaseGroundTruthLoader - abstract base for ground truth loading
- DropGroundTruthLoader - concrete loader for drop detection GT (JSON)
- FeatureDataset - manage features with caching via CacheRepository
- LabelAssigner - vectorized label assignment (nearest neighbor matching)
- BaseTrainer - abstract base for model training
- XGBoostTrainer - concrete XGBoost binary classifier
- Evaluator - model evaluation with vectorized NMS
- nms_vectorized - Non-Maximum Suppression without loops
- TrainingPipeline - orchestrates the entire training workflow

Key Design Decisions:
- ALL operations are vectorized (numpy broadcasting, no Python loops)
- Uses existing project components (AudioLoader, CacheRepository) - no duplication
- Abstract base classes allow extending for new GT formats and model types
- Nearest neighbor labeling ensures exactly 1 sample per GT label (avoids duplicates)
- Post-NMS evaluation gives true precision/recall for detection tasks

Usage:
    from src.training.pipelines.training import TrainingPipeline, DropGroundTruthLoader, XGBoostTrainer
    from src.core.tasks.drop_detector_ml import DropDetectorML

    # Create feature extractor using existing Task
    detector = DropDetectorML(auto_load_model=False)

    def my_extractor(y, sr, metadata):
        bpm = metadata.get('bpm', 132)
        interval = (60.0 / bpm) * 4  # sample every bar
        times = np.arange(interval, len(y)/sr - interval, interval)
        return detector._extract_features_at_boundaries(y, sr, times, tempo=bpm)

    # Build and run pipeline
    pipeline = TrainingPipeline(
        gt_loader=DropGroundTruthLoader('data/ground_truth_drops.json'),
        feature_extractor=my_extractor,
        trainer=XGBoostTrainer(feature_cols=['rms_change', 'bass_change', ...]),
        cache_dir='cache',
        label_tolerance=2.0,
    )
    result = pipeline.run(output_path='models/model.pkl', project_root=Path.cwd())
    print(f"F1={result.metrics['f1']:.3f}, CV_F1={result.metrics['cv_f1']:.3f}")

Extending for a new model type:
    from src.training.pipelines.training import BaseGroundTruthLoader, BaseTrainer, TrainingResult

    class MyGroundTruthLoader(BaseGroundTruthLoader):
        def load(self) -> Dict[str, Dict]:
            # Return: {name: {'file': path, 'labels': [times], 'metadata': {...}}}
            pass

    class MyTrainer(BaseTrainer):
        def train(self, df, label_col='is_positive') -> TrainingResult:
            # Train model, return TrainingResult with model, metrics, feature_cols
            pass
        def save(self, result, path):
            # Save model to disk
            pass

    pipeline = TrainingPipeline(
        gt_loader=MyGroundTruthLoader(...),
        feature_extractor=my_feature_fn,
        trainer=MyTrainer(...),
    )
"""

import json
import logging
import numpy as np
import pandas as pd
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from src.core.primitives import STFTCache
from src.core.cache import CacheRepository
from src.audio.loader import AudioLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Ground Truth Loaders
# =============================================================================

class BaseGroundTruthLoader(ABC):
    """Abstract base for ground truth loading."""

    @abstractmethod
    def load(self) -> Dict[str, Dict]:
        """
        Load ground truth data.

        Returns:
            Dict mapping set_name -> {
                'file': str,
                'labels': List[float],  # e.g., drop times
                'metadata': Dict,       # e.g., bpm, music_start
            }
        """
        pass


class DropGroundTruthLoader(BaseGroundTruthLoader):
    """Load drop detection ground truth from JSON."""

    def __init__(self, gt_path: str):
        self.gt_path = gt_path

    def load(self) -> Dict[str, Dict]:
        with open(self.gt_path) as f:
            gt = json.load(f)

        result = {}
        for name, data in gt.items():
            # Extract all drop times (all labels are drops)
            drop_times = []
            for d in data.get('drops', []):
                t = d['time'] if isinstance(d, dict) else d
                drop_times.append(t)
            for t in data.get('memory_cues', []):
                if t not in drop_times:
                    drop_times.append(t)

            result[name] = {
                'file': data.get('file'),
                'labels': sorted(drop_times),
                'metadata': {
                    'music_start': data.get('music_start', 0),
                    'bpm': data.get('bpm', 132),
                },
            }
        return result


# =============================================================================
# Feature Dataset (with caching)
# =============================================================================

@dataclass
class FeatureDataset:
    """
    Manage feature extraction with caching.

    Uses CacheRepository from project architecture.
    """
    cache: CacheRepository
    loader: AudioLoader

    def extract(
        self,
        audio_path: str,
        extractor_fn: Callable[[np.ndarray, int, Dict], List[dict]],
        metadata: Dict,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Extract features with caching.

        Args:
            audio_path: Path to audio file
            extractor_fn: Function(y, sr, metadata) -> List[dict]
            metadata: Metadata dict (bpm, music_start, etc.)
            use_cache: Whether to use cache

        Returns:
            DataFrame with features
        """
        file_hash = self.cache.compute_file_hash(audio_path)

        # Try cache
        if use_cache:
            cached = self.cache.get_features(file_hash)
            if cached is not None:
                logger.info(f"Cache hit: {file_hash[:8]}")
                return pd.DataFrame(cached)

        # Load audio via AudioLoader
        y, sr = self.loader.load(audio_path)
        logger.info(f"Loaded: {len(y)/sr/60:.1f} min")

        # Extract features
        features = extractor_fn(y, sr, metadata)
        if not features:
            return pd.DataFrame()

        # Cache
        self.cache.save_features(file_hash, features)

        return pd.DataFrame(features)


# =============================================================================
# Label Assigner (vectorized)
# =============================================================================

class LabelAssigner:
    """
    Assign labels using vectorized nearest neighbor matching.

    For each ground truth label, finds nearest sample and marks it.
    Ensures exactly one sample per GT label.
    """

    def __init__(self, tolerance: float = 2.0, label_col: str = 'is_positive'):
        self.tolerance = tolerance
        self.label_col = label_col

    def assign(self, df: pd.DataFrame, labels: List[float], time_col: str = 'time') -> pd.DataFrame:
        """
        Assign labels (VECTORIZED).

        Args:
            df: DataFrame with time_col column
            labels: List of ground truth times
            time_col: Name of time column

        Returns:
            DataFrame with label_col added
        """
        times = df[time_col].values
        gt = np.array(labels)

        if len(gt) == 0:
            result = df.copy()
            result[self.label_col] = 0
            return result

        # Pairwise distances: (n_samples, n_gt)
        diffs = np.abs(times[:, np.newaxis] - gt[np.newaxis, :])

        # Nearest sample for each GT
        nearest_idx = np.argmin(diffs, axis=0)
        nearest_dist = diffs[nearest_idx, np.arange(len(gt))]

        # Only label if within tolerance
        valid = nearest_idx[nearest_dist <= self.tolerance]

        is_positive = np.zeros(len(df), dtype=np.int32)
        is_positive[valid] = 1

        result = df.copy()
        result[self.label_col] = is_positive
        return result


# =============================================================================
# NMS (vectorized)
# =============================================================================

def nms_vectorized(times: np.ndarray, scores: np.ndarray, min_gap: float) -> np.ndarray:
    """
    Non-Maximum Suppression - keep local maxima (FULLY VECTORIZED).

    Args:
        times: Array of detection times
        scores: Array of detection scores/probabilities
        min_gap: Minimum gap between kept detections

    Returns:
        Indices of kept detections
    """
    if len(times) == 0:
        return np.array([], dtype=np.int64)

    n = len(times)
    # Pairwise distances
    dist = np.abs(times[:, np.newaxis] - times[np.newaxis, :])
    # Neighbors within gap (excluding self)
    neighbors = (dist < min_gap) & (np.eye(n) == 0)
    # Max score among neighbors
    max_neighbor = np.max(np.where(neighbors, scores[np.newaxis, :], -np.inf), axis=1)
    # Keep if own score >= all neighbor scores (local maximum)
    return np.where(scores >= max_neighbor)[0]


# =============================================================================
# Model Trainers
# =============================================================================

@dataclass
class TrainingResult:
    """Result of model training."""
    model: Any
    metrics: Dict[str, float]
    feature_cols: List[str]
    importance: Optional[pd.DataFrame] = None


class BaseTrainer(ABC):
    """Abstract base for model trainers."""

    @abstractmethod
    def train(self, df: pd.DataFrame, label_col: str = 'is_positive') -> TrainingResult:
        """Train model and return result."""
        pass

    @abstractmethod
    def save(self, result: TrainingResult, path: str):
        """Save trained model."""
        pass


class XGBoostTrainer(BaseTrainer):
    """XGBoost trainer for binary classification."""

    def __init__(
        self,
        feature_cols: List[str],
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
    ):
        self.feature_cols = feature_cols
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def train(self, df: pd.DataFrame, label_col: str = 'is_positive') -> TrainingResult:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0).values
        y = df[label_col].values

        n_pos, n_neg = y.sum(), len(y) - y.sum()
        scale = n_neg / n_pos if n_pos > 0 else 1
        logger.info(f"Classes: {n_pos} positive, {n_neg} negative")

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        print("\n" + "="*60 + "\nTEST RESULTS\n" + "="*60)
        print(classification_report(y_te, y_pred, target_names=['Negative', 'Positive']))

        cv = cross_val_score(model, X, y, cv=5, scoring='f1')
        print(f"CV F1: {cv.mean():.3f} (+/- {cv.std()*2:.3f})")

        importance = pd.DataFrame({'feature': cols, 'importance': model.feature_importances_})
        importance = importance.sort_values('importance', ascending=False)
        print("\nFeature Importance:\n" + importance.to_string(index=False))

        return TrainingResult(
            model=model,
            metrics={
                'precision': precision_score(y_te, y_pred),
                'recall': recall_score(y_te, y_pred),
                'f1': f1_score(y_te, y_pred),
                'cv_f1': cv.mean(),
            },
            feature_cols=cols,
            importance=importance,
        )

    def save(self, result: TrainingResult, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': result.model,
                'feature_cols': result.feature_cols,
                'metrics': result.metrics,
            }, f)
        logger.info(f"Saved: {path}")


# =============================================================================
# Evaluator
# =============================================================================

class Evaluator:
    """Evaluate model with NMS against ground truth."""

    def __init__(self, min_gap: float = 15.0, tolerance: float = 5.0):
        self.min_gap = min_gap
        self.tolerance = tolerance

    def evaluate(
        self,
        model,
        df: pd.DataFrame,
        feature_cols: List[str],
        gt_times: np.ndarray,
        time_col: str = 'time',
        thresholds: List[float] = None,
    ):
        """Evaluate with NMS (VECTORIZED)."""
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        cols = [c for c in feature_cols if c in df.columns]
        X = df[cols].fillna(0).values
        times = df[time_col].values
        proba = model.predict_proba(X)[:, 1]

        print(f"\n{'='*60}\nPOST-NMS (gap={self.min_gap}s, tol={self.tolerance}s)\n{'='*60}")
        print(f"GT: {len(gt_times)} items")

        for th in thresholds:
            mask = proba >= th
            det_t, det_p = times[mask], proba[mask]

            if len(det_t) == 0:
                print(f"th={th:.1f}: No detections")
                continue

            keep = nms_vectorized(det_t, det_p, self.min_gap)
            nms_t = det_t[keep]

            # Vectorized matching
            d = np.abs(gt_times[:, np.newaxis] - nms_t[np.newaxis, :])
            gt_hit = d.min(axis=1) <= self.tolerance if len(nms_t) else np.zeros(len(gt_times), dtype=bool)
            det_hit = d.min(axis=0) <= self.tolerance if len(gt_times) else np.zeros(len(nms_t), dtype=bool)

            tp, fp, fn = det_hit.sum(), (~det_hit).sum(), (~gt_hit).sum()
            p = tp / (tp + fp) if tp + fp else 0
            r = tp / (tp + fn) if tp + fn else 0
            f1 = 2*p*r / (p + r) if p + r else 0

            print(f"th={th:.1f}: P={p:.2f} R={r:.2f} F1={f1:.2f} (TP={tp} FP={fp} FN={fn} det={len(nms_t)})")


# =============================================================================
# Training Pipeline
# =============================================================================

class TrainingPipeline:
    """
    Universal training pipeline following project architecture.

    Orchestrates the complete ML training workflow:
    1. Load ground truth via gt_loader
    2. For each audio file:
       - Load audio via AudioLoader (from src/audio/loader.py)
       - Extract features via feature_extractor (uses Tasks like DropDetectorML)
       - Cache features via CacheManager (from src/core/pipelines/cache_manager.py)
    3. Assign labels via LabelAssigner (vectorized nearest neighbor)
    4. Train model via trainer (XGBoostTrainer or custom)
    5. Evaluate with NMS via Evaluator

    Key Features:
    - Single Responsibility: each component has one job
    - No code duplication: uses existing AudioLoader, CacheManager
    - Vectorized operations: no Python loops in hot paths
    - Extensible: abstract bases for new GT formats and models

    Components:
    - gt_loader: Ground truth loader (DropGroundTruthLoader, etc.)
    - feature_extractor: Callable[[np.ndarray, int, Dict], List[dict]]
    - trainer: Model trainer (XGBoostTrainer, etc.)
    - labeler: Vectorized label assigner (nearest neighbor)
    - evaluator: Post-NMS evaluator for detection tasks

    Example:
        pipeline = TrainingPipeline(
            gt_loader=DropGroundTruthLoader('data/ground_truth_drops.json'),
            feature_extractor=my_extractor_fn,
            trainer=XGBoostTrainer(feature_cols=[...]),
        )
        result = pipeline.run('models/model.pkl', project_root=Path.cwd())
        # result.metrics = {'f1': 0.80, 'cv_f1': 0.29, ...}
    """

    def __init__(
        self,
        gt_loader: BaseGroundTruthLoader,
        feature_extractor: Callable[[np.ndarray, int, Dict], List[dict]],
        trainer: BaseTrainer,
        cache_dir: str = 'cache',
        sr: int = 22050,
        label_tolerance: float = 2.0,
    ):
        self.gt_loader = gt_loader
        self.feature_extractor = feature_extractor
        self.trainer = trainer

        # Use project components (CacheRepository as single entry point)
        self.cache = CacheRepository(cache_dir=cache_dir)
        self.loader = AudioLoader(sample_rate=sr)
        self.dataset = FeatureDataset(cache=self.cache, loader=self.loader)
        self.labeler = LabelAssigner(tolerance=label_tolerance, label_col='is_positive')
        self.evaluator = Evaluator()

    def run(
        self,
        output_path: str,
        project_root: Path = None,
        use_cache: bool = True,
    ) -> TrainingResult:
        """
        Run training pipeline.

        Args:
            output_path: Path to save model
            project_root: Project root for resolving relative paths
            use_cache: Whether to use feature cache

        Returns:
            TrainingResult
        """
        if project_root is None:
            project_root = Path.cwd()

        print("="*60 + "\nTRAINING PIPELINE\n" + "="*60)

        # Load ground truth
        gt = self.gt_loader.load()
        all_labels = []
        print(f"\nSets: {len(gt)}")
        for name, data in gt.items():
            all_labels.extend(data['labels'])
            print(f"  {name}: {len(data['labels'])} labels")

        # Process sets
        all_dfs = []
        for name, data in gt.items():
            if not data['file']:
                continue
            full_path = project_root / data['file']
            if not full_path.exists():
                logger.warning(f"Skip {name}: not found")
                continue

            print(f"\n{'='*60}\n{name}\n{'='*60}")

            df = self.dataset.extract(
                str(full_path),
                lambda y, sr, m: self.feature_extractor(y, sr, m),
                data['metadata'],
                use_cache=use_cache,
            )

            if len(df):
                df = self.labeler.assign(df, data['labels'])
                df['set_name'] = name
                all_dfs.append(df)
                logger.info(f"Labeled: {df['is_positive'].sum()} positive")

        if not all_dfs:
            raise ValueError("No data extracted")

        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n{'='*60}\nCOMBINED\n{'='*60}")
        print(f"Samples: {len(combined)}, Positive: {combined['is_positive'].sum()}")

        # Train
        result = self.trainer.train(combined, label_col='is_positive')
        self.trainer.save(result, output_path)

        # Evaluate
        self.evaluator.evaluate(
            result.model, combined, result.feature_cols, np.array(all_labels)
        )

        print(f"\n{'='*60}\nSaved: {output_path}\n{'='*60}")
        return result
