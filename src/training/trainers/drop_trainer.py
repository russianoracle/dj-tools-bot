"""
Drop Detector Trainer - Training logic extracted from DropDetectorML.

This module separates training concerns from inference, following SRP.
The production DropDetectorML only handles inference (load_model + execute).

Usage:
    from src.training.trainers import DropDetectorTrainer

    trainer = DropDetectorTrainer()
    model, metrics = trainer.train(features_df, label_column='is_drop')
    trainer.save(model, 'models/drop_detector.pkl')
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DropDetectorTrainer:
    """
    Trainer for XGBoost drop detection model.

    Extracted from DropDetectorML.train_model() to separate training from inference.
    """

    # Default feature columns for drop detection
    DEFAULT_FEATURE_COLS = [
        'rms_change', 'rms_change_short', 'bass_change',
        'high_change', 'centroid_change',
        'bass_before', 'bass_after', 'rms_before', 'rms_after'
    ]

    # Extended feature columns (if available in training data)
    EXTENDED_FEATURE_COLS = [
        'rms_change', 'rms_change_short', 'bass_change', 'high_change', 'centroid_change',
        'bass_before', 'bass_after', 'rms_before', 'rms_after',
        'valley_depth', 'drop_contrast', 'buildup_ratio', 'has_buildup_pattern',
    ]

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        n_estimators: int = 100,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        """
        Initialize trainer.

        Args:
            feature_cols: Feature columns to use (default: DEFAULT_FEATURE_COLS)
            n_estimators: XGBoost n_estimators
            max_depth: XGBoost max_depth
            random_state: Random seed for reproducibility
        """
        self.feature_cols = feature_cols or self.DEFAULT_FEATURE_COLS
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def train(
        self,
        features_df: pd.DataFrame,
        label_column: str = 'is_drop',
    ) -> Tuple[Any, Dict]:
        """
        Train XGBoost model on labeled data.

        Args:
            features_df: DataFrame with features and labels
            label_column: Column name for drop labels (0/1)

        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score

        # Use available feature columns
        available_cols = [c for c in self.feature_cols if c in features_df.columns]
        if not available_cols:
            raise ValueError(f"No feature columns found. Expected: {self.feature_cols}")

        X = features_df[available_cols].values
        y = features_df[label_column].values

        # Handle class imbalance
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        scale_pos = n_neg / n_pos if n_pos > 0 else 1

        logger.info(f"Training data: {n_pos} positive, {n_neg} negative samples")

        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            scale_pos_weight=scale_pos,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
        )

        # Cross-validation
        scores = cross_val_score(model, X, y, cv=3, scoring='f1')
        logger.info(f"F1 Score (3-fold CV): {scores.mean():.3f} (+/- {scores.std():.3f})")

        # Train on all data
        model.fit(X, y)

        # Feature importance
        importance = dict(zip(available_cols, model.feature_importances_))
        logger.info(f"Feature importance: {importance}")

        metrics = {
            'f1_mean': float(scores.mean()),
            'f1_std': float(scores.std()),
            'importance': importance,
            'n_positive': int(n_pos),
            'n_negative': int(n_neg),
            'feature_cols': available_cols,
        }

        return model, metrics

    def save(self, model: Any, save_path: str) -> None:
        """
        Save trained model to disk.

        Args:
            model: Trained XGBoost model
            save_path: Path to save model
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {save_path}")

    def load(self, load_path: str) -> Any:
        """
        Load trained model from disk.

        Args:
            load_path: Path to model file

        Returns:
            Loaded model
        """
        with open(load_path, 'rb') as f:
            return pickle.load(f)
