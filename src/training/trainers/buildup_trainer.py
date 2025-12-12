"""
Buildup Detector Trainer - Training logic extracted from BuildupDetectorML.

This module separates training concerns from inference, following SRP.
The production BuildupDetectorML only handles inference (load_model + execute).

Usage:
    from src.training.trainers import BuildupDetectorTrainer

    trainer = BuildupDetectorTrainer()
    model, metrics = trainer.train(features_df, label_column='label')
    trainer.save(model, 'models/buildup_detector.pkl')
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BuildupDetectorTrainer:
    """
    Trainer for XGBoost buildup phase classification model.

    Extracted from BuildupDetectorML.train_model() to separate training from inference.
    Handles multi-class classification for buildup phases.
    """

    # Default feature columns for buildup detection
    DEFAULT_FEATURE_COLS = [
        'bass_before', 'bass_after', 'bass_change',
        'high_before', 'high_after', 'high_change',
        'rms_before', 'rms_after', 'rms_change',
        'centroid_before', 'centroid_after', 'centroid_change',
    ]

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        n_estimators: int = 100,
        max_depth: int = 4,
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
        label_column: str = 'label',
    ) -> Tuple[Any, Dict]:
        """
        Train XGBoost model on labeled data.

        Args:
            features_df: DataFrame with features and phase labels
            label_column: Column name for phase labels (categorical)

        Returns:
            Tuple of (trained_model_dict, metrics_dict)
            model_dict contains: model, label_encoder, feature_cols
        """
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score

        # Use available feature columns
        available_cols = [c for c in self.feature_cols if c in features_df.columns]
        if not available_cols:
            raise ValueError(f"No feature columns found. Expected: {self.feature_cols}")

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(features_df[label_column].values)
        X = features_df[available_cols].values

        logger.info(f"Training data: {len(X)} samples, {len(le.classes_)} classes: {le.classes_}")

        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
        )

        # Cross-validation
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        logger.info(f"Accuracy (3-fold CV): {scores.mean():.3f} (+/- {scores.std():.3f})")

        # Train on all data
        model.fit(X, y)

        # Feature importance
        importance = dict(zip(available_cols, model.feature_importances_))
        logger.info(f"Feature importance: {importance}")

        # Return model bundle (for compatibility with existing loading)
        model_bundle = {
            'model': model,
            'label_encoder': le,
            'feature_cols': available_cols,
        }

        metrics = {
            'accuracy_mean': float(scores.mean()),
            'accuracy_std': float(scores.std()),
            'importance': importance,
            'classes': le.classes_.tolist(),
            'n_samples': len(X),
        }

        return model_bundle, metrics

    def save(self, model_bundle: Dict, save_path: str) -> None:
        """
        Save trained model bundle to disk.

        Args:
            model_bundle: Dict with model, label_encoder, feature_cols
            save_path: Path to save model
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(model_bundle, f)
        logger.info(f"Saved model to {save_path}")

    def load(self, load_path: str) -> Dict:
        """
        Load trained model bundle from disk.

        Args:
            load_path: Path to model file

        Returns:
            Dict with model, label_encoder, feature_cols
        """
        with open(load_path, 'rb') as f:
            return pickle.load(f)
