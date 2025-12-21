"""
Zone Classification Task - Classify track into energy zones.

Zones:
- YELLOW: Low energy, calm (rest zone)
- GREEN: Medium energy (transitional)
- PURPLE: High energy with drops (peak zone)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
import time
import pickle
from pathlib import Path

from .base import AudioContext, TaskResult, BaseTask
from .feature_extraction import FeatureExtractionTask, FEATURE_NAMES
from app.common.logging import get_logger

logger = get_logger(__name__)


ZONE_YELLOW = 'yellow'
ZONE_GREEN = 'green'
ZONE_PURPLE = 'purple'

ZONE_COLORS = {
    ZONE_YELLOW: '#FFD700',
    ZONE_GREEN: '#32CD32',
    ZONE_PURPLE: '#9370DB',
}

ZONE_DESCRIPTIONS = {
    ZONE_YELLOW: 'Low energy, calm - rest zone',
    ZONE_GREEN: 'Medium energy - transitional',
    ZONE_PURPLE: 'High energy with drops - peak zone',
}


@dataclass
class ZoneClassificationResult(TaskResult):
    """
    Result of zone classification.

    Attributes:
        zone: Predicted zone ('yellow', 'green', 'purple')
        confidence: Prediction confidence (0-1)
        zone_scores: Scores for each zone
        key_features: Important features for this prediction
        probabilities: Class probabilities if using ML model
    """
    zone: str = ZONE_GREEN
    confidence: float = 0.0
    zone_scores: Dict[str, float] = field(default_factory=dict)
    key_features: Dict[str, float] = field(default_factory=dict)
    probabilities: Dict[str, float] = field(default_factory=dict)

    @property
    def color(self) -> str:
        """Get zone color code."""
        return ZONE_COLORS.get(self.zone, '#808080')

    @property
    def description(self) -> str:
        """Get zone description."""
        return ZONE_DESCRIPTIONS.get(self.zone, 'Unknown zone')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            'zone': self.zone,
            'confidence': self.confidence,
            'zone_scores': self.zone_scores,
            'key_features': self.key_features,
            'probabilities': self.probabilities,
            'color': self.color,
            'description': self.description,
        })
        return base


class ZoneClassificationTask(BaseTask):
    """
    Classify track into energy zone.

    Can use:
    1. Rule-based classification (no model needed)
    2. ML model (XGBoost) if model_path provided

    Rule-based criteria:
    - YELLOW: Low tempo (<110), low energy variance, low brightness
    - PURPLE: High tempo (>128), high drops, high energy variance
    - GREEN: Everything else

    Usage:
        # Rule-based
        task = ZoneClassificationTask()
        result = task.execute(context)

        # With ML model
        task = ZoneClassificationTask(model_path="models/zone_classifier.pkl")
        result = task.execute(context)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_rules_fallback: bool = True
    ):
        """
        Initialize zone classification task.

        Args:
            model_path: Path to trained ML model (pkl)
            use_rules_fallback: Fall back to rules if model fails
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.use_rules_fallback = use_rules_fallback

        if model_path:
            self._load_model(model_path)

        # Feature extraction for input preparation
        self._feature_task = FeatureExtractionTask()

    def _load_model(self, model_path: str):
        """Load ML model from file."""
        try:
            path = Path(model_path)
            if path.exists():
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                if isinstance(data, dict):
                    self.model = data.get('model')
                    self.scaler = data.get('scaler')
                else:
                    self.model = data
        except Exception as e:
            logger.warning("Could not load model", data={"error": str(e)})

    def execute(self, context: AudioContext) -> ZoneClassificationResult:
        """Classify the track into an energy zone."""
        start_time = time.time()

        try:
            # Extract features
            feature_result = self._feature_task.execute(context)
            if not feature_result.success:
                raise ValueError(f"Feature extraction failed: {feature_result.error}")

            features = feature_result.features
            feature_vector = feature_result.to_vector()

            # Try ML classification first
            if self.model is not None:
                try:
                    result = self._classify_ml(feature_vector, features)
                    result.processing_time_sec = time.time() - start_time
                    return result
                except Exception as e:
                    if not self.use_rules_fallback:
                        raise
                    logger.warning("ML classification failed, using rules", data={"error": str(e)})

            # Rule-based classification
            result = self._classify_rules(features)
            result.processing_time_sec = time.time() - start_time
            return result

        except Exception as e:
            return ZoneClassificationResult(
                success=False,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                error=str(e)
            )

    def _classify_ml(
        self,
        feature_vector: np.ndarray,
        features: Dict[str, float]
    ) -> ZoneClassificationResult:
        """Classify using ML model."""
        X = feature_vector.reshape(1, -1)

        # Scale if scaler available
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Predict
        zone_idx = self.model.predict(X)[0]
        zones = [ZONE_YELLOW, ZONE_GREEN, ZONE_PURPLE]
        zone = zones[zone_idx] if isinstance(zone_idx, int) else str(zone_idx)

        # Get probabilities if available
        probabilities = {}
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[0]
            for i, z in enumerate(zones):
                probabilities[z] = float(probs[i]) if i < len(probs) else 0.0
            confidence = float(max(probs))
        else:
            confidence = 0.8  # Default confidence for non-probabilistic models
            probabilities = {zone: confidence}

        # Key features (top 5 by importance if available)
        key_features = self._get_key_features(features)

        return ZoneClassificationResult(
            success=True,
            task_name=self.name,
            processing_time_sec=0.0,
            zone=zone,
            confidence=confidence,
            zone_scores={zone: confidence},
            key_features=key_features,
            probabilities=probabilities
        )

    def _classify_rules(self, features: Dict[str, float]) -> ZoneClassificationResult:
        """Classify using rule-based logic."""
        # Extract key features
        tempo = features.get('tempo', 120.0)
        energy_var = features.get('rms_energy_delta', 0.0)
        brightness = features.get('brightness', 0.5)
        drop_count = features.get('drop_count', 0)
        drop_intensity = features.get('drop_intensity', 0.0)
        low_energy = features.get('low_energy_ratio', 0.5)
        bass_ratio = features.get('bass_energy_ratio', 0.5)

        # Calculate zone scores
        scores = {
            ZONE_YELLOW: 0.0,
            ZONE_GREEN: 0.0,
            ZONE_PURPLE: 0.0,
        }

        # YELLOW criteria
        if tempo < 110:
            scores[ZONE_YELLOW] += 0.3
        if energy_var < 0.1:
            scores[ZONE_YELLOW] += 0.2
        if brightness < 0.3:
            scores[ZONE_YELLOW] += 0.2
        if low_energy > 0.6:
            scores[ZONE_YELLOW] += 0.2
        if drop_count == 0:
            scores[ZONE_YELLOW] += 0.1

        # PURPLE criteria
        if tempo > 128:
            scores[ZONE_PURPLE] += 0.25
        if drop_count >= 2:
            scores[ZONE_PURPLE] += 0.25
        if drop_intensity > 0.5:
            scores[ZONE_PURPLE] += 0.2
        if energy_var > 0.2:
            scores[ZONE_PURPLE] += 0.15
        if brightness > 0.4:
            scores[ZONE_PURPLE] += 0.15

        # GREEN gets remaining score
        scores[ZONE_GREEN] = max(0.0, 1.0 - scores[ZONE_YELLOW] - scores[ZONE_PURPLE])

        # Normalize
        total = sum(scores.values()) + 1e-10
        scores = {k: v / total for k, v in scores.items()}

        # Select zone
        zone = max(scores, key=scores.get)
        confidence = scores[zone]

        # Key features
        key_features = {
            'tempo': tempo,
            'drop_count': drop_count,
            'energy_variance': energy_var,
            'brightness': brightness,
            'low_energy_ratio': low_energy,
        }

        return ZoneClassificationResult(
            success=True,
            task_name=self.name,
            processing_time_sec=0.0,
            zone=zone,
            confidence=confidence,
            zone_scores=scores,
            key_features=key_features,
            probabilities=scores
        )

    def _get_key_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get the most important features for classification."""
        important_features = [
            'tempo', 'drop_count', 'rms_energy_delta',
            'brightness', 'low_energy_ratio'
        ]
        return {k: features.get(k, 0.0) for k in important_features}
