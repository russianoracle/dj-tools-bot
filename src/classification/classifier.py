"""Main classification logic for energy zones."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import numpy as np

from ..audio.extractors import AudioFeatures
from ..utils import get_logger, get_config
from .rules import RuleBasedClassifier

logger = get_logger(__name__)


class EnergyZone(Enum):
    """Energy zone classifications."""
    YELLOW = "yellow"
    GREEN = "green"
    PURPLE = "purple"
    UNCERTAIN = "uncertain"

    def __str__(self):
        return self.value

    @property
    def emoji(self):
        """Get emoji for zone."""
        emoji_map = {
            EnergyZone.YELLOW: "🟨",
            EnergyZone.GREEN: "🟩",
            EnergyZone.PURPLE: "🟪",
            EnergyZone.UNCERTAIN: "❓"
        }
        return emoji_map[self]

    @property
    def display_name(self):
        """Get display name for zone."""
        name_map = {
            EnergyZone.YELLOW: "Yellow (Rest)",
            EnergyZone.GREEN: "Green (Transition)",
            EnergyZone.PURPLE: "Purple (Energy)",
            EnergyZone.UNCERTAIN: "Uncertain"
        }
        return name_map[self]


@dataclass
class ClassificationResult:
    """Result of energy zone classification."""
    zone: EnergyZone
    confidence: float
    features: AudioFeatures
    method: str  # 'rule-based' or 'ml'

    def __str__(self):
        return f"{self.zone.emoji} {self.zone.display_name} (confidence: {self.confidence:.2%}, method: {self.method})"


class EnergyZoneClassifier:
    """
    Hybrid classifier for energy zones.

    Uses rule-based classification with optional ML enhancement.
    """

    def __init__(self, config: Any = None, model_path: Optional[str] = None):
        """
        Initialize classifier.

        Args:
            config: Configuration object
            model_path: Path to trained ML model (optional)
        """
        if config is None:
            config = get_config()

        self.config = config
        self.rule_classifier = RuleBasedClassifier(config)
        self.ml_model = None

        if model_path:
            self._load_model(model_path)

    def classify(self, features: AudioFeatures) -> ClassificationResult:
        """
        Classify track into energy zone.

        Args:
            features: Extracted audio features

        Returns:
            ClassificationResult with zone and confidence
        """
        # Try rule-based classification first
        zone, confidence = self.rule_classifier.classify(features)

        # If confidence is low and ML model available, use it
        if confidence < 0.7 and self.ml_model is not None:
            ml_zone, ml_confidence = self._classify_ml(features)
            if ml_confidence > confidence:
                logger.info(f"Using ML prediction: {ml_zone} (conf: {ml_confidence:.2%})")
                return ClassificationResult(
                    zone=ml_zone,
                    confidence=ml_confidence,
                    features=features,
                    method='ml'
                )

        return ClassificationResult(
            zone=zone,
            confidence=confidence,
            features=features,
            method='rule-based'
        )

    def _classify_ml(self, features: AudioFeatures) -> tuple[EnergyZone, float]:
        """
        Classify using ML model.

        Args:
            features: Audio features

        Returns:
            Tuple of (zone, confidence)
        """
        if self.ml_model is None:
            return EnergyZone.UNCERTAIN, 0.0

        try:
            # Convert features to vector
            X = features.to_vector().reshape(1, -1)

            # Predict
            prediction = self.ml_model.predict(X)[0]
            probabilities = self.ml_model.predict_proba(X)[0]

            # Map prediction to zone
            zone_map = {0: EnergyZone.YELLOW, 1: EnergyZone.GREEN, 2: EnergyZone.PURPLE}
            zone = zone_map.get(prediction, EnergyZone.UNCERTAIN)
            confidence = float(np.max(probabilities))

            return zone, confidence

        except Exception as e:
            logger.error(f"ML classification failed: {e}")
            return EnergyZone.UNCERTAIN, 0.0

    def _load_model(self, model_path: str):
        """Load trained ML model."""
        try:
            import pickle
            with open(model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            logger.info(f"Loaded ML model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
            self.ml_model = None

    def save_model(self, model_path: str):
        """Save trained ML model."""
        if self.ml_model is None:
            raise ValueError("No model to save")

        try:
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(self.ml_model, f)
            logger.info(f"Saved ML model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def train_model(self, X: np.ndarray, y: np.ndarray):
        """
        Train ML model on labeled data.

        Args:
            X: Feature vectors (N, 16)
            y: Labels (N,) - 0=yellow, 1=green, 2=purple
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        logger.info(f"Training ML model on {len(X)} samples...")

        # Train Random Forest
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )

        self.ml_model.fit(X, y)

        # Evaluate with cross-validation
        scores = cross_val_score(self.ml_model, X, y, cv=5)
        logger.info(f"Cross-validation accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})")

        return scores.mean()
