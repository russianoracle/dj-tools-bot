"""Rule-based classification logic."""

from typing import Tuple
from ..audio.extractors import AudioFeatures
from ..utils import get_logger

logger = get_logger(__name__)


class RuleBasedClassifier:
    """Rule-based classifier using audio feature thresholds."""

    def __init__(self, config):
        """
        Initialize rule-based classifier.

        Args:
            config: Configuration object with classification thresholds
        """
        self.config = config

        # Load thresholds from config
        self.yellow_max_bpm = config.get('classification.yellow_max_bpm', 110)
        self.purple_min_bpm = config.get('classification.purple_min_bpm', 128)

        self.yellow_max_energy_var = config.get('classification.yellow_max_energy_variance', 0.15)
        self.purple_min_energy_var = config.get('classification.purple_min_energy_variance', 0.40)

        self.yellow_max_brightness = config.get('classification.yellow_max_brightness', 0.20)
        self.purple_min_spectral_centroid = config.get('classification.purple_min_spectral_centroid', 2000)

        self.drop_intensity_threshold = config.get('classification.drop_intensity_threshold', 0.6)
        self.min_confidence = config.get('classification.min_confidence', 0.6)

    def classify(self, features: AudioFeatures) -> Tuple['EnergyZone', float]:
        """
        Classify track based on rules.

        Args:
            features: Extracted audio features

        Returns:
            Tuple of (EnergyZone, confidence)
        """
        # Import here to avoid circular import
        from .classifier import EnergyZone

        # Score for each zone (0-1)
        yellow_score = self._score_yellow(features)
        green_score = self._score_green(features)
        purple_score = self._score_purple(features)

        logger.debug(f"Zone scores - Yellow: {yellow_score:.2f}, Green: {green_score:.2f}, Purple: {purple_score:.2f}")

        # Select zone with highest score
        scores = {
            EnergyZone.YELLOW: yellow_score,
            EnergyZone.GREEN: green_score,
            EnergyZone.PURPLE: purple_score
        }

        best_zone = max(scores, key=scores.get)
        confidence = scores[best_zone]

        # Check if confidence is too low
        if confidence < self.min_confidence:
            logger.warning(f"Low confidence classification: {confidence:.2%}")
            return EnergyZone.UNCERTAIN, confidence

        logger.info(f"Classified as {best_zone.display_name} (confidence: {confidence:.2%})")
        return best_zone, confidence

    def _score_yellow(self, features: AudioFeatures) -> float:
        """
        Calculate yellow zone score.

        Yellow tracks: Low tempo, low energy variance, low brightness.
        """
        score = 0.0
        count = 0

        # BPM criteria (lower is better for yellow)
        # Smooth scoring function: highest score at low BPM, decreasing as tempo increases
        # score = 1.0 - (tempo / (yellow_max + penalty_range))
        # This ensures continuous scoring across the threshold
        penalty_range = 50.0
        max_tempo = self.yellow_max_bpm + penalty_range
        score += max(0, 1.0 - (features.tempo / max_tempo))
        count += 1

        # Energy variance (lower is better)
        # Continuous scoring function
        variance_penalty_range = 0.15  # Additional range for penalty
        max_variance = self.yellow_max_energy_var + variance_penalty_range
        score += max(0, 1.0 - (features.energy_variance / max_variance))
        count += 1

        # Brightness (lower is better)
        # Continuous scoring function (brightness is 0.0-1.0 normalized)
        brightness_penalty_range = 0.3
        max_brightness = min(1.0, self.yellow_max_brightness + brightness_penalty_range)
        score += max(0, 1.0 - (features.brightness / max_brightness))
        count += 1

        # Drop intensity (should be low)
        score += 1.0 - min(1.0, features.drop_intensity / self.drop_intensity_threshold)
        count += 1

        # Low energy percentage (higher is better for yellow)
        score += features.low_energy
        count += 1

        return score / count if count > 0 else 0.0

    def _score_green(self, features: AudioFeatures) -> float:
        """
        Calculate green zone score.

        Green tracks: Medium tempo, medium energy, transitional characteristics.
        """
        score = 0.0
        count = 0

        # BPM in middle range (110-128)
        mid_bpm = (self.yellow_max_bpm + self.purple_min_bpm) / 2
        bpm_range = self.purple_min_bpm - self.yellow_max_bpm

        # Distance from middle (closer = higher score)
        bpm_distance = abs(features.tempo - mid_bpm)
        if bpm_distance < bpm_range / 2:
            score += 1.0 - (bpm_distance / (bpm_range / 2))
            count += 1
        else:
            score += max(0, 0.5 - (bpm_distance / bpm_range))
            count += 1

        # Energy variance in middle range
        mid_energy_var = (self.yellow_max_energy_var + self.purple_min_energy_var) / 2
        energy_range = self.purple_min_energy_var - self.yellow_max_energy_var

        energy_distance = abs(features.energy_variance - mid_energy_var)
        if energy_distance < energy_range / 2:
            score += 1.0 - (energy_distance / (energy_range / 2))
            count += 1
        else:
            score += max(0, 0.5 - (energy_distance / energy_range))
            count += 1

        # Moderate brightness
        if 0.2 < features.brightness < 0.6:
            score += 1.0 - abs(features.brightness - 0.4) / 0.2
            count += 1
        else:
            score += 0.3
            count += 1

        # Moderate drop intensity
        if features.drop_intensity < self.drop_intensity_threshold:
            score += features.drop_intensity / self.drop_intensity_threshold
            count += 1
        else:
            score += 0.5
            count += 1

        return score / count if count > 0 else 0.0

    def _score_purple(self, features: AudioFeatures) -> float:
        """
        Calculate purple zone score.

        Purple tracks: High tempo, high energy variance, pronounced drops.
        """
        score = 0.0
        count = 0

        # BPM criteria (higher is better for purple)
        # More strict: require BPM > purple_min for good scores
        if features.tempo >= self.purple_min_bpm:
            # Bonus for very high BPM
            score += min(1.0, (features.tempo - self.purple_min_bpm) / 40 + 0.7)
            count += 1
        else:
            # Moderate penalty for low BPM
            bpm_ratio = features.tempo / self.purple_min_bpm
            score += max(0, bpm_ratio * 0.7)  # Balanced penalty
            count += 1

        # Energy variance (higher is better)
        # More strict: require variance > purple_min for good scores
        if features.energy_variance >= self.purple_min_energy_var:
            score += min(1.0, features.energy_variance / 0.6)
            count += 1
        else:
            # Moderate penalty for low variance
            var_ratio = features.energy_variance / self.purple_min_energy_var
            score += max(0, var_ratio * 0.8)  # Balanced penalty
            count += 1

        # Spectral centroid (higher is better)
        if features.spectral_centroid > self.purple_min_spectral_centroid:
            score += min(1.0, features.spectral_centroid / 3000)
            count += 1
        else:
            score += max(0, features.spectral_centroid / self.purple_min_spectral_centroid * 0.7)
            count += 1

        # Drop intensity (higher is better)
        if features.drop_intensity > self.drop_intensity_threshold:
            score += min(1.0, features.drop_intensity / 0.8 + 0.3)
            count += 1
        else:
            score += features.drop_intensity / self.drop_intensity_threshold * 0.8
            count += 1

        # Brightness (higher is better for energetic tracks)
        score += min(1.0, features.brightness / 0.7)
        count += 1

        # Low energy percentage (lower is better for purple)
        score += 1.0 - features.low_energy
        count += 1

        return score / count if count > 0 else 0.0
