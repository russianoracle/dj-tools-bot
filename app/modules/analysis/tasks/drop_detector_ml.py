"""
ML-based Drop Detection Task.

Uses XGBoost trained on signal features at phrase boundaries:
- Bass energy change (most important for EDM)
- Spectral centroid change (brightness drop)
- RMS energy change (overall loudness)
- High-frequency energy change

Key insight from analysis:
- Drops have bass_change > +200% (5-25x increase)
- Drops have centroid_change < -40% (sound gets darker)
- Drops happen on phrase boundaries (every 16 beats)

Architecture:
- Requires AdaptiveBeatGridTask to be run first
- Analyzes signal at each phrase boundary
- Returns drops with confidence scores
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import pickle

from .base import BaseTask, AudioContext

logger = logging.getLogger(__name__)


@dataclass
class DetectedDrop:
    """A detected drop with confidence and features."""
    time_sec: float
    confidence: float
    bass_change: float      # % change in bass energy
    centroid_change: float  # % change in spectral centroid
    rms_change: float       # % change in RMS energy
    phrase_idx: int = 0     # Which phrase boundary


@dataclass
class DropDetectorMLResult:
    """Result of ML drop detection."""
    drops: List[DetectedDrop] = field(default_factory=list)
    n_phrase_boundaries: int = 0
    detection_threshold: float = 0.5

    def get_drop_times(self) -> List[float]:
        """Get list of drop times."""
        return [d.time_sec for d in self.drops]

    def get_high_confidence_drops(self, min_confidence: float = 0.8) -> List[DetectedDrop]:
        """Get drops with confidence above threshold."""
        return [d for d in self.drops if d.confidence >= min_confidence]

    def to_dict(self) -> dict:
        return {
            'n_drops': len(self.drops),
            'n_phrase_boundaries': self.n_phrase_boundaries,
            'detection_threshold': self.detection_threshold,
            'drops': [
                {
                    'time_sec': d.time_sec,
                    'confidence': d.confidence,
                    'bass_change': d.bass_change,
                    'centroid_change': d.centroid_change,
                    'rms_change': d.rms_change,
                }
                for d in self.drops
            ],
        }


class DropDetectorML(BaseTask):
    """
    ML-based drop detector using XGBoost.

    Analyzes signal features at phrase boundaries and classifies
    each as drop/non-drop using a trained model.

    Features used:
    - rms_change: Overall energy change
    - rms_change_short: Energy change in 500ms window
    - bass_change: Sub-bass (< 150Hz) energy change
    - high_change: High-freq (> 3kHz) energy change
    - centroid_change: Spectral brightness change

    Usage:
        # Basic usage (needs AdaptiveBeatGridResult)
        grid_task = AdaptiveBeatGridTask()
        grid_result = grid_task.execute(audio_context)

        detector = DropDetectorML()
        result = detector.execute(audio_context, grid_result.phrase_boundaries)

        for drop in result.drops:
            print(f"Drop at {drop.time_sec:.1f}s (conf: {drop.confidence:.2f})")
    """

    name = "drop_detector_ml"

    # Default XGBoost model path (trained on Josh Baker 26 drops with 100% recall)
    DEFAULT_MODEL_PATH = 'models/drop_detector_xgb.pkl'

    # Feature weights from XGBoost importance analysis (used for rule fallback)
    DEFAULT_FEATURE_WEIGHTS = {
        'rms_change': 0.29,          # Most important - overall energy change
        'bass_change': 0.21,         # Bass change
        'bass_after': 0.11,          # Bass level after drop
        'mid_change': 0.11,          # Mid-frequency change
        'bass_before': 0.06,         # Bass level before drop
        'flux_before': 0.05,         # Spectral flux before
        'high_before': 0.05,         # High-freq before
        'high_change': 0.03,         # High-freq change
    }

    # Thresholds learned from XGBoost analysis
    DROP_THRESHOLDS = {
        'bass_change_min': 1.0,      # 100% increase (lowered from 200%)
        'rms_change_min': 0.3,       # 30% increase
        'centroid_change_max': -0.1,  # 10% decrease (softened)
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        detection_threshold: float = 0.5,
        window_sec: float = 3.0,
        use_rule_fallback: bool = True,
        auto_load_model: bool = True,
    ):
        """
        Initialize ML drop detector.

        Args:
            model_path: Path to trained XGBoost model (pickle)
            detection_threshold: Minimum probability to consider a drop
            window_sec: Analysis window before/after boundary
            use_rule_fallback: Use rule-based fallback if no model
            auto_load_model: If True, try to load default model if no model_path given
        """
        self.detection_threshold = detection_threshold
        self.window_sec = window_sec
        self.use_rule_fallback = use_rule_fallback

        self._model = None
        self._feature_cols = None
        self._essentia_available = False

        # Determine model path
        if model_path is None and auto_load_model:
            model_path = self.DEFAULT_MODEL_PATH

        self.model_path = model_path

        # Try to load model
        if model_path and Path(model_path).exists():
            try:
                with open(model_path, 'rb') as f:
                    saved = pickle.load(f)
                    if isinstance(saved, dict):
                        self._model = saved.get('model')
                        self._feature_cols = saved.get('feature_cols')
                    else:
                        self._model = saved
                logger.info(f"Loaded drop detection model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

        # Try to import essentia
        try:
            import essentia.standard as es
            self._essentia_available = True
            self._es = es
        except ImportError:
            logger.warning("Essentia not available, using librosa fallback")

    def execute(
        self,
        audio_context: AudioContext,
        phrase_boundaries: np.ndarray,
    ) -> DropDetectorMLResult:
        """
        Detect drops at phrase boundaries.

        Args:
            audio_context: Audio context with loaded audio
            phrase_boundaries: Array of phrase boundary times

        Returns:
            DropDetectorMLResult with detected drops
        """
        y = audio_context.y
        sr = audio_context.sr

        if len(phrase_boundaries) == 0:
            return DropDetectorMLResult()

        # Get tempo from beat grid if available, otherwise use default
        tempo = 130.0
        if audio_context.beat_grid is not None:
            tempo = audio_context.beat_grid.tempo

        # Extract features at each boundary (with tempo for bar-based windows)
        features = self._extract_features_at_boundaries(y, sr, phrase_boundaries, tempo=tempo)

        if len(features) == 0:
            return DropDetectorMLResult(n_phrase_boundaries=len(phrase_boundaries))

        # Classify each boundary
        drops = []

        if self._model is not None:
            # Use ML model
            drops = self._classify_with_model(features, phrase_boundaries)
        elif self.use_rule_fallback:
            # Use rule-based fallback
            drops = self._classify_with_rules(features, phrase_boundaries)

        return DropDetectorMLResult(
            drops=drops,
            n_phrase_boundaries=len(phrase_boundaries),
            detection_threshold=self.detection_threshold,
        )

    def execute_for_dj_set(
        self,
        audio_context: AudioContext,
        tempo: float = 132.0,
        beats_per_sample: int = 4,  # Sample every bar (4 beats) instead of phrase (16)
        min_drop_gap_sec: float = 30.0,  # Minimum gap between drops (~2 drops/min max)
    ) -> DropDetectorMLResult:
        """
        Detect drops in DJ set using dense sampling.

        In DJ sets, phrase boundaries from beat grid may not align with
        actual drops due to track layering during transitions. This method
        samples at every bar (4 beats) to catch drops that fall on any
        downbeat, not just phrase boundaries.

        Args:
            audio_context: Audio context with loaded audio
            tempo: Estimated BPM (default 132 for techno)
            beats_per_sample: Sample interval in beats (4=bar, 16=phrase)
            min_drop_gap_sec: Minimum time between drops to avoid duplicates

        Returns:
            DropDetectorMLResult with detected drops
        """
        y = audio_context.y
        sr = audio_context.sr
        duration = len(y) / sr

        # Calculate sample interval
        beat_duration = 60.0 / tempo
        sample_interval = beat_duration * beats_per_sample

        # Generate dense grid of potential drop points
        sample_times = np.arange(sample_interval, duration - sample_interval, sample_interval)

        logger.info(f"DJ Set mode: sampling every {beats_per_sample} beats "
                   f"({len(sample_times)} points at {tempo:.1f} BPM)")

        if len(sample_times) == 0:
            return DropDetectorMLResult()

        # Extract features at each sample point (with tempo for bar-based windows)
        features = self._extract_features_at_boundaries(y, sr, sample_times, tempo=tempo)

        if len(features) == 0:
            return DropDetectorMLResult(n_phrase_boundaries=len(sample_times))

        # Classify
        if self._model is not None:
            all_drops = self._classify_with_model(features, sample_times)
        elif self.use_rule_fallback:
            all_drops = self._classify_with_rules(features, sample_times)
        else:
            all_drops = []

        # Post-process 1: validate drops have proper valley (≥4 bars)
        validated_drops = self._validate_drops_have_valley(
            all_drops, features, min_valley_bars=4
        )

        # Post-process 2: remove drops too close together (NMS)
        if len(validated_drops) > 1 and min_drop_gap_sec > 0:
            drops = self._filter_close_drops(validated_drops, min_drop_gap_sec)
        else:
            drops = validated_drops

        logger.info(f"DJ Set drops: {len(drops)} after validation "
                   f"(raw={len(all_drops)}, with_valley={len(validated_drops)})")

        return DropDetectorMLResult(
            drops=drops,
            n_phrase_boundaries=len(sample_times),
            detection_threshold=self.detection_threshold,
        )

    def _validate_drops_have_valley(
        self,
        drops: List['DetectedDrop'],
        features: List[dict],
        min_valley_bars: int = 4,
        valley_threshold: float = 0.1,  # valley_depth must be > this
    ) -> List['DetectedDrop']:
        """
        Validate that drops have a proper valley (breakdown) before them.

        Rule: A real drop MUST have ≥4 bars of low energy ("valley") before it.
        If the valley is shorter, it's just an energy spike, not a real drop.

        Args:
            drops: Detected drops to validate
            features: Feature dicts (must have 'valley_depth' and 'phrase_idx')
            min_valley_bars: Minimum valley length (default 4 bars)
            valley_threshold: Minimum valley_depth to count as valley

        Returns:
            List of validated drops
        """
        if not drops or not features:
            return drops

        # Build feature lookup by phrase_idx
        feature_by_idx = {f['phrase_idx']: f for f in features}

        validated = []
        for drop in drops:
            feat = feature_by_idx.get(drop.phrase_idx)
            if feat is None:
                continue

            valley_depth = feat.get('valley_depth', 0)
            drop_contrast = feat.get('drop_contrast', 0)

            # Validation rule:
            # - valley_depth > threshold means there IS a valley before
            # - The valley is measured over 4 bars (valley_window_samples)
            # - If valley_depth is low, the "valley" wasn't really there
            has_valley = valley_depth > valley_threshold
            has_explosion = drop_contrast > 0.3

            if has_valley and has_explosion:
                validated.append(drop)
            else:
                logger.debug(
                    f"Drop at {drop.time_sec:.1f}s rejected: "
                    f"valley_depth={valley_depth:.2f} (<{valley_threshold}), "
                    f"drop_contrast={drop_contrast:.2f}"
                )

        return validated

    def _filter_close_drops(
        self,
        drops: List['DetectedDrop'],
        min_gap_sec: float
    ) -> List['DetectedDrop']:
        """
        Filter drops using non-maximum suppression (NMS).

        For each window of min_gap_sec, keeps only the drop with highest confidence.
        This ensures we don't miss the actual drop if it has lower time index
        but higher confidence than adjacent false positives.

        Algorithm:
        1. Sort by confidence (descending)
        2. Greedily select highest confidence drop
        3. Suppress all drops within min_gap_sec window
        4. Repeat until no more drops

        This is similar to NMS in object detection.
        """
        if len(drops) <= 1:
            return drops

        # Sort by confidence descending (greedy: take best first)
        sorted_by_conf = sorted(drops, key=lambda d: -d.confidence)

        selected = []
        suppressed = set()  # indices of suppressed drops

        for i, drop in enumerate(sorted_by_conf):
            if i in suppressed:
                continue

            # Select this drop (highest remaining confidence)
            selected.append(drop)

            # Suppress all drops within min_gap_sec of this one
            for j, other in enumerate(sorted_by_conf):
                if j != i and j not in suppressed:
                    gap = abs(other.time_sec - drop.time_sec)
                    if gap < min_gap_sec:
                        suppressed.add(j)

        # Sort selected by time for output
        return sorted(selected, key=lambda d: d.time_sec)

    def _extract_features_at_boundaries(
        self,
        y: np.ndarray,
        sr: int,
        phrase_boundaries: np.ndarray,
        tempo: float = 130.0,  # BPM for musical units
    ) -> List[dict]:
        """
        Extract signal features at each phrase boundary (VECTORIZED).

        Windows are measured in musical units (bars) not seconds:
        - 1 bar = 4 beats = 60*4/BPM seconds
        - "before" window = 8 bars before drop
        - "valley" window = 4 bars immediately before drop
        - "after" window = 2 bars after drop

        Optimizations (M2 Apple Silicon):
        - Pre-compute filtered signals (bass/mid/high) ONCE
        - Use cumsum for O(1) window energy computation
        - Vectorize all operations with numpy broadcasting
        """
        from scipy.signal import butter, sosfilt

        # Ensure contiguous float32 for M2 optimization
        y = np.ascontiguousarray(y, dtype=np.float32)
        phrase_boundaries = np.ascontiguousarray(phrase_boundaries, dtype=np.float64)

        # Convert BPM to bar duration in samples
        beat_sec = 60.0 / tempo
        bar_sec = beat_sec * 4  # 4 beats per bar

        # Define windows in bars (musical units)
        before_bars = 8   # buildup zone
        valley_bars = 4   # breakdown/valley before drop
        after_bars = 2    # drop explosion
        short_bars = 0.5  # short window for transient detection

        before_samples = int(before_bars * bar_sec * sr)
        valley_samples = int(valley_bars * bar_sec * sr)
        after_samples = int(after_bars * bar_sec * sr)
        short_samples = int(short_bars * bar_sec * sr)

        # === STEP 1: Pre-compute filtered signals ONCE ===
        nyq = sr / 2

        # Bass filter (< 150 Hz)
        sos_bass = butter(4, 150 / nyq, btype='low', output='sos')
        y_bass = sosfilt(sos_bass, y).astype(np.float32)

        # High filter (> 3000 Hz)
        high_cutoff = min(3000, nyq * 0.9)
        sos_high = butter(4, high_cutoff / nyq, btype='high', output='sos')
        y_high = sosfilt(sos_high, y).astype(np.float32)

        # Mid filter (300-3000 Hz)
        mid_high = min(3000, nyq * 0.9)
        sos_mid = butter(4, [300 / nyq, mid_high / nyq], btype='band', output='sos')
        y_mid = sosfilt(sos_mid, y).astype(np.float32)

        # === STEP 2: Compute cumulative sum of squared signals for O(1) RMS ===
        # RMS over window [a, b] = sqrt(sum(x[a:b]**2) / (b-a))
        # Using cumsum: sum(x[a:b]**2) = cumsum[b] - cumsum[a]
        y_sq_cumsum = np.concatenate([[0], np.cumsum(y ** 2)])
        bass_sq_cumsum = np.concatenate([[0], np.cumsum(y_bass ** 2)])
        high_sq_cumsum = np.concatenate([[0], np.cumsum(y_high ** 2)])
        mid_sq_cumsum = np.concatenate([[0], np.cumsum(y_mid ** 2)])

        def rms_from_cumsum_vectorized(cumsum, starts, ends):
            """Fully vectorized O(1) RMS computation using cumulative sum."""
            # Clip to valid range
            starts = np.clip(starts, 0, len(cumsum) - 1)
            ends = np.clip(ends, 0, len(cumsum) - 1)
            # Compute RMS: sqrt((cumsum[end] - cumsum[start]) / (end - start))
            lengths = ends - starts
            lengths = np.maximum(lengths, 1)  # Avoid division by zero
            sums = cumsum[ends] - cumsum[starts]
            return np.sqrt(sums / lengths).astype(np.float32)

        # === STEP 3: Vectorized boundary processing ===
        pb_samples = (phrase_boundaries * sr).astype(np.int64)

        # Filter valid boundaries (enough samples before/after)
        min_sample = before_samples
        max_sample = len(y) - after_samples
        valid_mask = (pb_samples >= min_sample) & (pb_samples < max_sample)
        valid_indices = np.where(valid_mask)[0]
        valid_pb = pb_samples[valid_mask]

        if len(valid_indices) == 0:
            return []

        n_valid = len(valid_indices)

        # === FULLY VECTORIZED RMS computation (no loops!) ===
        # Window layout for drop detection:
        #   [----buildup (4 bars)----][----valley (4 bars)----][DROP][----after (2 bars)----]
        #   ^before_starts            ^valley_starts           ^valid_pb                    ^after_ends
        #
        before_starts = valid_pb - before_samples  # 8 bars before drop
        valley_starts = valid_pb - valley_samples  # 4 bars before drop (start of valley)
        after_ends = valid_pb + after_samples      # 2 bars after drop
        short_before_starts = valid_pb - short_samples

        # RMS for main windows (vectorized)
        # "before" = full 8 bars before drop (includes buildup + valley)
        rms_before = rms_from_cumsum_vectorized(y_sq_cumsum, before_starts, valid_pb)
        rms_after = rms_from_cumsum_vectorized(y_sq_cumsum, valid_pb, after_ends)
        rms_before_short = rms_from_cumsum_vectorized(y_sq_cumsum, short_before_starts, valid_pb)
        rms_after_short = rms_from_cumsum_vectorized(y_sq_cumsum, valid_pb, valid_pb + short_samples)

        bass_before = rms_from_cumsum_vectorized(bass_sq_cumsum, before_starts, valid_pb)
        bass_after = rms_from_cumsum_vectorized(bass_sq_cumsum, valid_pb, after_ends)

        high_before = rms_from_cumsum_vectorized(high_sq_cumsum, before_starts, valid_pb)
        high_after = rms_from_cumsum_vectorized(high_sq_cumsum, valid_pb, after_ends)

        mid_before = rms_from_cumsum_vectorized(mid_sq_cumsum, before_starts, valid_pb)
        mid_after = rms_from_cumsum_vectorized(mid_sq_cumsum, valid_pb, after_ends)

        # Valley = 4 bars immediately before drop (breakdown/silence)
        # Buildup = 4 bars before valley (where energy builds up)
        valley_rms = rms_from_cumsum_vectorized(y_sq_cumsum, valley_starts, valid_pb)
        buildup_rms = rms_from_cumsum_vectorized(y_sq_cumsum, before_starts, valley_starts)

        # === STEP 4: Vectorized change computation ===
        eps = 1e-6

        rms_change = (rms_after - rms_before) / (rms_before + eps)
        rms_change_short = (rms_after_short - rms_before_short) / (rms_before_short + eps)
        bass_change = (bass_after - bass_before) / (bass_before + eps)
        high_change = (high_after - high_before) / (high_before + eps)
        mid_change = (mid_after - mid_before) / (mid_before + eps)

        # Valley/buildup features
        valley_depth = (buildup_rms - valley_rms) / (buildup_rms + eps)
        drop_contrast = (rms_after - valley_rms) / (valley_rms + eps)
        buildup_ratio = buildup_rms / (rms_before + eps)
        has_buildup_pattern = ((valley_depth > 0.1) & (drop_contrast > 0.3)).astype(np.float32)

        # === STEP 5: Spectral centroid (simplified vectorized) ===
        # Use pre-computed STFT if available, otherwise skip for speed
        centroid_before = np.zeros(n_valid, dtype=np.float32)
        centroid_after = np.zeros(n_valid, dtype=np.float32)

        # Simple centroid approximation: ratio of high to total energy
        # This correlates with spectral centroid without FFT per boundary
        total_before = rms_before + eps
        total_after = rms_after + eps
        centroid_before = high_before / total_before  # Proxy for brightness
        centroid_after = high_after / total_after
        centroid_change = (centroid_after - centroid_before) / (centroid_before + eps)

        # === STEP 6: Build feature list ===
        features = []
        for i, orig_idx in enumerate(valid_indices):
            features.append({
                'time': phrase_boundaries[orig_idx],
                'phrase_idx': int(orig_idx),
                # Changes
                'rms_change': float(rms_change[i]),
                'rms_change_short': float(rms_change_short[i]),
                'bass_change': float(bass_change[i]),
                'high_change': float(high_change[i]),
                'mid_change': float(mid_change[i]),
                'flux_change': 0.0,  # Skipped for speed (low importance)
                'centroid_change': float(centroid_change[i]),
                # Absolute values
                'bass_before': float(bass_before[i]),
                'bass_after': float(bass_after[i]),
                'rms_before': float(rms_before[i]),
                'rms_after': float(rms_after[i]),
                'high_before': float(high_before[i]),
                'high_after': float(high_after[i]),
                'mid_before': float(mid_before[i]),
                'mid_after': float(mid_after[i]),
                'flux_before': 0.0,
                'flux_after': 0.0,
                'centroid_before': float(centroid_before[i]),
                'centroid_after': float(centroid_after[i]),
                # Valley/Buildup features
                'valley_rms': float(valley_rms[i]),
                'buildup_rms': float(buildup_rms[i]),
                'valley_depth': float(valley_depth[i]),
                'drop_contrast': float(drop_contrast[i]),
                'buildup_ratio': float(buildup_ratio[i]),
                'has_buildup_pattern': float(has_buildup_pattern[i]),
            })

        return features

    def _classify_with_model(
        self,
        features: List[dict],
        phrase_boundaries: np.ndarray,
    ) -> List[DetectedDrop]:
        """Classify using trained XGBoost model."""
        # Use feature columns from model if available, otherwise use default
        if self._feature_cols is not None:
            feature_cols = self._feature_cols
        else:
            feature_cols = [
                'bass_before', 'bass_after', 'bass_change',
                'rms_before', 'rms_after', 'rms_change',
                'centroid_before', 'centroid_after', 'centroid_change',
                'high_before', 'high_after', 'high_change',
                'mid_before', 'mid_after', 'mid_change',
                'flux_before', 'flux_after', 'flux_change',
            ]

        # Build feature matrix, using 0 for missing features
        X = np.array([[f.get(col, 0.0) for col in feature_cols] for f in features])
        probabilities = self._model.predict_proba(X)[:, 1]

        drops = []
        for i, (feat, prob) in enumerate(zip(features, probabilities)):
            if prob >= self.detection_threshold:
                drops.append(DetectedDrop(
                    time_sec=feat['time'],
                    confidence=float(prob),
                    bass_change=feat.get('bass_change', 0),
                    centroid_change=feat.get('centroid_change', 0),
                    rms_change=feat.get('rms_change', 0),
                    phrase_idx=feat['phrase_idx'],
                ))

        return drops

    def _classify_with_rules(
        self,
        features: List[dict],
        phrase_boundaries: np.ndarray,
    ) -> List[DetectedDrop]:
        """
        Rule-based classification when no model available.

        Uses weighted feature scores based on learned importance.
        """
        drops = []

        for feat in features:
            # Score based on feature weights
            score = 0.0

            # Bass change (normalized to 0-1 scale)
            # > 200% change = 1.0, 0% = 0
            bass_score = min(1.0, max(0.0, feat['bass_change'] / 5.0))
            score += bass_score * self.DEFAULT_FEATURE_WEIGHTS['bass_change']

            # Centroid change (negative = good for drop)
            # < -50% = 1.0, > 0% = 0
            centroid_score = min(1.0, max(0.0, -feat['centroid_change'] * 2))
            score += centroid_score * self.DEFAULT_FEATURE_WEIGHTS['centroid_change']

            # RMS change short
            rms_score = min(1.0, max(0.0, feat['rms_change_short']))
            score += rms_score * self.DEFAULT_FEATURE_WEIGHTS['rms_change_short']

            # High freq change (negative = good)
            high_score = min(1.0, max(0.0, -feat['high_change']))
            score += high_score * self.DEFAULT_FEATURE_WEIGHTS['high_change']

            # Normalize to 0-1
            confidence = min(1.0, score / sum(self.DEFAULT_FEATURE_WEIGHTS.values()))

            # Apply threshold
            if confidence >= self.detection_threshold:
                drops.append(DetectedDrop(
                    time_sec=feat['time'],
                    confidence=confidence,
                    bass_change=feat['bass_change'],
                    centroid_change=feat['centroid_change'],
                    rms_change=feat['rms_change'],
                    phrase_idx=feat['phrase_idx'],
                ))

        return drops

    # NOTE: train_model() has been removed.
    # Use src.training.trainers.DropDetectorTrainer instead.