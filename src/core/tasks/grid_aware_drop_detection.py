"""
Grid-Aware Drop Detection Task.

Combines XGBoost ML model with beat grid and buildup awareness for
accurate drop detection in DJ sets.

Architecture (SOLID/GRASP compliant):
- Single Responsibility: Only detects drops, delegates to primitives
- Open/Closed: Extensible via strategy pattern for different detection modes
- Dependency Inversion: Depends on abstractions (BaseTask, AudioContext)

Features:
1. XGBoost-based classification on audio features
2. Beat grid alignment (drops happen on bar/phrase boundaries)
3. Buildup detection (true drops have preceding buildup phase)
4. Confidence scoring with multiple signals

Usage:
    from src.core.tasks import GridAwareDropDetectionTask

    detector = GridAwareDropDetectionTask(
        model_path='models/drop_detector_xgb.pkl',
        beat_grid=beat_grid_result,  # From BeatGridTask
    )
    result = detector.execute(audio_context)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
from enum import Enum, auto
import pickle

from .base import BaseTask, AudioContext

logger = logging.getLogger(__name__)


class DropType(Enum):
    """Type of drop based on buildup pattern."""
    CLASSIC = auto()     # Full breakdown before drop (bass_before < 0.02)
    SOFT = auto()        # Partial buildup (bass_before 0.02-0.05)
    GROOVE_TO_GROOVE = auto()  # No breakdown (bass_before > 0.05)


@dataclass
class GridAlignedDrop:
    """A detected drop with grid and buildup info."""
    time_sec: float
    confidence: float
    drop_type: DropType

    # ML features
    bass_change: float
    rms_change: float

    # Grid alignment
    grid_aligned: bool = False
    beat_position: int = 0  # Beat within phrase (1-16)

    # Buildup info
    has_buildup: bool = False
    buildup_bass: float = 0.0  # Average bass in buildup zone
    buildup_score: int = 0     # 0-3 buildup indicators

    def to_dict(self) -> dict:
        return {
            'time_sec': self.time_sec,
            'confidence': self.confidence,
            'drop_type': self.drop_type.name,
            'bass_change': self.bass_change,
            'rms_change': self.rms_change,
            'grid_aligned': self.grid_aligned,
            'beat_position': self.beat_position,
            'has_buildup': self.has_buildup,
            'buildup_score': self.buildup_score,
        }


@dataclass
class GridAwareDropResult:
    """Result of grid-aware drop detection."""
    drops: List[GridAlignedDrop] = field(default_factory=list)
    n_candidates: int = 0
    n_filtered: int = 0
    processing_time_sec: float = 0.0

    def get_drop_times(self) -> np.ndarray:
        """Get array of drop times."""
        return np.array([d.time_sec for d in self.drops]) if self.drops else np.array([])

    def get_classic_drops(self) -> List[GridAlignedDrop]:
        """Get only classic drops with full breakdown."""
        return [d for d in self.drops if d.drop_type == DropType.CLASSIC]

    def get_high_confidence_drops(self, min_conf: float = 0.8) -> List[GridAlignedDrop]:
        """Get drops above confidence threshold."""
        return [d for d in self.drops if d.confidence >= min_conf]

    def to_dict(self) -> dict:
        return {
            'n_drops': len(self.drops),
            'n_candidates': self.n_candidates,
            'n_filtered': self.n_filtered,
            'processing_time_sec': self.processing_time_sec,
            'drops': [d.to_dict() for d in self.drops],
            'drop_types': {
                t.name: len([d for d in self.drops if d.drop_type == t])
                for t in DropType
            }
        }


class GridAwareDropDetectionTask(BaseTask):
    """
    Grid-aware drop detection combining ML with musical structure.

    Detection pipeline:
    1. Extract features at beat grid boundaries
    2. Run XGBoost model to get initial candidates
    3. Filter by buildup presence (true drops have buildup phase)
    4. Adjust confidence based on grid alignment

    Args:
        model_path: Path to XGBoost model pickle
        beat_grid: Optional beat grid result for alignment
        detection_threshold: Minimum ML probability (default 0.5)
        require_buildup: If True, filter out drops without buildup
        window_sec: Feature extraction window size
    """

    name = "grid_aware_drop_detection"

    # Default model path
    DEFAULT_MODEL_PATH = 'models/drop_detector_xgb.pkl'

    # Buildup detection thresholds
    BUILDUP_BASS_THRESHOLD = 0.03  # Low bass = breakdown
    BUILDUP_WINDOW_SEC = 8.0       # How far back to check for buildup

    def __init__(
        self,
        model_path: Optional[str] = None,
        beat_grid: Optional[any] = None,
        detection_threshold: float = 0.5,
        require_buildup: bool = False,
        window_sec: float = 3.0,
    ):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.beat_grid = beat_grid
        self.detection_threshold = detection_threshold
        self.require_buildup = require_buildup
        self.window_sec = window_sec

        self._model = None
        self._feature_cols = None

        # Load model
        if Path(self.model_path).exists():
            try:
                with open(self.model_path, 'rb') as f:
                    saved = pickle.load(f)
                    if isinstance(saved, dict):
                        self._model = saved.get('model')
                        self._feature_cols = saved.get('feature_cols')
                    else:
                        self._model = saved
                logger.info(f"Loaded model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def execute(
        self,
        audio_context: AudioContext,
        boundaries: Optional[np.ndarray] = None,
    ) -> GridAwareDropResult:
        """
        Detect drops in audio.

        Args:
            audio_context: Audio context with loaded audio
            boundaries: Analysis boundaries (default: use beat_grid or generate)

        Returns:
            GridAwareDropResult with detected drops
        """
        y = audio_context.y
        sr = audio_context.sr

        # Report start
        audio_context.report_progress("drop_detection", 0.0, "Starting drop detection...")

        # Get analysis boundaries
        if boundaries is None:
            if self.beat_grid is not None:
                # Use beat grid phrase boundaries
                boundaries = self.beat_grid.get_phrase_boundaries()
            else:
                # Generate uniform boundaries (every 4 beats at 130 BPM)
                beat_duration = 60.0 / 130.0
                boundaries = np.arange(0, len(y)/sr, beat_duration * 4)

        if len(boundaries) == 0:
            audio_context.report_progress("drop_detection", 1.0, "No boundaries found")
            return GridAwareDropResult()

        audio_context.report_progress("drop_detection", 0.1, f"Analyzing {len(boundaries)} boundaries...")

        # Extract features at boundaries (vectorized)
        features = self._extract_features_vectorized(y, sr, boundaries)

        audio_context.report_progress("drop_detection", 0.5, "Features extracted, running ML...")

        if len(features) == 0:
            audio_context.report_progress("drop_detection", 1.0, "No features extracted")
            return GridAwareDropResult(n_candidates=len(boundaries))

        # Get ML predictions
        candidates = self._get_ml_candidates(features, boundaries)
        n_candidates = len(candidates)

        audio_context.report_progress("drop_detection", 0.7, f"Found {n_candidates} candidates, analyzing buildups...")

        # Add buildup info and filter
        drops = self._add_buildup_info(y, sr, candidates)

        if self.require_buildup:
            drops = [d for d in drops if d.has_buildup]

        audio_context.report_progress("drop_detection", 0.9, f"Filtering: {len(drops)} drops after buildup filter")

        # Add grid alignment info
        if self.beat_grid is not None:
            drops = self._add_grid_alignment(drops)

        audio_context.report_progress("drop_detection", 1.0, f"Done: {len(drops)} drops detected")

        return GridAwareDropResult(
            drops=drops,
            n_candidates=n_candidates,
            n_filtered=n_candidates - len(drops),
        )

    def _extract_features_vectorized(
        self,
        y: np.ndarray,
        sr: int,
        boundaries: np.ndarray,
    ) -> List[dict]:
        """Extract features at boundaries using vectorized operations."""
        from scipy.signal import butter, filtfilt

        window_samples = int(self.window_sec * sr)

        # Pre-compute filter coefficients
        nyq = sr / 2
        b_bass, a_bass = butter(4, 150 / nyq, btype='low')
        b_high, a_high = butter(4, min(3000, nyq * 0.9) / nyq, btype='high')
        b_mid, a_mid = butter(4, [300/nyq, min(3000, nyq * 0.9)/nyq], btype='band')

        features = []

        # Process in batches for memory efficiency
        for pb in boundaries:
            pb_sample = int(pb * sr)

            before_start = max(0, pb_sample - window_samples)
            after_end = min(len(y), pb_sample + window_samples)

            if pb_sample - before_start < window_samples // 2:
                continue
            if after_end - pb_sample < window_samples // 2:
                continue

            audio_before = y[before_start:pb_sample]
            audio_after = y[pb_sample:after_end]

            # Extract features
            feat = self._compute_features(
                audio_before, audio_after, sr,
                b_bass, a_bass, b_high, a_high, b_mid, a_mid
            )
            feat['time'] = pb
            features.append(feat)

        return features

    def _compute_features(
        self,
        before: np.ndarray,
        after: np.ndarray,
        sr: int,
        b_bass, a_bass, b_high, a_high, b_mid, a_mid,
    ) -> dict:
        """Compute all features for a single boundary."""
        from scipy.signal import filtfilt

        eps = 1e-6

        # RMS
        rms_before = np.sqrt(np.mean(before**2))
        rms_after = np.sqrt(np.mean(after**2))

        # Bass
        bass_before = np.sqrt(np.mean(filtfilt(b_bass, a_bass, before)**2))
        bass_after = np.sqrt(np.mean(filtfilt(b_bass, a_bass, after)**2))

        # High
        high_before = np.sqrt(np.mean(filtfilt(b_high, a_high, before)**2))
        high_after = np.sqrt(np.mean(filtfilt(b_high, a_high, after)**2))

        # Mid
        mid_before = np.sqrt(np.mean(filtfilt(b_mid, a_mid, before)**2))
        mid_after = np.sqrt(np.mean(filtfilt(b_mid, a_mid, after)**2))

        # Centroid
        def centroid(seg):
            if len(seg) < 2048:
                return 0.0
            fft = np.abs(np.fft.rfft(seg[:2048]))
            freqs = np.fft.rfftfreq(2048, 1/sr)
            return np.sum(freqs * fft) / (np.sum(fft) + eps)

        centroid_before = centroid(before)
        centroid_after = centroid(after)

        # Spectral flux
        def flux(seg):
            if len(seg) < 4096:
                return 0.0
            hop = 512
            n_frames = (len(seg) - 2048) // hop
            if n_frames < 2:
                return 0.0
            fluxes = []
            prev = None
            for i in range(n_frames):
                start = i * hop
                spec = np.abs(np.fft.rfft(seg[start:start+2048]))
                if prev is not None:
                    fluxes.append(np.sum(np.maximum(0, spec - prev)))
                prev = spec
            return np.mean(fluxes) if fluxes else 0.0

        flux_before = flux(before)
        flux_after = flux(after)

        return {
            'bass_before': float(bass_before),
            'bass_after': float(bass_after),
            'bass_change': float((bass_after - bass_before) / (bass_before + eps)),
            'rms_before': float(rms_before),
            'rms_after': float(rms_after),
            'rms_change': float((rms_after - rms_before) / (rms_before + eps)),
            'high_before': float(high_before),
            'high_after': float(high_after),
            'high_change': float((high_after - high_before) / (high_before + eps)),
            'mid_before': float(mid_before),
            'mid_after': float(mid_after),
            'mid_change': float((mid_after - mid_before) / (mid_before + eps)),
            'centroid_before': float(centroid_before),
            'centroid_after': float(centroid_after),
            'centroid_change': float((centroid_after - centroid_before) / (centroid_before + eps)),
            'flux_before': float(flux_before),
            'flux_after': float(flux_after),
            'flux_change': float((flux_after - flux_before) / (flux_before + eps)),
        }

    def _get_ml_candidates(
        self,
        features: List[dict],
        boundaries: np.ndarray,
    ) -> List[GridAlignedDrop]:
        """Get drop candidates using ML model."""
        if self._model is None:
            # Fall back to rule-based
            return self._get_rule_candidates(features)

        # Get feature columns
        if self._feature_cols is None:
            self._feature_cols = [
                'bass_before', 'bass_after', 'bass_change',
                'rms_before', 'rms_after', 'rms_change',
                'centroid_before', 'centroid_after', 'centroid_change',
                'high_before', 'high_after', 'high_change',
                'mid_before', 'mid_after', 'mid_change',
                'flux_before', 'flux_after', 'flux_change',
            ]

        # Build feature matrix
        X = np.array([[f.get(col, 0.0) for col in self._feature_cols] for f in features])

        # Get predictions
        proba = self._model.predict_proba(X)[:, 1]

        candidates = []
        for feat, prob in zip(features, proba):
            if prob >= self.detection_threshold:
                # Determine drop type based on bass_before
                bass_before = feat['bass_before']
                if bass_before < 0.02:
                    drop_type = DropType.CLASSIC
                elif bass_before < 0.05:
                    drop_type = DropType.SOFT
                else:
                    drop_type = DropType.GROOVE_TO_GROOVE

                candidates.append(GridAlignedDrop(
                    time_sec=feat['time'],
                    confidence=float(prob),
                    drop_type=drop_type,
                    bass_change=feat['bass_change'],
                    rms_change=feat['rms_change'],
                ))

        return candidates

    def _get_rule_candidates(self, features: List[dict]) -> List[GridAlignedDrop]:
        """Fallback rule-based detection."""
        candidates = []

        for feat in features:
            bass_change = feat['bass_change']
            rms_change = feat['rms_change']
            bass_before = feat['bass_before']

            # Simple rules
            score = 0.0
            if bass_change > 5.0:
                score = 0.9
            elif bass_change > 2.0 and rms_change > 0.5:
                score = 0.7
            elif rms_change > 1.0:
                score = 0.5

            if score >= self.detection_threshold:
                if bass_before < 0.02:
                    drop_type = DropType.CLASSIC
                elif bass_before < 0.05:
                    drop_type = DropType.SOFT
                else:
                    drop_type = DropType.GROOVE_TO_GROOVE

                candidates.append(GridAlignedDrop(
                    time_sec=feat['time'],
                    confidence=score,
                    drop_type=drop_type,
                    bass_change=bass_change,
                    rms_change=rms_change,
                ))

        return candidates

    def _add_buildup_info(
        self,
        y: np.ndarray,
        sr: int,
        candidates: List[GridAlignedDrop],
    ) -> List[GridAlignedDrop]:
        """Add buildup detection info to candidates."""
        from scipy.signal import butter, filtfilt

        buildup_samples = int(self.BUILDUP_WINDOW_SEC * sr)

        # Pre-compute bass filter
        nyq = sr / 2
        b, a = butter(4, 150 / nyq, btype='low')

        for drop in candidates:
            drop_sample = int(drop.time_sec * sr)
            buildup_start = max(0, drop_sample - buildup_samples)

            audio_buildup = y[buildup_start:drop_sample]

            if len(audio_buildup) < 100:
                continue

            # Compute buildup bass
            bass_buildup = np.sqrt(np.mean(filtfilt(b, a, audio_buildup)**2))
            rms_buildup = np.sqrt(np.mean(audio_buildup**2))

            # Buildup indicators
            has_low_bass = bass_buildup < self.BUILDUP_BASS_THRESHOLD
            has_energy_drop = rms_buildup < drop.rms_change * 0.5 if drop.rms_change > 0 else False
            has_classic = drop.drop_type == DropType.CLASSIC

            buildup_score = sum([has_low_bass, has_energy_drop, has_classic])

            drop.buildup_bass = float(bass_buildup)
            drop.buildup_score = buildup_score
            drop.has_buildup = buildup_score >= 1

            # Adjust confidence based on buildup
            if drop.has_buildup:
                drop.confidence = min(1.0, drop.confidence * 1.1)
            else:
                drop.confidence *= 0.8

        return candidates

    def _add_grid_alignment(self, drops: List[GridAlignedDrop]) -> List[GridAlignedDrop]:
        """Add beat grid alignment info."""
        if self.beat_grid is None:
            return drops

        for drop in drops:
            # Check if on phrase boundary
            phrase_boundaries = self.beat_grid.get_phrase_boundaries()
            min_dist = np.min(np.abs(phrase_boundaries - drop.time_sec))

            drop.grid_aligned = min_dist < 1.0

            # Get beat position
            if hasattr(self.beat_grid, 'time_to_phrase_position'):
                pos = self.beat_grid.time_to_phrase_position(drop.time_sec)
                drop.beat_position = pos[2] if pos else 0

            # Boost confidence for grid-aligned drops
            if drop.grid_aligned:
                drop.confidence = min(1.0, drop.confidence * 1.05)

        return drops
