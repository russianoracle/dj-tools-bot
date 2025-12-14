"""
ML-based Buildup Detection Task.

Detects musical buildup zones before drops using XGBoost trained on signal features.

Buildup Structure in EDM (4-16 bars before drop):
- GROOVE: Normal track section (32-64 beats before drop)
- BUILDUP_EARLY: Tension starts building (32-64 beats before)
- BUILDUP_LATE: Peak tension, filters rising (16-32 beats before)
- PRE_DROP/BREAKDOWN: Maximum tension, often stripped (0-16 beats before)
- DROP: Energy release

Key Features from Analysis:
- bass_before: Low during buildup (0.02), high during groove (0.10)
- centroid_before: Rising during buildup (2400 → 3600 Hz)
- bass_change: Negative during early buildup, massive positive at drop (+13x)
- centroid_change: Positive during buildup (rising), negative at drop (-57%)

Architecture:
- Requires beat grid or phrase boundaries
- Analyzes signal at each phrase boundary looking back
- Returns buildup zones with confidence and phase classification
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from enum import Enum, auto
import pickle

from .base import BaseTask, AudioContext

logger = logging.getLogger(__name__)


class BuildupPhase(Enum):
    """Phase within buildup structure."""
    GROOVE = auto()        # Normal groove, no buildup
    BUILDUP_EARLY = auto() # Tension starting (32-64 beats before drop)
    BUILDUP_LATE = auto()  # Peak tension (16-32 beats before drop)
    PRE_DROP = auto()      # Breakdown/maximum tension (0-16 beats before)
    DROP = auto()          # Energy release point


@dataclass
class BuildupZone:
    """A detected buildup zone with phase classification."""
    start_time_sec: float
    end_time_sec: float      # Usually the drop time
    phase: BuildupPhase
    confidence: float

    # Signal characteristics
    bass_level: float        # Average bass energy
    centroid_level: float    # Average spectral centroid
    bass_change: float       # Bass change at boundary
    centroid_change: float   # Centroid change at boundary
    rms_level: float         # Average RMS energy

    # Structural info
    phrase_idx: int = 0      # Which phrase boundary
    beats_to_drop: int = 0   # Beats until next drop

    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec

    def to_dict(self) -> dict:
        return {
            'start_time_sec': self.start_time_sec,
            'end_time_sec': self.end_time_sec,
            'duration_sec': self.duration_sec,
            'phase': self.phase.name,
            'confidence': self.confidence,
            'bass_level': self.bass_level,
            'centroid_level': self.centroid_level,
            'bass_change': self.bass_change,
            'centroid_change': self.centroid_change,
            'rms_level': self.rms_level,
            'beats_to_drop': self.beats_to_drop,
        }


@dataclass
class BuildupDetectorResult:
    """Result of buildup detection."""
    zones: List[BuildupZone] = field(default_factory=list)
    n_phrase_boundaries: int = 0
    n_drops_found: int = 0

    def get_buildups_before_drop(self, drop_time: float, tolerance_sec: float = 1.0) -> List[BuildupZone]:
        """Get all buildup zones leading to a specific drop."""
        return [
            z for z in self.zones
            if abs(z.end_time_sec - drop_time) < tolerance_sec
            and z.phase != BuildupPhase.DROP
        ]

    def get_zones_by_phase(self, phase: BuildupPhase) -> List[BuildupZone]:
        """Get all zones of a specific phase."""
        return [z for z in self.zones if z.phase == phase]

    def get_complete_buildups(self) -> List[List[BuildupZone]]:
        """Get complete buildup sequences (early → late → pre_drop → drop)."""
        buildups = []
        drops = self.get_zones_by_phase(BuildupPhase.DROP)

        for drop in drops:
            sequence = self.get_buildups_before_drop(drop.end_time_sec) + [drop]
            # Sort by time
            sequence.sort(key=lambda z: z.start_time_sec)
            if len(sequence) > 1:  # At least buildup + drop
                buildups.append(sequence)

        return buildups

    def to_dict(self) -> dict:
        return {
            'n_zones': len(self.zones),
            'n_phrase_boundaries': self.n_phrase_boundaries,
            'n_drops_found': self.n_drops_found,
            'zones': [z.to_dict() for z in self.zones],
            'phase_counts': {
                phase.name: len(self.get_zones_by_phase(phase))
                for phase in BuildupPhase
            },
        }


class BuildupDetectorML(BaseTask):
    """
    ML-based buildup detector using XGBoost.

    Analyzes signal features at phrase boundaries and classifies each
    into buildup phases: groove, buildup_early, buildup_late, pre_drop, drop.

    Features used:
    - bass_before/after: Sub-bass energy levels
    - centroid_before/after: Spectral brightness
    - rms_before/after: Overall energy
    - bass_change, centroid_change, rms_change: Signal transitions

    Usage:
        # With known drop times (from DropDetectorML or ground truth)
        detector = BuildupDetectorML()
        result = detector.execute(audio_context, phrase_boundaries, drop_times)

        for zone in result.zones:
            print(f"{zone.phase.name} at {zone.start_time_sec:.1f}s")

        # Get complete buildup sequences
        for sequence in result.get_complete_buildups():
            phases = [z.phase.name for z in sequence]
            print(f"Sequence: {' → '.join(phases)}")
    """

    name = "buildup_detector_ml"

    # Feature thresholds learned from Josh Baker Boiler Room analysis
    PHASE_THRESHOLDS = {
        'groove': {
            'bass_before_min': 0.05,    # High bass in groove
            'centroid_before_max': 2000, # Lower brightness
        },
        'buildup_early': {
            'bass_before_max': 0.06,     # Bass drops
            'centroid_before_range': (2000, 3000),
            'bass_change_range': (-0.5, 0.2),
        },
        'buildup_late': {
            'bass_before_max': 0.04,     # Very low bass
            'centroid_before_range': (2500, 3500),
            'centroid_change_min': 0.0,  # Rising brightness
        },
        'pre_drop': {
            'bass_before_max': 0.03,     # Minimal bass
            'centroid_before_min': 2500,
            'bass_change_range': (0.5, 5.0),  # Bass starting to return
        },
        'drop': {
            'bass_change_min': 5.0,      # Massive bass increase
            'centroid_change_max': -0.3,  # Sound gets darker
        },
    }

    # Beats before drop for each phase (at 128-133 BPM, phrase = 16 beats)
    PHASE_BEAT_RANGES = {
        BuildupPhase.GROOVE: (64, 128),      # 4-8 phrases before
        BuildupPhase.BUILDUP_EARLY: (32, 64), # 2-4 phrases before
        BuildupPhase.BUILDUP_LATE: (16, 32),  # 1-2 phrases before
        BuildupPhase.PRE_DROP: (0, 16),       # 0-1 phrase before
        BuildupPhase.DROP: (0, 0),            # At the drop
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        window_sec: float = 3.0,
        use_rule_fallback: bool = True,
        tempo_bpm: float = 130.0,
    ):
        """
        Initialize buildup detector.

        Args:
            model_path: Path to trained XGBoost model (pickle)
            window_sec: Analysis window before/after boundary
            use_rule_fallback: Use rule-based fallback if no model
            tempo_bpm: Estimated tempo for beat calculations
        """
        self.model_path = model_path
        self.window_sec = window_sec
        self.use_rule_fallback = use_rule_fallback
        self.tempo_bpm = tempo_bpm

        self._model = None
        self._label_encoder = None

        # Try to load model
        if model_path and Path(model_path).exists():
            try:
                with open(model_path, 'rb') as f:
                    saved = pickle.load(f)
                    if isinstance(saved, dict):
                        self._model = saved.get('model')
                        self._label_encoder = saved.get('label_encoder')
                    else:
                        self._model = saved
                logger.info(f"Loaded buildup detection model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def execute(
        self,
        audio_context: AudioContext,
        phrase_boundaries: np.ndarray,
        drop_times: Optional[np.ndarray] = None,
    ) -> BuildupDetectorResult:
        """
        Detect buildup zones at phrase boundaries.

        Args:
            audio_context: Audio context with loaded audio
            phrase_boundaries: Array of phrase boundary times
            drop_times: Known drop times (for structural labeling)

        Returns:
            BuildupDetectorResult with detected zones
        """
        y = audio_context.y
        sr = audio_context.sr

        if len(phrase_boundaries) == 0:
            return BuildupDetectorResult()

        # Extract features at each boundary
        features = self._extract_features_at_boundaries(y, sr, phrase_boundaries)

        if len(features) == 0:
            return BuildupDetectorResult(n_phrase_boundaries=len(phrase_boundaries))

        # Add structural info if drop times provided
        if drop_times is not None and len(drop_times) > 0:
            features = self._add_structural_features(features, drop_times)

        # Classify each boundary
        if self._model is not None:
            zones = self._classify_with_model(features)
        elif self.use_rule_fallback:
            zones = self._classify_with_rules(features, drop_times)
        else:
            zones = []

        return BuildupDetectorResult(
            zones=zones,
            n_phrase_boundaries=len(phrase_boundaries),
            n_drops_found=len(drop_times) if drop_times is not None else 0,
        )

    def _extract_features_at_boundaries(
        self,
        y: np.ndarray,
        sr: int,
        phrase_boundaries: np.ndarray,
    ) -> List[dict]:
        """Extract signal features at each phrase boundary."""
        from scipy.signal import butter, filtfilt

        window_samples = int(self.window_sec * sr)
        short_window_samples = int(1.0 * sr)  # 1 second window for drop detection

        def get_bass_energy(segment, cutoff=150):
            if len(segment) < 100:
                return 0.0
            nyq = sr / 2
            b, a = butter(4, cutoff / nyq, btype='low')
            try:
                bass = filtfilt(b, a, segment)
                return float(np.sqrt(np.mean(bass**2)))
            except:
                return 0.0

        def get_high_energy(segment, cutoff=3000):
            if len(segment) < 100:
                return 0.0
            nyq = sr / 2
            if cutoff >= nyq:
                cutoff = nyq * 0.9
            b, a = butter(4, cutoff / nyq, btype='high')
            try:
                high = filtfilt(b, a, segment)
                return float(np.sqrt(np.mean(high**2)))
            except:
                return 0.0

        def get_rms(segment):
            if len(segment) < 100:
                return 0.0
            return float(np.sqrt(np.mean(segment**2)))

        def get_centroid(segment):
            """Approximate spectral centroid."""
            if len(segment) < 2048:
                return 0.0
            # Simple FFT-based centroid
            fft = np.abs(np.fft.rfft(segment[:2048]))
            freqs = np.fft.rfftfreq(2048, 1/sr)
            if fft.sum() > 0:
                return float(np.sum(freqs * fft) / np.sum(fft))
            return 0.0

        features = []

        for i, pb in enumerate(phrase_boundaries):
            pb_sample = int(pb * sr)

            # Get windows
            before_start = max(0, pb_sample - window_samples)
            after_end = min(len(y), pb_sample + window_samples)

            if pb_sample - before_start < window_samples // 2:
                continue
            if after_end - pb_sample < window_samples // 2:
                continue

            audio_before = y[before_start:pb_sample]
            audio_after = y[pb_sample:after_end]

            # Extract features
            bass_before = get_bass_energy(audio_before)
            bass_after = get_bass_energy(audio_after)

            high_before = get_high_energy(audio_before)
            high_after = get_high_energy(audio_after)

            rms_before = get_rms(audio_before)
            rms_after = get_rms(audio_after)

            centroid_before = get_centroid(audio_before)
            centroid_after = get_centroid(audio_after)

            # Short window analysis (better for drop detection)
            short_before_start = max(0, pb_sample - short_window_samples)
            short_after_end = min(len(y), pb_sample + short_window_samples)
            audio_before_short = y[short_before_start:pb_sample]
            audio_after_short = y[pb_sample:short_after_end]

            bass_before_short = get_bass_energy(audio_before_short) if len(audio_before_short) > 100 else bass_before
            bass_after_short = get_bass_energy(audio_after_short) if len(audio_after_short) > 100 else bass_after

            # Compute changes
            eps = 1e-6
            bass_change = (bass_after - bass_before) / (bass_before + eps)
            bass_change_short = (bass_after_short - bass_before_short) / (bass_before_short + eps)
            high_change = (high_after - high_before) / (high_before + eps)
            rms_change = (rms_after - rms_before) / (rms_before + eps)
            centroid_change = (centroid_after - centroid_before) / (centroid_before + eps)

            features.append({
                'time': pb,
                'phrase_idx': i,
                'bass_before': bass_before,
                'bass_after': bass_after,
                'bass_change': bass_change,
                'bass_change_short': bass_change_short,
                'high_before': high_before,
                'high_after': high_after,
                'high_change': high_change,
                'rms_before': rms_before,
                'rms_after': rms_after,
                'rms_change': rms_change,
                'centroid_before': centroid_before,
                'centroid_after': centroid_after,
                'centroid_change': centroid_change,
            })

        return features

    def _add_structural_features(
        self,
        features: List[dict],
        drop_times: np.ndarray,
    ) -> List[dict]:
        """Add structural features based on known drop times."""
        beat_duration = 60.0 / self.tempo_bpm

        for feat in features:
            # Find closest drop AFTER this boundary
            future_drops = drop_times[drop_times > feat['time']]
            if len(future_drops) > 0:
                next_drop = future_drops[0]
                time_to_drop = next_drop - feat['time']
                beats_to_drop = int(time_to_drop / beat_duration)
            else:
                beats_to_drop = 999  # No upcoming drop

            feat['beats_to_drop'] = beats_to_drop
            feat['time_to_drop'] = beats_to_drop * beat_duration if beats_to_drop < 999 else float('inf')

        return features

    def _classify_with_model(self, features: List[dict]) -> List[BuildupZone]:
        """Classify using trained XGBoost model."""
        feature_cols = [
            'bass_before', 'bass_after', 'bass_change',
            'high_before', 'high_after', 'high_change',
            'rms_before', 'rms_after', 'rms_change',
            'centroid_before', 'centroid_after', 'centroid_change',
        ]

        X = np.array([[f.get(col, 0) for col in feature_cols] for f in features])

        if hasattr(self._model, 'predict_proba'):
            probabilities = self._model.predict_proba(X)
            predictions = self._model.predict(X)
        else:
            predictions = self._model.predict(X)
            probabilities = None

        zones = []
        for i, (feat, pred) in enumerate(zip(features, predictions)):
            # Decode label
            if self._label_encoder is not None:
                phase_name = self._label_encoder.inverse_transform([pred])[0]
            else:
                phase_name = pred

            try:
                phase = BuildupPhase[phase_name.upper()]
            except KeyError:
                phase = BuildupPhase.GROOVE

            # Get confidence
            if probabilities is not None:
                confidence = float(np.max(probabilities[i]))
            else:
                confidence = 0.8

            # Calculate end time (next boundary or +window)
            if i + 1 < len(features):
                end_time = features[i + 1]['time']
            else:
                end_time = feat['time'] + self.window_sec

            zones.append(BuildupZone(
                start_time_sec=feat['time'],
                end_time_sec=end_time,
                phase=phase,
                confidence=confidence,
                bass_level=feat['bass_before'],
                centroid_level=feat['centroid_before'],
                bass_change=feat['bass_change'],
                centroid_change=feat['centroid_change'],
                rms_level=feat['rms_before'],
                phrase_idx=feat['phrase_idx'],
                beats_to_drop=feat.get('beats_to_drop', 0),
            ))

        return zones

    def _classify_with_rules(
        self,
        features: List[dict],
        drop_times: Optional[np.ndarray] = None,
    ) -> List[BuildupZone]:
        """
        Rule-based classification using learned thresholds.

        Priority:
        1. Signal-based DROP detection (bass_change > 5.0)
        2. Structural position (beats to drop) when drops are known
        3. Pure signal analysis as fallback
        """
        zones = []

        # Create set of drop times for fast lookup
        drop_time_set = set()
        if drop_times is not None:
            for dt in drop_times:
                drop_time_set.add(round(dt, 1))

        for i, feat in enumerate(features):
            beats_to_drop = feat.get('beats_to_drop', 999)
            current_time = feat['time']

            # Check if this boundary IS a known drop (within 1 second tolerance)
            is_known_drop = any(abs(current_time - dt) < 1.0 for dt in drop_times) if drop_times is not None else False

            # PRIORITY 1: Signal-based DROP detection
            # Key features from XGBoost analysis (F1=0.844):
            #   1. bass_change: 46.2% importance - main feature
            #   2. onset_change: 16.4% - onset strength change
            #   3. rms_change: 11.7% - overall energy change
            #   4. bass_before: 9.5% - LOW bass before = classic drop, HIGH = soft drop
            #   5. mid_change: 6.8% - mid-frequency change

            bass_change = feat['bass_change']
            bass_change_short = feat.get('bass_change_short', bass_change)
            max_bass_change = max(bass_change, bass_change_short)
            rms_change = feat.get('rms_change', 0)
            rms_before = feat.get('rms_before', 0)
            bass_before = feat.get('bass_before', 0)
            centroid_change = feat.get('centroid_change', 0)
            high_change = feat.get('high_change', 0)

            # Compute drop score using weighted features
            # Analysis on Josh Baker 26 drops showed:
            #   - Classic drops: bass_change > 5, bass_before < 0.03
            #   - Soft drops: bass_before > 0.05, but rms_change > 0.3 or centroid_change < -0.2
            #   - All drops: some combination of energy increase signals

            drop_score = 0.0

            # 1. Bass change - primary signal (weight: 35%)
            # Normalized: bass_change of 5 = 0.175, 10 = 0.35, 20+ = 0.35
            if max_bass_change > 1.0:
                drop_score += min(0.35, max_bass_change / 20.0 * 0.35)

            # 2. RMS/energy change (weight: 20%)
            # rms_change of 0.5 = 0.10, 1.0+ = 0.20
            if rms_change > 0.3:
                drop_score += min(0.20, rms_change * 0.20)

            # 3. Bass before - breakdown indicator (weight: 15%)
            # Low bass_before = classic drop with breakdown
            if bass_before < 0.015:  # Very low - strong breakdown
                drop_score += 0.15
            elif bass_before < 0.03:  # Low - typical breakdown
                drop_score += 0.10
            elif bass_before < 0.05:  # Medium - partial breakdown
                drop_score += 0.05

            # 4. Centroid change - "sound gets darker" (weight: 15%)
            if centroid_change < -0.40:  # Strong darkening
                drop_score += 0.15
            elif centroid_change < -0.20:  # Moderate darkening
                drop_score += 0.10
            elif centroid_change < -0.10:  # Slight darkening
                drop_score += 0.05

            # 5. High-frequency change (weight: 10%)
            if high_change < -0.30:  # Strong high cut
                drop_score += 0.10
            elif high_change < -0.15:  # Moderate high cut
                drop_score += 0.05

            # 6. Combined RMS + bass for soft drops (bonus: 5%)
            # Soft drops have high bass_before but still energy increase
            if bass_before > 0.05 and rms_change > 0.5:
                drop_score += 0.05

            # DETECTION THRESHOLDS
            # Classic drop: bass_change > 5 (immediate detection)
            if max_bass_change > 5.0:
                phase = BuildupPhase.DROP
                confidence = min(1.0, 0.5 + max_bass_change / 20.0)
            # High confidence drop: combined score >= 0.30
            elif drop_score >= 0.30:
                phase = BuildupPhase.DROP
                confidence = min(1.0, 0.5 + drop_score)
            # Known drop with moderate signals (for calibration mode)
            elif is_known_drop and (max_bass_change > 2.0 or drop_score >= 0.20):
                phase = BuildupPhase.DROP
                confidence = min(1.0, 0.6 + drop_score)
            # PRIORITY 2: Structural classification (when drops are known)
            elif beats_to_drop < 999 and beats_to_drop >= 0:
                if beats_to_drop <= 16:
                    phase = BuildupPhase.PRE_DROP
                    confidence = 0.8 + (16 - beats_to_drop) / 80
                elif beats_to_drop <= 32:
                    phase = BuildupPhase.BUILDUP_LATE
                    confidence = 0.7 + (32 - beats_to_drop) / 160
                elif beats_to_drop <= 64:
                    phase = BuildupPhase.BUILDUP_EARLY
                    confidence = 0.6 + (64 - beats_to_drop) / 320
                else:
                    phase = BuildupPhase.GROOVE
                    confidence = 0.5
            else:
                # PRIORITY 3: Pure signal-based classification
                phase, confidence = self._classify_by_signal(feat)

            # Adjust confidence based on signal characteristics
            confidence = self._adjust_confidence_by_signal(phase, feat, confidence)

            # Calculate end time
            if i + 1 < len(features):
                end_time = features[i + 1]['time']
            else:
                end_time = feat['time'] + self.window_sec

            zones.append(BuildupZone(
                start_time_sec=feat['time'],
                end_time_sec=end_time,
                phase=phase,
                confidence=confidence,
                bass_level=feat['bass_before'],
                centroid_level=feat['centroid_before'],
                bass_change=feat['bass_change'],
                centroid_change=feat['centroid_change'],
                rms_level=feat['rms_before'],
                phrase_idx=feat['phrase_idx'],
                beats_to_drop=feat.get('beats_to_drop', 0),
            ))

        return zones

    def _classify_by_signal(self, feat: dict) -> Tuple[BuildupPhase, float]:
        """Classify phase using only signal features."""
        bass_change = feat['bass_change']
        centroid_change = feat['centroid_change']
        bass_before = feat['bass_before']
        centroid_before = feat['centroid_before']

        # Drop detection (highest priority)
        if bass_change > 5.0 and centroid_change < -0.3:
            return BuildupPhase.DROP, min(1.0, bass_change / 10.0)

        # Pre-drop: bass starting to return, high tension
        if 0.5 < bass_change < 5.0 and centroid_before > 2500:
            return BuildupPhase.PRE_DROP, 0.7

        # Buildup late: very low bass, high centroid, rising
        if bass_before < 0.04 and centroid_before > 2500 and centroid_change > 0:
            return BuildupPhase.BUILDUP_LATE, 0.65

        # Buildup early: low bass, moderate centroid
        if bass_before < 0.06 and 2000 < centroid_before < 3000:
            return BuildupPhase.BUILDUP_EARLY, 0.6

        # Default: groove
        return BuildupPhase.GROOVE, 0.5

    def _adjust_confidence_by_signal(
        self,
        phase: BuildupPhase,
        feat: dict,
        base_confidence: float,
    ) -> float:
        """Adjust confidence based on how well signal matches expected characteristics."""
        bass_before = feat['bass_before']
        centroid_before = feat['centroid_before']
        bass_change = feat['bass_change']
        centroid_change = feat['centroid_change']

        adjustment = 0.0

        if phase == BuildupPhase.DROP:
            # Drops should have massive bass increase
            if bass_change > 10.0:
                adjustment += 0.1
            if centroid_change < -0.5:
                adjustment += 0.05

        elif phase == BuildupPhase.PRE_DROP:
            # Pre-drop should have low bass, high tension
            if bass_before < 0.03:
                adjustment += 0.05
            if centroid_before > 2800:
                adjustment += 0.05

        elif phase == BuildupPhase.BUILDUP_LATE:
            # Late buildup should have minimal bass
            if bass_before < 0.03:
                adjustment += 0.05
            if centroid_change > 0.1:
                adjustment += 0.05

        elif phase == BuildupPhase.BUILDUP_EARLY:
            # Early buildup shows bass dropping
            if bass_change < -0.2:
                adjustment += 0.05

        return min(1.0, base_confidence + adjustment)

    # NOTE: train_model() has been removed.
    # Use src.training.trainers.BuildupDetectorTrainer instead.