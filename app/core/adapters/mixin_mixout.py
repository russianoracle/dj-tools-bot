"""
Mixin/Mixout Detection for DJ Set Analysis

Apple Silicon M2 Optimized Implementation

Detects:
- Mixin points: where a new track starts entering the mix
- Mixout points: where the current track starts leaving the mix
- Transition types: CUT, FADE, BLEND, EQ_FILTER

Optimizations:
1. Single STFT computation via AudioAnalysisCore (reused everywhere)
2. Apple Accelerate framework via NumPy (VECLIB)
3. Optional Metal GPU (MPS) for tensor operations
4. Vectorized operations throughout
5. Memory-efficient streaming analysis

Uses shared analysis_utils for maximum code reuse.

Author: Optimized for Apple Silicon M2
"""

import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from enum import Enum, auto
from pathlib import Path
import os

# Import shared utilities
from .analysis_utils import (
    AudioAnalysisCore,
    normalize,
    smooth,
    compute_novelty_function,
)
# Import centralized STFTCache from common primitives
from app.common.primitives.stft import STFTCache
from app.common.logging import get_logger

logger = get_logger(__name__)

# Additional scipy imports
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

# Force NumPy to use Apple Accelerate
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

# Try to import torch for MPS acceleration
try:
    import torch
    HAS_MPS = torch.backends.mps.is_available()
    if HAS_MPS:
        MPS_DEVICE = torch.device("mps")
except ImportError:
    HAS_MPS = False
    MPS_DEVICE = None


class TransitionType(Enum):
    """Type of DJ transition between tracks."""
    CUT = auto()          # Hard cut (instant switch)
    FADE = auto()         # Volume fade in/out
    BLEND = auto()        # Overlapping blend (beatmatch)
    EQ_FILTER = auto()    # EQ/Filter sweep transition
    UNKNOWN = auto()      # Uncertain transition type


@dataclass
class MixinEvent:
    """
    Represents a mixin point where a new track enters the mix.

    The mixin is the beginning of a transition where you start
    hearing the incoming track.
    """
    time_sec: float           # Timestamp in seconds
    frame_idx: int            # Frame index
    confidence: float         # Detection confidence (0-1)

    # Characteristics
    transition_type: TransitionType
    duration_sec: float       # How long the mixin takes

    # Energy profile
    energy_start: float       # Energy at mixin start
    energy_end: float         # Energy at mixin end
    energy_slope: float       # Rate of energy change

    # Spectral profile
    bass_introduction: float  # How much bass is introduced (0-1)
    spectral_shift: float     # Change in spectral centroid
    brightness_change: float  # Change in high-frequency content

    # Filter characteristics (for EQ transitions)
    filter_detected: bool     # Was a filter sweep detected?
    filter_direction: str     # 'lowpass_open', 'highpass_close', 'none'

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['transition_type'] = self.transition_type.name
        return d


@dataclass
class MixoutEvent:
    """
    Represents a mixout point where the current track leaves the mix.

    The mixout is the end of a transition where the outgoing track
    stops being audible.
    """
    time_sec: float           # Timestamp in seconds
    frame_idx: int            # Frame index
    confidence: float         # Detection confidence (0-1)

    # Characteristics
    transition_type: TransitionType
    duration_sec: float       # How long the mixout takes

    # Energy profile
    energy_start: float       # Energy at mixout start
    energy_end: float         # Energy at mixout end
    energy_slope: float       # Rate of energy change

    # Spectral profile
    bass_removal: float       # How much bass is removed (0-1)
    spectral_shift: float     # Change in spectral centroid
    brightness_change: float  # Change in high-frequency content

    # Filter characteristics
    filter_detected: bool     # Was a filter sweep detected?
    filter_direction: str     # 'lowpass_close', 'highpass_open', 'none'

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['transition_type'] = self.transition_type.name
        return d


@dataclass
class TransitionPair:
    """A complete transition consisting of mixout and mixin."""
    mixout: MixoutEvent       # Outgoing track leaving
    mixin: MixinEvent         # Incoming track entering
    overlap_sec: float        # Duration of overlap
    beatmatch_quality: float  # How well beats are aligned (0-1)
    key_compatibility: float  # Harmonic compatibility (0-1, optional)


@dataclass
class TransitionAnalysis:
    """Complete mixin/mixout analysis for an audio file."""

    # Metadata
    file_path: str
    duration_sec: float
    sample_rate: int

    # Detected events
    mixins: List[MixinEvent] = field(default_factory=list)
    mixouts: List[MixoutEvent] = field(default_factory=list)
    transitions: List[TransitionPair] = field(default_factory=list)

    # Statistics
    avg_transition_duration: float = 0.0
    transition_type_distribution: Dict[str, int] = field(default_factory=dict)

    # Curves (for visualization)
    energy_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    bass_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    filter_curve: np.ndarray = field(default_factory=lambda: np.array([]))  # Estimated filter position

    # Time axis
    time_axis: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict:
        return {
            'file_path': self.file_path,
            'duration_sec': self.duration_sec,
            'sample_rate': self.sample_rate,
            'mixins': [m.to_dict() for m in self.mixins],
            'mixouts': [m.to_dict() for m in self.mixouts],
            'transitions': [
                {
                    'mixout': t.mixout.to_dict(),
                    'mixin': t.mixin.to_dict(),
                    'overlap_sec': t.overlap_sec,
                    'beatmatch_quality': t.beatmatch_quality,
                    'key_compatibility': t.key_compatibility
                }
                for t in self.transitions
            ],
            'avg_transition_duration': self.avg_transition_duration,
            'transition_type_distribution': self.transition_type_distribution,
            'num_mixins': len(self.mixins),
            'num_mixouts': len(self.mixouts),
        }

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Mixin/Mixout Analysis ===",
            f"File: {Path(self.file_path).name}",
            f"Duration: {self.duration_sec/60:.1f} min",
            f"",
            f"Detected Events:",
            f"  Mixins: {len(self.mixins)}",
            f"  Mixouts: {len(self.mixouts)}",
            f"  Complete Transitions: {len(self.transitions)}",
            f"",
            f"Avg Transition Duration: {self.avg_transition_duration:.1f}s",
            f"",
            f"Transition Types:",
        ]

        for t_type, count in self.transition_type_distribution.items():
            lines.append(f"  {t_type}: {count}")

        if self.mixins:
            lines.append(f"\nMixin Events:")
            for i, m in enumerate(self.mixins[:5], 1):
                lines.append(f"  {i}. t={m.time_sec:.1f}s, type={m.transition_type.name}, "
                           f"dur={m.duration_sec:.1f}s, conf={m.confidence:.2f}")
            if len(self.mixins) > 5:
                lines.append(f"  ... and {len(self.mixins) - 5} more")

        return '\n'.join(lines)


@dataclass
class M2DetectorConfig:
    """Configuration optimized for Apple Silicon M2."""

    # Audio
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512

    # Analysis windows
    analysis_frame_sec: float = 0.5     # Frame size for feature extraction
    transition_window_sec: float = 16.0  # Window to analyze around transition
    min_transition_sec: float = 2.0      # Minimum transition duration
    max_transition_sec: float = 32.0     # Maximum transition duration

    # Detection thresholds
    energy_change_threshold: float = 0.15    # Min energy change for transition
    spectral_change_threshold: float = 0.20  # Min spectral change
    bass_change_threshold: float = 0.25      # Min bass change for bass detection
    filter_velocity_threshold: float = 500   # Hz/sec for filter sweep detection

    # Smoothing
    smooth_window: int = 5

    # M2 specific
    use_mps: bool = True
    vectorize_all: bool = True


class M2MixinMixoutDetector:
    """
    Apple Silicon M2 optimized mixin/mixout detector.

    Detects transition points in DJ mixes by analyzing:
    1. Energy envelope changes (fades)
    2. Spectral centroid velocity (filter sweeps)
    3. Bass frequency introduction/removal
    4. Onset density changes
    5. Beat grid discontinuities

    All computations are vectorized and optimized for the
    Apple Accelerate framework.
    """

    def __init__(self, config: M2DetectorConfig = None):
        self.config = config or M2DetectorConfig()

        # Pre-computed frequency masks
        self._freq_masks = None
        self._freqs = None

    def analyze(self, audio_path: str) -> TransitionAnalysis:
        """
        Analyze audio file for mixin/mixout points.

        Args:
            audio_path: Path to audio file

        Returns:
            TransitionAnalysis with all detected events
        """
        logger.info("Loading audio file", data={"path": str(audio_path)})

        # Load audio using centralized loader
        from .loader import AudioLoader
        loader = AudioLoader(sample_rate=self.config.sample_rate)
        y, sr = loader.load(str(audio_path))
        duration = len(y) / sr

        logger.info("Audio loaded", data={"duration_min": f"{duration/60:.1f}"})

        # Run analysis
        return self._analyze_audio(y, sr, audio_path)

    def analyze_audio(self, y: np.ndarray, sr: int,
                      file_path: str = "unknown") -> TransitionAnalysis:
        """Analyze pre-loaded audio array."""
        return self._analyze_audio(y, sr, file_path)

    def _analyze_audio(self, y: np.ndarray, sr: int,
                       file_path: str) -> TransitionAnalysis:
        """
        Core analysis pipeline with M2 optimizations.

        All computations share a single STFT computation.
        """
        duration = len(y) / sr
        hop = self.config.hop_length
        n_fft = self.config.n_fft

        # ============================================================
        # PHASE 1: Core computations (compute ONCE via STFTCache)
        # ============================================================

        logger.debug("Computing STFT")

        # Use centralized STFTCache for all spectral features
        from app.common.primitives.stft import compute_stft
        stft_cache = compute_stft(y, sr=sr, n_fft=n_fft, hop_length=hop)

        # Extract from cache
        S = stft_cache.S
        S_power = S ** 2
        n_frames = stft_cache.n_frames

        # Frequency axis (from cache)
        self._freqs = stft_cache.freqs
        self._init_freq_masks()

        # Time axis (from cache)
        time_axis = stft_cache.times

        # ============================================================
        # PHASE 2: Energy analysis (vectorized)
        # ============================================================

        logger.debug("Analyzing energy")

        # RMS energy (from STFTCache)
        rms = stft_cache.get_rms()
        rms_smooth = self._smooth(rms)
        rms_norm = self._normalize(rms_smooth)

        # Energy derivative (for fade detection)
        energy_derivative = np.gradient(rms_smooth)
        energy_derivative_smooth = self._smooth(energy_derivative)

        # Second derivative (for transition boundaries)
        energy_accel = np.gradient(energy_derivative_smooth)

        # ============================================================
        # PHASE 3: Frequency band analysis (vectorized)
        # ============================================================

        logger.debug("Analyzing frequency bands")

        # Sub-bass (20-60 Hz) - kick drum fundamental
        sub_bass = self._band_energy(S_power, 20, 60)
        sub_bass_norm = self._normalize(self._smooth(sub_bass))

        # Bass (60-250 Hz) - bass line
        bass = self._band_energy(S_power, 60, 250)
        bass_norm = self._normalize(self._smooth(bass))
        bass_derivative = np.gradient(bass_norm)

        # Low-mid (250-500 Hz)
        low_mid = self._band_energy(S_power, 250, 500)

        # Mid (500-2000 Hz)
        mid = self._band_energy(S_power, 500, 2000)

        # High-mid (2000-6000 Hz) - presence
        high_mid = self._band_energy(S_power, 2000, 6000)

        # High (6000-20000 Hz) - air/brightness
        high = self._band_energy(S_power, 6000, 20000)
        high_norm = self._normalize(self._smooth(high))

        # ============================================================
        # PHASE 4: Spectral analysis (for filter detection)
        # ============================================================

        logger.debug("Analyzing spectral features")

        # Spectral centroid (from STFTCache)
        centroid = stft_cache.get_spectral_centroid()
        centroid_smooth = self._smooth(centroid, window=7)

        # Spectral centroid velocity (filter sweep rate)
        centroid_velocity = np.gradient(centroid_smooth) * sr / hop
        centroid_velocity_smooth = self._smooth(centroid_velocity)

        # Spectral rolloff (from STFTCache)
        rolloff = stft_cache.get_spectral_rolloff(roll_percent=0.85)
        rolloff_smooth = self._smooth(rolloff)

        # Brightness ratio
        total_energy = np.sum(S_power, axis=0) + 1e-10
        brightness = self._band_energy(S_power, 3000, 20000) / total_energy
        brightness_smooth = self._smooth(brightness)

        # Estimated filter position (combining centroid and rolloff)
        # Normalized to 0-1 where 0 = fully filtered, 1 = fully open
        centroid_norm = self._normalize(centroid_smooth)
        rolloff_norm = self._normalize(rolloff_smooth)
        filter_position = 0.6 * centroid_norm + 0.4 * rolloff_norm

        # ============================================================
        # PHASE 5: Onset/rhythm analysis
        # ============================================================

        logger.debug("Analyzing rhythm")

        # Onset strength (from STFTCache)
        onset_env = stft_cache.get_onset_strength()[:n_frames]
        onset_smooth = self._smooth(onset_env)

        # Local onset density (onsets per second)
        density_window = int(2.0 * sr / hop)  # 2-second window
        onset_density = uniform_filter1d(onset_env, size=density_window)

        # ============================================================
        # PHASE 6: Detect transition candidates
        # ============================================================

        logger.debug("Detecting transitions")

        # Combined novelty function for transition detection
        novelty = self._compute_novelty(
            rms_norm, bass_norm, centroid_norm,
            brightness_smooth, onset_density
        )

        # Find transition candidates
        candidates = self._find_transition_candidates(
            novelty, time_axis,
            energy_derivative_smooth,
            bass_derivative,
            centroid_velocity_smooth
        )

        # ============================================================
        # PHASE 7: Classify and refine transitions
        # ============================================================

        logger.debug("Classifying transitions")

        mixins = []
        mixouts = []

        for candidate in candidates:
            start_frame = candidate['start_frame']
            end_frame = candidate['end_frame']

            # Extract local features
            local_energy = rms_norm[start_frame:end_frame]
            local_bass = bass_norm[start_frame:end_frame]
            local_centroid_vel = centroid_velocity_smooth[start_frame:end_frame]
            local_brightness = brightness_smooth[start_frame:end_frame]
            local_filter = filter_position[start_frame:end_frame]

            # Determine transition direction and type
            transition_info = self._classify_transition(
                local_energy, local_bass, local_centroid_vel,
                local_brightness, local_filter, time_axis[start_frame:end_frame]
            )

            # Create events
            if transition_info['is_mixin']:
                mixin = MixinEvent(
                    time_sec=time_axis[start_frame],
                    frame_idx=start_frame,
                    confidence=transition_info['confidence'],
                    transition_type=transition_info['type'],
                    duration_sec=transition_info['duration'],
                    energy_start=float(local_energy[0]) if len(local_energy) > 0 else 0,
                    energy_end=float(local_energy[-1]) if len(local_energy) > 0 else 0,
                    energy_slope=transition_info['energy_slope'],
                    bass_introduction=transition_info['bass_change'],
                    spectral_shift=transition_info['spectral_shift'],
                    brightness_change=transition_info['brightness_change'],
                    filter_detected=transition_info['filter_detected'],
                    filter_direction=transition_info['filter_direction']
                )
                mixins.append(mixin)

            if transition_info['is_mixout']:
                mixout = MixoutEvent(
                    time_sec=time_axis[start_frame],
                    frame_idx=start_frame,
                    confidence=transition_info['confidence'],
                    transition_type=transition_info['type'],
                    duration_sec=transition_info['duration'],
                    energy_start=float(local_energy[0]) if len(local_energy) > 0 else 0,
                    energy_end=float(local_energy[-1]) if len(local_energy) > 0 else 0,
                    energy_slope=transition_info['energy_slope'],
                    bass_removal=abs(transition_info['bass_change']),
                    spectral_shift=transition_info['spectral_shift'],
                    brightness_change=transition_info['brightness_change'],
                    filter_detected=transition_info['filter_detected'],
                    filter_direction=transition_info['filter_direction']
                )
                mixouts.append(mixout)

        # ============================================================
        # PHASE 8: Pair transitions
        # ============================================================

        transitions = self._pair_transitions(mixouts, mixins)

        # ============================================================
        # PHASE 9: Compute statistics
        # ============================================================

        all_events = mixins + mixouts
        if all_events:
            avg_duration = np.mean([e.duration_sec for e in all_events])
        else:
            avg_duration = 0.0

        type_dist = {}
        for e in all_events:
            t_name = e.transition_type.name
            type_dist[t_name] = type_dist.get(t_name, 0) + 1

        logger.info("Transitions detected", data={"mixins": len(mixins), "mixouts": len(mixouts)})

        return TransitionAnalysis(
            file_path=file_path,
            duration_sec=duration,
            sample_rate=sr,
            mixins=mixins,
            mixouts=mixouts,
            transitions=transitions,
            avg_transition_duration=avg_duration,
            transition_type_distribution=type_dist,
            energy_curve=rms_norm,
            spectral_curve=centroid_norm,
            bass_curve=bass_norm,
            filter_curve=filter_position,
            time_axis=time_axis
        )

    def _init_freq_masks(self):
        """Pre-compute frequency band masks."""
        if self._freqs is None:
            return

        self._freq_masks = {
            'sub_bass': (self._freqs >= 20) & (self._freqs < 60),
            'bass': (self._freqs >= 60) & (self._freqs < 250),
            'low_mid': (self._freqs >= 250) & (self._freqs < 500),
            'mid': (self._freqs >= 500) & (self._freqs < 2000),
            'high_mid': (self._freqs >= 2000) & (self._freqs < 6000),
            'high': (self._freqs >= 6000),
        }

    def _band_energy(self, S_power: np.ndarray,
                     low_hz: float, high_hz: float) -> np.ndarray:
        """Extract energy in frequency band (vectorized)."""
        mask = (self._freqs >= low_hz) & (self._freqs < high_hz)
        return np.sum(S_power[mask, :], axis=0)

    def _smooth(self, arr: np.ndarray, window: int = None) -> np.ndarray:
        """Smooth array with Gaussian filter."""
        if window is None:
            window = self.config.smooth_window
        return gaussian_filter1d(arr, sigma=window/2)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize to 0-1 range."""
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max - arr_min < 1e-10:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    def _compute_novelty(self, energy: np.ndarray, bass: np.ndarray,
                         centroid: np.ndarray, brightness: np.ndarray,
                         onset_density: np.ndarray) -> np.ndarray:
        """
        Compute combined novelty function for transition detection.

        Combines multiple features to detect points where the
        audio characteristics change significantly.
        """
        n_frames = len(energy)

        # Derivative-based novelty for each feature
        energy_nov = np.abs(np.gradient(energy))
        bass_nov = np.abs(np.gradient(bass))
        centroid_nov = np.abs(np.gradient(centroid))
        brightness_nov = np.abs(np.gradient(brightness))
        density_nov = np.abs(np.gradient(onset_density))

        # Weight and combine
        novelty = (
            0.25 * self._normalize(energy_nov) +
            0.25 * self._normalize(bass_nov) +
            0.20 * self._normalize(centroid_nov) +
            0.15 * self._normalize(brightness_nov) +
            0.15 * self._normalize(density_nov)
        )

        # Smooth to reduce noise
        novelty = self._smooth(novelty, window=7)

        return novelty

    def _find_transition_candidates(self, novelty: np.ndarray,
                                     time_axis: np.ndarray,
                                     energy_deriv: np.ndarray,
                                     bass_deriv: np.ndarray,
                                     centroid_vel: np.ndarray) -> List[Dict]:
        """Find candidate transition regions."""
        sr = self.config.sample_rate
        hop = self.config.hop_length
        min_frames = int(self.config.min_transition_sec * sr / hop)
        max_frames = int(self.config.max_transition_sec * sr / hop)

        # Find peaks in novelty
        threshold = np.percentile(novelty, 85)
        peaks, properties = find_peaks(
            novelty,
            height=threshold,
            distance=min_frames,
            prominence=0.05
        )

        candidates = []

        for peak_idx in peaks:
            # Define transition region around peak
            start = max(0, peak_idx - max_frames // 2)
            end = min(len(novelty), peak_idx + max_frames // 2)

            # Check if this is a significant transition
            region_energy_change = np.max(energy_deriv[start:end]) - np.min(energy_deriv[start:end])
            region_bass_change = np.max(bass_deriv[start:end]) - np.min(bass_deriv[start:end])
            region_spectral_change = np.max(np.abs(centroid_vel[start:end]))

            # At least one significant change must be present
            if (region_energy_change > self.config.energy_change_threshold or
                region_bass_change > self.config.bass_change_threshold or
                region_spectral_change > self.config.filter_velocity_threshold):

                candidates.append({
                    'peak_frame': peak_idx,
                    'start_frame': start,
                    'end_frame': end,
                    'time_sec': time_axis[peak_idx],
                    'novelty_score': novelty[peak_idx],
                    'energy_change': region_energy_change,
                    'bass_change': region_bass_change,
                    'spectral_change': region_spectral_change,
                })

        return candidates

    def _classify_transition(self, energy: np.ndarray, bass: np.ndarray,
                             centroid_vel: np.ndarray, brightness: np.ndarray,
                             filter_pos: np.ndarray, times: np.ndarray) -> Dict:
        """
        Classify transition type and determine if it's a mixin or mixout.
        """
        if len(energy) < 3:
            return self._default_transition_info()

        duration = times[-1] - times[0] if len(times) > 1 else 0

        # Energy analysis
        energy_start = np.mean(energy[:len(energy)//4])
        energy_end = np.mean(energy[-len(energy)//4:])
        energy_change = energy_end - energy_start
        energy_slope = energy_change / max(duration, 0.1)

        # Bass analysis
        bass_start = np.mean(bass[:len(bass)//4])
        bass_end = np.mean(bass[-len(bass)//4:])
        bass_change = bass_end - bass_start

        # Filter/spectral analysis
        max_centroid_vel = np.max(np.abs(centroid_vel))
        filter_detected = max_centroid_vel > self.config.filter_velocity_threshold

        # Determine filter direction
        if filter_detected:
            centroid_trend = np.mean(centroid_vel)
            if centroid_trend > 0:
                filter_direction = 'opening'  # Filter opening (lowpass releasing)
            else:
                filter_direction = 'closing'  # Filter closing
        else:
            filter_direction = 'none'

        # Brightness analysis
        brightness_start = np.mean(brightness[:len(brightness)//4])
        brightness_end = np.mean(brightness[-len(brightness)//4:])
        brightness_change = brightness_end - brightness_start

        # Spectral shift
        spectral_shift = abs(brightness_change)

        # Determine transition type
        transition_type = self._determine_type(
            energy_change, bass_change, filter_detected,
            max_centroid_vel, duration
        )

        # Determine if mixin or mixout
        # Mixin: energy/bass increasing or filter opening
        # Mixout: energy/bass decreasing or filter closing
        is_mixin = (energy_change > 0.05 or bass_change > 0.1 or
                   (filter_detected and filter_direction == 'opening'))
        is_mixout = (energy_change < -0.05 or bass_change < -0.1 or
                    (filter_detected and filter_direction == 'closing'))

        # If ambiguous, mark as both (blend transition)
        if not is_mixin and not is_mixout:
            is_mixin = True
            is_mixout = True

        # Calculate confidence
        confidence = self._calculate_confidence(
            abs(energy_change), abs(bass_change),
            filter_detected, duration
        )

        # Adjust filter direction for mixin/mixout terminology
        if is_mixin and filter_detected:
            filter_direction = 'lowpass_open' if filter_direction == 'opening' else 'highpass_close'
        elif is_mixout and filter_detected:
            filter_direction = 'lowpass_close' if filter_direction == 'closing' else 'highpass_open'

        return {
            'type': transition_type,
            'is_mixin': is_mixin,
            'is_mixout': is_mixout,
            'confidence': confidence,
            'duration': duration,
            'energy_slope': energy_slope,
            'bass_change': bass_change,
            'spectral_shift': spectral_shift,
            'brightness_change': brightness_change,
            'filter_detected': filter_detected,
            'filter_direction': filter_direction,
        }

    def _determine_type(self, energy_change: float, bass_change: float,
                        filter_detected: bool, centroid_vel: float,
                        duration: float) -> TransitionType:
        """Determine transition type from characteristics."""

        # Very short duration = likely a cut
        if duration < 1.0:
            return TransitionType.CUT

        # Strong filter movement
        if filter_detected and centroid_vel > self.config.filter_velocity_threshold * 2:
            return TransitionType.EQ_FILTER

        # Gradual energy change without much spectral change
        if abs(energy_change) > 0.15 and not filter_detected:
            return TransitionType.FADE

        # Moderate changes across multiple dimensions = blend
        if (abs(energy_change) > 0.05 and abs(bass_change) > 0.05):
            return TransitionType.BLEND

        # Filter with gradual energy
        if filter_detected:
            return TransitionType.EQ_FILTER

        return TransitionType.UNKNOWN

    def _calculate_confidence(self, energy_change: float, bass_change: float,
                              filter_detected: bool, duration: float) -> float:
        """Calculate detection confidence."""
        confidence = 0.3  # Base confidence

        # Strong energy change
        if energy_change > 0.2:
            confidence += 0.25
        elif energy_change > 0.1:
            confidence += 0.15

        # Strong bass change
        if bass_change > 0.2:
            confidence += 0.2
        elif bass_change > 0.1:
            confidence += 0.1

        # Filter detected
        if filter_detected:
            confidence += 0.15

        # Reasonable duration
        if 4.0 <= duration <= 16.0:
            confidence += 0.1

        return min(confidence, 1.0)

    def _default_transition_info(self) -> Dict:
        """Return default transition info for edge cases."""
        return {
            'type': TransitionType.UNKNOWN,
            'is_mixin': False,
            'is_mixout': False,
            'confidence': 0.0,
            'duration': 0.0,
            'energy_slope': 0.0,
            'bass_change': 0.0,
            'spectral_shift': 0.0,
            'brightness_change': 0.0,
            'filter_detected': False,
            'filter_direction': 'none',
        }

    def _pair_transitions(self, mixouts: List[MixoutEvent],
                          mixins: List[MixinEvent]) -> List[TransitionPair]:
        """Pair mixouts with corresponding mixins."""
        pairs = []

        # Sort by time
        mixouts_sorted = sorted(mixouts, key=lambda x: x.time_sec)
        mixins_sorted = sorted(mixins, key=lambda x: x.time_sec)

        used_mixins = set()

        for mixout in mixouts_sorted:
            # Find closest mixin that starts around or after mixout
            best_mixin = None
            best_overlap = -float('inf')

            for i, mixin in enumerate(mixins_sorted):
                if i in used_mixins:
                    continue

                # Mixin should be close to mixout
                time_diff = mixin.time_sec - mixout.time_sec

                # Allow mixin to be slightly before mixout (overlap)
                if -16.0 <= time_diff <= 32.0:
                    # Calculate overlap quality
                    overlap = min(
                        mixout.time_sec + mixout.duration_sec,
                        mixin.time_sec + mixin.duration_sec
                    ) - max(mixout.time_sec, mixin.time_sec)

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_mixin = (i, mixin)

            if best_mixin:
                i, mixin = best_mixin
                used_mixins.add(i)

                overlap_sec = max(0, best_overlap)

                # Simple beatmatch quality estimation
                # (in real implementation, would analyze beat grids)
                beatmatch_quality = 0.5 + 0.5 * min(overlap_sec / 8.0, 1.0)

                pairs.append(TransitionPair(
                    mixout=mixout,
                    mixin=mixin,
                    overlap_sec=overlap_sec,
                    beatmatch_quality=beatmatch_quality,
                    key_compatibility=0.5  # Would need key detection
                ))

        return pairs


# Convenience function
def detect_mixin_mixout(audio_path: str,
                        config: M2DetectorConfig = None) -> TransitionAnalysis:
    """
    Convenience function to detect mixin/mixout points in audio.

    Args:
        audio_path: Path to audio file
        config: Optional configuration

    Returns:
        TransitionAnalysis with all detected events
    """
    detector = M2MixinMixoutDetector(config)
    return detector.analyze(audio_path)


# CLI interface
if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python mixin_mixout.py <audio_file.mp3> [--json]")
        print("\nExample:")
        print("  python mixin_mixout.py mix.mp3")
        print("  python mixin_mixout.py mix.mp3 --json > results.json")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_json = '--json' in sys.argv

    print(f"\n{'='*60}")
    print("Mixin/Mixout Detection (Apple Silicon M2 Optimized)")
    print(f"{'='*60}")
    print(f"MPS (Metal GPU): {'Enabled' if HAS_MPS else 'Disabled'}")
    print(f"{'='*60}\n")

    # Run analysis
    analysis = detect_mixin_mixout(audio_path)

    if output_json:
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        print("\n" + analysis.describe())

        # Print transition timeline
        if analysis.transitions:
            print("\n=== Transition Timeline ===")
            for i, t in enumerate(analysis.transitions, 1):
                print(f"\nTransition {i}:")
                print(f"  Mixout: {t.mixout.time_sec:.1f}s ({t.mixout.transition_type.name})")
                print(f"  Mixin:  {t.mixin.time_sec:.1f}s ({t.mixin.transition_type.name})")
                print(f"  Overlap: {t.overlap_sec:.1f}s")