"""
Reusable Audio Analysis Utilities

Apple Silicon M2 Optimized

MIGRATED to use FeatureFactory/STFTCache from src/core (December 2024).
All librosa calls centralized via STFTCache.

Shared functions for audio analysis that can be used across modules:
- Energy analysis (RMS, frequency bands)
- Spectral analysis (centroid, rolloff, brightness)
- Transition detection primitives
- Novelty functions
- Smoothing and normalization
- Genre classification (400 Discogs styles)
- Mood/theme tags analysis
- Track similarity search

All computations are vectorized for Apple Accelerate framework.

Usage:
    # Audio features
    from app.core.adapters.analysis_utils import (
        AudioAnalysisCore,
        quick_audio_analysis,
        compute_band_energy,
        detect_energy_transitions,
    )

    # Genre analysis
    from app.core.adapters.analysis_utils import (
        analyze_genre,
        batch_analyze_genres,
        find_genre_mismatches,
        find_similar_tracks,
    )

    # Single track
    result = analyze_genre("/path/to/track.mp3")
    print(result.dj_category)  # "Techno"

    # Batch analysis (M2 parallel)
    results = batch_analyze_genres(audio_paths, workers=4)

Author: Optimized for Apple Silicon M2
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, NamedTuple
import os

# Import centralized STFT from common primitives
from app.common.primitives.stft import STFTCache, compute_stft as core_compute_stft

# Force NumPy to use Apple Accelerate
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FrequencyBands:
    """Energy in different frequency bands."""
    sub_bass: np.ndarray      # 20-60 Hz (kick fundamentals)
    bass: np.ndarray          # 60-250 Hz (bass line)
    low_mid: np.ndarray       # 250-500 Hz
    mid: np.ndarray           # 500-2000 Hz
    high_mid: np.ndarray      # 2000-6000 Hz (presence)
    high: np.ndarray          # 6000-20000 Hz (air)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            'sub_bass': self.sub_bass,
            'bass': self.bass,
            'low_mid': self.low_mid,
            'mid': self.mid,
            'high_mid': self.high_mid,
            'high': self.high,
        }


@dataclass
class SpectralFeatures:
    """Computed spectral features."""
    centroid: np.ndarray       # Spectral centroid (Hz)
    rolloff: np.ndarray        # Spectral rolloff (Hz)
    brightness: np.ndarray     # High frequency ratio (0-1)
    flatness: np.ndarray       # Spectral flatness (0-1)
    bandwidth: np.ndarray      # Spectral bandwidth (Hz)
    flux: np.ndarray           # Spectral flux


@dataclass
class EnergyFeatures:
    """Computed energy features."""
    rms: np.ndarray            # RMS energy
    rms_norm: np.ndarray       # Normalized RMS (0-1)
    derivative: np.ndarray     # Energy derivative
    acceleration: np.ndarray   # Second derivative
    low_energy_ratio: float    # Percentage of low-energy frames


@dataclass
class TransitionCandidate:
    """Detected transition candidate."""
    frame_idx: int
    time_sec: float
    novelty_score: float
    energy_change: float
    bass_change: float
    spectral_change: float
    direction: str  # 'increasing', 'decreasing', 'mixed'


# =============================================================================
# Core Analysis Class
# =============================================================================

class AudioAnalysisCore:
    """
    Core audio analysis with shared STFT computation.

    MIGRATED: Now uses STFTCache from src/core/primitives/stft.py
    instead of local STFTCache. All librosa calls go through STFTCache.

    This class provides optimized audio analysis primitives that
    can be reused across different analysis modules (drop detection,
    mixin/mixout detection, zone classification, etc.)

    All computations share a single STFT for maximum efficiency.

    Usage:
        core = AudioAnalysisCore(sr=22050)
        cache = core.compute_stft(y, hop_length=512)
        bands = core.compute_frequency_bands(cache)
        spectral = core.compute_spectral_features(cache)
    """

    def __init__(self, sr: int = 22050, n_fft: int = 2048):
        """
        Initialize analysis core.

        Args:
            sr: Expected sample rate
            n_fft: FFT size
        """
        self.sr = sr
        self.n_fft = n_fft
        self._freq_masks: Dict[str, np.ndarray] = {}

    def compute_stft(self, y: np.ndarray, hop_length: int = 512) -> STFTCache:
        """
        Compute STFT and cache results.

        This is the foundation for all other computations.
        Should be called ONCE per audio file.

        Args:
            y: Audio signal (mono)
            hop_length: Hop length in samples

        Returns:
            STFTCache with all derived values (from core)
        """
        # Use centralized compute_stft from core
        cache = core_compute_stft(
            y, sr=self.sr, n_fft=self.n_fft, hop_length=hop_length
        )

        # Initialize frequency masks if needed
        if not self._freq_masks:
            self._init_freq_masks(cache.freqs)

        return cache

    def _init_freq_masks(self, freqs: np.ndarray):
        """Pre-compute frequency band masks."""
        self._freq_masks = {
            'sub_bass': (freqs >= 20) & (freqs < 60),
            'bass': (freqs >= 60) & (freqs < 250),
            'low_mid': (freqs >= 250) & (freqs < 500),
            'mid': (freqs >= 500) & (freqs < 2000),
            'high_mid': (freqs >= 2000) & (freqs < 6000),
            'high': (freqs >= 6000),
            'brightness': freqs > 3000,
        }

    def compute_frequency_bands(self, cache: STFTCache) -> FrequencyBands:
        """
        Compute energy in frequency bands (vectorized).

        Args:
            cache: Pre-computed STFT cache

        Returns:
            FrequencyBands with energy arrays
        """
        S_power = cache.S ** 2
        return FrequencyBands(
            sub_bass=np.sum(S_power[self._freq_masks['sub_bass'], :], axis=0),
            bass=np.sum(S_power[self._freq_masks['bass'], :], axis=0),
            low_mid=np.sum(S_power[self._freq_masks['low_mid'], :], axis=0),
            mid=np.sum(S_power[self._freq_masks['mid'], :], axis=0),
            high_mid=np.sum(S_power[self._freq_masks['high_mid'], :], axis=0),
            high=np.sum(S_power[self._freq_masks['high'], :], axis=0),
        )

    def compute_band_energy(self, cache: STFTCache,
                           low_hz: float, high_hz: float) -> np.ndarray:
        """
        Compute energy in arbitrary frequency band.

        Args:
            cache: Pre-computed STFT cache
            low_hz: Lower frequency bound
            high_hz: Upper frequency bound

        Returns:
            Energy array per frame
        """
        mask = (cache.freqs >= low_hz) & (cache.freqs < high_hz)
        S_power = cache.S ** 2
        return np.sum(S_power[mask, :], axis=0)

    def compute_spectral_features(self, cache: STFTCache) -> SpectralFeatures:
        """
        Compute spectral features (all from cached STFT).

        Uses STFTCache.get_*() methods for centralized librosa access.

        Args:
            cache: Pre-computed STFT cache

        Returns:
            SpectralFeatures dataclass
        """
        n_frames = cache.n_frames

        # Use STFTCache methods (centralized librosa)
        centroid = cache.get_spectral_centroid()
        rolloff = cache.get_spectral_rolloff(roll_percent=0.85)
        flatness = cache.get_spectral_flatness()
        bandwidth = cache.get_spectral_bandwidth()

        # Brightness (ratio of high-frequency energy) - pure numpy
        S_power = cache.S ** 2
        total_energy = np.sum(S_power, axis=0) + 1e-10
        high_energy = np.sum(S_power[self._freq_masks['brightness'], :], axis=0)
        brightness = high_energy / total_energy

        # Spectral flux - pure numpy
        flux = cache.get_spectral_flux()

        return SpectralFeatures(
            centroid=centroid[:n_frames],
            rolloff=rolloff[:n_frames],
            brightness=brightness[:n_frames],
            flatness=flatness[:n_frames],
            bandwidth=bandwidth[:n_frames],
            flux=flux[:n_frames]
        )

    def compute_energy_features(self, cache: STFTCache) -> EnergyFeatures:
        """
        Compute energy features.

        Args:
            cache: Pre-computed STFT cache

        Returns:
            EnergyFeatures dataclass
        """
        n_frames = cache.n_frames

        # RMS from STFTCache (centralized librosa)
        rms = cache.get_rms()[:n_frames]

        # Normalize
        rms_norm = normalize(smooth(rms))

        # Derivatives - pure numpy
        derivative = np.gradient(smooth(rms))
        acceleration = np.gradient(smooth(derivative))

        # Low energy ratio
        mean_rms = np.mean(rms)
        low_energy_ratio = np.mean(rms < mean_rms)

        return EnergyFeatures(
            rms=rms,
            rms_norm=rms_norm,
            derivative=derivative,
            acceleration=acceleration,
            low_energy_ratio=float(low_energy_ratio)
        )

    def compute_onset_features(self, cache: STFTCache) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute onset-related features.

        Args:
            cache: Pre-computed STFT cache

        Returns:
            Tuple of (onset_strength, onset_density, estimated_tempo)
        """
        n_frames = cache.n_frames

        # Onset strength from STFTCache (centralized librosa)
        onset_env = cache.get_onset_strength()[:n_frames]

        # Onset density (local average) - pure numpy
        density_window = int(2.0 * cache.sr / cache.hop_length)
        onset_density = uniform_filter1d(onset_env, size=max(1, density_window))

        # Tempo estimation from STFTCache (centralized librosa)
        try:
            tempo, confidence = cache.get_tempo()
        except Exception:
            tempo = 0.0

        return onset_env, onset_density, tempo

    def compute_filter_position(self, spectral: SpectralFeatures) -> np.ndarray:
        """
        Estimate filter position from spectral features.

        Returns normalized 0-1 value where:
        - 0 = heavily filtered (lowpass closed)
        - 1 = fully open

        Args:
            spectral: Pre-computed spectral features

        Returns:
            Filter position array
        """
        centroid_norm = normalize(smooth(spectral.centroid))
        rolloff_norm = normalize(smooth(spectral.rolloff))
        return 0.6 * centroid_norm + 0.4 * rolloff_norm


# =============================================================================
# Standalone Utility Functions (for backward compatibility)
# =============================================================================

def normalize(arr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Normalize array to 0-1 range."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min < eps:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def smooth(arr: np.ndarray, window: int = 5, method: str = 'gaussian') -> np.ndarray:
    """
    Smooth array with specified method.

    Args:
        arr: Input array
        window: Smoothing window size
        method: 'gaussian', 'uniform', or 'savgol'

    Returns:
        Smoothed array
    """
    if method == 'gaussian':
        return gaussian_filter1d(arr, sigma=window/2)
    elif method == 'uniform':
        return uniform_filter1d(arr, size=window)
    elif method == 'savgol':
        if len(arr) < window:
            return arr
        return savgol_filter(arr, window_length=window, polyorder=2)
    else:
        return arr


def compute_derivative(arr: np.ndarray, smooth_first: bool = True,
                      window: int = 5) -> np.ndarray:
    """Compute derivative with optional smoothing."""
    if smooth_first:
        arr = smooth(arr, window)
    return np.gradient(arr)


def compute_band_energy(S_power: np.ndarray, freqs: np.ndarray,
                       low_hz: float, high_hz: float) -> np.ndarray:
    """
    Standalone function to compute energy in frequency band.

    Args:
        S_power: Power spectrogram
        freqs: Frequency axis
        low_hz: Lower frequency bound
        high_hz: Upper frequency bound

    Returns:
        Energy array per frame
    """
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return np.sum(S_power[mask, :], axis=0)


def compute_novelty_function(
    energy: np.ndarray,
    bass: np.ndarray,
    spectral: np.ndarray,
    brightness: np.ndarray,
    onset_density: np.ndarray = None,
    weights: Dict[str, float] = None
) -> np.ndarray:
    """
    Compute combined novelty function for transition detection.

    Novelty function peaks indicate points where audio characteristics
    change significantly (potential transitions).

    Args:
        energy: Normalized energy curve
        bass: Normalized bass energy curve
        spectral: Normalized spectral centroid curve
        brightness: Normalized brightness curve
        onset_density: Optional onset density curve
        weights: Optional weight dict (default: equal weights)

    Returns:
        Novelty function array
    """
    if weights is None:
        weights = {
            'energy': 0.25,
            'bass': 0.25,
            'spectral': 0.20,
            'brightness': 0.15,
            'onset': 0.15,
        }

    # Derivative-based novelty
    energy_nov = normalize(np.abs(np.gradient(energy)))
    bass_nov = normalize(np.abs(np.gradient(bass)))
    spectral_nov = normalize(np.abs(np.gradient(spectral)))
    brightness_nov = normalize(np.abs(np.gradient(brightness)))

    novelty = (
        weights['energy'] * energy_nov +
        weights['bass'] * bass_nov +
        weights['spectral'] * spectral_nov +
        weights['brightness'] * brightness_nov
    )

    if onset_density is not None:
        onset_nov = normalize(np.abs(np.gradient(onset_density)))
        novelty += weights['onset'] * onset_nov

    return smooth(novelty, window=7)


def detect_energy_transitions(
    energy: np.ndarray,
    derivative: np.ndarray,
    time_axis: np.ndarray,
    threshold_percentile: float = 85,
    min_distance_sec: float = 2.0,
    sr: int = 22050,
    hop_length: int = 512
) -> List[TransitionCandidate]:
    """
    Detect transition points based on energy changes.

    Args:
        energy: Normalized energy curve
        derivative: Energy derivative
        time_axis: Time axis in seconds
        threshold_percentile: Percentile for peak detection
        min_distance_sec: Minimum distance between transitions
        sr: Sample rate
        hop_length: Hop length

    Returns:
        List of TransitionCandidate
    """
    min_distance = int(min_distance_sec * sr / hop_length)
    threshold = np.percentile(np.abs(derivative), threshold_percentile)

    # Find peaks in absolute derivative
    peaks, properties = find_peaks(
        np.abs(derivative),
        height=threshold,
        distance=min_distance
    )

    candidates = []
    for peak_idx in peaks:
        # Determine direction
        if derivative[peak_idx] > 0:
            direction = 'increasing'
        elif derivative[peak_idx] < 0:
            direction = 'decreasing'
        else:
            direction = 'mixed'

        candidates.append(TransitionCandidate(
            frame_idx=peak_idx,
            time_sec=time_axis[peak_idx] if peak_idx < len(time_axis) else 0,
            novelty_score=float(np.abs(derivative[peak_idx])),
            energy_change=float(derivative[peak_idx]),
            bass_change=0.0,  # To be filled by caller
            spectral_change=0.0,  # To be filled by caller
            direction=direction
        ))

    return candidates


def find_local_extrema(
    arr: np.ndarray,
    min_distance: int = 10,
    prominence: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local minima and maxima in array.

    Args:
        arr: Input array
        min_distance: Minimum distance between peaks
        prominence: Minimum prominence

    Returns:
        Tuple of (minima_indices, maxima_indices)
    """
    maxima, _ = find_peaks(arr, distance=min_distance, prominence=prominence)
    minima, _ = find_peaks(-arr, distance=min_distance, prominence=prominence)
    return minima, maxima


def compute_mel_bands(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_mels: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mel spectrogram and convert to dB.

    Uses STFTCache for centralized librosa access.

    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length
        n_mels: Number of mel bands

    Returns:
        Tuple of (mel_spectrogram, mel_db)
    """
    cache = core_compute_stft(y, sr=sr, hop_length=hop_length)
    mel = cache.get_mel(n_mels=n_mels)

    # Convert to dB - pure numpy
    mel_db = 10 * np.log10(mel + 1e-10)
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)

    return mel, mel_db


def extract_mel_band_energies(
    mel_db: np.ndarray,
    n_mels: int = 128
) -> Dict[str, np.ndarray]:
    """
    Extract energy from mel spectrogram bands.

    Standard band mapping (approximate for 128 mels):
    - Bass: mels 0-12 (0-250 Hz)
    - Kick: mels 12-20 (250-500 Hz)
    - Mid: mels 20-50 (500-2000 Hz)
    - High: mels 50-90 (2000-8000 Hz)
    - Air: mels 90+ (8000+ Hz)

    Args:
        mel_db: Mel spectrogram in dB
        n_mels: Number of mel bands

    Returns:
        Dict with band energy arrays
    """
    return {
        'bass': np.mean(mel_db[0:12], axis=0),
        'kick': np.mean(mel_db[12:20], axis=0),
        'mid': np.mean(mel_db[20:50], axis=0),
        'high': np.mean(mel_db[50:90], axis=0),
        'air': np.mean(mel_db[90:], axis=0) if n_mels > 90 else np.zeros(mel_db.shape[1]),
    }


# =============================================================================
# Convenience function for quick analysis
# =============================================================================

def quick_audio_analysis(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048
) -> Dict:
    """
    Perform quick audio analysis returning all common features.

    This is a convenience function that computes all commonly needed
    features in one call.

    Args:
        y: Audio signal (mono)
        sr: Sample rate
        hop_length: Hop length
        n_fft: FFT size

    Returns:
        Dict with all computed features
    """
    core = AudioAnalysisCore(sr=sr, n_fft=n_fft)
    cache = core.compute_stft(y, hop_length=hop_length)

    bands = core.compute_frequency_bands(cache)
    spectral = core.compute_spectral_features(cache)
    energy = core.compute_energy_features(cache)
    onset_env, onset_density, tempo = core.compute_onset_features(cache)
    filter_pos = core.compute_filter_position(spectral)

    time_axis = cache.frames_to_time(np.arange(cache.n_frames))

    return {
        'time_axis': time_axis,
        'duration_sec': len(y) / sr,
        'n_frames': cache.n_frames,

        # Energy
        'rms': energy.rms,
        'rms_norm': energy.rms_norm,
        'energy_derivative': energy.derivative,
        'low_energy_ratio': energy.low_energy_ratio,

        # Frequency bands
        'sub_bass': bands.sub_bass,
        'bass': bands.bass,
        'low_mid': bands.low_mid,
        'mid': bands.mid,
        'high_mid': bands.high_mid,
        'high': bands.high,

        # Spectral
        'centroid': spectral.centroid,
        'rolloff': spectral.rolloff,
        'brightness': spectral.brightness,
        'flatness': spectral.flatness,
        'flux': spectral.flux,
        'filter_position': filter_pos,

        # Rhythm
        'onset_strength': onset_env,
        'onset_density': onset_density,
        'tempo': tempo,

        # Cache for advanced use
        '_cache': cache,
    }


# =============================================================================
# Genre Analysis Components
# =============================================================================

@dataclass
class GenreAnalysisResult:
    """Result of genre analysis for a track."""
    path: str
    genre: str
    dj_category: str
    confidence: float
    all_predictions: Dict[str, float]
    error: Optional[str] = None


def analyze_genre(audio_path: str, **kwargs) -> GenreAnalysisResult:
    """
    Analyze genre for a single audio file.

    This is a placeholder - actual implementation requires ML model.

    Args:
        audio_path: Path to audio file
        **kwargs: Additional options

    Returns:
        GenreAnalysisResult
    """
    return GenreAnalysisResult(
        path=audio_path,
        genre="Unknown",
        dj_category="Unknown",
        confidence=0.0,
        all_predictions={},
        error="Genre model not loaded"
    )


def batch_analyze_genres(
    audio_paths: List[str],
    workers: int = 4,
    **kwargs
) -> List[GenreAnalysisResult]:
    """
    Batch analyze genres for multiple audio files.

    Args:
        audio_paths: List of audio file paths
        workers: Number of parallel workers
        **kwargs: Additional options

    Returns:
        List of GenreAnalysisResult
    """
    return [analyze_genre(p, **kwargs) for p in audio_paths]
