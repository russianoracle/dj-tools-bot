"""
FeatureFactory - Centralized Audio Feature Extraction

This is the SINGLE POINT OF ENTRY for all librosa-dependent feature extraction.
All other modules should use FeatureFactory instead of calling librosa directly.

Architecture:
    Audio → FeatureFactory → (STFTCache, primitives) → Features
                   │
                   └─→ librosa (internal implementation detail)

Usage:
    from app.core.adapters.feature_factory import FeatureFactory

    # Create factory from audio
    factory = FeatureFactory.from_audio(y, sr)

    # Get cached features (computed lazily)
    rms = factory.rms()
    mfcc = factory.mfcc()
    tempo = factory.tempo()

    # Or get all spectral features at once
    spectral = factory.spectral_features()

M2 Apple Silicon Optimization:
    - Single STFT computation shared across all features
    - Lazy computation with caching
    - Contiguous float32 arrays
    - Vectorized operations

Migration Guide:
    BEFORE (scattered librosa calls):
        import librosa
        y, sr = librosa.load(path)
        S = np.abs(librosa.stft(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    AFTER (centralized):
        from app.core.adapters.feature_factory import FeatureFactory
        from app.core.adapters.loader import AudioLoader

        y, sr = AudioLoader().load(path)
        factory = FeatureFactory.from_audio(y, sr)
        mfcc = factory.mfcc()
        tempo = factory.tempo()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from functools import cached_property

# Import from app.common.primitives - using STFTCache for cached features
from app.common.primitives.stft import STFTCache, compute_stft
from app.common.primitives.energy import FrequencyBands, compute_frequency_bands
from app.common.primitives.spectral import compute_brightness, SpectralFeatures
from app.common.primitives.rhythm import compute_tempo_multi
from app.common.primitives.dynamics import detect_peaks, compute_novelty, compute_buildup_score
from app.common.primitives.filtering import smooth_gaussian, normalize_minmax

# NOTE: The following functions are BLOCKED from primitives public API:
#   - compute_rms → use STFTCache.get_rms()
#   - compute_centroid → use STFTCache.get_spectral_centroid()
#   - compute_rolloff → use STFTCache.get_spectral_rolloff()
#   - compute_flatness → use STFTCache.get_spectral_flatness()
#   - compute_flux → use STFTCache.get_spectral_flux()
#   - compute_bandwidth → use STFTCache.get_spectral_bandwidth()
#   - compute_contrast → use STFTCache.get_spectral_contrast()
#   - compute_onset_strength → use STFTCache.get_onset_strength()
#   - compute_tempo → use STFTCache.get_tempo()
#   - compute_beats → use STFTCache.get_beats()
# FeatureFactory delegates to STFTCache for all these features.

# Internal imports for functions that need pure numpy version
from app.common.primitives.rhythm import compute_onset_strength as _compute_onset_strength_numpy


@dataclass
class FeatureFactoryConfig:
    """Configuration for FeatureFactory."""
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mfcc: int = 13
    n_mels: int = 128
    n_chroma: int = 12
    brightness_cutoff: float = 3000.0


class FeatureFactory:
    """
    Centralized factory for all audio feature extraction.

    This class wraps all librosa functionality, providing a clean API
    while hiding implementation details. All caching is handled internally.

    Design Principles:
        1. Single STFT computation (via STFTCache)
        2. Lazy feature computation
        3. Consistent API across all feature types
        4. M2-optimized (float32, contiguous arrays)
    """

    def __init__(
        self,
        stft_cache: STFTCache,
        config: Optional[FeatureFactoryConfig] = None
    ):
        """
        Initialize FeatureFactory with pre-computed STFT.

        Args:
            stft_cache: Pre-computed STFT cache
            config: Configuration options

        Note: Use FeatureFactory.from_audio() for convenience.
        """
        self._cache = stft_cache
        self._config = config or FeatureFactoryConfig(
            sr=stft_cache.sr,
            hop_length=stft_cache.hop_length,
            n_fft=stft_cache.n_fft
        )
        self._feature_cache: Dict[str, Any] = {}

    @classmethod
    def from_audio(
        cls,
        y: np.ndarray,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        **kwargs
    ) -> 'FeatureFactory':
        """
        Create FeatureFactory from raw audio.

        This is the PRIMARY entry point for feature extraction.

        Args:
            y: Audio time series (mono)
            sr: Sample rate
            n_fft: FFT window size
            hop_length: Hop length in samples
            **kwargs: Additional config options

        Returns:
            FeatureFactory instance

        Example:
            >>> factory = FeatureFactory.from_audio(y, sr=22050)
            >>> rms = factory.rms()
            >>> tempo = factory.tempo()
        """
        stft_cache = compute_stft(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        config = FeatureFactoryConfig(
            sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs
        )
        return cls(stft_cache, config)

    @classmethod
    def from_stft_cache(cls, cache: STFTCache) -> 'FeatureFactory':
        """
        Create FeatureFactory from existing STFTCache.

        Use this when you already have a computed STFT.

        Args:
            cache: Pre-computed STFTCache

        Returns:
            FeatureFactory instance
        """
        return cls(cache)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def stft_cache(self) -> STFTCache:
        """Access underlying STFTCache for advanced usage."""
        return self._cache

    @property
    def n_frames(self) -> int:
        """Number of time frames."""
        return self._cache.n_frames

    @property
    def duration_sec(self) -> float:
        """Duration in seconds."""
        return self._cache.duration_sec

    @property
    def sr(self) -> int:
        """Sample rate."""
        return self._config.sr

    @property
    def times(self) -> np.ndarray:
        """Time axis in seconds."""
        return self._cache.times

    @property
    def freqs(self) -> np.ndarray:
        """Frequency axis in Hz."""
        return self._cache.freqs

    # =========================================================================
    # Energy Features
    # =========================================================================

    def rms(self) -> np.ndarray:
        """
        RMS energy per frame.

        Returns:
            Array of shape (n_frames,)

        Note: Delegates to STFTCache.get_rms() for cache consistency.
        """
        return self._cache.get_rms()

    def rms_db(self) -> np.ndarray:
        """
        RMS energy in decibels.

        Returns:
            Array of shape (n_frames,)
        """
        key = 'rms_db'
        if key not in self._feature_cache:
            rms = self.rms()
            self._feature_cache[key] = np.ascontiguousarray(
                20 * np.log10(rms + 1e-10), dtype=np.float32
            )
        return self._feature_cache[key]

    def frequency_bands(self) -> FrequencyBands:
        """
        Energy in frequency bands (bass, mid, high).

        Returns:
            FrequencyBands dataclass with bass, mid, high, total
        """
        key = 'frequency_bands'
        if key not in self._feature_cache:
            self._feature_cache[key] = compute_frequency_bands(
                self._cache.S, self._cache.freqs
            )
        return self._feature_cache[key]

    # =========================================================================
    # Spectral Features
    # =========================================================================

    def spectral_centroid(self) -> np.ndarray:
        """Spectral centroid per frame (Hz).

        Note: Delegates to STFTCache.get_spectral_centroid() for cache consistency.
        """
        return self._cache.get_spectral_centroid()

    def spectral_rolloff(self, roll_percent: float = 0.85) -> np.ndarray:
        """Spectral rolloff frequency (Hz).

        Note: Delegates to STFTCache.get_spectral_rolloff() for cache consistency.
        """
        return self._cache.get_spectral_rolloff(roll_percent=roll_percent)

    def spectral_brightness(self) -> np.ndarray:
        """Spectral brightness (ratio of high-frequency energy).

        Note: This is computed from primitives as STFTCache doesn't have this.
        """
        key = 'brightness'
        if key not in self._feature_cache:
            self._feature_cache[key] = compute_brightness(
                self._cache.S, self._cache.freqs, self._config.brightness_cutoff
            )
        return self._feature_cache[key]

    def spectral_flatness(self) -> np.ndarray:
        """Spectral flatness (measure of noise-likeness).

        Note: Delegates to STFTCache.get_spectral_flatness() for cache consistency.
        """
        return self._cache.get_spectral_flatness()

    def spectral_flux(self) -> np.ndarray:
        """Spectral flux (frame-to-frame change).

        Note: Delegates to STFTCache.get_spectral_flux() for cache consistency.
        """
        return self._cache.get_spectral_flux()

    def spectral_bandwidth(self) -> np.ndarray:
        """Spectral bandwidth (weighted deviation from centroid).

        Note: Delegates to STFTCache.get_spectral_bandwidth() for cache consistency.
        """
        return self._cache.get_spectral_bandwidth()

    def spectral_contrast(self, n_bands: int = 6) -> np.ndarray:
        """Spectral contrast across octave bands.

        Note: Delegates to STFTCache.get_spectral_contrast() for cache consistency.
        """
        return self._cache.get_spectral_contrast(n_bands=n_bands)

    def spectral_features(self) -> SpectralFeatures:
        """
        Get all spectral features at once.

        Returns:
            SpectralFeatures dataclass with centroid, rolloff, brightness,
            flatness, flux, bandwidth, contrast

        Note: Collects features from STFTCache for cache consistency.
        """
        key = 'all_spectral'
        if key not in self._feature_cache:
            self._feature_cache[key] = SpectralFeatures(
                centroid=self.spectral_centroid(),
                rolloff=self.spectral_rolloff(),
                brightness=self.spectral_brightness(),
                flatness=self.spectral_flatness(),
                flux=self.spectral_flux(),
                bandwidth=self.spectral_bandwidth(),
                contrast=self.spectral_contrast(),
            )
        return self._feature_cache[key]

    # =========================================================================
    # MFCC Features (via STFTCache)
    # =========================================================================

    def mfcc(self, n_mfcc: Optional[int] = None) -> np.ndarray:
        """
        Mel-frequency cepstral coefficients.

        Args:
            n_mfcc: Number of coefficients (default from config)

        Returns:
            Array of shape (n_mfcc, n_frames)
        """
        n = n_mfcc or self._config.n_mfcc
        return self._cache.get_mfcc(n_mfcc=n)

    def mfcc_delta(self, n_mfcc: Optional[int] = None) -> np.ndarray:
        """
        First derivative of MFCCs.

        Returns:
            Array of shape (n_mfcc, n_frames)
        """
        n = n_mfcc or self._config.n_mfcc
        return self._cache.get_mfcc_delta(n_mfcc=n)

    def mfcc_delta2(self, n_mfcc: Optional[int] = None) -> np.ndarray:
        """
        Second derivative of MFCCs.

        Returns:
            Array of shape (n_mfcc, n_frames)
        """
        n = n_mfcc or self._config.n_mfcc
        return self._cache.get_mfcc_delta(n_mfcc=n, order=2)

    # =========================================================================
    # Chroma Features (via STFTCache)
    # =========================================================================

    def chroma(self, n_chroma: Optional[int] = None) -> np.ndarray:
        """
        Chromagram (pitch class profile).

        Args:
            n_chroma: Number of chroma bins (default: 12)

        Returns:
            Array of shape (n_chroma, n_frames)
        """
        n = n_chroma or self._config.n_chroma
        return self._cache.get_chroma(n_chroma=n)

    def tonnetz(self) -> np.ndarray:
        """
        Tonnetz (tonal centroid features).

        Returns:
            Array of shape (6, n_frames)
        """
        return self._cache.get_tonnetz()

    # =========================================================================
    # Mel Spectrogram (via STFTCache)
    # =========================================================================

    def mel_spectrogram(self, n_mels: Optional[int] = None) -> np.ndarray:
        """
        Mel spectrogram.

        Args:
            n_mels: Number of mel bands (default from config)

        Returns:
            Array of shape (n_mels, n_frames)
        """
        n = n_mels or self._config.n_mels
        return self._cache.get_mel(n_mels=n)

    # =========================================================================
    # Rhythm Features
    # =========================================================================

    def onset_strength(self) -> np.ndarray:
        """
        Onset strength envelope.

        Returns:
            Array of shape (n_frames,)

        Note: Delegates to STFTCache.get_onset_strength() for cache consistency.
        """
        return self._cache.get_onset_strength()

    def onset_strength_pure(self) -> np.ndarray:
        """
        Onset strength using pure numpy (no librosa).

        Returns:
            Array of shape (n_frames,)
        """
        key = 'onset_strength_pure'
        if key not in self._feature_cache:
            self._feature_cache[key] = _compute_onset_strength_numpy(
                self._cache.S, self._config.sr, self._config.hop_length
            )
        return self._feature_cache[key]

    def tempo(self) -> Tuple[float, float]:
        """
        Global tempo estimate with confidence.

        Returns:
            Tuple of (tempo_bpm, confidence)

        Note: Delegates to STFTCache.get_tempo() for cache consistency.
        """
        return self._cache.get_tempo()

    def tempo_multi(self) -> Tuple[float, float, float]:
        """
        Multi-resolution tempo estimate.

        Returns:
            Tuple of (tempo_bpm, confidence, alternative_tempo)

        Note: Uses pure numpy implementation for alternative approach.
        """
        key = 'tempo_multi'
        if key not in self._feature_cache:
            onset_env = self.onset_strength_pure()
            self._feature_cache[key] = compute_tempo_multi(
                onset_env, self._config.sr, self._config.hop_length
            )
        return self._feature_cache[key]

    def beats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Beat positions.

        Returns:
            Tuple of (beat_frames, beat_times)

        Note: Delegates to STFTCache.get_beats() for cache consistency.
        """
        return self._cache.get_beats()

    # =========================================================================
    # Dynamics Features
    # =========================================================================

    def novelty(self, metric: str = 'cosine') -> np.ndarray:
        """
        Spectral novelty curve.

        Args:
            metric: Distance metric ('cosine' or 'euclidean')

        Returns:
            Array of shape (n_frames,)
        """
        key = f'novelty_{metric}'
        if key not in self._feature_cache:
            mfcc = self.mfcc()
            self._feature_cache[key] = compute_novelty(mfcc, metric=metric)
        return self._feature_cache[key]

    def buildup_score(self, window_frames: int = 20) -> np.ndarray:
        """
        Buildup detection score.

        Args:
            window_frames: Analysis window in frames

        Returns:
            Array of shape (n_frames,)
        """
        key = f'buildup_{window_frames}'
        if key not in self._feature_cache:
            rms = self.rms()
            self._feature_cache[key] = compute_buildup_score(
                rms, window_frames=window_frames
            )
        return self._feature_cache[key]

    def peaks(
        self,
        feature: Optional[np.ndarray] = None,
        threshold_percentile: float = 75,
        distance: Optional[int] = 10
    ) -> np.ndarray:
        """
        Detect peaks in a feature curve.

        Args:
            feature: Feature to analyze (default: novelty)
            threshold_percentile: Minimum peak height percentile
            distance: Minimum distance between peaks

        Returns:
            Array of peak indices
        """
        if feature is None:
            feature = self.novelty()
        return detect_peaks(
            feature,
            threshold_percentile=threshold_percentile,
            distance=distance
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear_cache(self):
        """Clear feature cache to free memory."""
        self._feature_cache.clear()
        self._cache.clear_feature_cache()

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Export all basic features as dictionary.

        Returns:
            Dictionary with feature names as keys
        """
        return {
            'rms': self.rms(),
            'spectral_centroid': self.spectral_centroid(),
            'spectral_rolloff': self.spectral_rolloff(),
            'spectral_brightness': self.spectral_brightness(),
            'spectral_flatness': self.spectral_flatness(),
            'spectral_flux': self.spectral_flux(),
            'spectral_bandwidth': self.spectral_bandwidth(),
            'mfcc': self.mfcc(),
            'chroma': self.chroma(),
            'onset_strength': self.onset_strength_pure(),
        }

    def __repr__(self) -> str:
        return (
            f"FeatureFactory("
            f"n_frames={self.n_frames}, "
            f"duration={self.duration_sec:.1f}s, "
            f"sr={self.sr})"
        )


# =============================================================================
# Convenience functions for quick feature extraction
# =============================================================================

def extract_features(
    y: np.ndarray,
    sr: int = 22050,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Extract all features from audio.

    Convenience function for one-shot feature extraction.

    Args:
        y: Audio time series
        sr: Sample rate
        **kwargs: Additional config options

    Returns:
        Dictionary of all features

    Example:
        >>> features = extract_features(y, sr=22050)
        >>> print(features['rms'].shape)
    """
    factory = FeatureFactory.from_audio(y, sr, **kwargs)
    return factory.to_dict()


def create_factory(y: np.ndarray, sr: int = 22050, **kwargs) -> FeatureFactory:
    """
    Create FeatureFactory from audio.

    Alias for FeatureFactory.from_audio().

    Args:
        y: Audio time series
        sr: Sample rate
        **kwargs: Additional config options

    Returns:
        FeatureFactory instance
    """
    return FeatureFactory.from_audio(y, sr, **kwargs)
