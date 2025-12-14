"""
Spectral Primitives - Centroid, rolloff, brightness, etc.

Apple Silicon M2 Optimized:
- Vectorized NumPy operations
- Reuse pre-computed STFT
- Contiguous memory patterns

DEPRECATION WARNING:
    Many functions here have equivalents in STFTCache that use librosa.
    For cache consistency, prefer STFTCache methods:
    - compute_centroid() → cache.get_spectral_centroid()
    - compute_rolloff() → cache.get_spectral_rolloff()
    - compute_flatness() → cache.get_spectral_flatness()
    - compute_flux() → cache.get_spectral_flux()
    - compute_bandwidth() → cache.get_spectral_bandwidth()
    - compute_contrast() → cache.get_spectral_contrast()
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpectralFeatures:
    """
    Collection of spectral features per frame.

    All arrays have shape (n_frames,) except contrast (n_bands, n_frames).
    """
    centroid: np.ndarray     # Spectral center of mass
    rolloff: np.ndarray      # 85% energy frequency
    brightness: np.ndarray   # High-frequency ratio
    flatness: np.ndarray     # Tonality measure
    flux: np.ndarray         # Frame-to-frame change
    bandwidth: np.ndarray    # Spectral spread
    contrast: np.ndarray     # Band contrast (n_bands, n_frames)

    def to_dict(self) -> dict:
        """Convert to dictionary of mean values."""
        return {
            'spectral_centroid': float(np.mean(self.centroid)),
            'spectral_rolloff': float(np.mean(self.rolloff)),
            'brightness': float(np.mean(self.brightness)),
            'spectral_flatness': float(np.mean(self.flatness)),
            'spectral_flux': float(np.mean(self.flux)),
            'spectral_bandwidth': float(np.mean(self.bandwidth)),
            'spectral_contrast_mean': float(np.mean(self.contrast)),
        }


def compute_centroid(
    S: np.ndarray,
    freqs: np.ndarray,
    _warn: bool = True
) -> np.ndarray:
    """
    Compute spectral centroid (center of mass).

    DEPRECATED: Use STFTCache.get_spectral_centroid() for consistency.

    M2-optimized: Ensures contiguous arrays for Apple Accelerate.
    Higher centroid = brighter sound.

    Args:
        S: Magnitude spectrogram (n_freq, n_frames)
        freqs: Frequency bins in Hz
        _warn: Show deprecation warning (internal use)

    Returns:
        Centroid per frame in Hz
    """
    if _warn:
        warnings.warn(
            "compute_centroid() is deprecated. Use STFTCache.get_spectral_centroid() for consistency.",
            DeprecationWarning,
            stacklevel=2
        )
    # Ensure contiguous for Apple Accelerate
    S = np.ascontiguousarray(S, dtype=np.float32)
    freqs = np.ascontiguousarray(freqs, dtype=np.float32)

    # Weighted mean of frequencies
    norm = np.sum(S, axis=0) + 1e-10
    centroid = np.sum(freqs[:, np.newaxis] * S, axis=0) / norm
    return centroid


def compute_rolloff(
    S: np.ndarray,
    freqs: np.ndarray,
    roll_percent: float = 0.85
) -> np.ndarray:
    """
    Compute spectral rolloff frequency.

    M2-optimized: Ensures contiguous arrays for Apple Accelerate.
    The frequency below which roll_percent of energy is contained.

    Args:
        S: Magnitude spectrogram
        freqs: Frequency bins
        roll_percent: Energy threshold (default: 0.85)

    Returns:
        Rolloff frequency per frame in Hz
    """
    # Ensure contiguous for Apple Accelerate
    S = np.ascontiguousarray(S, dtype=np.float32)
    freqs = np.ascontiguousarray(freqs, dtype=np.float32)

    # Cumulative energy
    total_energy = np.sum(S ** 2, axis=0) + 1e-10
    cumsum = np.cumsum(S ** 2, axis=0)

    # Find rolloff index for each frame
    threshold = roll_percent * total_energy
    rolloff_idx = np.argmax(cumsum >= threshold, axis=0)

    return freqs[rolloff_idx]


def compute_brightness(
    S: np.ndarray,
    freqs: np.ndarray,
    cutoff: float = 3000.0
) -> np.ndarray:
    """
    Compute brightness (high-frequency energy ratio).

    Args:
        S: Magnitude spectrogram
        freqs: Frequency bins
        cutoff: Cutoff frequency for "bright" (default: 3000 Hz)

    Returns:
        Brightness ratio per frame (0-1)
    """
    bright_mask = freqs >= cutoff

    total_energy = np.sum(S ** 2, axis=0) + 1e-10
    bright_energy = np.sum(S[bright_mask, :] ** 2, axis=0)

    return bright_energy / total_energy


def compute_flatness(S: np.ndarray) -> np.ndarray:
    """
    Compute spectral flatness (Wiener entropy).

    Measures how noise-like vs tonal the signal is.
    - 0 = pure tone
    - 1 = white noise

    Args:
        S: Magnitude spectrogram

    Returns:
        Flatness per frame (0-1)
    """
    # Geometric mean / arithmetic mean
    S_pos = S + 1e-10  # Avoid log(0)

    geo_mean = np.exp(np.mean(np.log(S_pos), axis=0))
    arith_mean = np.mean(S_pos, axis=0)

    return geo_mean / (arith_mean + 1e-10)


def compute_flux(S: np.ndarray) -> np.ndarray:
    """
    Compute spectral flux (frame-to-frame change).

    High flux = rapid timbral changes (useful for drop detection).

    Args:
        S: Magnitude spectrogram

    Returns:
        Flux per frame
    """
    # L2 norm of difference between consecutive frames
    diff = np.diff(S, axis=1, prepend=S[:, :1])
    flux = np.sqrt(np.sum(diff ** 2, axis=0))
    return flux


def compute_bandwidth(
    S: np.ndarray,
    freqs: np.ndarray,
    centroid: Optional[np.ndarray] = None,
    p: int = 2
) -> np.ndarray:
    """
    Compute spectral bandwidth around centroid.

    Args:
        S: Magnitude spectrogram
        freqs: Frequency bins
        centroid: Pre-computed centroid (computed if None)
        p: Power for p-th order bandwidth

    Returns:
        Bandwidth per frame in Hz
    """
    if centroid is None:
        centroid = compute_centroid(S, freqs)

    # Weighted deviation from centroid
    deviation = np.abs(freqs[:, np.newaxis] - centroid) ** p
    norm = np.sum(S, axis=0) + 1e-10

    bandwidth = (np.sum(S * deviation, axis=0) / norm) ** (1/p)
    return bandwidth


def compute_contrast(
    S: np.ndarray,
    freqs: np.ndarray,
    n_bands: int = 7,
    fmin: float = 200.0
) -> np.ndarray:
    """
    Compute spectral contrast per octave band (VECTORIZED).

    Measures peak-to-valley difference in each band.

    M2 Optimized: Uses advanced indexing to avoid Python loops.

    Args:
        S: Magnitude spectrogram
        freqs: Frequency bins
        n_bands: Number of octave bands
        fmin: Minimum frequency

    Returns:
        Contrast (n_bands, n_frames)
    """
    S = np.ascontiguousarray(S, dtype=np.float32)
    n_frames = S.shape[1]

    # Define octave bands
    fmax = freqs[-1]
    band_edges = fmin * (2 ** np.arange(n_bands + 1))
    band_edges = np.clip(band_edges, fmin, fmax)

    contrast = np.zeros((n_bands, n_frames), dtype=np.float32)

    # Precompute band indices for all bands (vectorized boundary search)
    # Find frequency bin indices for each band edge
    edge_indices = np.searchsorted(freqs, band_edges)

    # Process all bands using vectorized slicing
    for i in range(n_bands):
        low_idx, high_idx = edge_indices[i], edge_indices[i + 1]

        if high_idx > low_idx:
            band_S = S[low_idx:high_idx, :]
            # Vectorized peak/valley along frequency axis
            peak = np.max(band_S, axis=0)
            valley = np.min(band_S, axis=0) + 1e-10
            contrast[i] = np.log10(peak / valley + 1e-10)

    return contrast


def compute_all_spectral(
    S: np.ndarray,
    freqs: np.ndarray,
    brightness_cutoff: float = 3000.0
) -> SpectralFeatures:
    """
    Compute all spectral features at once.

    More efficient than calling each function separately
    as centroid is reused for bandwidth.

    Args:
        S: Magnitude spectrogram
        freqs: Frequency bins
        brightness_cutoff: Cutoff for brightness calculation

    Returns:
        SpectralFeatures dataclass
    """
    centroid = compute_centroid(S, freqs)

    return SpectralFeatures(
        centroid=centroid,
        rolloff=compute_rolloff(S, freqs),
        brightness=compute_brightness(S, freqs, brightness_cutoff),
        flatness=compute_flatness(S),
        flux=compute_flux(S),
        bandwidth=compute_bandwidth(S, freqs, centroid),
        contrast=compute_contrast(S, freqs)
    )


def compute_spectral_slope(
    S: np.ndarray,
    freqs: np.ndarray
) -> np.ndarray:
    """
    Compute spectral slope (linear regression of spectrum).

    Negative slope = more energy in low frequencies (bass-heavy).

    Args:
        S: Magnitude spectrogram
        freqs: Frequency bins

    Returns:
        Slope per frame
    """
    # Normalize frequencies for numerical stability
    f_norm = (freqs - np.mean(freqs)) / (np.std(freqs) + 1e-10)

    # Linear regression slope for each frame
    f_var = np.sum(f_norm ** 2)
    S_log = np.log10(S + 1e-10)

    slopes = np.sum(f_norm[:, np.newaxis] * S_log, axis=0) / (f_var + 1e-10)
    return slopes


def compute_spectral_velocity(
    centroid: np.ndarray,
    sr: int,
    hop_length: int
) -> np.ndarray:
    """
    Compute rate of change of spectral centroid (Hz/sec).

    Used for filter sweep detection in DJ transitions.
    High velocity indicates rapid timbral changes (filter sweeps).

    Args:
        centroid: Spectral centroid per frame (Hz)
        sr: Sample rate
        hop_length: Hop length

    Returns:
        Centroid velocity per frame (Hz/sec)
    """
    # Time between frames
    frame_time = hop_length / sr

    # Compute gradient (Hz per frame) and convert to Hz/sec
    velocity = np.gradient(centroid) / frame_time

    return velocity


def compute_filter_position(
    centroid: np.ndarray,
    rolloff: np.ndarray
) -> np.ndarray:
    """
    Estimate filter position (0=closed/dark, 1=open/bright).

    Combines spectral centroid and rolloff to estimate
    the apparent filter position in DJ mixing.

    Useful for detecting:
    - Low-pass filter opening (mixin transitions)
    - High-pass filter closing (mixout transitions)
    - Filter sweeps during buildups

    Args:
        centroid: Spectral centroid per frame
        rolloff: Spectral rolloff per frame

    Returns:
        Estimated filter position per frame (0-1)
    """
    from .filtering import normalize_minmax

    # Normalize both to [0, 1]
    c_norm = normalize_minmax(centroid)
    r_norm = normalize_minmax(rolloff)

    # Weighted combination (centroid more important)
    filter_pos = 0.6 * c_norm + 0.4 * r_norm

    return filter_pos


def detect_filter_sweeps(
    centroid_velocity: np.ndarray,
    sr: int,
    hop_length: int,
    velocity_threshold: float = 500.0,
    min_duration_sec: float = 0.5
) -> np.ndarray:
    """
    Detect filter sweep regions in the audio (vectorized).

    Filter sweeps are characterized by sustained high
    centroid velocity (rapid brightness changes).

    Args:
        centroid_velocity: From compute_spectral_velocity()
        sr: Sample rate
        hop_length: Hop length
        velocity_threshold: Minimum velocity to consider (Hz/sec)
        min_duration_sec: Minimum sweep duration

    Returns:
        Boolean mask of filter sweep frames
    """
    from scipy.ndimage import binary_opening, generate_binary_structure

    # Detect high velocity regions (vectorized)
    high_velocity = np.abs(centroid_velocity) > velocity_threshold

    # Minimum frames for a valid sweep
    min_frames = max(1, int(min_duration_sec * sr / hop_length))

    # Use morphological opening to remove short segments (vectorized)
    # This removes regions shorter than min_frames
    structure = np.ones(min_frames, dtype=bool)
    sweep_mask = binary_opening(high_velocity, structure=structure)

    return sweep_mask
