"""
Energy Primitives - RMS, frequency bands, energy analysis.

Apple Silicon M2 Optimized:
- Vectorized operations using Apple Accelerate
- Avoid Python loops
- Contiguous memory access patterns

All functions are pure mathematical operations on numpy arrays.
No external audio library calls (librosa, essentia, etc.)

DEPRECATION WARNING:
    compute_rms() uses different algorithm than STFTCache.get_rms()
    For consistency, use: cache.get_rms() instead
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

from .filtering import normalize_minmax


@dataclass
class FrequencyBands:
    """
    Energy in different frequency bands.

    Standard DJ-relevant frequency ranges:
        sub_bass: 20-60 Hz (subwoofer territory)
        bass: 60-250 Hz (kick drums, bass)
        low_mid: 250-500 Hz (warmth)
        mid: 500-2000 Hz (vocals, snares)
        high_mid: 2000-4000 Hz (presence)
        high: 4000-20000 Hz (brightness, air)
    """
    sub_bass: np.ndarray    # 20-60 Hz
    bass: np.ndarray        # 60-250 Hz
    low_mid: np.ndarray     # 250-500 Hz
    mid: np.ndarray         # 500-2000 Hz
    high_mid: np.ndarray    # 2000-4000 Hz
    high: np.ndarray        # 4000-20000 Hz

    def to_array(self) -> np.ndarray:
        """Stack all bands into (6, n_frames) array."""
        return np.vstack([
            self.sub_bass, self.bass, self.low_mid,
            self.mid, self.high_mid, self.high
        ])

    @property
    def bass_to_high_ratio(self) -> np.ndarray:
        """Ratio of bass energy to high frequency energy."""
        high_energy = self.high_mid + self.high + 1e-10
        bass_energy = self.sub_bass + self.bass + 1e-10
        return bass_energy / high_energy


def compute_rms(S: np.ndarray, _warn: bool = True) -> np.ndarray:
    """
    Compute RMS energy from magnitude spectrogram.

    DEPRECATED: This is a legacy wrapper. Use STFTCache.get_rms() directly.
    For backward compatibility, this function uses the SAME algorithm as STFTCache.get_rms():
        RMS = sqrt(mean(S^2)) along frequency axis

    M2-optimized: Ensures contiguous array for Apple Accelerate BLAS.

    Args:
        S: Magnitude spectrogram (n_freq, n_frames)
        _warn: Show deprecation warning (internal use)

    Returns:
        RMS energy per frame (n_frames,)
    """
    if _warn:
        warnings.warn(
            "compute_rms() is deprecated. Use STFTCache.get_rms() for consistency.",
            DeprecationWarning,
            stacklevel=2
        )
    # Ensure contiguous for Apple Accelerate
    S = np.ascontiguousarray(S, dtype=np.float32)

    # RMS = sqrt(mean(S^2)) along frequency axis
    # Same algorithm as STFTCache.get_rms() for backward compatibility
    return np.sqrt(np.mean(S ** 2, axis=0))


def compute_rms_from_audio(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Compute RMS directly from audio signal.

    M2-optimized: Ensures contiguous array for Apple Accelerate.

    Args:
        y: Audio time series
        frame_length: Frame length for analysis
        hop_length: Hop between frames

    Returns:
        RMS energy per frame
    """
    # Ensure contiguous for Apple Accelerate
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Pad signal
    y_padded = np.pad(y, (frame_length // 2, frame_length // 2), mode='reflect')

    # Frame the signal
    n_frames = 1 + (len(y_padded) - frame_length) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(
        y_padded, frame_length
    )[::hop_length][:n_frames]

    # Compute RMS per frame
    return np.sqrt(np.mean(frames ** 2, axis=1))


def compute_band_energy(
    S: np.ndarray,
    freqs: np.ndarray,
    low: float,
    high: float
) -> np.ndarray:
    """
    Compute energy in a specific frequency band.

    Args:
        S: Magnitude spectrogram (n_freq, n_frames)
        freqs: Frequency bins in Hz
        low: Lower frequency bound
        high: Upper frequency bound

    Returns:
        Energy per frame in the specified band
    """
    # Find frequency indices
    band_mask = (freqs >= low) & (freqs <= high)

    if not np.any(band_mask):
        return np.zeros(S.shape[1])

    # Sum energy in band
    return np.sum(S[band_mask, :] ** 2, axis=0)


def compute_frequency_bands(
    S: np.ndarray,
    freqs: np.ndarray
) -> FrequencyBands:
    """
    Compute energy in all standard frequency bands.

    Args:
        S: Magnitude spectrogram
        freqs: Frequency bins

    Returns:
        FrequencyBands dataclass with energy per band
    """
    return FrequencyBands(
        sub_bass=compute_band_energy(S, freqs, 20, 60),
        bass=compute_band_energy(S, freqs, 60, 250),
        low_mid=compute_band_energy(S, freqs, 250, 500),
        mid=compute_band_energy(S, freqs, 500, 2000),
        high_mid=compute_band_energy(S, freqs, 2000, 4000),
        high=compute_band_energy(S, freqs, 4000, 20000)
    )


def compute_energy_derivative(
    energy: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    Compute derivative of energy curve.

    Args:
        energy: Energy curve (n_frames,)
        order: Derivative order (1 or 2)

    Returns:
        Energy derivative
    """
    if order == 1:
        return np.diff(energy, prepend=energy[0])
    elif order == 2:
        d1 = np.diff(energy, prepend=energy[0])
        return np.diff(d1, prepend=d1[0])
    else:
        raise ValueError(f"Order must be 1 or 2, got {order}")


def detect_low_energy_frames(
    rms: np.ndarray,
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Detect frames with energy below threshold.

    Args:
        rms: RMS energy per frame
        threshold: Energy threshold (default: mean energy)

    Returns:
        Boolean mask of low-energy frames
    """
    if threshold is None:
        threshold = np.mean(rms)

    return rms < threshold


def compute_low_energy_ratio(rms: np.ndarray) -> float:
    """
    Compute ratio of low-energy frames.

    This is a key feature for zone classification:
    - Yellow (calm): high low-energy ratio
    - Purple (energetic): low low-energy ratio

    Args:
        rms: RMS energy per frame

    Returns:
        Fraction of frames below mean energy
    """
    low_mask = detect_low_energy_frames(rms)
    return float(np.sum(low_mask) / len(rms))


def compute_energy_variance(rms: np.ndarray) -> float:
    """
    Compute normalized energy variance.

    Uses coefficient of variation (std/mean) squared.
    This is scale-independent and good for classification.

    Args:
        rms: RMS energy per frame

    Returns:
        Normalized energy variance
    """
    mean_rms = np.mean(rms)
    if mean_rms > 0:
        cv = np.std(rms) / mean_rms
        return float(cv ** 2)
    return 0.0


def compute_dynamic_range(rms: np.ndarray, percentile: float = 95) -> float:
    """
    Compute dynamic range of the signal.

    Args:
        rms: RMS energy per frame
        percentile: Upper percentile for range calculation

    Returns:
        Dynamic range in dB
    """
    rms_nonzero = rms[rms > 0]
    if len(rms_nonzero) == 0:
        return 0.0

    high = np.percentile(rms_nonzero, percentile)
    low = np.percentile(rms_nonzero, 100 - percentile)

    if low > 0:
        return float(20 * np.log10(high / low))
    return 0.0


@dataclass
class MelBandEnergies:
    """
    Energy in DJ-relevant mel frequency bands.

    Optimized for drop detection in electronic music:
        bass: 0-250 Hz (kick drums, sub bass) - mels 0-12
        kick: 250-500 Hz (punch, body) - mels 12-20
        mid: 500-2000 Hz (vocals, snares) - mels 20-50
        high: 2000-8000 Hz (brightness, hats) - mels 50-90
        presence: 8000+ Hz (air, shimmer) - mels 90+
    """
    bass: np.ndarray        # 0-250 Hz
    kick: np.ndarray        # 250-500 Hz
    mid: np.ndarray         # 500-2000 Hz
    high: np.ndarray        # 2000-8000 Hz
    presence: np.ndarray    # 8000+ Hz

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            'bass': self.bass,
            'kick': self.kick,
            'mid': self.mid,
            'high': self.high,
            'presence': self.presence
        }


def compute_mel_band_energies(
    S_mel_db: np.ndarray,
    n_mels: int = 128
) -> MelBandEnergies:
    """
    Extract energy from mel spectrogram frequency bands.

    Pure mathematical operation on pre-computed mel spectrogram.
    Mel spectrogram should be computed in Task layer using librosa.

    Args:
        S_mel_db: Mel spectrogram in dB scale (n_mels, n_frames)
        n_mels: Number of mel bands (for band boundary calculation)

    Returns:
        MelBandEnergies dataclass with energy per band
    """
    # Mel band indices (approximate for 128 mels at 22050 Hz)
    # These indices correspond to DJ-relevant frequency ranges
    return MelBandEnergies(
        bass=np.mean(S_mel_db[0:12], axis=0),        # 0-250 Hz
        kick=np.mean(S_mel_db[12:20], axis=0),       # 250-500 Hz
        mid=np.mean(S_mel_db[20:50], axis=0),        # 500-2000 Hz
        high=np.mean(S_mel_db[50:90], axis=0),       # 2000-8000 Hz
        presence=np.mean(S_mel_db[90:], axis=0) if n_mels > 90 else np.zeros(S_mel_db.shape[1])
    )


def compute_weighted_energy(
    mel_bands: MelBandEnergies,
    rms: np.ndarray,
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Compute weighted combination of band energies for drop detection.

    Pure mathematical operation combining normalized band energies.

    Default weights optimized for electronic music drops:
    - bass: 0.35 (most important for drops)
    - kick: 0.25 (punch of the drop)
    - mid: 0.20 (fullness)
    - high: 0.10 (brightness)
    - rms: 0.10 (overall energy)

    Args:
        mel_bands: MelBandEnergies from compute_mel_band_energies()
        rms: RMS energy per frame
        weights: Optional custom weights dict

    Returns:
        Weighted combined energy per frame (n_frames,)
    """
    if weights is None:
        weights = {
            'bass': 0.35,
            'kick': 0.25,
            'mid': 0.20,
            'high': 0.10,
            'rms': 0.10
        }

    # Normalize each component to [0, 1] before combining
    combined = np.zeros_like(rms, dtype=np.float64)

    if weights.get('bass', 0) > 0:
        combined += weights['bass'] * normalize_minmax(mel_bands.bass)

    if weights.get('kick', 0) > 0:
        combined += weights['kick'] * normalize_minmax(mel_bands.kick)

    if weights.get('mid', 0) > 0:
        combined += weights['mid'] * normalize_minmax(mel_bands.mid)

    if weights.get('high', 0) > 0:
        combined += weights['high'] * normalize_minmax(mel_bands.high)

    if weights.get('rms', 0) > 0:
        combined += weights['rms'] * normalize_minmax(rms)

    return combined
