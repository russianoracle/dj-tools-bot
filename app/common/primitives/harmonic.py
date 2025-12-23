"""
Harmonic Primitives - HPSS, harmonic ratio, key detection.

NOTE: MFCC, chroma, tonnetz computation moved to STFTCache.
Use context.stft_cache.get_mfcc(), get_chroma(), get_tonnetz() instead.

Apple Silicon M2 Optimized:
- Vectorized chroma computation
- Pre-built rotation matrices for key detection
"""

import numpy as np
from typing import Tuple, Optional


# =============================================================================
# DEPRECATED: Use STFTCache.get_mfcc(), get_chroma(), get_tonnetz() instead
# These functions are kept for backwards compatibility but should not be used.
# =============================================================================

def compute_mfcc(
    S: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    DEPRECATED: Use context.stft_cache.get_mfcc() instead.

    This function required librosa which violates the primitives architecture.
    """
    raise NotImplementedError(
        "compute_mfcc() is deprecated. Use context.stft_cache.get_mfcc() instead. "
        "See STFTCache in src/core/primitives/stft.py for lazy MFCC computation."
    )


def compute_mfcc_from_audio(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    DEPRECATED: Use compute_stft() + stft_cache.get_mfcc() instead.
    """
    raise NotImplementedError(
        "compute_mfcc_from_audio() is deprecated. "
        "Use compute_stft(y, sr).get_mfcc() instead."
    )


def compute_mfcc_delta(
    mfcc: np.ndarray,
    width: int = 9,
    order: int = 1
) -> np.ndarray:
    """
    DEPRECATED: Use context.stft_cache.get_mfcc_delta() instead.
    """
    raise NotImplementedError(
        "compute_mfcc_delta() is deprecated. "
        "Use context.stft_cache.get_mfcc_delta() instead."
    )


def compute_mfcc_stats(
    mfcc: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MFCC mean and std across time.

    Args:
        mfcc: MFCC features (n_mfcc, n_frames)

    Returns:
        Tuple of (mean, std) each of shape (n_mfcc,)
    """
    return np.mean(mfcc, axis=1), np.std(mfcc, axis=1)


def compute_chroma(
    S: np.ndarray,
    sr: int,
    n_chroma: int = 12,
    tuning: Optional[float] = None
) -> np.ndarray:
    """
    DEPRECATED: Use context.stft_cache.get_chroma() instead.

    This function required librosa which violates the primitives architecture.
    """
    raise NotImplementedError(
        "compute_chroma() is deprecated. Use context.stft_cache.get_chroma() instead. "
        "See STFTCache in src/core/primitives/stft.py for lazy chroma computation."
    )


def compute_chroma_from_audio(
    y: np.ndarray,
    sr: int,
    n_chroma: int = 12,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    DEPRECATED: Use compute_stft(y, sr).get_chroma() instead.
    """
    raise NotImplementedError(
        "compute_chroma_from_audio() is deprecated. "
        "Use compute_stft(y, sr).get_chroma() instead."
    )


def compute_chroma_cens(
    chroma: np.ndarray,
    win_len: int = 41,
    n_octaves: int = 7
) -> np.ndarray:
    """
    Compute chroma CENS (Chroma Energy Normalized Statistics).

    More robust to timbre variations than raw chroma.

    Args:
        chroma: Raw chromagram
        win_len: Smoothing window length
        n_octaves: Number of octaves for energy quantization

    Returns:
        CENS chromagram
    """
    # L1 normalize
    chroma_norm = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-10)

    # Quantize to discrete levels
    quantized = np.round(chroma_norm * n_octaves) / n_octaves

    # Smooth with Hann window
    from scipy.ndimage import uniform_filter1d
    cens = uniform_filter1d(quantized, size=win_len, axis=1)

    # L2 normalize final result
    cens = cens / (np.linalg.norm(cens, axis=0, keepdims=True) + 1e-10)

    return cens


def compute_tonnetz(
    chroma: np.ndarray
) -> np.ndarray:
    """
    DEPRECATED: Use context.stft_cache.get_tonnetz() instead.

    This function required librosa which violates the primitives architecture.
    """
    raise NotImplementedError(
        "compute_tonnetz() is deprecated. Use context.stft_cache.get_tonnetz() instead. "
        "See STFTCache in src/core/primitives/stft.py for lazy tonnetz computation."
    )


def compute_hpss(
    S: np.ndarray,
    kernel_size: int = 31,
    power: float = 2.0,
    margin: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Harmonic-Percussive Source Separation.

    Separates spectrogram into harmonic and percussive components.

    Args:
        S: Magnitude spectrogram
        kernel_size: Size of median filter kernels
        power: Exponent for Wiener filtering
        margin: Margin for soft masking

    Returns:
        Tuple of (S_harmonic, S_percussive)
    """
    from scipy.ndimage import median_filter

    # Median filter along time (harmonic)
    S_harmonic_mask = median_filter(S, size=(1, kernel_size))

    # Median filter along frequency (percussive)
    S_percussive_mask = median_filter(S, size=(kernel_size, 1))

    # Soft masks with margin
    mask_harmonic = S_harmonic_mask ** power / (
        S_harmonic_mask ** power + S_percussive_mask ** power + 1e-10
    )
    mask_percussive = S_percussive_mask ** power / (
        S_harmonic_mask ** power + S_percussive_mask ** power + 1e-10
    )

    # Apply margin
    mask_harmonic = np.minimum(1.0, mask_harmonic * margin)
    mask_percussive = np.minimum(1.0, mask_percussive * margin)

    S_harmonic = mask_harmonic * S
    S_percussive = mask_percussive * S

    return S_harmonic, S_percussive


def compute_harmonic_ratio(
    S_harmonic: np.ndarray,
    S_percussive: np.ndarray
) -> np.ndarray:
    """
    Compute harmonic-to-percussive ratio per frame.

    Args:
        S_harmonic: Harmonic spectrogram
        S_percussive: Percussive spectrogram

    Returns:
        Ratio per frame (>1 = more harmonic, <1 = more percussive)
    """
    h_energy = np.sum(S_harmonic ** 2, axis=0)
    p_energy = np.sum(S_percussive ** 2, axis=0) + 1e-10

    return h_energy / p_energy


def compute_key(chroma: np.ndarray) -> Tuple[str, float]:
    """
    Estimate musical key from chromagram.

    M2 Optimized: Vectorized correlation using pre-built rotation matrices.

    Args:
        chroma: Chromagram (12, n_frames)

    Returns:
        Tuple of (key_name, confidence)
    """
    # Major and minor key profiles (Krumhansl-Schmuckler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Average chroma over time
    chroma_avg = np.mean(chroma, axis=1).astype(np.float32)

    # Build all 12 rotations at once: (12, 12) matrices
    # Each row i is np.roll(profile, i)
    indices = np.arange(12)
    roll_indices = (indices[:, np.newaxis] - indices[np.newaxis, :]) % 12

    major_rotations = major_profile[roll_indices]  # (12, 12)
    minor_rotations = minor_profile[roll_indices]  # (12, 12)

    # Vectorized Pearson correlation for all 24 keys at once
    # corr(x, y) = cov(x,y) / (std(x) * std(y))
    # For normalized vectors: corr = dot(x_centered, y_centered) / (n * std_x * std_y)

    # Normalize chroma_avg
    chroma_centered = chroma_avg - np.mean(chroma_avg)
    chroma_std = np.std(chroma_avg) + 1e-10

    # Normalize all rotations (row-wise)
    major_centered = major_rotations - np.mean(major_rotations, axis=1, keepdims=True)
    major_std = np.std(major_rotations, axis=1) + 1e-10  # (12,)

    minor_centered = minor_rotations - np.mean(minor_rotations, axis=1, keepdims=True)
    minor_std = np.std(minor_rotations, axis=1) + 1e-10  # (12,)

    # Vectorized dot products: (12,) correlations each
    major_corrs = np.dot(major_centered, chroma_centered) / (12 * major_std * chroma_std)
    minor_corrs = np.dot(minor_centered, chroma_centered) / (12 * minor_std * chroma_std)

    # Find best key across all 24 options
    best_major_idx = np.argmax(major_corrs)
    best_minor_idx = np.argmax(minor_corrs)

    best_major_corr = major_corrs[best_major_idx]
    best_minor_corr = minor_corrs[best_minor_idx]

    if best_major_corr >= best_minor_corr:
        best_key = key_names[best_major_idx]
        best_corr = best_major_corr
        is_minor = False
    else:
        best_key = key_names[best_minor_idx]
        best_corr = best_minor_corr
        is_minor = True

    key_name = f"{best_key}m" if is_minor else best_key
    return key_name, float(max(0, best_corr))
