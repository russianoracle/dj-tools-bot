"""
Filtering Primitives - Smoothing, normalization, delta computation.

Apple Silicon M2 Optimized:
- Use scipy.ndimage for efficient filtering
- Vectorized normalization
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, convolve1d
from scipy.signal import savgol_filter
from typing import Optional


def smooth_gaussian(
    x: np.ndarray,
    sigma: float = 3.0,
    axis: int = -1
) -> np.ndarray:
    """
    Apply Gaussian smoothing.

    Args:
        x: Input array
        sigma: Gaussian standard deviation
        axis: Axis along which to smooth

    Returns:
        Smoothed array
    """
    return gaussian_filter1d(x, sigma=sigma, axis=axis)


def smooth_uniform(
    x: np.ndarray,
    size: int = 5,
    axis: int = -1
) -> np.ndarray:
    """
    Apply uniform (box) smoothing.

    Args:
        x: Input array
        size: Window size
        axis: Axis along which to smooth

    Returns:
        Smoothed array
    """
    return uniform_filter1d(x, size=size, axis=axis)


def smooth_savgol(
    x: np.ndarray,
    window: int = 11,
    order: int = 3,
    axis: int = -1
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing.

    Preserves peaks better than Gaussian/uniform smoothing.

    Args:
        x: Input array
        window: Window length (must be odd)
        order: Polynomial order
        axis: Axis along which to smooth

    Returns:
        Smoothed array
    """
    if window % 2 == 0:
        window += 1

    if window <= order:
        window = order + 2
        if window % 2 == 0:
            window += 1

    return savgol_filter(x, window_length=window, polyorder=order, axis=axis)


def normalize_minmax(
    x: np.ndarray,
    axis: Optional[int] = None,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Min-max normalization to [0, 1] range.

    Args:
        x: Input array
        axis: Axis along which to normalize (None = global)
        eps: Small value to prevent division by zero

    Returns:
        Normalized array
    """
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)

    return (x - x_min) / (x_max - x_min + eps)


def normalize_zscore(
    x: np.ndarray,
    axis: Optional[int] = None,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Z-score (standard) normalization.

    Args:
        x: Input array
        axis: Axis along which to normalize
        eps: Small value to prevent division by zero

    Returns:
        Normalized array with mean=0, std=1
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)

    return (x - mean) / (std + eps)


def normalize_l2(
    x: np.ndarray,
    axis: int = -1,
    eps: float = 1e-10
) -> np.ndarray:
    """
    L2 (unit vector) normalization.

    Args:
        x: Input array
        axis: Axis along which to normalize
        eps: Small value to prevent division by zero

    Returns:
        Array with unit L2 norm along specified axis
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def compute_delta(
    x: np.ndarray,
    width: int = 9,
    axis: int = -1
) -> np.ndarray:
    """
    Compute first derivative (delta) features.

    Uses local regression with specified width.
    Vectorized implementation using convolve1d ~5-10x faster than loop version.

    Args:
        x: Input array (features x frames)
        width: Window width for delta computation
        axis: Time axis

    Returns:
        Delta features (same shape as input)
    """
    # Ensure width is odd
    if width % 2 == 0:
        width += 1

    half = width // 2

    # Compute weights for linear regression (Bartlett window)
    t = np.arange(-half, half + 1, dtype=np.float32)
    denom = np.sum(t ** 2) + 1e-10

    # Normalize weights
    weights = t / denom

    # Use scipy's convolve1d for efficient convolution
    # mode='nearest' equivalent to 'edge' padding
    delta = convolve1d(x, weights, axis=axis, mode='nearest')

    return delta


def compute_delta2(
    x: np.ndarray,
    width: int = 9,
    axis: int = -1
) -> np.ndarray:
    """
    Compute second derivative (delta-delta) features.

    Args:
        x: Input array
        width: Window width
        axis: Time axis

    Returns:
        Delta-delta features
    """
    delta1 = compute_delta(x, width=width, axis=axis)
    return compute_delta(delta1, width=width, axis=axis)


def clip_outliers(
    x: np.ndarray,
    percentile: float = 99,
    axis: Optional[int] = None
) -> np.ndarray:
    """
    Clip outliers beyond specified percentile.

    Args:
        x: Input array
        percentile: Upper percentile (e.g., 99 clips top 1%)
        axis: Axis along which to compute percentiles

    Returns:
        Clipped array
    """
    lower = np.percentile(x, 100 - percentile, axis=axis, keepdims=True)
    upper = np.percentile(x, percentile, axis=axis, keepdims=True)

    return np.clip(x, lower, upper)


def interpolate_nans(
    x: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate NaN values in an array.

    Args:
        x: Input array (1D)
        method: Interpolation method ('linear', 'nearest', 'zero')

    Returns:
        Array with NaNs interpolated
    """
    if x.ndim != 1:
        raise ValueError("interpolate_nans only works on 1D arrays")

    nans = np.isnan(x)
    if not np.any(nans):
        return x

    indices = np.arange(len(x))
    valid = ~nans

    if not np.any(valid):
        return np.zeros_like(x)

    result = x.copy()
    result[nans] = np.interp(indices[nans], indices[valid], x[valid])

    return result


def pad_or_truncate(
    x: np.ndarray,
    target_length: int,
    axis: int = -1,
    mode: str = 'constant'
) -> np.ndarray:
    """
    Pad or truncate array to target length.

    Args:
        x: Input array
        target_length: Desired length
        axis: Axis to pad/truncate
        mode: Padding mode ('constant', 'edge', 'reflect')

    Returns:
        Array with target length
    """
    current_length = x.shape[axis]

    if current_length == target_length:
        return x

    if current_length > target_length:
        # Truncate
        idx = [slice(None)] * x.ndim
        idx[axis] = slice(0, target_length)
        return x[tuple(idx)]
    else:
        # Pad
        pad_amount = target_length - current_length
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (0, pad_amount)
        return np.pad(x, pad_width, mode=mode)


def resample_features(
    x: np.ndarray,
    target_frames: int,
    axis: int = -1
) -> np.ndarray:
    """
    Resample features to target number of frames (VECTORIZED).

    M2 Optimized: Uses scipy.interpolate for vectorized multi-dimensional
    interpolation instead of Python loops.

    Args:
        x: Input features (... x n_frames)
        target_frames: Target number of frames
        axis: Time axis

    Returns:
        Resampled features
    """
    current_frames = x.shape[axis]

    if current_frames == target_frames:
        return x

    # Use linear interpolation
    old_indices = np.linspace(0, 1, current_frames)
    new_indices = np.linspace(0, 1, target_frames)

    if x.ndim == 1:
        return np.interp(new_indices, old_indices, x)

    # Handle multi-dimensional arrays (VECTORIZED)
    if axis == -1 or axis == x.ndim - 1:
        # Reshape to 2D: (all_other_dims, time_frames)
        original_shape = x.shape
        x_2d = x.reshape(-1, current_frames)  # (N, current_frames)

        # Vectorized interpolation using numpy interp with apply_along_axis
        # More efficient: use scipy.ndimage.map_coordinates concept
        # Map new indices to old space
        scale = (current_frames - 1) / (target_frames - 1) if target_frames > 1 else 0
        mapped_indices = np.arange(target_frames) * scale

        # Get integer and fractional parts for linear interpolation
        idx_low = np.floor(mapped_indices).astype(np.int64)
        idx_high = np.minimum(idx_low + 1, current_frames - 1)
        frac = mapped_indices - idx_low

        # Vectorized interpolation: result = (1-frac) * x[low] + frac * x[high]
        # Shape: (N, target_frames)
        result_2d = (1 - frac) * x_2d[:, idx_low] + frac * x_2d[:, idx_high]

        # Reshape back to original shape (except last dim)
        result = result_2d.reshape(original_shape[:-1] + (target_frames,))
    else:
        # Move axis to end, resample, move back
        x_moved = np.moveaxis(x, axis, -1)
        result_moved = resample_features(x_moved, target_frames, axis=-1)
        result = np.moveaxis(result_moved, -1, axis)

    return result
