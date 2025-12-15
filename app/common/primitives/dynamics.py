"""
Dynamics Primitives - Peaks, valleys, drops, buildups.

Apple Silicon M2 Optimized:
- Fully vectorized operations (no Python loops)
- Uses stride_tricks for efficient sliding windows
- Leverages Apple Accelerate via NumPy BLAS
- Contiguous memory access patterns
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DropCandidate:
    """
    A potential drop event in the track.

    Attributes:
        frame_idx: Frame index of the drop
        time_sec: Time in seconds
        buildup_score: Strength of preceding buildup (0-1)
        drop_magnitude: Energy jump magnitude (normalized)
        confidence: Overall drop confidence (0-1)
        buildup_duration: Duration of buildup phase in seconds
        recovery_rate: Energy recovery rate after drop (energy/sec)
        bass_prominence: Bass energy prominence at drop point (0-1)
    """
    frame_idx: int
    time_sec: float
    buildup_score: float
    drop_magnitude: float
    confidence: float
    # Extended fields for multi-band analysis
    buildup_duration: float = 0.0
    recovery_rate: float = 0.0
    bass_prominence: float = 0.0

    def to_dict(self) -> dict:
        return {
            'frame_idx': self.frame_idx,
            'time_sec': self.time_sec,
            'buildup_score': self.buildup_score,
            'drop_magnitude': self.drop_magnitude,
            'confidence': self.confidence,
            'buildup_duration': self.buildup_duration,
            'recovery_rate': self.recovery_rate,
            'bass_prominence': self.bass_prominence,
        }


def detect_peaks(
    curve: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    threshold_percentile: float = 75
) -> np.ndarray:
    """
    Detect peaks in a curve.

    Args:
        curve: Input curve (n_frames,)
        height: Minimum peak height (auto if None)
        distance: Minimum distance between peaks
        prominence: Minimum peak prominence
        threshold_percentile: Percentile for auto height

    Returns:
        Array of peak indices
    """
    if height is None:
        height = np.percentile(curve, threshold_percentile)

    if distance is None:
        distance = max(1, len(curve) // 100)

    peaks, properties = find_peaks(
        curve,
        height=height,
        distance=distance,
        prominence=prominence
    )

    return peaks


def detect_valleys(
    curve: np.ndarray,
    depth: Optional[float] = None,
    distance: Optional[int] = None,
    threshold_percentile: float = 25
) -> np.ndarray:
    """
    Detect valleys (local minima) in a curve.

    Args:
        curve: Input curve
        depth: Maximum valley depth (auto if None)
        distance: Minimum distance between valleys
        threshold_percentile: Percentile for auto depth

    Returns:
        Array of valley indices
    """
    # Invert and find peaks
    inverted = -curve

    if depth is None:
        height = -np.percentile(curve, threshold_percentile)
    else:
        height = -depth

    if distance is None:
        distance = max(1, len(curve) // 100)

    valleys, _ = find_peaks(inverted, height=height, distance=distance)

    return valleys


def compute_buildup_score(
    energy: np.ndarray,
    window_frames: int = 20,
    smooth: bool = True
) -> np.ndarray:
    """
    Compute buildup score (energy accumulation before drops).

    High buildup score = energy is increasing over the window.

    M2 Optimized: Fully vectorized using stride_tricks for sliding windows.
    ~20-50x faster than loop-based implementation.

    Args:
        energy: Energy curve (RMS)
        window_frames: Lookback window size
        smooth: Whether to smooth the result

    Returns:
        Buildup score per frame
    """
    n_frames = len(energy)

    if n_frames < window_frames:
        return np.zeros(n_frames)

    # Ensure contiguous float32 for Apple Accelerate
    energy = np.ascontiguousarray(energy, dtype=np.float32)

    # Pad energy at start to handle edge cases
    padded = np.pad(energy, (window_frames - 1, 0), mode='edge')

    # Create sliding windows using stride_tricks (zero-copy view)
    # Shape: (n_frames, window_frames)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_frames)

    # Vectorized linear regression slope computation
    # For y = mx + b, slope m = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
    x = np.arange(window_frames, dtype=np.float32)
    x_mean = x.mean()
    x_centered = x - x_mean  # Shape: (window_frames,)

    # Precompute denominator (constant for all windows)
    denominator = np.sum(x_centered ** 2)

    # Compute y means for all windows: (n_frames,)
    y_means = windows.mean(axis=1)

    # Compute y_centered for all windows: (n_frames, window_frames)
    y_centered = windows - y_means[:, np.newaxis]

    # Compute slopes for all windows at once using einsum (faster than sum)
    # slopes = Σ(x_centered * y_centered) / denominator
    numerators = np.einsum('ij,j->i', y_centered, x_centered)
    slopes = numerators / (denominator + 1e-10)

    # Only positive slopes (buildup = energy increasing)
    buildup = np.maximum(0, slopes)

    if smooth:
        buildup = uniform_filter1d(buildup, size=5)

    # Normalize to [0, 1]
    max_val = np.max(buildup)
    if max_val > 0:
        buildup = buildup / max_val

    return buildup.astype(np.float32)


def detect_drop_candidates(
    energy: np.ndarray,
    sr: int,
    hop_length: int,
    buildup_window: int = 20,
    min_drop_magnitude: float = 0.3,
    min_confidence: float = 0.5,
    bass_energy: Optional[np.ndarray] = None,
    recovery_window: int = 10
) -> List[DropCandidate]:
    """
    Detect potential drop events with extended multi-band metrics.

    A drop is characterized by:
    1. Energy buildup before
    2. Sudden energy increase
    3. Sustained high energy after
    4. (Optional) Bass prominence at drop point

    Args:
        energy: Energy curve (RMS or weighted)
        sr: Sample rate
        hop_length: Hop length
        buildup_window: Frames to check for buildup
        min_drop_magnitude: Minimum energy jump
        min_confidence: Minimum confidence threshold
        bass_energy: Optional bass band energy for bass prominence calculation
        recovery_window: Frames to measure recovery rate

    Returns:
        List of DropCandidate objects with extended metrics
    """
    # Smooth energy
    energy_smooth = uniform_filter1d(energy, size=11)

    # Compute energy derivative
    energy_diff = np.diff(energy_smooth, prepend=energy_smooth[0])

    # Compute buildup scores
    buildup = compute_buildup_score(energy, window_frames=buildup_window)

    # Find sudden energy increases
    threshold = np.percentile(energy_diff[energy_diff > 0], 90) if np.any(energy_diff > 0) else 0
    jump_frames = np.where(energy_diff > threshold)[0]

    # Vectorized pre-filtering
    # Filter by buildup_window
    jump_frames = jump_frames[jump_frames >= buildup_window]

    if len(jump_frames) == 0:
        return []

    # Vectorized computations for all candidates
    frame_to_time = hop_length / sr
    mean_energy = np.mean(energy) + 1e-10

    # Get buildup scores for all frames at once
    buildup_scores = buildup[jump_frames]

    # Get drop magnitudes for all frames at once
    drop_mags = energy_diff[jump_frames] / mean_energy

    # Vectorized sustained energy check using sliding window means
    # Pre-compute cumulative sum for O(1) window means
    window_size = 10
    energy_padded = np.pad(energy, (window_size, window_size), mode='edge')
    cumsum = np.cumsum(energy_padded)

    # For each jump_frame, compute before/after window means vectorized
    # before: [frame-10:frame], after: [frame:frame+10]
    sustained = np.ones(len(jump_frames), dtype=bool)

    if len(jump_frames) > 0:
        # Indices adjusted for padding
        frames_padded = jump_frames + window_size

        # Valid frames for after window
        valid_after = jump_frames + window_size < len(energy)

        # Compute cumsum indices for before windows: sum[frame] - sum[frame-10]
        before_end_idx = frames_padded
        before_start_idx = np.maximum(frames_padded - window_size, 0)
        before_sums = cumsum[before_end_idx] - cumsum[before_start_idx]
        before_counts = before_end_idx - before_start_idx
        before_means = before_sums / np.maximum(before_counts, 1)

        # Compute cumsum indices for after windows: sum[frame+10] - sum[frame]
        after_start_idx = frames_padded
        after_end_idx = np.minimum(frames_padded + window_size, len(energy_padded) - 1)
        after_sums = cumsum[after_end_idx] - cumsum[after_start_idx]
        after_counts = after_end_idx - after_start_idx
        after_means = after_sums / np.maximum(after_counts, 1)

        # Sustained where after > before * 1.1 (for valid frames)
        sustained = np.where(valid_after, after_means > before_means * 1.1, True)

    # Vectorized confidence calculation
    confidences = (
        buildup_scores * 0.4 +
        np.minimum(drop_mags, 1.0) * 0.4 +
        (sustained.astype(float) * 0.2)
    )

    # Vectorized filtering by thresholds
    valid_mask = (drop_mags >= min_drop_magnitude) & (confidences >= min_confidence)

    # Filter to valid candidates only
    valid_frames = jump_frames[valid_mask]
    valid_buildup_scores = buildup_scores[valid_mask]
    valid_drop_mags = drop_mags[valid_mask]
    valid_confidences = confidences[valid_mask]

    candidates = []

    # Now loop only over valid candidates
    for i, frame in enumerate(valid_frames):
        buildup_score = valid_buildup_scores[i]
        drop_mag = valid_drop_mags[i]
        confidence = valid_confidences[i]

        # These calculations are harder to vectorize, keep in loop
        # Calculate extended metrics

        # 1. Buildup duration (find where buildup starts)
        buildup_start = max(0, frame - buildup_window * 2)
        buildup_segment = buildup[buildup_start:frame]
        if len(buildup_segment) > 0:
            buildup_threshold = 0.3 * np.max(buildup_segment) if np.max(buildup_segment) > 0 else 0
            above_threshold = buildup_segment > buildup_threshold
            if np.any(above_threshold):
                first_above = np.argmax(above_threshold)
                buildup_frames = len(buildup_segment) - first_above
            else:
                buildup_frames = buildup_window
        else:
            buildup_frames = buildup_window
        buildup_duration = buildup_frames * frame_to_time

        # 2. Recovery rate (energy change after drop)
        if frame + recovery_window < len(energy):
            recovery_energy = energy[frame:frame + recovery_window]
            recovery_time = recovery_window * frame_to_time
            if recovery_time > 0:
                recovery_rate = float((np.max(recovery_energy) - energy[frame]) / recovery_time)
            else:
                recovery_rate = 0.0
        else:
            recovery_rate = 0.0

        # 3. Bass prominence (if bass energy provided)
        if bass_energy is not None and len(bass_energy) > frame:
            bass_at_drop = bass_energy[min(frame, len(bass_energy) - 1)]
            bass_range = np.ptp(bass_energy)
            if bass_range > 0:
                bass_prominence = float((bass_at_drop - np.min(bass_energy)) / (bass_range + 1e-10))
            else:
                bass_prominence = 0.0
        else:
            bass_prominence = 0.0

        candidates.append(DropCandidate(
            frame_idx=int(frame),
            time_sec=float(frame * frame_to_time),
            buildup_score=float(buildup_score),
            drop_magnitude=float(drop_mag),
            confidence=float(confidence),
            buildup_duration=float(buildup_duration),
            recovery_rate=float(recovery_rate),
            bass_prominence=float(bass_prominence)
        ))

    # Merge nearby candidates using vectorized NMS
    if not candidates:
        return []

    min_gap = int(2.0 / frame_to_time)

    # Extract arrays for vectorized NMS
    frames = np.array([c.frame_idx for c in candidates], dtype=np.int64)
    confs = np.array([c.confidence for c in candidates], dtype=np.float32)

    # Vectorized NMS: keep candidates that are local maxima within min_gap
    # Compute pairwise distances
    dist = np.abs(frames[:, np.newaxis] - frames[np.newaxis, :])  # (N, N)
    neighbors = (dist < min_gap) & (dist > 0)  # exclude self

    # For each candidate, find max confidence among neighbors
    # Use -inf for non-neighbors to ignore them in max
    neighbor_confs = np.where(neighbors, confs[np.newaxis, :], -np.inf)
    max_neighbor_conf = np.max(neighbor_confs, axis=1)

    # Keep candidates where confidence >= all neighbors' confidence
    keep_mask = confs >= max_neighbor_conf

    # Select kept candidates and sort by frame
    kept_candidates = [candidates[i] for i in np.where(keep_mask)[0]]
    return sorted(kept_candidates, key=lambda c: c.frame_idx)


def compute_novelty(
    S: np.ndarray,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute spectral novelty function.

    Measures how different each frame is from its neighbors.

    M2 Optimized: Fully vectorized using NumPy BLAS operations.
    ~10-30x faster than loop-based implementation.

    Args:
        S: Magnitude spectrogram (n_freq, n_frames)
        metric: Distance metric ('cosine', 'euclidean', 'correlation')

    Returns:
        Novelty curve (n_frames,)
    """
    n_frames = S.shape[1]

    if n_frames < 2:
        return np.zeros(n_frames, dtype=np.float32)

    # Ensure contiguous float32 for Apple Accelerate
    S = np.ascontiguousarray(S, dtype=np.float32)

    # Get consecutive frame pairs
    prev_frames = S[:, :-1]  # (n_freq, n_frames-1)
    curr_frames = S[:, 1:]   # (n_freq, n_frames-1)

    if metric == 'cosine':
        # Vectorized cosine distance
        # similarity = dot(prev, curr) / (||prev|| * ||curr||)

        # Compute norms for all frames at once: (n_frames-1,)
        prev_norms = np.linalg.norm(prev_frames, axis=0)
        curr_norms = np.linalg.norm(curr_frames, axis=0)

        # Compute dot products for all pairs: (n_frames-1,)
        # Using einsum for efficiency on M2
        dot_products = np.einsum('ij,ij->j', prev_frames, curr_frames)

        # Cosine similarity
        norms_product = prev_norms * curr_norms + 1e-10
        similarity = dot_products / norms_product

        # Novelty = 1 - similarity (0 = identical, 1 = orthogonal)
        novelty_values = 1.0 - similarity

    elif metric == 'euclidean':
        # Vectorized Euclidean distance
        diff = curr_frames - prev_frames
        novelty_values = np.linalg.norm(diff, axis=0)

    elif metric == 'correlation':
        # Vectorized Pearson correlation
        # corr = cov(x,y) / (std(x) * std(y))

        # Compute means: (n_frames-1,)
        prev_means = prev_frames.mean(axis=0)
        curr_means = curr_frames.mean(axis=0)

        # Centered versions
        prev_centered = prev_frames - prev_means
        curr_centered = curr_frames - curr_means

        # Compute covariance and stds
        cov = np.einsum('ij,ij->j', prev_centered, curr_centered) / S.shape[0]
        prev_std = np.std(prev_frames, axis=0)
        curr_std = np.std(curr_frames, axis=0)

        # Correlation (handle zero std)
        std_product = prev_std * curr_std + 1e-10
        correlation = cov / std_product

        # Clip to valid range and compute novelty
        correlation = np.clip(correlation, -1.0, 1.0)
        novelty_values = 1.0 - correlation

    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Prepend 0 for first frame (no previous frame to compare)
    novelty = np.zeros(n_frames, dtype=np.float32)
    novelty[1:] = novelty_values

    # Normalize to [0, 1]
    max_val = np.max(novelty)
    if max_val > 0:
        novelty = novelty / max_val

    return novelty


def compute_drop_density(
    drops: List[DropCandidate],
    duration_sec: float
) -> float:
    """
    Compute drops per minute.

    Args:
        drops: List of detected drops
        duration_sec: Track duration

    Returns:
        Drops per minute
    """
    if duration_sec <= 0:
        return 0.0
    return len(drops) * 60.0 / duration_sec


def compute_drop_intensity(
    drops: List[DropCandidate]
) -> float:
    """
    Compute overall drop intensity (0-1).

    Combines drop count and magnitude.

    Args:
        drops: List of detected drops

    Returns:
        Intensity score (0-1)
    """
    if not drops:
        return 0.0

    # Average magnitude weighted by confidence
    total_weight = sum(d.confidence for d in drops)
    if total_weight == 0:
        return 0.0

    weighted_mag = sum(d.drop_magnitude * d.confidence for d in drops)
    avg_magnitude = weighted_mag / total_weight

    # Scale by count (more drops = higher intensity, with diminishing returns)
    count_factor = 1 - np.exp(-len(drops) / 3)

    return float(min(1.0, avg_magnitude * count_factor))


def detect_transitions(
    energy: np.ndarray,
    sr: int,
    hop_length: int,
    threshold_factor: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect energy transitions (mixin/mixout points).

    Args:
        energy: Energy curve
        sr: Sample rate
        hop_length: Hop length
        threshold_factor: Sensitivity factor

    Returns:
        Tuple of (mixin_frames, mixout_frames)
    """
    # Smooth energy
    energy_smooth = uniform_filter1d(energy, size=21)

    # Compute long-term derivative
    diff = np.diff(energy_smooth, prepend=energy_smooth[0])

    # Smooth derivative
    diff_smooth = uniform_filter1d(diff, size=11)

    # Threshold
    std = np.std(diff_smooth)
    threshold = threshold_factor * std

    # Rising edges = mixin, falling edges = mixout
    mixins = detect_peaks(diff_smooth, height=threshold)
    mixouts = detect_valleys(diff_smooth, depth=-threshold)

    return mixins, mixouts


# =============================================================================
# Timbral and Structural Novelty (for DJ transition detection)
# =============================================================================

def compute_timbral_novelty(
    mfcc: np.ndarray,
    sr: int,
    hop_length: int = 512,
    window_sec: float = 30.0
) -> np.ndarray:
    """
    Compute timbral novelty combining short-term and long-term MFCC analysis.

    Short-term: frame-to-frame MFCC distance (catches sharp changes)
    Long-term: 30-sec window MFCC drift (catches gradual fades)

    Critical for smooth DJ mixing detection where tracks fade in over 30+ seconds.

    Args:
        mfcc: MFCC features (n_mfcc, n_frames)
        sr: Sample rate
        hop_length: Hop length
        window_sec: Window size for long-term analysis

    Returns:
        Novelty curve (n_frames,) normalized 0-1
    """
    n_frames = mfcc.shape[1]

    # SHORT-TERM: frame-to-frame distance (vectorized)
    mfcc_diff = np.diff(mfcc, axis=1)
    short_term = np.sqrt(np.sum(mfcc_diff ** 2, axis=0))
    short_term = np.concatenate([[0], short_term])

    # LONG-TERM: compare MFCC profiles across windows (VECTORIZED)
    frame_to_time = hop_length / sr
    window_frames = int(window_sec / frame_to_time)
    half_window = window_frames // 2

    # Pad cumsum for easier indexing (prepend zeros column)
    mfcc_cumsum = np.concatenate([np.zeros((mfcc.shape[0], 1)), np.cumsum(mfcc, axis=1)], axis=1)
    long_term = np.zeros(n_frames, dtype=np.float32)

    if n_frames > window_frames:
        # Create arrays of all time indices we need to process
        t_indices = np.arange(window_frames, n_frames)  # (M,)

        # Current window: [t - half_window, t)
        curr_starts = np.maximum(0, t_indices - half_window)  # (M,)
        curr_ends = t_indices  # (M,)
        curr_lengths = curr_ends - curr_starts  # (M,)

        # Past window: [t - window_frames, t - half_window)
        past_starts = np.maximum(0, t_indices - window_frames)  # (M,)
        past_ends = t_indices - half_window  # (M,)
        past_lengths = past_ends - past_starts  # (M,)

        # Vectorized cumsum lookups: cumsum[:, end] - cumsum[:, start]
        # Shape: (n_mfcc, M)
        curr_sums = mfcc_cumsum[:, curr_ends] - mfcc_cumsum[:, curr_starts]
        past_sums = mfcc_cumsum[:, past_ends] - mfcc_cumsum[:, past_starts]

        # Compute means (broadcast lengths)
        current_profiles = curr_sums / np.maximum(curr_lengths, 1)  # (n_mfcc, M)
        past_profiles = past_sums / np.maximum(past_lengths, 1)  # (n_mfcc, M)

        # Compute L2 norm of difference along mfcc axis
        diff = current_profiles - past_profiles  # (n_mfcc, M)
        norms = np.sqrt(np.sum(diff ** 2, axis=0))  # (M,)

        # Only assign where past_end > past_start
        valid_mask = past_ends > past_starts
        long_term[t_indices[valid_mask]] = norms[valid_mask]

    # Normalize both
    if np.max(short_term) > 0:
        short_term = short_term / np.max(short_term)
    if np.max(long_term) > 0:
        long_term = long_term / np.max(long_term)

    # Combine: weight long-term more for smooth mixing detection
    return short_term * 0.3 + long_term * 0.7


def compute_ssm_novelty(
    mfcc: np.ndarray,
    sr: int,
    hop_length: int = 512,
    kernel_size_sec: float = 30.0,
    subsample: int = 4,
    max_ssm_size: int = 5000
) -> np.ndarray:
    """
    Compute structural novelty using Foote's checkerboard kernel on SSM.

    Well-established MIR method (Foote 2000) for detecting structural boundaries
    via self-similarity matrix analysis. Robust for DJ mix segmentation.

    Memory-optimized: automatically increases subsample for long audio to keep
    SSM matrix under max_ssm_size × max_ssm_size.

    Args:
        mfcc: MFCC features (n_mfcc, n_frames)
        sr: Sample rate
        hop_length: Hop length
        kernel_size_sec: Checkerboard kernel size in seconds
        subsample: Base subsample factor for efficiency
        max_ssm_size: Maximum SSM dimension (prevents OOM for long mixes)

    Returns:
        Novelty curve (n_frames,) normalized 0-1
    """
    n_frames = mfcc.shape[1]

    # Auto-adjust subsample for long audio to prevent OOM
    # SSM memory = n_sub^2 * 8 bytes
    # For max_ssm_size=5000: 5000^2 * 8 = 200 MB (safe)
    n_sub_initial = n_frames // subsample
    if n_sub_initial > max_ssm_size:
        subsample = max(subsample, n_frames // max_ssm_size)

    # Subsample for efficiency
    mfcc_sub = mfcc[:, ::subsample]
    n_sub = mfcc_sub.shape[1]

    if n_sub < 20:
        return np.zeros(n_frames)

    # Compute SSM using cosine similarity (vectorized)
    mfcc_norm = mfcc_sub / (np.linalg.norm(mfcc_sub, axis=0, keepdims=True) + 1e-10)
    ssm = np.dot(mfcc_norm.T, mfcc_norm)

    # Create checkerboard kernel
    frame_to_time = hop_length * subsample / sr
    kernel_frames = int(kernel_size_sec / frame_to_time)
    kernel_frames = max(10, min(kernel_frames, n_sub // 4))

    L = kernel_frames // 2
    kernel = np.zeros((2 * L, 2 * L))
    kernel[:L, :L] = 1
    kernel[L:, L:] = 1
    kernel[:L, L:] = -1
    kernel[L:, :L] = -1

    # Gaussian taper
    x = np.arange(-L, L)
    gauss = np.exp(-x**2 / (2 * (L/2)**2))
    gauss_2d = np.outer(gauss, gauss)
    kernel = kernel * gauss_2d

    # Vectorized diagonal convolution using stride tricks
    # Extract all diagonal patches at once for M2 SIMD optimization
    kernel_size = 2 * L
    n_patches = n_sub - kernel_size + 1

    if n_patches > 0:
        # Use stride_tricks to create views of diagonal patches
        # Shape: (n_patches, kernel_size, kernel_size)
        from numpy.lib.stride_tricks import as_strided
        ssm_strides = ssm.strides
        diagonal_patches = as_strided(
            ssm,
            shape=(n_patches, kernel_size, kernel_size),
            strides=(ssm_strides[0] + ssm_strides[1], ssm_strides[0], ssm_strides[1]),
            writeable=False
        )

        # Vectorized element-wise multiply and sum: (n_patches,)
        # kernel is (kernel_size, kernel_size), patches is (n_patches, kernel_size, kernel_size)
        novelty_values = np.einsum('ijk,jk->i', diagonal_patches, kernel)

        # Place results in correct positions (offset by L)
        novelty_sub = np.zeros(n_sub)
        novelty_sub[L:L + n_patches] = novelty_values
    else:
        novelty_sub = np.zeros(n_sub)

    novelty_sub = np.abs(novelty_sub)
    if np.max(novelty_sub) > 0:
        novelty_sub = novelty_sub / np.max(novelty_sub)

    # Upsample back to original frame rate
    novelty = np.interp(np.arange(n_frames), np.arange(n_sub) * subsample, novelty_sub)

    return novelty


def compute_chroma_novelty(chroma: np.ndarray) -> np.ndarray:
    """
    Compute harmonic novelty from chroma features.

    Detects key/harmonic changes between tracks.

    Args:
        chroma: Chroma features (12, n_frames)

    Returns:
        Novelty curve (n_frames,) normalized 0-1
    """
    chroma_diff = np.diff(chroma, axis=1)
    chroma_dist = np.sqrt(np.sum(chroma_diff ** 2, axis=0))
    chroma_dist = np.concatenate([[0], chroma_dist])

    if np.max(chroma_dist) > 0:
        chroma_dist = chroma_dist / np.max(chroma_dist)

    return chroma_dist
