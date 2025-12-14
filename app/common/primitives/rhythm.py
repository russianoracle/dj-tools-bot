"""
Rhythm Primitives - Tempo, beats, onset detection.

NOTE: Beat tracking (compute_beats, compute_tempo_multi, compute_plp_tempo)
uses librosa which is centralized in STFTCache.
Use stft_cache.get_beats(), get_tempo(), get_tempogram(), get_plp() instead.

Apple Silicon M2 Optimized:
- Reuse onset envelope across functions
- Vectorized tempo analysis
"""

import numpy as np
from typing import Tuple, Optional


def _power_to_db(S: np.ndarray, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    """
    Convert power spectrogram to dB scale (pure numpy replacement for librosa.power_to_db).

    Args:
        S: Power spectrogram
        ref: Reference power
        amin: Minimum amplitude to avoid log(0)
        top_db: Maximum dB range

    Returns:
        dB-scaled spectrogram
    """
    log_spec = 10.0 * np.log10(np.maximum(amin, S) / ref)
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec.astype(np.float32)


def compute_onset_strength(
    S: np.ndarray,
    sr: int,
    hop_length: int,
    aggregate: bool = True
) -> np.ndarray:
    """
    Compute onset strength envelope from spectrogram.

    Pure numpy implementation (no librosa).

    Args:
        S: Magnitude spectrogram
        sr: Sample rate
        hop_length: Hop length
        aggregate: If True, return single envelope; else per-band

    Returns:
        Onset strength envelope (n_frames,) or (n_bands, n_frames)
    """
    # Compute log-power spectrogram (pure numpy)
    S_db = _power_to_db(S ** 2)

    # First-order difference (positive part only)
    onset = np.maximum(0, np.diff(S_db, axis=1, prepend=S_db[:, :1]))

    if aggregate:
        # Mean across frequency
        return np.mean(onset, axis=0).astype(np.float32)
    return onset.astype(np.float32)


def compute_onset_density(
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
    window_sec: float = 1.0,
    threshold_percentile: float = 75
) -> np.ndarray:
    """
    Compute onset density (onsets per second).

    Args:
        onset_env: Onset strength envelope
        sr: Sample rate
        hop_length: Hop length
        window_sec: Window size in seconds
        threshold_percentile: Percentile for onset detection

    Returns:
        Onset density per frame
    """
    # Find onset peaks
    threshold = np.percentile(onset_env, threshold_percentile)
    onsets = onset_env > threshold

    # Sliding window density
    window_frames = int(window_sec * sr / hop_length)

    # Use convolution for efficiency
    kernel = np.ones(window_frames) / window_sec
    density = np.convolve(onsets.astype(float), kernel, mode='same')

    return density


def _autocorrelate(x: np.ndarray, max_size: Optional[int] = None) -> np.ndarray:
    """
    Autocorrelation (pure numpy replacement for librosa.autocorrelate).

    Args:
        x: Input signal
        max_size: Maximum lag to compute

    Returns:
        Autocorrelation coefficients
    """
    from scipy.signal import correlate
    n = len(x)
    if max_size is None:
        max_size = n
    # Full autocorrelation then take positive lags
    ac_full = correlate(x, x, mode='full')
    # Center is at n-1, take positive lags up to max_size
    ac = ac_full[n-1:n-1+max_size]
    # Normalize by n
    return ac / n


def compute_tempo(
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
    start_bpm: float = 120.0,
    std_bpm: float = 1.0,
    ac_size: float = 8.0,
    max_tempo: float = 200.0,
    prior_weight: float = 1.0
) -> Tuple[float, float]:
    """
    Compute tempo with confidence score.

    Uses autocorrelation with a prior centered at start_bpm.
    Pure numpy/scipy implementation (no librosa).

    Args:
        onset_env: Onset strength envelope
        sr: Sample rate
        hop_length: Hop length
        start_bpm: Prior tempo center
        std_bpm: Prior standard deviation
        ac_size: Autocorrelation window in seconds
        max_tempo: Maximum tempo to consider
        prior_weight: Weight of tempo prior

    Returns:
        Tuple of (tempo_bpm, confidence)
    """
    # Autocorrelation (pure numpy/scipy)
    ac = _autocorrelate(onset_env, max_size=int(ac_size * sr / hop_length))

    # Convert lag to BPM
    frame_rate = sr / hop_length
    min_lag = int(60 * frame_rate / max_tempo)
    max_lag = min(len(ac) - 1, int(60 * frame_rate / 30))  # 30 BPM minimum

    if max_lag <= min_lag:
        return start_bpm, 0.0

    # Apply tempo prior
    lags = np.arange(min_lag, max_lag)
    tempos = 60 * frame_rate / lags

    # Gaussian prior
    prior = np.exp(-0.5 * ((tempos - start_bpm) / std_bpm) ** 2)

    # Weighted autocorrelation
    ac_range = ac[min_lag:max_lag]
    weighted_ac = ac_range * (prior ** prior_weight)

    # Find peak
    peak_idx = np.argmax(weighted_ac)
    tempo = tempos[peak_idx]

    # Confidence from peak prominence
    if np.max(weighted_ac) > 0:
        confidence = float(weighted_ac[peak_idx] / np.sum(weighted_ac))
    else:
        confidence = 0.0

    return float(tempo), min(confidence * 2, 1.0)  # Scale confidence


def compute_tempo_multi(
    onset_env: np.ndarray,
    sr: int,
    hop_length: int
) -> Tuple[float, float, float]:
    """
    Compute tempo using multiple autocorrelation methods and return best estimate.

    Pure numpy/scipy implementation. Uses two autocorrelation priors
    (neutral 120 BPM and dance 128 BPM) without tempogram.

    For tempogram-based tempo estimation, use stft_cache.get_tempogram() instead.

    Args:
        onset_env: Onset strength envelope
        sr: Sample rate
        hop_length: Hop length

    Returns:
        Tuple of (tempo, confidence, tempo_std)
    """
    # Method 1: Autocorrelation with neutral prior
    tempo1, conf1 = compute_tempo(
        onset_env, sr, hop_length,
        start_bpm=120, std_bpm=50, prior_weight=0.5
    )

    # Method 2: Autocorrelation with dance music prior
    tempo2, conf2 = compute_tempo(
        onset_env, sr, hop_length,
        start_bpm=128, std_bpm=20, prior_weight=1.0
    )

    # Method 3: Autocorrelation with fast tempo prior (for drum & bass, etc)
    tempo3, conf3 = compute_tempo(
        onset_env, sr, hop_length,
        start_bpm=170, std_bpm=30, prior_weight=0.8
    )

    # Weighted average based on confidence
    total_conf = conf1 + conf2 + conf3 + 1e-10
    weighted_tempo = (tempo1 * conf1 + tempo2 * conf2 + tempo3 * conf3) / total_conf

    # Tempo std across methods
    tempo_std = np.std([tempo1, tempo2, tempo3])

    # Overall confidence
    overall_conf = max(conf1, conf2, conf3)
    if tempo_std > 10:
        overall_conf *= 0.5  # Penalize disagreement

    return float(weighted_tempo), float(overall_conf), float(tempo_std)


def _frames_to_time(frames: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """
    Convert frame indices to time in seconds (pure numpy replacement for librosa.frames_to_time).

    Args:
        frames: Frame indices
        sr: Sample rate
        hop_length: Hop length

    Returns:
        Time in seconds for each frame
    """
    return np.asarray(frames) * hop_length / sr


def compute_beats(
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
    start_bpm: float = 120.0,
    tightness: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DEPRECATED: Use stft_cache.get_beats() instead for librosa beat tracking.

    This function now uses a simplified peak-picking approach.
    For accurate beat tracking, use STFTCache.get_beats() which uses librosa.

    Args:
        onset_env: Onset strength envelope
        sr: Sample rate
        hop_length: Hop length
        start_bpm: Starting tempo estimate
        tightness: How tightly to follow tempo estimate (ignored in simplified version)

    Returns:
        Tuple of (beat_frames, beat_times)
    """
    # Estimate tempo
    tempo, _ = compute_tempo(onset_env, sr, hop_length, start_bpm=start_bpm)

    # Simplified beat tracking: place beats at regular intervals
    # adjusted by onset peaks
    frame_rate = sr / hop_length
    beat_period_frames = 60 * frame_rate / tempo

    # Find onset peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(onset_env, distance=int(beat_period_frames * 0.7))

    if len(peaks) == 0:
        # No peaks found - use regular grid
        n_beats = int(len(onset_env) / beat_period_frames)
        beats = np.round(np.arange(n_beats) * beat_period_frames).astype(np.int32)
    else:
        # Use peaks, filtered to roughly match expected beat period
        beats = peaks

    beat_times = _frames_to_time(beats, sr, hop_length)

    return beats.astype(np.int32), beat_times.astype(np.float32)


def compute_beat_sync_mask(
    n_frames: int,
    beat_frames: np.ndarray,
    tolerance: int = 2
) -> np.ndarray:
    """
    Create a mask for frames near beats (VECTORIZED).

    Useful for beat-synchronous feature aggregation.

    M2 Optimized: Uses broadcasting instead of Python loop.

    Args:
        n_frames: Total number of frames
        beat_frames: Beat frame indices
        tolerance: Frames within tolerance of beat are marked

    Returns:
        Boolean mask (n_frames,)
    """
    if len(beat_frames) == 0:
        return np.zeros(n_frames, dtype=bool)

    # Vectorized: compute distance from each frame to all beats
    # Shape: (n_frames,) vs (n_beats,) -> broadcasting
    frame_indices = np.arange(n_frames)[:, np.newaxis]  # (n_frames, 1)
    beat_frames = np.asarray(beat_frames)[np.newaxis, :]  # (1, n_beats)

    # Distance from each frame to each beat
    distances = np.abs(frame_indices - beat_frames)  # (n_frames, n_beats)

    # Frame is near beat if min distance <= tolerance
    min_distances = np.min(distances, axis=1)  # (n_frames,)
    mask = min_distances <= tolerance

    return mask


def compute_beat_strength(
    onset_env: np.ndarray,
    beat_frames: np.ndarray
) -> np.ndarray:
    """
    Compute strength of each beat.

    Args:
        onset_env: Onset strength envelope
        beat_frames: Beat frame indices

    Returns:
        Strength per beat
    """
    valid_beats = beat_frames[beat_frames < len(onset_env)]
    return onset_env[valid_beats]


def compute_groove(
    onset_env: np.ndarray,
    beat_frames: np.ndarray
) -> float:
    """
    Compute groove metric (how much activity between beats).

    Higher groove = more syncopation/swing.

    Args:
        onset_env: Onset strength envelope
        beat_frames: Beat frame indices

    Returns:
        Groove score (0-1)
    """
    if len(beat_frames) < 2:
        return 0.0

    # Energy on beats vs off beats
    beat_mask = np.zeros(len(onset_env), dtype=bool)
    beat_mask[beat_frames[beat_frames < len(onset_env)]] = True

    on_beat_energy = np.mean(onset_env[beat_mask]) if np.any(beat_mask) else 0
    off_beat_energy = np.mean(onset_env[~beat_mask]) if np.any(~beat_mask) else 0

    # Groove = ratio of off-beat to total energy
    total = on_beat_energy + off_beat_energy + 1e-10
    return float(off_beat_energy / total)


# ============== PLP: Predominant Local Pulse ==============
# For DJ sets with varying tempo (unlike single tracks with fixed BPM)

from dataclasses import dataclass
from typing import List


@dataclass
class PLPResult:
    """
    Result of PLP (Predominant Local Pulse) analysis.

    PLP provides LOCAL tempo estimates per frame, unlike global tempo.
    Essential for DJ sets where BPM varies (118 → 145 BPM transitions).
    """
    # Local tempo per frame (BPM)
    local_tempo: np.ndarray  # shape: (n_frames,)

    # Pulse strength per frame (0-1)
    pulse_strength: np.ndarray  # shape: (n_frames,)

    # Time axis in seconds
    times: np.ndarray  # shape: (n_frames,)

    # Metadata
    sr: int
    hop_length: int

    def get_tempo_at_time(self, time_sec: float) -> float:
        """Get local tempo at specific time."""
        if len(self.times) == 0:
            return 120.0
        idx = np.searchsorted(self.times, time_sec)
        idx = min(idx, len(self.local_tempo) - 1)
        return float(self.local_tempo[idx])

    def get_mean_tempo(self, start_sec: float = 0.0, end_sec: float = None) -> float:
        """Get mean tempo in time range."""
        if end_sec is None:
            end_sec = self.times[-1] if len(self.times) > 0 else 0.0

        mask = (self.times >= start_sec) & (self.times <= end_sec)
        if not np.any(mask):
            return 120.0

        # Weight by pulse strength
        weighted_tempo = np.average(
            self.local_tempo[mask],
            weights=self.pulse_strength[mask] + 1e-10
        )
        return float(weighted_tempo)


def compute_plp_tempo(
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
    tempo_min: float = 60.0,
    tempo_max: float = 200.0,
    prior_bpm: float = 128.0,
    prior_weight: float = 0.5,
) -> PLPResult:
    """
    Compute Predominant Local Pulse (PLP) - local tempo per frame.

    SIMPLIFIED VERSION: Uses windowed autocorrelation instead of librosa.
    For accurate PLP analysis, use stft_cache.get_plp() and get_tempogram() instead.

    M2 Optimized: Vectorized operations, single pass.

    Args:
        onset_env: Onset strength envelope
        sr: Sample rate
        hop_length: Hop length
        tempo_min: Minimum tempo to consider (BPM)
        tempo_max: Maximum tempo to consider (BPM)
        prior_bpm: Prior tempo for regularization
        prior_weight: Weight of prior (0 = no prior, 1 = strong prior)

    Returns:
        PLPResult with local tempo and pulse strength per frame

    Example:
        >>> plp = compute_plp_tempo(onset_env, sr, hop_length)
        >>> tempo_at_5min = plp.get_tempo_at_time(300.0)  # 300 sec = 5 min
        >>> mean_tempo = plp.get_mean_tempo(0, 600)  # First 10 minutes
    """
    n_frames = len(onset_env)

    if n_frames < 10:
        # Too short for meaningful PLP
        return PLPResult(
            local_tempo=np.full(n_frames, prior_bpm, dtype=np.float32),
            pulse_strength=np.zeros(n_frames, dtype=np.float32),
            times=np.arange(n_frames, dtype=np.float32) * hop_length / sr,
            sr=sr,
            hop_length=hop_length,
        )

    # Simplified PLP using windowed autocorrelation
    # Window size in frames (8 seconds)
    window_frames = int(8.0 * sr / hop_length)
    half_window = window_frames // 2

    frame_rate = sr / hop_length
    local_tempo = np.full(n_frames, prior_bpm, dtype=np.float32)
    pulse_strength = np.zeros(n_frames, dtype=np.float32)

    # Process in chunks for efficiency
    step = max(1, window_frames // 4)
    for center in range(half_window, n_frames - half_window, step):
        start = center - half_window
        end = center + half_window
        window = onset_env[start:end]

        # Local tempo via autocorrelation
        tempo, conf = compute_tempo(
            window, sr, hop_length,
            start_bpm=prior_bpm, std_bpm=30, prior_weight=prior_weight
        )

        # Clamp to valid range
        if tempo_min <= tempo <= tempo_max:
            # Fill surrounding frames
            fill_start = max(0, center - step // 2)
            fill_end = min(n_frames, center + step // 2)
            local_tempo[fill_start:fill_end] = tempo
            pulse_strength[fill_start:fill_end] = conf

    # Fill edges
    local_tempo[:half_window] = local_tempo[half_window]
    local_tempo[-half_window:] = local_tempo[-half_window - 1]
    pulse_strength[:half_window] = pulse_strength[half_window]
    pulse_strength[-half_window:] = pulse_strength[-half_window - 1]

    times = np.arange(n_frames, dtype=np.float32) * hop_length / sr

    return PLPResult(
        local_tempo=local_tempo,
        pulse_strength=pulse_strength,
        times=times,
        sr=sr,
        hop_length=hop_length,
    )


@dataclass
class TempoSegment:
    """A segment with consistent tempo."""
    start_sec: float
    end_sec: float
    mean_tempo: float
    tempo_std: float
    confidence: float  # Based on pulse strength


def segment_by_tempo_changes(
    plp_result: PLPResult,
    min_segment_sec: float = 30.0,
    tempo_change_threshold: float = 5.0,
    smooth_window_sec: float = 5.0,
) -> List[TempoSegment]:
    """
    Segment audio by tempo changes using PLP results.

    Finds regions with consistent tempo, useful for:
    - DJ set track boundary detection
    - Tempo transition identification
    - Per-segment beat grid computation

    M2 Optimized: Vectorized change detection.

    Args:
        plp_result: PLPResult from compute_plp_tempo()
        min_segment_sec: Minimum segment duration
        tempo_change_threshold: BPM change to consider boundary
        smooth_window_sec: Smoothing window for tempo

    Returns:
        List of TempoSegment with consistent tempo regions

    Example:
        >>> plp = compute_plp_tempo(onset_env, sr, hop_length)
        >>> segments = segment_by_tempo_changes(plp)
        >>> for seg in segments:
        ...     print(f"{seg.start_sec:.1f}-{seg.end_sec:.1f}: {seg.mean_tempo:.1f} BPM")
    """
    if len(plp_result.local_tempo) < 10:
        # Too short - single segment
        mean_tempo = float(np.mean(plp_result.local_tempo)) if len(plp_result.local_tempo) > 0 else 120.0
        end_time = plp_result.times[-1] if len(plp_result.times) > 0 else 0.0
        return [TempoSegment(
            start_sec=0.0,
            end_sec=end_time,
            mean_tempo=mean_tempo,
            tempo_std=0.0,
            confidence=1.0,
        )]

    times = plp_result.times
    local_tempo = plp_result.local_tempo.copy()
    pulse_strength = plp_result.pulse_strength

    # Smooth tempo to reduce noise
    frame_rate = 1.0 / (times[1] - times[0]) if len(times) > 1 else 1.0
    smooth_frames = max(1, int(smooth_window_sec * frame_rate))

    if smooth_frames > 1:
        # Uniform smoothing (vectorized)
        kernel = np.ones(smooth_frames) / smooth_frames
        local_tempo_smooth = np.convolve(local_tempo, kernel, mode='same')
    else:
        local_tempo_smooth = local_tempo

    # Detect tempo changes (derivative)
    tempo_diff = np.abs(np.diff(local_tempo_smooth, prepend=local_tempo_smooth[0]))

    # Find change points where tempo shifts significantly (VECTORIZED)
    min_frames = max(1, int(min_segment_sec * frame_rate))

    # Vectorized boundary detection:
    # Find all indices where tempo_diff > threshold
    above_threshold = np.where(tempo_diff > tempo_change_threshold)[0]

    # Filter by minimum segment length using vectorized diff
    if len(above_threshold) > 0:
        # Always include 0 as first boundary
        # Then filter candidates to ensure min_frames gap
        candidates = above_threshold[above_threshold > 0]  # exclude 0

        if len(candidates) > 0:
            # Compute gaps between consecutive candidates
            gaps = np.diff(candidates, prepend=0)
            # Keep only candidates with sufficient gap from previous
            valid_mask = gaps >= min_frames
            change_points = np.concatenate([[0], candidates[valid_mask], [len(times) - 1]])
        else:
            change_points = np.array([0, len(times) - 1])
    else:
        change_points = np.array([0, len(times) - 1])

    # Remove duplicates and sort
    change_points = np.unique(change_points)

    # Build segments (vectorized statistics where possible)
    segments = []
    n_segments = len(change_points) - 1

    if n_segments > 0:
        start_indices = change_points[:-1]
        end_indices = change_points[1:]

        # Precompute cumulative sums for fast segment statistics
        tempo_cumsum = np.concatenate([[0], np.cumsum(local_tempo_smooth)])
        tempo_sq_cumsum = np.concatenate([[0], np.cumsum(local_tempo_smooth ** 2)])
        pulse_cumsum = np.concatenate([[0], np.cumsum(pulse_strength)])

        for i in range(n_segments):
            start_idx = int(start_indices[i])
            end_idx = int(end_indices[i])

            if end_idx <= start_idx:
                continue

            length = end_idx - start_idx

            # Vectorized mean/std using cumsum
            tempo_sum = tempo_cumsum[end_idx] - tempo_cumsum[start_idx]
            tempo_sq_sum = tempo_sq_cumsum[end_idx] - tempo_sq_cumsum[start_idx]
            pulse_sum = pulse_cumsum[end_idx] - pulse_cumsum[start_idx]

            mean_tempo = tempo_sum / length
            variance = (tempo_sq_sum / length) - (mean_tempo ** 2)
            tempo_std = np.sqrt(max(0, variance))
            confidence = pulse_sum / length

            segments.append(TempoSegment(
                start_sec=float(times[start_idx]),
                end_sec=float(times[end_idx]),
                mean_tempo=float(mean_tempo),
                tempo_std=float(tempo_std),
                confidence=float(confidence),
            ))

    # Merge very short segments (keep loop - segment merging is inherently sequential)
    merged_segments = []
    for seg in segments:
        if seg.end_sec - seg.start_sec < min_segment_sec and merged_segments:
            # Merge with previous
            prev = merged_segments[-1]
            merged_segments[-1] = TempoSegment(
                start_sec=prev.start_sec,
                end_sec=seg.end_sec,
                mean_tempo=(prev.mean_tempo + seg.mean_tempo) / 2,
                tempo_std=max(prev.tempo_std, seg.tempo_std),
                confidence=(prev.confidence + seg.confidence) / 2,
            )
        else:
            merged_segments.append(seg)

    return merged_segments if merged_segments else [TempoSegment(
        start_sec=0.0,
        end_sec=times[-1] if len(times) > 0 else 0.0,
        mean_tempo=float(np.mean(local_tempo)),
        tempo_std=float(np.std(local_tempo)),
        confidence=float(np.mean(pulse_strength)),
    )]
