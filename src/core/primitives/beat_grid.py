"""
Beat Grid Primitives - Musical structure alignment.

Builds hierarchical beat grid:
- Beats (individual hits)
- Bars/Такты (4 beats in 4/4)
- Phrases/Квадраты (4 bars = 16 beats)

This is fundamental for accurate event detection in electronic music,
where all significant events (drops, transitions, buildups) align to
phrase boundaries.

Apple Silicon M2 Optimized:
- Vectorized downbeat detection
- Vectorized snap operations with numpy broadcasting
- Cached boundary arrays for O(1) repeated access
- Contiguous float32 arrays for Apple Accelerate

ARCHITECTURE NOTE:
This is PRIMITIVES layer - pure numpy math only, NO librosa.
All librosa calls (STFT, beat tracking) happen in Tasks layer.
"""

import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Tuple, Optional

from .rhythm import compute_beats, compute_onset_strength, compute_tempo_multi


@dataclass
class BeatInfo:
    """Single beat information."""
    time_sec: float           # Time in seconds
    frame_idx: int            # STFT frame index
    bar_position: int         # Position in bar (1-4)
    phrase_position: int      # Position in phrase (1-16)
    strength: float           # Beat strength (onset magnitude)


@dataclass
class BarInfo:
    """Bar (такт) = 4 beats."""
    index: int                # Bar number (0-indexed)
    start_time: float         # Start time in seconds
    end_time: float           # End time in seconds
    beat_indices: List[int]   # Indices into beats list
    phrase_idx: int           # Which phrase this bar belongs to
    bar_in_phrase: int        # Position in phrase (1-4)


@dataclass
class PhraseInfo:
    """Phrase (квадрат) = 4 bars = 16 beats."""
    index: int                # Phrase number (0-indexed)
    start_time: float         # Start time in seconds
    end_time: float           # End time in seconds
    bar_indices: List[int]    # Indices into bars list
    duration_sec: float       # Duration in seconds
    avg_energy: float = 0.0   # Average energy in this phrase


@dataclass
class BeatGridResult:
    """
    Complete beat grid structure.

    Hierarchical structure:
        phrases[i].bar_indices → bars[j].beat_indices → beats[k]

    Example for 128 BPM track:
        - 1 beat = 0.469 sec
        - 1 bar = 4 beats = 1.875 sec
        - 1 phrase = 16 beats = 7.5 sec

    M2 Optimization: Boundary arrays are cached on first access.
    """
    beats: List[BeatInfo]
    bars: List[BarInfo]
    phrases: List[PhraseInfo]

    tempo: float                    # Detected tempo in BPM
    tempo_confidence: float         # Confidence in tempo detection
    downbeat_idx: int = 0           # Index of first downbeat (first beat of bar)

    # Timing info
    beat_duration_sec: float = 0.0  # Average beat duration
    bar_duration_sec: float = 0.0   # Average bar duration
    phrase_duration_sec: float = 0.0  # Average phrase duration

    # For frame conversion
    sr: int = 22050
    hop_length: int = 512

    # Private cache fields (not init, not compared)
    _phrase_boundaries_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _bar_boundaries_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _beat_times_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def get_phrase_boundaries(self) -> np.ndarray:
        """Get phrase boundary times - key points for events (cached)."""
        if self._phrase_boundaries_cache is None:
            if not self.phrases:
                self._phrase_boundaries_cache = np.array([], dtype=np.float32)
            else:
                boundaries = [p.start_time for p in self.phrases] + [self.phrases[-1].end_time]
                self._phrase_boundaries_cache = np.ascontiguousarray(boundaries, dtype=np.float32)
        return self._phrase_boundaries_cache

    def get_bar_boundaries(self) -> np.ndarray:
        """Get bar boundary times (cached)."""
        if self._bar_boundaries_cache is None:
            if not self.bars:
                self._bar_boundaries_cache = np.array([], dtype=np.float32)
            else:
                boundaries = [b.start_time for b in self.bars] + [self.bars[-1].end_time]
                self._bar_boundaries_cache = np.ascontiguousarray(boundaries, dtype=np.float32)
        return self._bar_boundaries_cache

    def get_beat_times(self) -> np.ndarray:
        """Get all beat times as numpy array (cached)."""
        if self._beat_times_cache is None:
            if not self.beats:
                self._beat_times_cache = np.array([], dtype=np.float32)
            else:
                self._beat_times_cache = np.ascontiguousarray(
                    [b.time_sec for b in self.beats], dtype=np.float32
                )
        return self._beat_times_cache

    def snap_to_phrase(self, time_sec: float) -> float:
        """Snap time to nearest phrase boundary."""
        boundaries = self.get_phrase_boundaries()
        if len(boundaries) == 0:
            return time_sec
        idx = np.argmin(np.abs(boundaries - time_sec))
        return float(boundaries[idx])

    def snap_to_bar(self, time_sec: float) -> float:
        """Snap time to nearest bar boundary."""
        boundaries = self.get_bar_boundaries()
        if len(boundaries) == 0:
            return time_sec
        idx = np.argmin(np.abs(boundaries - time_sec))
        return float(boundaries[idx])

    def snap_to_beat(self, time_sec: float) -> float:
        """Snap time to nearest beat."""
        beat_times = self.get_beat_times()
        if len(beat_times) == 0:
            return time_sec
        idx = np.argmin(np.abs(beat_times - time_sec))
        return float(beat_times[idx])

    def is_on_phrase_boundary(self, time_sec: float, tolerance_beats: int = 2) -> bool:
        """Check if time is near a phrase boundary."""
        tolerance_sec = tolerance_beats * self.beat_duration_sec
        boundaries = self.get_phrase_boundaries()
        if len(boundaries) == 0:
            return False
        min_distance = np.min(np.abs(boundaries - time_sec))
        return min_distance <= tolerance_sec

    def is_on_bar_boundary(self, time_sec: float, tolerance_beats: int = 1) -> bool:
        """Check if time is near a bar boundary."""
        tolerance_sec = tolerance_beats * self.beat_duration_sec
        boundaries = self.get_bar_boundaries()
        if len(boundaries) == 0:
            return False
        min_distance = np.min(np.abs(boundaries - time_sec))
        return min_distance <= tolerance_sec

    def get_phrase_at_time(self, time_sec: float) -> Optional[PhraseInfo]:
        """Get phrase containing the given time."""
        for phrase in self.phrases:
            if phrase.start_time <= time_sec < phrase.end_time:
                return phrase
        return None

    def get_bar_at_time(self, time_sec: float) -> Optional[BarInfo]:
        """Get bar containing the given time."""
        for bar in self.bars:
            if bar.start_time <= time_sec < bar.end_time:
                return bar
        return None

    def time_to_phrase_position(self, time_sec: float) -> Tuple[int, int, int]:
        """
        Convert time to musical position.

        Returns:
            (phrase_number, bar_in_phrase, beat_in_bar) - all 1-indexed
        """
        phrase = self.get_phrase_at_time(time_sec)
        bar = self.get_bar_at_time(time_sec)

        if phrase is None or bar is None:
            return (0, 0, 0)

        # Find beat position within bar
        beat_times = self.get_beat_times()
        beat_idx = np.searchsorted(beat_times, time_sec) - 1
        beat_idx = max(0, min(beat_idx, len(self.beats) - 1))

        beat_in_bar = self.beats[beat_idx].bar_position if beat_idx < len(self.beats) else 1

        return (phrase.index + 1, bar.bar_in_phrase, beat_in_bar)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'tempo': float(self.tempo),
            'tempo_confidence': float(self.tempo_confidence),
            'n_beats': len(self.beats),
            'n_bars': len(self.bars),
            'n_phrases': len(self.phrases),
            'beat_duration_sec': float(self.beat_duration_sec),
            'bar_duration_sec': float(self.bar_duration_sec),
            'phrase_duration_sec': float(self.phrase_duration_sec),
            'phrase_boundaries': self.get_phrase_boundaries().tolist(),
            'downbeat_idx': self.downbeat_idx,
        }


def detect_downbeat(
    onset_env: np.ndarray,
    beat_frames: np.ndarray,
    sr: int,
    hop_length: int
) -> int:
    """
    Detect the first downbeat (first beat of a bar).

    In 4/4 time, downbeats typically have stronger accents.
    We look for a pattern where every 4th beat is stronger.

    M2 Optimized: Vectorized across all 4 offsets simultaneously.

    Args:
        onset_env: Onset strength envelope
        beat_frames: Beat frame indices
        sr: Sample rate
        hop_length: Hop length

    Returns:
        Index offset for first downbeat (0-3)
    """
    if len(beat_frames) < 8:
        return 0

    # Get beat strengths - vectorized
    valid_mask = beat_frames < len(onset_env)
    valid_beats = beat_frames[valid_mask]
    strengths = onset_env[valid_beats].astype(np.float32)

    if len(strengths) < 8:
        return 0

    n_beats = len(strengths)

    # Vectorized: compute scores for all 4 offsets at once
    # Create mask arrays for each offset
    scores = np.zeros(4, dtype=np.float32)

    for offset in range(4):
        # Vectorized mask: True for downbeats (every 4th starting at offset)
        downbeat_mask = np.zeros(n_beats, dtype=bool)
        downbeat_mask[offset::4] = True

        n_downbeats = np.sum(downbeat_mask)
        n_others = n_beats - n_downbeats

        if n_downbeats < 2 or n_others < 2:
            scores[offset] = -np.inf
            continue

        # Vectorized mean computation
        downbeat_strength = np.sum(strengths[downbeat_mask]) / n_downbeats
        other_strength = np.sum(strengths[~downbeat_mask]) / n_others

        scores[offset] = downbeat_strength - other_strength

    return int(np.argmax(scores))


def compute_beat_grid(
    S: np.ndarray,
    sr: int,
    hop_length: int = 512,
    beats_per_bar: int = 4,
    bars_per_phrase: int = 4,
) -> BeatGridResult:
    """
    Compute complete beat grid from precomputed spectrogram.

    Builds hierarchical structure:
        Beats → Bars (4 beats) → Phrases (4 bars = 16 beats)

    ARCHITECTURE NOTE: This is PRIMITIVES layer.
    S (spectrogram) must be precomputed by Task layer.
    No librosa calls allowed here.

    Args:
        S: Magnitude spectrogram (REQUIRED, computed by Task layer)
        sr: Sample rate
        hop_length: Hop length for analysis
        beats_per_bar: Beats per bar (default 4 for 4/4 time)
        bars_per_phrase: Bars per phrase (default 4, standard in techno/house)

    Returns:
        BeatGridResult with full hierarchical structure
    """

    # Compute onset envelope
    onset_env = compute_onset_strength(S, sr, hop_length)

    # Compute tempo
    tempo, tempo_conf, _ = compute_tempo_multi(onset_env, sr, hop_length)

    # Compute beat positions
    beat_frames, beat_times = compute_beats(
        onset_env, sr, hop_length,
        start_bpm=tempo,
        tightness=100.0
    )

    if len(beat_frames) == 0:
        # No beats detected - return empty grid
        return BeatGridResult(
            beats=[],
            bars=[],
            phrases=[],
            tempo=tempo,
            tempo_confidence=tempo_conf,
            sr=sr,
            hop_length=hop_length
        )

    # Detect downbeat offset
    downbeat_offset = detect_downbeat(onset_env, beat_frames, sr, hop_length)

    # Get beat strengths
    valid_beats = beat_frames[beat_frames < len(onset_env)]
    beat_strengths = np.zeros(len(beat_frames))
    beat_strengths[:len(valid_beats)] = onset_env[valid_beats]

    # Build beat list
    beats = []
    beats_per_phrase = beats_per_bar * bars_per_phrase

    for i, (frame, time) in enumerate(zip(beat_frames, beat_times)):
        # Adjust index based on downbeat offset
        adjusted_idx = i - downbeat_offset

        if adjusted_idx < 0:
            bar_pos = (adjusted_idx % beats_per_bar) + 1
            phrase_pos = (adjusted_idx % beats_per_phrase) + 1
        else:
            bar_pos = (adjusted_idx % beats_per_bar) + 1
            phrase_pos = (adjusted_idx % beats_per_phrase) + 1

        beats.append(BeatInfo(
            time_sec=float(time),
            frame_idx=int(frame),
            bar_position=bar_pos,
            phrase_position=phrase_pos,
            strength=float(beat_strengths[i]) if i < len(beat_strengths) else 0.0
        ))

    # Build bar list
    bars = []
    n_full_bars = (len(beats) - downbeat_offset) // beats_per_bar

    for bar_idx in range(n_full_bars):
        start_beat_idx = downbeat_offset + bar_idx * beats_per_bar
        end_beat_idx = start_beat_idx + beats_per_bar

        if end_beat_idx > len(beats):
            break

        beat_indices = list(range(start_beat_idx, end_beat_idx))
        phrase_idx = bar_idx // bars_per_phrase
        bar_in_phrase = (bar_idx % bars_per_phrase) + 1

        # Get timing
        start_time = beats[start_beat_idx].time_sec
        if end_beat_idx < len(beats):
            end_time = beats[end_beat_idx].time_sec
        else:
            # Extrapolate
            beat_duration = 60.0 / tempo
            end_time = beats[end_beat_idx - 1].time_sec + beat_duration

        bars.append(BarInfo(
            index=bar_idx,
            start_time=start_time,
            end_time=end_time,
            beat_indices=beat_indices,
            phrase_idx=phrase_idx,
            bar_in_phrase=bar_in_phrase
        ))

    # Build phrase list
    phrases = []
    n_full_phrases = len(bars) // bars_per_phrase

    for phrase_idx in range(n_full_phrases):
        start_bar_idx = phrase_idx * bars_per_phrase
        end_bar_idx = start_bar_idx + bars_per_phrase

        if end_bar_idx > len(bars):
            break

        bar_indices = list(range(start_bar_idx, end_bar_idx))

        start_time = bars[start_bar_idx].start_time
        end_time = bars[end_bar_idx - 1].end_time

        phrases.append(PhraseInfo(
            index=phrase_idx,
            start_time=start_time,
            end_time=end_time,
            bar_indices=bar_indices,
            duration_sec=end_time - start_time
        ))

    # Calculate average durations
    beat_duration = 60.0 / tempo if tempo > 0 else 0.5
    bar_duration = beat_duration * beats_per_bar
    phrase_duration = bar_duration * bars_per_phrase

    return BeatGridResult(
        beats=beats,
        bars=bars,
        phrases=phrases,
        tempo=tempo,
        tempo_confidence=tempo_conf,
        downbeat_idx=downbeat_offset,
        beat_duration_sec=beat_duration,
        bar_duration_sec=bar_duration,
        phrase_duration_sec=phrase_duration,
        sr=sr,
        hop_length=hop_length
    )


def compute_beat_aligned_features(
    feature: np.ndarray,
    beat_grid: BeatGridResult,
    aggregation: str = 'mean',
    level: str = 'beat'
) -> np.ndarray:
    """
    Resample feature array to beat-aligned frames.

    Args:
        feature: Feature array (n_frames,) or (n_features, n_frames)
        beat_grid: Beat grid result
        aggregation: 'mean', 'max', 'median', or 'sum'
        level: 'beat', 'bar', or 'phrase'

    Returns:
        Beat-aligned feature array
    """
    if level == 'beat':
        boundaries = beat_grid.get_beat_times()
    elif level == 'bar':
        boundaries = beat_grid.get_bar_boundaries()
    elif level == 'phrase':
        boundaries = beat_grid.get_phrase_boundaries()
    else:
        raise ValueError(f"Unknown level: {level}")

    if len(boundaries) < 2:
        return feature

    # Convert boundaries to frames
    frame_rate = beat_grid.sr / beat_grid.hop_length
    boundary_frames = (boundaries * frame_rate).astype(int)
    boundary_frames = np.clip(boundary_frames, 0, feature.shape[-1] - 1)

    # Aggregate function
    agg_funcs = {
        'mean': np.mean,
        'max': np.max,
        'median': np.median,
        'sum': np.sum
    }
    agg_func = agg_funcs.get(aggregation, np.mean)

    # Handle 1D and 2D arrays
    is_1d = feature.ndim == 1
    if is_1d:
        feature = feature.reshape(1, -1)

    n_features = feature.shape[0]
    n_segments = len(boundary_frames) - 1
    result = np.zeros((n_features, n_segments))

    for i in range(n_segments):
        start = boundary_frames[i]
        end = boundary_frames[i + 1]
        if end > start:
            result[:, i] = agg_func(feature[:, start:end], axis=1)

    if is_1d:
        result = result.flatten()

    return result


def snap_events_to_grid(
    event_times: np.ndarray,
    beat_grid: BeatGridResult,
    snap_level: str = 'phrase',
    max_shift_beats: float = 2.0
) -> np.ndarray:
    """
    Snap event times to nearest grid boundary.

    M2 Optimized: Fully vectorized with numpy broadcasting.
    No Python loops - uses efficient matrix operations.

    Args:
        event_times: Array of event times in seconds
        beat_grid: Beat grid result
        snap_level: 'beat', 'bar', or 'phrase'
        max_shift_beats: Maximum allowed shift in beats

    Returns:
        Snapped event times
    """
    max_shift_sec = max_shift_beats * beat_grid.beat_duration_sec

    if snap_level == 'beat':
        boundaries = beat_grid.get_beat_times()
    elif snap_level == 'bar':
        boundaries = beat_grid.get_bar_boundaries()
    elif snap_level == 'phrase':
        boundaries = beat_grid.get_phrase_boundaries()
    else:
        raise ValueError(f"Unknown snap_level: {snap_level}")

    if len(boundaries) == 0:
        return event_times

    # Ensure contiguous float32 arrays for M2 optimization
    event_times = np.ascontiguousarray(event_times, dtype=np.float32)
    boundaries = np.ascontiguousarray(boundaries, dtype=np.float32)

    # Vectorized: compute distances matrix (n_events x n_boundaries)
    # Using broadcasting: event_times[:, None] - boundaries[None, :]
    distances = np.abs(event_times[:, np.newaxis] - boundaries[np.newaxis, :])

    # Find nearest boundary for each event
    nearest_indices = np.argmin(distances, axis=1)
    min_distances = distances[np.arange(len(event_times)), nearest_indices]

    # Get snapped values
    snapped = boundaries[nearest_indices]

    # Keep original where shift exceeds max
    too_far_mask = min_distances > max_shift_sec
    snapped[too_far_mask] = event_times[too_far_mask]

    return snapped


# =============================================================================
# Grid Calibration - Align grid to musical events
# =============================================================================

@dataclass
class GridCalibrationResult:
    """
    Result of grid calibration against anchor events.

    Calibration aligns the beat grid so that musical events (drops, transitions)
    fall on phrase boundaries as they should in electronic music.
    """
    # Phase offset correction
    phase_offset_sec: float         # Correction to apply to grid (seconds)
    phase_offset_beats: float       # Same in beats

    # Quality metrics
    alignment_score_before: float   # % events on boundaries before calibration
    alignment_score_after: float    # % events on boundaries after calibration
    n_anchor_events: int            # Number of events used for calibration

    # Event analysis
    event_offsets: np.ndarray       # Offset of each event from nearest boundary
    median_offset: float            # Median offset (used for correction)
    std_offset: float               # Standard deviation of offsets

    # Confidence
    calibration_confidence: float   # 0-1, higher = more reliable calibration


def compute_event_offsets(
    event_times: np.ndarray,
    phrase_duration_sec: float,
    first_phrase_time: float = 0.0
) -> np.ndarray:
    """
    Compute offset of each event from the nearest phrase boundary.

    M2 Optimized: Fully vectorized computation.

    In properly calibrated music:
    - Drops should have offset ≈ 0 (they start ON phrase boundaries)
    - Buildups should have offset ≈ -phrase_duration (they start 1 phrase BEFORE drop)

    Args:
        event_times: Array of event times in seconds
        phrase_duration_sec: Duration of one phrase
        first_phrase_time: Time of first phrase boundary

    Returns:
        Array of offsets in seconds (positive = after boundary, negative = before)
    """
    if len(event_times) == 0 or phrase_duration_sec <= 0:
        return np.array([], dtype=np.float32)

    # Ensure contiguous float32
    event_times = np.ascontiguousarray(event_times, dtype=np.float32)

    # For each event, find which phrase it's in and compute offset from phrase start
    # offset = (event_time - first_phrase) mod phrase_duration
    relative_times = event_times - first_phrase_time
    phrase_indices = relative_times / phrase_duration_sec

    # Distance to nearest integer phrase boundary
    # e.g., if phrase_idx = 2.1, nearest is 2.0, offset = 0.1 * phrase_duration
    nearest_phrase = np.round(phrase_indices)
    offsets = (phrase_indices - nearest_phrase) * phrase_duration_sec

    return offsets.astype(np.float32)


def compute_alignment_score(
    event_times: np.ndarray,
    beat_grid: BeatGridResult,
    tolerance_beats: float = 2.0
) -> float:
    """
    Compute percentage of events that fall on phrase boundaries.

    M2 Optimized: Vectorized boundary distance computation.

    Args:
        event_times: Array of event times
        beat_grid: Beat grid to check against
        tolerance_beats: Tolerance in beats

    Returns:
        Score 0-1 (1 = all events on boundaries)
    """
    if len(event_times) == 0:
        return 1.0

    boundaries = beat_grid.get_phrase_boundaries()
    if len(boundaries) == 0:
        return 0.0

    tolerance_sec = tolerance_beats * beat_grid.beat_duration_sec

    # Ensure contiguous float32
    event_times = np.ascontiguousarray(event_times, dtype=np.float32)
    boundaries = np.ascontiguousarray(boundaries, dtype=np.float32)

    # Vectorized: compute min distance to any boundary for each event
    # Shape: (n_events, n_boundaries)
    distances = np.abs(event_times[:, np.newaxis] - boundaries[np.newaxis, :])
    min_distances = np.min(distances, axis=1)

    # Count events within tolerance
    n_aligned = np.sum(min_distances <= tolerance_sec)

    return float(n_aligned / len(event_times))


def calibrate_grid_phase(
    beat_grid: BeatGridResult,
    anchor_events: np.ndarray,
    tolerance_beats: float = 2.0,
    min_events: int = 2
) -> GridCalibrationResult:
    """
    Calibrate grid phase offset using anchor events (drops, transitions).

    In electronic music, drops ALWAYS occur on phrase boundaries.
    This function computes the phase correction needed to align
    the beat grid with actual musical events.

    M2 Optimized: Vectorized offset computation and statistics.

    Algorithm:
    1. For each anchor event, compute offset from nearest phrase boundary
    2. Take median offset (robust to outliers)
    3. Phase correction = -median_offset
    4. Apply correction: new_boundary = old_boundary + phase_correction

    Args:
        beat_grid: Original beat grid
        anchor_events: Array of anchor event times (drops, transitions)
        tolerance_beats: Tolerance for "on boundary" check
        min_events: Minimum events required for calibration

    Returns:
        GridCalibrationResult with phase correction and metrics
    """
    # Handle empty or insufficient events
    if len(anchor_events) < min_events:
        return GridCalibrationResult(
            phase_offset_sec=0.0,
            phase_offset_beats=0.0,
            alignment_score_before=0.0,
            alignment_score_after=0.0,
            n_anchor_events=len(anchor_events),
            event_offsets=np.array([], dtype=np.float32),
            median_offset=0.0,
            std_offset=0.0,
            calibration_confidence=0.0
        )

    # Ensure contiguous float32
    anchor_events = np.ascontiguousarray(anchor_events, dtype=np.float32)

    # Compute alignment score BEFORE calibration
    alignment_before = compute_alignment_score(
        anchor_events, beat_grid, tolerance_beats
    )

    # Compute offsets from phrase boundaries
    first_phrase_time = beat_grid.phrases[0].start_time if beat_grid.phrases else 0.0
    offsets = compute_event_offsets(
        anchor_events,
        beat_grid.phrase_duration_sec,
        first_phrase_time
    )

    # Compute statistics
    median_offset = float(np.median(offsets))
    std_offset = float(np.std(offsets))

    # Phase correction = negative of median offset
    # If events are 0.3s AFTER boundaries, we need to shift grid by +0.3s
    phase_correction_sec = -median_offset
    phase_correction_beats = phase_correction_sec / beat_grid.beat_duration_sec

    # Compute alignment score AFTER calibration (simulated)
    # Shift events by correction and check alignment
    corrected_events = anchor_events + phase_correction_sec
    alignment_after = compute_alignment_score(
        corrected_events, beat_grid, tolerance_beats
    )

    # Compute confidence based on:
    # 1. Number of events (more = more confident)
    # 2. Consistency of offsets (lower std = more confident)
    # 3. Improvement in alignment (higher = more confident)
    n_events_factor = min(1.0, len(anchor_events) / 5.0)  # saturates at 5 events
    consistency_factor = max(0.0, 1.0 - std_offset / beat_grid.beat_duration_sec)
    improvement_factor = alignment_after - alignment_before + 0.5  # +0.5 baseline

    calibration_confidence = float(np.clip(
        n_events_factor * 0.3 + consistency_factor * 0.4 + improvement_factor * 0.3,
        0.0, 1.0
    ))

    return GridCalibrationResult(
        phase_offset_sec=phase_correction_sec,
        phase_offset_beats=phase_correction_beats,
        alignment_score_before=alignment_before,
        alignment_score_after=alignment_after,
        n_anchor_events=len(anchor_events),
        event_offsets=offsets,
        median_offset=median_offset,
        std_offset=std_offset,
        calibration_confidence=calibration_confidence
    )


def apply_phase_correction(
    beat_grid: BeatGridResult,
    phase_offset_sec: float
) -> BeatGridResult:
    """
    Create a new BeatGridResult with phase-corrected times.

    M2 Optimized: Vectorized time calculations before object creation.

    Args:
        beat_grid: Original beat grid
        phase_offset_sec: Phase correction to apply (seconds)

    Returns:
        New BeatGridResult with corrected times
    """
    if abs(phase_offset_sec) < 1e-6:
        return beat_grid  # No correction needed

    # Precompute conversion factor once
    frame_factor = beat_grid.sr / beat_grid.hop_length

    # Vectorized: compute all new beat times and frame indices at once
    if beat_grid.beats:
        old_beat_times = np.array([b.time_sec for b in beat_grid.beats], dtype=np.float32)
        new_beat_times = old_beat_times + phase_offset_sec
        new_frame_indices = (new_beat_times * frame_factor).astype(np.int32)

        new_beats = [
            BeatInfo(
                time_sec=float(new_beat_times[i]),
                frame_idx=int(new_frame_indices[i]),
                bar_position=beat_grid.beats[i].bar_position,
                phrase_position=beat_grid.beats[i].phrase_position,
                strength=beat_grid.beats[i].strength
            )
            for i in range(len(beat_grid.beats))
        ]
    else:
        new_beats = []

    # Vectorized: compute all bar times at once
    if beat_grid.bars:
        bar_starts = np.array([b.start_time for b in beat_grid.bars], dtype=np.float32)
        bar_ends = np.array([b.end_time for b in beat_grid.bars], dtype=np.float32)
        new_bar_starts = bar_starts + phase_offset_sec
        new_bar_ends = bar_ends + phase_offset_sec

        new_bars = [
            BarInfo(
                index=beat_grid.bars[i].index,
                start_time=float(new_bar_starts[i]),
                end_time=float(new_bar_ends[i]),
                beat_indices=beat_grid.bars[i].beat_indices,
                phrase_idx=beat_grid.bars[i].phrase_idx,
                bar_in_phrase=beat_grid.bars[i].bar_in_phrase
            )
            for i in range(len(beat_grid.bars))
        ]
    else:
        new_bars = []

    # Vectorized: compute all phrase times at once
    if beat_grid.phrases:
        phrase_starts = np.array([p.start_time for p in beat_grid.phrases], dtype=np.float32)
        phrase_ends = np.array([p.end_time for p in beat_grid.phrases], dtype=np.float32)
        new_phrase_starts = phrase_starts + phase_offset_sec
        new_phrase_ends = phrase_ends + phase_offset_sec

        new_phrases = [
            PhraseInfo(
                index=beat_grid.phrases[i].index,
                start_time=float(new_phrase_starts[i]),
                end_time=float(new_phrase_ends[i]),
                bar_indices=beat_grid.phrases[i].bar_indices,
                duration_sec=beat_grid.phrases[i].duration_sec,
                avg_energy=beat_grid.phrases[i].avg_energy
            )
            for i in range(len(beat_grid.phrases))
        ]
    else:
        new_phrases = []

    return BeatGridResult(
        beats=new_beats,
        bars=new_bars,
        phrases=new_phrases,
        tempo=beat_grid.tempo,
        tempo_confidence=beat_grid.tempo_confidence,
        downbeat_idx=beat_grid.downbeat_idx,
        beat_duration_sec=beat_grid.beat_duration_sec,
        bar_duration_sec=beat_grid.bar_duration_sec,
        phrase_duration_sec=beat_grid.phrase_duration_sec,
        sr=beat_grid.sr,
        hop_length=beat_grid.hop_length
    )
