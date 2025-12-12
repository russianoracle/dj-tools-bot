"""
Beat Grid Analysis Task.

Builds musical structure grid (beats → bars → phrases) for accurate
event detection aligned to musical boundaries.

In electronic music:
- Drops happen on phrase boundaries (every 16 beats)
- Transitions happen on phrase boundaries
- Buildups last 1-2 phrases (16-32 beats)

This task provides the foundation for beat-aligned analysis.

Supports two modes:
- STATIC: Single global tempo (for single tracks)
- PLP: Local tempo per frame (for DJ sets with tempo variations)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum, auto

from .base import BaseTask, AudioContext
from ..primitives import (
    compute_beat_grid,
    BeatGridResult,
    compute_onset_strength,
    compute_plp_tempo,
    segment_by_tempo_changes,
    PLPResult,
    TempoSegment,
)


class BeatGridMode(Enum):
    """Beat grid computation mode."""
    STATIC = auto()   # Single global tempo (single tracks)
    PLP = auto()      # Local tempo (DJ sets with tempo variations)

logger = logging.getLogger(__name__)


@dataclass
class BeatGridAnalysisResult:
    """
    Result of beat grid analysis.

    Contains the full BeatGridResult plus additional metadata.
    For DJ sets with PLP mode, includes tempo segments.
    """
    beat_grid: BeatGridResult

    # Summary stats
    tempo: float
    tempo_confidence: float
    n_beats: int
    n_bars: int
    n_phrases: int

    # Timing
    beat_duration_sec: float
    bar_duration_sec: float
    phrase_duration_sec: float
    total_duration_sec: float

    # Quality metrics
    beat_regularity: float    # How regular are beat intervals (0-1)
    downbeat_confidence: float  # Confidence in downbeat detection

    # PLP mode (DJ sets)
    mode: BeatGridMode = BeatGridMode.STATIC
    tempo_segments: List[TempoSegment] = field(default_factory=list)
    segment_grids: List[BeatGridResult] = field(default_factory=list)

    def get_tempo_at_time(self, time_sec: float) -> float:
        """Get tempo at specific time (uses segments if PLP mode)."""
        if self.mode == BeatGridMode.STATIC or not self.tempo_segments:
            return self.tempo

        for seg in self.tempo_segments:
            if seg.start_sec <= time_sec <= seg.end_sec:
                return seg.mean_tempo
        return self.tempo

    def get_grid_at_time(self, time_sec: float) -> BeatGridResult:
        """Get beat grid for segment at specific time."""
        if self.mode == BeatGridMode.STATIC or not self.segment_grids:
            return self.beat_grid

        for i, seg in enumerate(self.tempo_segments):
            if seg.start_sec <= time_sec <= seg.end_sec:
                if i < len(self.segment_grids):
                    return self.segment_grids[i]
        return self.beat_grid

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            'tempo': float(self.tempo),
            'tempo_confidence': float(self.tempo_confidence),
            'n_beats': self.n_beats,
            'n_bars': self.n_bars,
            'n_phrases': self.n_phrases,
            'beat_duration_sec': float(self.beat_duration_sec),
            'bar_duration_sec': float(self.bar_duration_sec),
            'phrase_duration_sec': float(self.phrase_duration_sec),
            'total_duration_sec': float(self.total_duration_sec),
            'beat_regularity': float(self.beat_regularity),
            'downbeat_confidence': float(self.downbeat_confidence),
            'phrase_boundaries': self.beat_grid.get_phrase_boundaries().tolist(),
            'mode': self.mode.name,
        }
        if self.tempo_segments:
            result['tempo_segments'] = [
                {
                    'start_sec': seg.start_sec,
                    'end_sec': seg.end_sec,
                    'mean_tempo': seg.mean_tempo,
                    'tempo_std': seg.tempo_std,
                }
                for seg in self.tempo_segments
            ]
        return result


class BeatGridTask(BaseTask):
    """
    Task to compute beat grid from audio.

    Supports two modes:
    - STATIC: Single global tempo (for single tracks)
    - PLP: Local tempo per frame (for DJ sets with tempo variations 118→145 BPM)

    Usage:
        # Single track (STATIC mode - default)
        task = BeatGridTask()
        result = task.execute(audio_context)

        # DJ set (PLP mode - local tempo per segment)
        task = BeatGridTask(mode=BeatGridMode.PLP)
        result = task.execute(audio_context)

        # Get tempo at specific time in DJ set
        tempo = result.get_tempo_at_time(300.0)  # at 5 minutes

        # Use beat grid for event snapping
        snapped_time = result.beat_grid.snap_to_phrase(event_time)

        # Check if event is on phrase boundary
        is_aligned = result.beat_grid.is_on_phrase_boundary(event_time)
    """

    name = "beat_grid"

    def __init__(
        self,
        beats_per_bar: int = 4,
        bars_per_phrase: int = 4,
        min_tempo: float = 60.0,
        max_tempo: float = 200.0,
        mode: BeatGridMode = BeatGridMode.STATIC,
        # PLP mode parameters
        min_segment_sec: float = 30.0,
        tempo_change_threshold: float = 5.0,
    ):
        """
        Initialize beat grid task.

        Args:
            beats_per_bar: Beats per bar (4 for 4/4 time signature)
            bars_per_phrase: Bars per phrase (4 = 16 beats, standard in techno/house)
            min_tempo: Minimum expected tempo
            max_tempo: Maximum expected tempo
            mode: STATIC (single tempo) or PLP (local tempo for DJ sets)
            min_segment_sec: (PLP mode) Minimum segment duration
            tempo_change_threshold: (PLP mode) BPM change to consider boundary
        """
        self.beats_per_bar = beats_per_bar
        self.bars_per_phrase = bars_per_phrase
        self.min_tempo = min_tempo
        self.max_tempo = max_tempo
        self.mode = mode
        self.min_segment_sec = min_segment_sec
        self.tempo_change_threshold = tempo_change_threshold

    def execute(self, audio_context: AudioContext) -> BeatGridAnalysisResult:
        """
        Execute beat grid analysis.

        ARCHITECTURE NOTE: This is TASKS layer.
        - Uses precomputed STFT from audio_context.stft_cache
        - Passes S to primitive (no librosa calls in primitives)
        - PLP mode computes per-segment grids for DJ sets

        Args:
            audio_context: Audio context with loaded audio and STFT

        Returns:
            BeatGridAnalysisResult with full grid structure
        """
        y = audio_context.y
        sr = audio_context.sr
        hop_length = audio_context.stft_cache.hop_length
        S = audio_context.stft_cache.S  # Precomputed by ComputeSTFTStage
        total_duration = len(y) / sr

        logger.debug(f"Computing beat grid ({self.mode.name}): {total_duration:.1f}s audio, hop={hop_length}")

        if self.mode == BeatGridMode.PLP:
            return self._execute_plp_mode(audio_context)

        # STATIC mode: single global tempo
        beat_grid = compute_beat_grid(
            S=S,
            sr=sr,
            hop_length=hop_length,
            beats_per_bar=self.beats_per_bar,
            bars_per_phrase=self.bars_per_phrase,
        )

        # Calculate beat regularity (how consistent are beat intervals)
        beat_regularity = self._compute_beat_regularity(beat_grid)

        # Estimate downbeat confidence
        downbeat_confidence = self._estimate_downbeat_confidence(beat_grid, S, sr, hop_length)

        result = BeatGridAnalysisResult(
            beat_grid=beat_grid,
            tempo=beat_grid.tempo,
            tempo_confidence=beat_grid.tempo_confidence,
            n_beats=len(beat_grid.beats),
            n_bars=len(beat_grid.bars),
            n_phrases=len(beat_grid.phrases),
            beat_duration_sec=beat_grid.beat_duration_sec,
            bar_duration_sec=beat_grid.bar_duration_sec,
            phrase_duration_sec=beat_grid.phrase_duration_sec,
            total_duration_sec=total_duration,
            beat_regularity=beat_regularity,
            downbeat_confidence=downbeat_confidence,
            mode=BeatGridMode.STATIC,
        )

        logger.info(
            f"Beat grid (STATIC): {beat_grid.tempo:.1f} BPM, "
            f"{len(beat_grid.beats)} beats, "
            f"{len(beat_grid.bars)} bars, "
            f"{len(beat_grid.phrases)} phrases"
        )

        return result

    def _execute_plp_mode(self, audio_context: AudioContext) -> BeatGridAnalysisResult:
        """
        Execute beat grid analysis with PLP (local tempo) for DJ sets.

        Computes:
        1. PLP for local tempo per frame
        2. Segments with consistent tempo
        3. Per-segment beat grids
        """
        y = audio_context.y
        sr = audio_context.sr
        hop_length = audio_context.stft_cache.hop_length
        S = audio_context.stft_cache.S
        total_duration = len(y) / sr

        # Step 1: Compute onset strength (once)
        onset_env = compute_onset_strength(S, sr, hop_length)

        # Step 2: Compute PLP (local tempo per frame)
        plp_result = compute_plp_tempo(
            onset_env=onset_env,
            sr=sr,
            hop_length=hop_length,
            tempo_min=self.min_tempo,
            tempo_max=self.max_tempo,
            prior_bpm=128.0,
            prior_weight=0.5,
        )

        # Step 3: Segment by tempo changes
        tempo_segments = segment_by_tempo_changes(
            plp_result=plp_result,
            min_segment_sec=self.min_segment_sec,
            tempo_change_threshold=self.tempo_change_threshold,
        )

        # Step 4: Compute beat grid for each segment
        segment_grids = []
        for seg in tempo_segments:
            # Extract segment audio samples
            start_sample = int(seg.start_sec * sr)
            end_sample = int(seg.end_sec * sr)
            y_segment = y[start_sample:end_sample]

            if len(y_segment) < sr:  # Less than 1 second
                continue

            # Compute segment STFT (local)
            from ..primitives import compute_stft
            seg_cache = compute_stft(y_segment, sr=sr, n_fft=2048, hop_length=hop_length)

            # Compute segment beat grid with known tempo
            seg_grid = compute_beat_grid(
                S=seg_cache.S,
                sr=sr,
                hop_length=hop_length,
                beats_per_bar=self.beats_per_bar,
                bars_per_phrase=self.bars_per_phrase,
            )

            # Adjust timestamps to global time
            seg_grid = self._offset_grid_times(seg_grid, seg.start_sec)
            segment_grids.append(seg_grid)

        # Compute global beat grid (for fallback and main result)
        global_grid = compute_beat_grid(
            S=S,
            sr=sr,
            hop_length=hop_length,
            beats_per_bar=self.beats_per_bar,
            bars_per_phrase=self.bars_per_phrase,
        )

        # Use weighted mean tempo from segments
        if tempo_segments:
            total_weight = sum(seg.end_sec - seg.start_sec for seg in tempo_segments)
            weighted_tempo = sum(
                seg.mean_tempo * (seg.end_sec - seg.start_sec)
                for seg in tempo_segments
            ) / total_weight if total_weight > 0 else global_grid.tempo
        else:
            weighted_tempo = global_grid.tempo

        beat_regularity = self._compute_beat_regularity(global_grid)
        downbeat_confidence = self._estimate_downbeat_confidence(global_grid, S, sr, hop_length)

        result = BeatGridAnalysisResult(
            beat_grid=global_grid,
            tempo=weighted_tempo,
            tempo_confidence=global_grid.tempo_confidence,
            n_beats=len(global_grid.beats),
            n_bars=len(global_grid.bars),
            n_phrases=len(global_grid.phrases),
            beat_duration_sec=60.0 / weighted_tempo,
            bar_duration_sec=60.0 / weighted_tempo * self.beats_per_bar,
            phrase_duration_sec=60.0 / weighted_tempo * self.beats_per_bar * self.bars_per_phrase,
            total_duration_sec=total_duration,
            beat_regularity=beat_regularity,
            downbeat_confidence=downbeat_confidence,
            mode=BeatGridMode.PLP,
            tempo_segments=tempo_segments,
            segment_grids=segment_grids,
        )

        logger.info(
            f"Beat grid (PLP): {len(tempo_segments)} tempo segments, "
            f"tempo range {min(s.mean_tempo for s in tempo_segments):.1f}-"
            f"{max(s.mean_tempo for s in tempo_segments):.1f} BPM"
            if tempo_segments else f"Beat grid (PLP): fallback to {weighted_tempo:.1f} BPM"
        )

        return result

    def _offset_grid_times(self, grid: BeatGridResult, offset_sec: float) -> BeatGridResult:
        """Offset all times in grid by offset_sec."""
        # Create new grid with offset times
        # Note: This is a shallow copy approach - for full implementation
        # would need to adjust BeatInfo/BarInfo/PhraseInfo times
        return grid  # TODO: implement proper time offset

    def _compute_beat_regularity(self, beat_grid: BeatGridResult) -> float:
        """
        Compute how regular the beat intervals are.

        Returns:
            Score 0-1 where 1 = perfectly regular beats
        """
        if len(beat_grid.beats) < 3:
            return 0.0

        beat_times = beat_grid.get_beat_times()
        intervals = np.diff(beat_times)

        if len(intervals) == 0:
            return 0.0

        # Expected interval from tempo
        expected_interval = beat_grid.beat_duration_sec

        # Measure deviation from expected
        deviations = np.abs(intervals - expected_interval) / expected_interval
        mean_deviation = np.mean(deviations)

        # Convert to 0-1 score (0.1 deviation = 0.9 regularity)
        regularity = max(0.0, 1.0 - mean_deviation * 10)

        return float(regularity)

    def _estimate_downbeat_confidence(
        self,
        beat_grid: BeatGridResult,
        S: np.ndarray,
        sr: int,
        hop_length: int
    ) -> float:
        """
        Estimate confidence in downbeat detection.

        Measures how much stronger downbeats are vs other beats.
        """
        if len(beat_grid.beats) < 8:
            return 0.5

        # Get onset envelope
        onset_env = compute_onset_strength(S, sr, hop_length)

        # Get beat frames
        beat_frames = np.array([b.frame_idx for b in beat_grid.beats])
        valid_mask = beat_frames < len(onset_env)
        beat_frames = beat_frames[valid_mask]

        if len(beat_frames) < 8:
            return 0.5

        # Get strengths
        strengths = onset_env[beat_frames]

        # Separate downbeats (bar_position == 1) from others
        bar_positions = np.array([b.bar_position for b in beat_grid.beats])[:len(beat_frames)]
        downbeat_mask = bar_positions == 1

        if not np.any(downbeat_mask) or not np.any(~downbeat_mask):
            return 0.5

        downbeat_strength = np.mean(strengths[downbeat_mask])
        other_strength = np.mean(strengths[~downbeat_mask])

        # Confidence based on ratio
        if other_strength > 0:
            ratio = downbeat_strength / other_strength
            confidence = min(1.0, (ratio - 1.0) * 2 + 0.5)
        else:
            confidence = 1.0

        return float(np.clip(confidence, 0.0, 1.0))
