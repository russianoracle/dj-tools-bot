"""
Track Boundary Detection Task - Detect track changes in DJ sets.

Uses multiple features to detect where one track ends and another begins:
1. Timbral distance (MFCC cosine distance)
2. Harmonic content change (chroma)
3. Spectral novelty (flux peaks)
4. Energy patterns

Architecture: Tasks layer - orchestrates primitives for track boundary detection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, auto
import logging

from .base import BaseTask, AudioContext, TaskResult
from app.common.primitives import (
    compute_novelty,
    detect_peaks,
    smooth_gaussian,
)
# INTERNAL imports from submodules (bypass guard for internal use)
# These tasks process audio segments that aren't in the main STFTCache
from app.common.primitives.harmonic import compute_mfcc, compute_chroma

logger = logging.getLogger(__name__)


class TransitionStyle(Enum):
    """Style of track transition detected."""
    CUT = auto()          # Hard cut, instant change
    BLEND = auto()        # Smooth beatmatch blend
    FILTER_SWEEP = auto() # Filter-based transition
    FADE = auto()         # Volume fade transition


@dataclass
class TrackBoundary:
    """A detected track boundary (transition point)."""
    time_sec: float
    confidence: float
    transition_style: TransitionStyle
    # Timbral features
    mfcc_distance: float = 0.0
    chroma_distance: float = 0.0
    # Transition zone
    transition_start_sec: float = 0.0
    transition_end_sec: float = 0.0
    # Track info
    prev_track_idx: int = 0
    next_track_idx: int = 0

    def to_dict(self) -> dict:
        return {
            'time_sec': self.time_sec,
            'confidence': self.confidence,
            'transition_style': self.transition_style.name,
            'mfcc_distance': self.mfcc_distance,
            'chroma_distance': self.chroma_distance,
            'transition_start_sec': self.transition_start_sec,
            'transition_end_sec': self.transition_end_sec,
            'prev_track_idx': self.prev_track_idx,
            'next_track_idx': self.next_track_idx,
        }


@dataclass
class TrackInfo:
    """Information about a detected track segment."""
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    # Aggregate features
    avg_mfcc: Optional[np.ndarray] = None
    avg_chroma: Optional[np.ndarray] = None
    avg_energy: float = 0.0

    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'start_sec': self.start_sec,
            'end_sec': self.end_sec,
            'duration_sec': self.duration_sec,
            'avg_energy': self.avg_energy,
        }


@dataclass
class TrackBoundaryResult(TaskResult):
    """Result of track boundary detection."""
    boundaries: List[TrackBoundary] = field(default_factory=list)
    tracks: List[TrackInfo] = field(default_factory=list)
    n_tracks: int = 0
    avg_track_duration_sec: float = 0.0
    # Raw features for debugging
    mfcc_distance_curve: Optional[np.ndarray] = None
    chroma_distance_curve: Optional[np.ndarray] = None
    novelty_curve: Optional[np.ndarray] = None

    def get_boundary_times(self) -> np.ndarray:
        """Get array of boundary times."""
        return np.array([b.time_sec for b in self.boundaries])

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'n_tracks': self.n_tracks,
            'n_boundaries': len(self.boundaries),
            'avg_track_duration_sec': self.avg_track_duration_sec,
            'boundaries': [b.to_dict() for b in self.boundaries],
            'tracks': [t.to_dict() for t in self.tracks],
        })
        return base


class TrackBoundaryDetectionTask(BaseTask):
    """
    Detect track boundaries in DJ sets using timbral analysis.

    Features used:
    - MFCC distance: Changes in timbre indicate different tracks
    - Chroma distance: Changes in harmonic content
    - Spectral novelty: Sudden spectral changes

    Usage:
        task = TrackBoundaryDetectionTask(
            min_track_duration_sec=60.0,
            sensitivity=0.5,
        )
        result = task.execute(audio_context)

        for track in result.tracks:
            print(f"Track {track.index}: {track.start_sec:.1f}s - {track.end_sec:.1f}s")
    """

    def __init__(
        self,
        min_track_duration_sec: float = 60.0,
        max_track_duration_sec: float = 480.0,
        window_sec: float = 10.0,
        hop_sec: float = 2.0,
        sensitivity: float = 0.5,
        use_mfcc: bool = True,
        use_chroma: bool = True,
        use_novelty: bool = True,
    ):
        """
        Initialize track boundary detector.

        Args:
            min_track_duration_sec: Minimum track length (default 60s)
            max_track_duration_sec: Maximum track length (default 480s = 8min)
            window_sec: Analysis window for feature comparison
            hop_sec: Hop between analysis windows
            sensitivity: Detection sensitivity 0-1 (higher = more boundaries)
            use_mfcc: Use MFCC distance for detection
            use_chroma: Use chroma distance for detection
            use_novelty: Use spectral novelty for detection
        """
        self.min_track_duration_sec = min_track_duration_sec
        self.max_track_duration_sec = max_track_duration_sec
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.sensitivity = sensitivity
        self.use_mfcc = use_mfcc
        self.use_chroma = use_chroma
        self.use_novelty = use_novelty

    def execute(self, context: AudioContext) -> TrackBoundaryResult:
        """
        Detect track boundaries in the audio.

        Args:
            context: AudioContext with precomputed STFT

        Returns:
            TrackBoundaryResult with detected boundaries and tracks
        """
        import time
        start_time = time.time()

        y = context.y
        sr = context.sr
        duration = context.duration_sec

        context.report_progress("track_detection", 0.0, "Starting track boundary detection...")

        # Compute feature curves
        hop_samples = int(self.hop_sec * sr)
        window_samples = int(self.window_sec * sr)
        n_frames = int((len(y) - window_samples) / hop_samples) + 1

        if n_frames < 3:
            return TrackBoundaryResult(
                success=True,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                n_tracks=1,
            )

        context.report_progress("track_detection", 0.1, "Computing timbral features...")

        # Extract features per window
        mfcc_features = []
        chroma_features = []
        frame_times = []

        for i in range(n_frames):
            start_sample = i * hop_samples
            end_sample = start_sample + window_samples
            segment = y[start_sample:end_sample]
            frame_time = start_sample / sr + self.window_sec / 2

            frame_times.append(frame_time)

            # MFCC (first 13 coefficients) - use primitive (from submodule)
            if self.use_mfcc:
                mfcc = compute_mfcc(segment, sr=sr, n_mfcc=13, _warn=False)
                mfcc_mean = np.mean(mfcc, axis=1) if mfcc.ndim > 1 else mfcc
                mfcc_features.append(mfcc_mean)

            # Chroma - use primitive (from submodule)
            if self.use_chroma:
                chroma = compute_chroma(segment, sr=sr, _warn=False)
                chroma_mean = np.mean(chroma, axis=1) if chroma.ndim > 1 else chroma
                chroma_features.append(chroma_mean)

        frame_times = np.array(frame_times)

        context.report_progress("track_detection", 0.4, "Computing distance curves...")

        # Compute distance curves (VECTORIZED cosine distance)
        mfcc_distance = np.zeros(n_frames - 1, dtype=np.float32)
        chroma_distance = np.zeros(n_frames - 1, dtype=np.float32)

        if self.use_mfcc and mfcc_features:
            mfcc_arr = np.array(mfcc_features, dtype=np.float32)  # (n_frames, n_mfcc)
            # Vectorized cosine distance between consecutive frames
            a = mfcc_arr[:-1]  # (n_frames-1, n_mfcc)
            b = mfcc_arr[1:]   # (n_frames-1, n_mfcc)
            # Norms
            norm_a = np.linalg.norm(a, axis=1)  # (n_frames-1,)
            norm_b = np.linalg.norm(b, axis=1)  # (n_frames-1,)
            # Dot products using einsum
            dots = np.einsum('ij,ij->i', a, b)  # (n_frames-1,)
            # Cosine distance
            valid = (norm_a > 0) & (norm_b > 0)
            mfcc_distance[valid] = 1 - dots[valid] / (norm_a[valid] * norm_b[valid])

        if self.use_chroma and chroma_features:
            chroma_arr = np.array(chroma_features, dtype=np.float32)  # (n_frames, 12)
            # Vectorized cosine distance
            a = chroma_arr[:-1]
            b = chroma_arr[1:]
            norm_a = np.linalg.norm(a, axis=1)
            norm_b = np.linalg.norm(b, axis=1)
            dots = np.einsum('ij,ij->i', a, b)
            valid = (norm_a > 0) & (norm_b > 0)
            chroma_distance[valid] = 1 - dots[valid] / (norm_a[valid] * norm_b[valid])

        context.report_progress("track_detection", 0.6, "Computing novelty...")

        # Spectral novelty from STFT cache
        novelty = None
        if self.use_novelty:
            S = context.stft_cache.S
            novelty = compute_novelty(S)
            # Resample to match our frame rate
            novelty_times = np.linspace(0, duration, len(novelty))
            novelty_resampled = np.interp(frame_times[:-1], novelty_times, novelty)
        else:
            novelty_resampled = np.zeros(n_frames - 1)

        context.report_progress("track_detection", 0.7, "Finding boundaries...")

        # Combine features into boundary score
        # Normalize each feature
        def normalize(arr):
            if arr.max() > arr.min():
                return (arr - arr.min()) / (arr.max() - arr.min())
            return arr

        combined_score = np.zeros(n_frames - 1)
        weights = []

        if self.use_mfcc:
            combined_score += normalize(mfcc_distance) * 0.5
            weights.append(0.5)
        if self.use_chroma:
            combined_score += normalize(chroma_distance) * 0.3
            weights.append(0.3)
        if self.use_novelty:
            combined_score += normalize(novelty_resampled) * 0.2
            weights.append(0.2)

        if sum(weights) > 0:
            combined_score /= sum(weights)

        # Smooth the score
        combined_score = smooth_gaussian(combined_score, sigma=2.0)

        # Find peaks (potential boundaries)
        # Threshold based on sensitivity
        threshold = np.percentile(combined_score, 100 - self.sensitivity * 50)

        # Detect peaks above threshold
        peak_indices = detect_peaks(
            combined_score,
            threshold=threshold,
            min_distance=int(self.min_track_duration_sec / self.hop_sec),
        )

        context.report_progress("track_detection", 0.85, f"Found {len(peak_indices)} candidates...")

        # Convert peaks to boundaries
        boundaries = []
        boundary_times = frame_times[peak_indices] if len(peak_indices) > 0 else np.array([])

        for idx in peak_indices:
            boundary_time = frame_times[idx]

            # Skip if too close to start/end
            if boundary_time < self.min_track_duration_sec:
                continue
            if boundary_time > duration - self.min_track_duration_sec:
                continue

            # Get feature values at this point
            mfcc_dist = mfcc_distance[idx] if idx < len(mfcc_distance) else 0
            chroma_dist = chroma_distance[idx] if idx < len(chroma_distance) else 0
            score = combined_score[idx]

            # Classify transition style
            if mfcc_dist > 0.5 and score > 0.7:
                style = TransitionStyle.CUT
            elif chroma_dist > 0.4:
                style = TransitionStyle.BLEND
            else:
                style = TransitionStyle.FADE

            boundaries.append(TrackBoundary(
                time_sec=boundary_time,
                confidence=float(score),
                transition_style=style,
                mfcc_distance=float(mfcc_dist),
                chroma_distance=float(chroma_dist),
                transition_start_sec=max(0, boundary_time - self.window_sec),
                transition_end_sec=min(duration, boundary_time + self.window_sec),
            ))

        # Sort by time
        boundaries.sort(key=lambda b: b.time_sec)

        context.report_progress("track_detection", 0.95, "Building track list...")

        # Build track list from boundaries
        tracks = []
        prev_time = 0.0

        for i, boundary in enumerate(boundaries):
            boundary.prev_track_idx = i
            boundary.next_track_idx = i + 1

            tracks.append(TrackInfo(
                index=i,
                start_sec=prev_time,
                end_sec=boundary.time_sec,
                duration_sec=boundary.time_sec - prev_time,
            ))
            prev_time = boundary.time_sec

        # Add final track
        if prev_time < duration:
            tracks.append(TrackInfo(
                index=len(boundaries),
                start_sec=prev_time,
                end_sec=duration,
                duration_sec=duration - prev_time,
            ))

        # Filter tracks that are too short
        valid_tracks = [t for t in tracks if t.duration_sec >= self.min_track_duration_sec * 0.5]

        # Recalculate average duration
        avg_duration = np.mean([t.duration_sec for t in valid_tracks]) if valid_tracks else duration

        context.report_progress("track_detection", 1.0, f"Done: {len(valid_tracks)} tracks detected")

        return TrackBoundaryResult(
            success=True,
            task_name=self.name,
            processing_time_sec=time.time() - start_time,
            boundaries=boundaries,
            tracks=valid_tracks,
            n_tracks=len(valid_tracks),
            avg_track_duration_sec=avg_duration,
            mfcc_distance_curve=mfcc_distance,
            chroma_distance_curve=chroma_distance,
            novelty_curve=novelty_resampled if self.use_novelty else None,
        )
