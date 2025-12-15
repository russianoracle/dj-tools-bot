"""
Segmentation Task - Automatic track boundary detection in DJ sets.

Uses Laplacian Segmentation (McFee & Ellis, 2014) with primitives
from app.common.primitives.segmentation.

The task orchestrates:
1. Beat-sync CQT features
2. Recurrence matrix (self-similarity)
3. Path similarity (MFCC sequential)
4. Spectral clustering on Laplacian eigenvectors
5. Boundary detection from cluster labels

NOTE: librosa is used here in TASKS layer for orchestration (allowed).
Uses STFTCache for cached features where available.
"""

import time
import numpy as np
import sklearn.cluster
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from .base import AudioContext, TaskResult, BaseTask
from app.common.primitives import (
    # Segmentation primitives (pure numpy/scipy)
    compute_recurrence_matrix,
    compute_path_similarity,
    compute_laplacian_eigenvectors,
    detect_boundaries_from_labels,
    enhance_recurrence_diagonals,
    combine_recurrence_and_path,
)


@dataclass
class SegmentBoundary:
    """A detected segment boundary."""
    time_sec: float
    beat_idx: int
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'time_sec': self.time_sec,
            'beat_idx': self.beat_idx,
            'confidence': self.confidence,
        }


@dataclass
class SegmentationTaskResult(TaskResult):
    """Result of automatic segmentation."""
    # Detected boundaries
    boundaries: List[SegmentBoundary] = field(default_factory=list)
    # Number of segments
    n_segments: int = 0
    # Segment labels for each beat
    segment_labels: Optional[np.ndarray] = None
    # Beat times for reference
    beat_times: Optional[np.ndarray] = None
    # Estimated tempo
    tempo: float = 0.0

    @property
    def boundary_times(self) -> List[float]:
        return [b.time_sec for b in self.boundaries]

    def get_segment_durations(self, total_duration: float) -> List[float]:
        """Get duration of each segment."""
        times = self.boundary_times + [total_duration]
        return [times[i+1] - times[i] for i in range(len(times)-1)]

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'boundaries': [b.to_dict() for b in self.boundaries],
            'n_segments': self.n_segments,
            'boundary_times': self.boundary_times,
            'tempo': self.tempo,
        })
        return base


class SegmentationTask(BaseTask):
    """
    Automatic track boundary detection using Laplacian Segmentation.

    Orchestrates segmentation primitives to detect track boundaries
    in DJ sets. Uses spectral clustering on combined recurrence and
    path similarity matrices.

    Configuration:
        n_segments: Fixed number of segments (auto if None)
        min_segment_sec: Minimum segment duration
        max_segment_sec: Maximum segment duration
        merge_threshold_sec: Merge boundaries closer than this

    Usage:
        task = SegmentationTask(min_segment_sec=120)
        result = task.execute(context)
        for boundary in result.boundaries:
            print(f"Boundary at {boundary.time_sec:.1f}s")
    """

    def __init__(
        self,
        n_segments: Optional[int] = None,
        min_segment_sec: float = 120.0,
        max_segment_sec: float = 480.0,
        merge_threshold_sec: float = 60.0,
        bar_downsample: int = 4,
        max_beats: int = 2000,
        # Beat grid alignment parameters
        snap_to_grid: bool = True,
        grid_snap_tolerance_beats: float = 8.0,  # Wider tolerance for segment boundaries
    ):
        """
        Initialize segmentation task.

        Args:
            n_segments: Fixed number of segments (auto-detect if None)
            min_segment_sec: Minimum segment duration in seconds
            max_segment_sec: Maximum segment duration in seconds
            merge_threshold_sec: Merge boundaries closer than this
            bar_downsample: Use every Nth beat (4=bars, 1=beats). Higher = faster.
            max_beats: Maximum beats to process (auto-increase downsample if exceeded)
            snap_to_grid: Snap boundaries to phrase boundaries if beat_grid available
            grid_snap_tolerance_beats: Max distance (in beats) to snap boundary (default 8 = 2 bars)
        """
        self.n_segments = n_segments
        self.min_segment_sec = min_segment_sec
        self.max_segment_sec = max_segment_sec
        self.merge_threshold_sec = merge_threshold_sec
        self.bar_downsample = bar_downsample
        self.max_beats = max_beats
        # Beat grid alignment
        self.snap_to_grid = snap_to_grid
        self.grid_snap_tolerance_beats = grid_snap_tolerance_beats

    def _sync_to_beats(self, data: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
        """
        Sync feature matrix to beat frames (vectorized, replaces librosa.util.sync).

        Args:
            data: Feature matrix (n_features, n_frames)
            beat_frames: Beat frame indices

        Returns:
            Synced features (n_features, n_beats)
        """
        beat_frames = np.asarray(beat_frames, dtype=np.int32)
        n_beats = len(beat_frames)
        n_features = data.shape[0] if data.ndim > 1 else 1
        n_frames = data.shape[-1]

        # Build segments between beat frames
        # First segment: 0 to first beat
        # Middle segments: beat[i] to beat[i+1]
        # Last segment: last beat to end
        segments = np.zeros(n_frames, dtype=np.int32)
        for i in range(n_beats):
            start = beat_frames[i] if i > 0 else 0
            end = beat_frames[i + 1] if i + 1 < n_beats else n_frames
            segments[int(start):int(end)] = i

        # Aggregate using median (vectorized per segment)
        if data.ndim == 1:
            result = np.zeros(n_beats, dtype=np.float32)
            for i in range(n_beats):
                mask = segments == i
                if np.any(mask):
                    result[i] = np.median(data[mask])
            return result
        else:
            result = np.zeros((n_features, n_beats), dtype=np.float32)
            for i in range(n_beats):
                mask = segments == i
                if np.any(mask):
                    result[:, i] = np.median(data[:, mask], axis=1)
            return result

    def execute(self, context: AudioContext) -> SegmentationTaskResult:
        """Execute segmentation on audio context."""
        start_time = time.time()

        try:
            sr = context.sr
            duration = context.duration_sec

            # 1. Beat tracking via STFTCache (librosa centralized)
            tempo = context.stft_cache.get_tempo()
            beat_frames = context.stft_cache.get_beats()
            hop_length = context.stft_cache.hop_length
            beat_times = beat_frames.astype(np.float32) * hop_length / sr

            if len(beat_frames) < 4:
                return self._single_segment_result(
                    start_time, duration, tempo, beat_times
                )

            # 2. Compute downsample factor to stay under max_beats
            # This dramatically speeds up eigendecomposition (O(n³))
            downsample = self.bar_downsample
            while len(beat_frames) // downsample > self.max_beats:
                downsample *= 2

            # Downsample beats (use bars instead of individual beats)
            if downsample > 1:
                ds_beats = beat_frames[::downsample]
                ds_beat_times = beat_times[::downsample]
            else:
                ds_beats = beat_frames
                ds_beat_times = beat_times

            # 3. Compute CQT via STFTCache and sync to downsampled beats
            C = context.stft_cache.get_cqt()  # Returns np.abs(cqt)
            C_db = 20.0 * np.log10(C + 1e-10)  # amplitude_to_db equivalent
            C_db -= np.max(C_db)  # ref=np.max

            # Sync CQT to beat frames (vectorized median aggregation)
            Csync = self._sync_to_beats(C_db, ds_beats)

            # 4. Build recurrence matrix using primitive
            R = compute_recurrence_matrix(Csync, width=3, mode='affinity', sym=True)
            Rf = enhance_recurrence_diagonals(R, filter_size=7)

            # 5. Compute MFCC and path similarity using STFTCache lazy computation
            mfcc = context.stft_cache.get_mfcc(n_mfcc=13)
            Msync = self._sync_to_beats(mfcc, ds_beats)
            R_path = compute_path_similarity(Msync)

            # 6. Combine matrices using primitive
            A = combine_recurrence_and_path(Rf, R_path)

            # 7. Determine number of clusters
            k = self._estimate_k(duration, len(ds_beats))

            # 8. Compute Laplacian eigenvectors using primitive
            X = compute_laplacian_eigenvectors(A, k=k, median_filter_size=9)

            # 9. Cluster beats
            KM = sklearn.cluster.KMeans(n_clusters=k, n_init="auto", random_state=42)
            seg_ids = KM.fit_predict(X)

            # 10. Detect boundaries using primitive
            bound_indices = detect_boundaries_from_labels(seg_ids)

            # 11. Convert to times (using downsampled beat times)
            raw_times = ds_beat_times[bound_indices] if len(bound_indices) > 0 else np.array([0.0])
            merged_times = self._merge_close_boundaries(raw_times)

            # 12. Build boundary objects (map back to original beat indices)
            boundaries = []
            for t in merged_times:
                # Find closest beat index in original beats
                beat_idx = int(np.argmin(np.abs(beat_times - t)))
                boundaries.append(SegmentBoundary(
                    time_sec=float(t),
                    beat_idx=beat_idx,
                    confidence=1.0
                ))

            # 13. Apply beat grid alignment if available
            beat_grid = context.beat_grid
            if beat_grid is not None and self.snap_to_grid and len(boundaries) > 0:
                boundaries = self._apply_grid_alignment(boundaries, beat_grid, beat_times)

            return SegmentationTaskResult(
                success=True,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                boundaries=boundaries,
                n_segments=len(boundaries),
                segment_labels=seg_ids,
                beat_times=beat_times,
                tempo=float(tempo) if np.isscalar(tempo) else float(tempo[0])
            )

        except Exception as e:
            return SegmentationTaskResult(
                success=False,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                error=str(e)
            )

    def _estimate_k(self, duration: float, n_beats: int) -> int:
        """Estimate number of segments based on duration."""
        if self.n_segments is not None:
            return min(self.n_segments, n_beats - 1)

        # Estimate: 1 track per 5 minutes on average
        expected = max(2, int(duration / 300))

        # Bound by min/max segment duration
        min_k = max(2, int(duration / self.max_segment_sec))
        max_k = min(50, int(duration / self.min_segment_sec))

        k = max(min_k, min(max_k, expected))
        return min(k, n_beats - 1)

    def _merge_close_boundaries(self, times: np.ndarray) -> np.ndarray:
        """Merge boundaries that are too close together."""
        if len(times) <= 1:
            return times

        merged = [times[0]]
        for t in times[1:]:
            if t - merged[-1] >= self.merge_threshold_sec:
                merged.append(t)

        return np.array(merged)

    def _single_segment_result(
        self,
        start_time: float,
        duration: float,
        tempo: float,
        beat_times: np.ndarray
    ) -> SegmentationTaskResult:
        """Return result for single-segment case."""
        return SegmentationTaskResult(
            success=True,
            task_name=self.name,
            processing_time_sec=time.time() - start_time,
            boundaries=[SegmentBoundary(time_sec=0.0, beat_idx=0)],
            n_segments=1,
            segment_labels=np.zeros(1, dtype=int),
            beat_times=beat_times,
            tempo=float(tempo) if np.isscalar(tempo) else float(tempo[0])
        )

    def _apply_grid_alignment(
        self,
        boundaries: List[SegmentBoundary],
        beat_grid,  # BeatGridResult
        beat_times: np.ndarray
    ) -> List[SegmentBoundary]:
        """
        Align segment boundaries to phrase boundaries using beat grid.

        Segment boundaries (track changes) typically happen on phrase
        boundaries in DJ sets.
        """
        if not boundaries or beat_grid is None:
            return boundaries

        aligned_boundaries = []

        for boundary in boundaries:
            original_time = boundary.time_sec

            # Check if boundary is near a phrase boundary
            is_on_phrase = beat_grid.is_on_phrase_boundary(
                original_time,
                tolerance_beats=self.grid_snap_tolerance_beats
            )

            if is_on_phrase:
                # Snap to exact phrase boundary
                snapped_time = beat_grid.snap_to_phrase(original_time)
                # Find closest beat index for snapped time
                beat_idx = int(np.argmin(np.abs(beat_times - snapped_time)))
                confidence = 1.0  # High confidence - on phrase boundary
            else:
                # Keep original time
                snapped_time = original_time
                beat_idx = boundary.beat_idx
                confidence = 0.8  # Lower confidence - not on phrase boundary

            aligned_boundaries.append(SegmentBoundary(
                time_sec=snapped_time,
                beat_idx=beat_idx,
                confidence=confidence
            ))

        return aligned_boundaries