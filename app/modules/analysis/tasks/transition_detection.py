"""
Transition Detection Task - Detect mixin/mixout points in DJ sets.

Transitions are characterized by:
- Energy changes
- Frequency content shifts (filter sweeps)
- Bass introduction/removal
- Tempo alignment points

TransitionType classification:
- CUT: Hard cut (instant, <1 sec)
- FADE: Volume fade
- BLEND: Overlapping beatmatch
- EQ_FILTER: Filter sweep transition
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time

from .base import AudioContext, TaskResult, BaseTask
from app.common.primitives import (
    # NOTE: compute_rms, compute_centroid, compute_rolloff are BLOCKED
    # Use context.stft_cache.get_rms(), get_spectral_centroid(), get_spectral_rolloff() instead
    compute_frequency_bands,
    compute_novelty,
    detect_peaks,
    detect_valleys,
    smooth_gaussian,
    # Filter detection primitives (unique, not in STFTCache)
    compute_spectral_velocity,
    compute_filter_position,
    detect_filter_sweeps,
)


class TransitionType(Enum):
    """Type of DJ transition."""
    CUT = auto()        # Hard cut (instant)
    FADE = auto()       # Volume fade
    BLEND = auto()      # Overlapping beatmatch
    EQ_FILTER = auto()  # Filter sweep
    UNKNOWN = auto()


@dataclass
class MixinEvent:
    """
    Extended mixin (track entry) event.

    Includes filter sweep detection for DJ transition analysis.
    """
    frame_idx: int
    time_sec: float
    confidence: float
    energy_change: float
    bass_change: float
    # Extended fields
    transition_type: TransitionType = TransitionType.UNKNOWN
    duration_sec: float = 0.0
    bass_introduction: float = 0.0   # 0-1, how much bass was added
    spectral_shift: float = 0.0      # Hz, change in spectral centroid
    brightness_change: float = 0.0   # Change in high-freq content
    filter_detected: bool = False
    filter_direction: str = 'none'   # 'lowpass_open', 'highpass_close', 'none'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'mixin',
            'frame_idx': self.frame_idx,
            'time_sec': self.time_sec,
            'confidence': self.confidence,
            'energy_change': self.energy_change,
            'bass_change': self.bass_change,
            'transition_type': self.transition_type.name,
            'duration_sec': self.duration_sec,
            'bass_introduction': self.bass_introduction,
            'spectral_shift': self.spectral_shift,
            'brightness_change': self.brightness_change,
            'filter_detected': self.filter_detected,
            'filter_direction': self.filter_direction,
        }


@dataclass
class MixoutEvent:
    """
    Extended mixout (track exit) event.

    Includes filter sweep detection for DJ transition analysis.
    """
    frame_idx: int
    time_sec: float
    confidence: float
    energy_change: float
    bass_change: float
    # Extended fields
    transition_type: TransitionType = TransitionType.UNKNOWN
    duration_sec: float = 0.0
    bass_removal: float = 0.0        # 0-1, how much bass was removed
    spectral_shift: float = 0.0      # Hz, change in spectral centroid
    brightness_change: float = 0.0   # Change in high-freq content
    filter_detected: bool = False
    filter_direction: str = 'none'   # 'lowpass_close', 'highpass_open', 'none'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'mixout',
            'frame_idx': self.frame_idx,
            'time_sec': self.time_sec,
            'confidence': self.confidence,
            'energy_change': self.energy_change,
            'bass_change': self.bass_change,
            'transition_type': self.transition_type.name,
            'duration_sec': self.duration_sec,
            'bass_removal': self.bass_removal,
            'spectral_shift': self.spectral_shift,
            'brightness_change': self.brightness_change,
            'filter_detected': self.filter_detected,
            'filter_direction': self.filter_direction,
        }


@dataclass
class TransitionPair:
    """A matched mixin-mixout transition pair."""
    mixin: MixinEvent
    mixout: MixoutEvent
    overlap_sec: float
    quality_score: float
    transition_type: TransitionType = TransitionType.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mixin': self.mixin.to_dict(),
            'mixout': self.mixout.to_dict(),
            'overlap_sec': self.overlap_sec,
            'quality_score': self.quality_score,
            'transition_type': self.transition_type.name,
        }


@dataclass
class TransitionDetectionResult(TaskResult):
    """
    Result of transition detection with filter analysis.

    Attributes:
        mixins: List of mixin events
        mixouts: List of mixout events
        transitions: Paired transitions
        transition_density: Transitions per minute

        # Extended metrics
        avg_transition_duration: Average transition duration (seconds)
        transition_type_distribution: Count per TransitionType
        energy_curve: Energy time series (optional)
        bass_curve: Bass energy time series (optional)
        filter_curve: Estimated filter position time series (optional)
    """
    mixins: List[MixinEvent] = field(default_factory=list)
    mixouts: List[MixoutEvent] = field(default_factory=list)
    transitions: List[TransitionPair] = field(default_factory=list)
    transition_density: float = 0.0

    # Extended metrics
    avg_transition_duration: float = 0.0
    transition_type_distribution: Dict[str, int] = field(default_factory=dict)
    energy_curve: Optional[np.ndarray] = None
    bass_curve: Optional[np.ndarray] = None
    filter_curve: Optional[np.ndarray] = None

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    def get_transition_times(self) -> List[Tuple[float, float]]:
        """Get (mixin_time, mixout_time) pairs."""
        return [(t.mixin.time_sec, t.mixout.time_sec) for t in self.transitions]

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'mixins': [m.to_dict() for m in self.mixins],
            'mixouts': [m.to_dict() for m in self.mixouts],
            'transitions': [t.to_dict() for t in self.transitions],
            'transition_density': self.transition_density,
            'n_transitions': self.n_transitions,
            'avg_transition_duration': self.avg_transition_duration,
            'transition_type_distribution': self.transition_type_distribution,
        })
        return base


class TransitionDetectionTask(BaseTask):
    """
    Detect mixin/mixout transition points in DJ sets with filter analysis.

    Analyzes:
    - Energy changes (new track entering/leaving)
    - Bass frequency changes (common mixing technique)
    - Spectral novelty (timbral changes)
    - Filter sweeps (spectral velocity analysis)

    TransitionType classification:
    - CUT: duration < 1 sec
    - EQ_FILTER: high spectral velocity detected
    - FADE: energy change without filter sweep
    - BLEND: gradual bass/energy change

    Configuration:
        min_transition_gap_sec: Minimum gap between transitions
        energy_threshold: Sensitivity for energy change detection
        bass_weight: Importance of bass frequency analysis
        detect_filters: Enable filter sweep detection

    Usage:
        task = TransitionDetectionTask(detect_filters=True)
        result = task.execute(context)
        for transition in result.transitions:
            print(f"Transition: {transition.transition_type.name}")
    """

    def __init__(
        self,
        min_transition_gap_sec: float = 30.0,
        energy_threshold: float = 0.3,
        bass_weight: float = 0.5,
        smooth_sigma: float = 5.0,
        detect_filters: bool = True,
        filter_velocity_threshold: float = 500.0,
        peak_percentile: float = 90.0,
        transition_merge_window_sec: float = 90.0,
        timbral_weight: float = 0.0,
        verbose: bool = False,
        drop_window_sec: float = 10.0,
        drop_confidence_threshold: float = 0.3,
        boundary_tolerance_sec: float = 15.0,
        # Beat grid alignment parameters
        snap_to_grid: bool = True,
        grid_confidence_boost: float = 1.25,
        grid_confidence_penalty: float = 0.85,
        grid_snap_tolerance_beats: float = 4.0,  # More tolerance for transitions
        # Dependency injection for context filters (DIP compliance)
        drop_detector: Optional['BaseTask'] = None,
        segmenter: Optional['BaseTask'] = None,
    ):
        """
        Initialize transition detection task.

        Args:
            min_transition_gap_sec: Minimum gap between transitions
            energy_threshold: Sensitivity for energy changes (0-1)
            bass_weight: Weight for bass frequency analysis (0-1)
            smooth_sigma: Gaussian smoothing sigma
            detect_filters: Enable filter sweep detection
            filter_velocity_threshold: Hz/sec threshold for filter detection
            peak_percentile: Percentile threshold for peak detection (lower = more sensitive)
            transition_merge_window_sec: Window to merge nearby mixin/mixout points into single transition zone
            timbral_weight: Weight for timbral novelty (0=energy-based, 1=timbral-only)
            verbose: Enable verbose logging for filter debugging
            drop_window_sec: Time window (seconds) around drops for filtering
            drop_confidence_threshold: Minimum drop confidence threshold for filtering
            boundary_tolerance_sec: Time window (seconds) around segment boundaries
            snap_to_grid: Snap transitions to phrase boundaries if beat_grid available
            grid_confidence_boost: Confidence multiplier for transitions on phrase boundaries
            grid_confidence_penalty: Confidence multiplier for transitions off phrase boundaries
            grid_snap_tolerance_beats: Max distance (in beats) to snap to boundary (default 4 = 1 bar)
            drop_detector: Optional injected DropDetectionTask (for testing/DI)
            segmenter: Optional injected SegmentationTask (for testing/DI)
        """
        self.min_transition_gap_sec = min_transition_gap_sec
        self.energy_threshold = energy_threshold
        self.bass_weight = bass_weight
        self.smooth_sigma = smooth_sigma
        self.detect_filters = detect_filters
        self.filter_velocity_threshold = filter_velocity_threshold
        self.peak_percentile = peak_percentile
        self.transition_merge_window_sec = transition_merge_window_sec
        self.timbral_weight = timbral_weight
        self.verbose = verbose
        self.drop_window_sec = drop_window_sec
        self.drop_confidence_threshold = drop_confidence_threshold
        self.boundary_tolerance_sec = boundary_tolerance_sec
        # Beat grid alignment
        self.snap_to_grid = snap_to_grid
        self.grid_confidence_boost = grid_confidence_boost
        self.grid_confidence_penalty = grid_confidence_penalty
        self.grid_snap_tolerance_beats = grid_snap_tolerance_beats
        # Dependency injection (DIP)
        self._injected_drop_detector = drop_detector
        self._injected_segmenter = segmenter

    def execute(self, context: AudioContext) -> TransitionDetectionResult:
        """Detect transitions with filter analysis."""
        start_time = time.time()

        try:
            S = context.stft_cache.S
            freqs = context.stft_cache.freqs
            sr = context.sr
            hop_length = context.hop_length
            duration = context.duration_sec

            # Compute features for transition detection (use STFTCache for consistency)
            rms = context.stft_cache.get_rms()
            rms_smooth = smooth_gaussian(rms, sigma=self.smooth_sigma)

            # Frequency bands (bass is key for DJ mixing)
            bands = compute_frequency_bands(S, freqs)
            bass_energy = bands.bass + bands.sub_bass
            bass_smooth = smooth_gaussian(bass_energy, sigma=self.smooth_sigma)

            # Spectral novelty
            novelty = compute_novelty(S, metric='cosine')
            novelty_smooth = smooth_gaussian(novelty, sigma=self.smooth_sigma)

            # Timbral novelty (MFCC-based) - crucial for smooth mixing detection
            mfcc = context.stft_cache.get_mfcc(n_mfcc=13)
            timbral_novelty = self._compute_timbral_novelty_from_mfcc(mfcc)
            timbral_smooth = smooth_gaussian(timbral_novelty, sigma=self.smooth_sigma)

            # Chroma novelty - harmonic changes between tracks
            chroma = context.stft_cache.get_chroma()
            chroma_novelty = self._compute_chroma_novelty(chroma)
            chroma_smooth = smooth_gaussian(chroma_novelty, sigma=self.smooth_sigma)

            # Combined novelty: spectral + timbral + chroma
            combined_novelty = (
                novelty_smooth * 0.3 +
                timbral_smooth * 0.5 +  # Timbral is key for smooth mixing
                chroma_smooth * 0.2
            )

            # Spectral features for filter detection (use STFTCache for consistency)
            centroid = context.stft_cache.get_spectral_centroid()
            rolloff = context.stft_cache.get_spectral_rolloff()
            filter_pos = None
            centroid_velocity = None
            sweep_mask = None

            if self.detect_filters:
                centroid_velocity = compute_spectral_velocity(centroid, sr, hop_length)
                filter_pos = compute_filter_position(centroid, rolloff)
                sweep_mask = detect_filter_sweeps(
                    centroid_velocity, sr, hop_length,
                    velocity_threshold=self.filter_velocity_threshold
                )

            # Detect mixins and mixouts with extended info
            # Use combined_novelty for better smooth mixing detection
            mixins = self._detect_mixins(
                rms_smooth, bass_smooth, combined_novelty,
                sr, hop_length,
                centroid=centroid,
                centroid_velocity=centroid_velocity,
                sweep_mask=sweep_mask
            )

            # CONTEXT-AWARE FILTERING: Use drop and segmentation context
            # to filter false positives (internal drops, buildups, etc.)
            mixins = self._filter_by_drop_context(mixins, context)
            mixins = self._filter_by_segmentation_context(mixins, context)

            mixouts = self._detect_mixouts(
                rms_smooth, bass_smooth, combined_novelty,
                sr, hop_length,
                centroid=centroid,
                centroid_velocity=centroid_velocity,
                sweep_mask=sweep_mask
            )

            # Apply beat grid alignment if available
            beat_grid = context.beat_grid
            if beat_grid is not None and self.snap_to_grid:
                if len(mixins) > 0:
                    mixins = self._apply_grid_alignment_mixins(mixins, beat_grid, sr, hop_length)
                if len(mixouts) > 0:
                    mixouts = self._apply_grid_alignment_mixouts(mixouts, beat_grid, sr, hop_length)

            # Pair transitions and classify types
            transitions = self._pair_transitions(mixins, mixouts, sr, hop_length)

            # Calculate density
            transition_density = len(transitions) * 60.0 / duration if duration > 0 else 0.0

            # Calculate extended metrics
            avg_duration = float(np.mean([t.overlap_sec for t in transitions])) if transitions else 0.0
            type_dist = {}
            for t in transitions:
                type_name = t.transition_type.name
                type_dist[type_name] = type_dist.get(type_name, 0) + 1

            return TransitionDetectionResult(
                success=True,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                mixins=mixins,
                mixouts=mixouts,
                transitions=transitions,
                transition_density=transition_density,
                avg_transition_duration=avg_duration,
                transition_type_distribution=type_dist,
                energy_curve=rms_smooth,
                bass_curve=bass_smooth,
                filter_curve=filter_pos
            )

        except Exception as e:
            return TransitionDetectionResult(
                success=False,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                error=str(e)
            )

    def _apply_grid_alignment_mixins(
        self,
        mixins: List[MixinEvent],
        beat_grid,  # BeatGridResult
        sr: int,
        hop_length: int
    ) -> List[MixinEvent]:
        """
        Align mixin events to phrase boundaries using beat grid.

        In electronic music, transitions (mixin points) almost always happen
        on phrase boundaries (every 16 beats / 4 bars). This method:
        1. Checks if each mixin is near a phrase boundary
        2. Snaps mixins to the nearest boundary if within tolerance
        3. Boosts confidence for mixins on boundaries
        4. Penalizes confidence for mixins far from boundaries
        """
        if not mixins or beat_grid is None:
            return mixins

        aligned_mixins = []

        for mixin in mixins:
            original_time = mixin.time_sec

            is_on_boundary = beat_grid.is_on_phrase_boundary(
                original_time,
                tolerance_beats=self.grid_snap_tolerance_beats
            )

            if is_on_boundary:
                snapped_time = beat_grid.snap_to_phrase(original_time)
                new_confidence = min(1.0, mixin.confidence * self.grid_confidence_boost)
                snapped_frame = int(snapped_time * sr / hop_length)
            else:
                snapped_time = original_time
                snapped_frame = mixin.frame_idx
                new_confidence = mixin.confidence * self.grid_confidence_penalty

            aligned_mixin = MixinEvent(
                frame_idx=snapped_frame,
                time_sec=snapped_time,
                confidence=new_confidence,
                energy_change=mixin.energy_change,
                bass_change=mixin.bass_change,
                transition_type=mixin.transition_type,
                duration_sec=mixin.duration_sec,
                bass_introduction=mixin.bass_introduction,
                spectral_shift=mixin.spectral_shift,
                brightness_change=mixin.brightness_change,
                filter_detected=mixin.filter_detected,
                filter_direction=mixin.filter_direction,
            )
            aligned_mixins.append(aligned_mixin)

        return aligned_mixins

    def _apply_grid_alignment_mixouts(
        self,
        mixouts: List[MixoutEvent],
        beat_grid,  # BeatGridResult
        sr: int,
        hop_length: int
    ) -> List[MixoutEvent]:
        """
        Align mixout events to phrase boundaries using beat grid.
        Similar to mixin alignment - transitions happen on phrase boundaries.
        """
        if not mixouts or beat_grid is None:
            return mixouts

        aligned_mixouts = []

        for mixout in mixouts:
            original_time = mixout.time_sec

            is_on_boundary = beat_grid.is_on_phrase_boundary(
                original_time,
                tolerance_beats=self.grid_snap_tolerance_beats
            )

            if is_on_boundary:
                snapped_time = beat_grid.snap_to_phrase(original_time)
                new_confidence = min(1.0, mixout.confidence * self.grid_confidence_boost)
                snapped_frame = int(snapped_time * sr / hop_length)
            else:
                snapped_time = original_time
                snapped_frame = mixout.frame_idx
                new_confidence = mixout.confidence * self.grid_confidence_penalty

            aligned_mixout = MixoutEvent(
                frame_idx=snapped_frame,
                time_sec=snapped_time,
                confidence=new_confidence,
                energy_change=mixout.energy_change,
                bass_change=mixout.bass_change,
                transition_type=mixout.transition_type,
                duration_sec=mixout.duration_sec,
                bass_removal=mixout.bass_removal,
                spectral_shift=mixout.spectral_shift,
                brightness_change=mixout.brightness_change,
                filter_detected=mixout.filter_detected,
                filter_direction=mixout.filter_direction,
            )
            aligned_mixouts.append(aligned_mixout)

        return aligned_mixouts

    def _detect_mixins(
        self,
        rms: np.ndarray,
        bass: np.ndarray,
        novelty: np.ndarray,
        sr: int,
        hop_length: int,
        centroid: Optional[np.ndarray] = None,
        centroid_velocity: Optional[np.ndarray] = None,
        sweep_mask: Optional[np.ndarray] = None
    ) -> List[MixinEvent]:
        """Detect mixin (track entry) points with filter analysis. Fully vectorized."""
        frame_to_time = hop_length / sr
        min_gap_frames = int(self.min_transition_gap_sec / frame_to_time)

        # Compute energy derivatives (vectorized)
        energy_diff = np.diff(rms, prepend=rms[0])
        bass_diff = np.diff(bass, prepend=bass[0])

        # Combined signal: energy increase + bass increase + novelty
        combined = (
            energy_diff * (1 - self.bass_weight) +
            bass_diff * self.bass_weight
        ) * novelty

        combined_smooth = smooth_gaussian(combined, sigma=3.0)

        # Find peaks (potential mixins) - use configurable percentile
        pos_mask = combined_smooth > 0
        threshold = np.percentile(combined_smooth[pos_mask], self.peak_percentile) if np.any(pos_mask) else 0
        peaks = detect_peaks(combined_smooth, height=threshold, distance=min_gap_frames)

        if len(peaks) == 0:
            return []

        # Vectorized computation for all peaks
        peaks = np.asarray(peaks)
        rms_mean = np.mean(rms) + 1e-10
        bass_mean = np.mean(bass) + 1e-10

        energy_changes = energy_diff[peaks] / rms_mean
        bass_changes = bass_diff[peaks] / bass_mean
        novelty_vals = novelty[peaks]

        # Vectorized confidence: emphasize novelty more for smooth mixing
        confidences = np.minimum(1.0,
            np.abs(energy_changes) * 0.3 +
            np.abs(bass_changes) * 0.3 +
            novelty_vals * 0.4   # Higher weight for timbral novelty
        )

        # Filter by confidence threshold
        valid_mask = confidences > self.energy_threshold
        valid_peaks = peaks[valid_mask]
        valid_energy = energy_changes[valid_mask]
        valid_bass = bass_changes[valid_mask]
        valid_conf = confidences[valid_mask]

        if len(valid_peaks) == 0:
            return []

        # Vectorized filter detection
        n_valid = len(valid_peaks)
        filter_detected = np.zeros(n_valid, dtype=bool)
        spectral_shifts = np.zeros(n_valid)

        if sweep_mask is not None:
            in_bounds = valid_peaks < len(sweep_mask)
            filter_detected[in_bounds] = sweep_mask[valid_peaks[in_bounds]]

            if centroid_velocity is not None:
                vel_mask = filter_detected & in_bounds
                spectral_shifts[vel_mask] = centroid_velocity[valid_peaks[vel_mask]] * frame_to_time

        # Bass introduction (positive bass change, clamped to [0,1])
        bass_intro = np.clip(np.maximum(0, valid_bass), 0, 1)

        # Build result list
        times = valid_peaks * frame_to_time
        mixins = [
            MixinEvent(
                frame_idx=int(valid_peaks[i]),
                time_sec=float(times[i]),
                confidence=float(valid_conf[i]),
                energy_change=float(valid_energy[i]),
                bass_change=float(valid_bass[i]),
                bass_introduction=float(bass_intro[i]),
                spectral_shift=float(spectral_shifts[i]),
                filter_detected=bool(filter_detected[i]),
                filter_direction=(
                    'lowpass_open' if filter_detected[i] and centroid_velocity is not None and centroid_velocity[valid_peaks[i]] > 0
                    else 'highpass_close' if filter_detected[i]
                    else 'none'
                )
            )
            for i in range(n_valid)
        ]

        return mixins

    def _detect_mixouts(
        self,
        rms: np.ndarray,
        bass: np.ndarray,
        novelty: np.ndarray,
        sr: int,
        hop_length: int,
        centroid: Optional[np.ndarray] = None,
        centroid_velocity: Optional[np.ndarray] = None,
        sweep_mask: Optional[np.ndarray] = None
    ) -> List[MixoutEvent]:
        """Detect mixout (track exit) points with filter analysis. Fully vectorized."""
        frame_to_time = hop_length / sr
        min_gap_frames = int(self.min_transition_gap_sec / frame_to_time)

        # Compute energy derivatives (vectorized)
        energy_diff = np.diff(rms, prepend=rms[0])
        bass_diff = np.diff(bass, prepend=bass[0])

        # Combined signal: energy decrease + bass decrease + novelty
        combined = -(
            energy_diff * (1 - self.bass_weight) +
            bass_diff * self.bass_weight
        ) * novelty

        combined_smooth = smooth_gaussian(combined, sigma=3.0)

        # Find peaks (potential mixouts) - use configurable percentile
        pos_mask = combined_smooth > 0
        threshold = np.percentile(combined_smooth[pos_mask], self.peak_percentile) if np.any(pos_mask) else 0
        peaks = detect_peaks(combined_smooth, height=threshold, distance=min_gap_frames)

        if len(peaks) == 0:
            return []

        # Vectorized computation for all peaks
        peaks = np.asarray(peaks)
        rms_mean = np.mean(rms) + 1e-10
        bass_mean = np.mean(bass) + 1e-10

        energy_changes = energy_diff[peaks] / rms_mean
        bass_changes = bass_diff[peaks] / bass_mean
        novelty_vals = novelty[peaks]

        # Vectorized confidence: emphasize novelty more for smooth mixing
        confidences = np.minimum(1.0,
            np.abs(energy_changes) * 0.3 +
            np.abs(bass_changes) * 0.3 +
            novelty_vals * 0.4   # Higher weight for timbral novelty
        )

        # Filter by confidence threshold
        valid_mask = confidences > self.energy_threshold
        valid_peaks = peaks[valid_mask]
        valid_energy = energy_changes[valid_mask]
        valid_bass = bass_changes[valid_mask]
        valid_conf = confidences[valid_mask]

        if len(valid_peaks) == 0:
            return []

        # Vectorized filter detection
        n_valid = len(valid_peaks)
        filter_detected = np.zeros(n_valid, dtype=bool)
        spectral_shifts = np.zeros(n_valid)

        if sweep_mask is not None:
            in_bounds = valid_peaks < len(sweep_mask)
            filter_detected[in_bounds] = sweep_mask[valid_peaks[in_bounds]]

            if centroid_velocity is not None:
                vel_mask = filter_detected & in_bounds
                spectral_shifts[vel_mask] = centroid_velocity[valid_peaks[vel_mask]] * frame_to_time

        # Bass removal (negative bass change, clamped to [0,1])
        bass_removal = np.clip(np.maximum(0, -valid_bass), 0, 1)

        # Build result list
        times = valid_peaks * frame_to_time
        mixouts = [
            MixoutEvent(
                frame_idx=int(valid_peaks[i]),
                time_sec=float(times[i]),
                confidence=float(valid_conf[i]),
                energy_change=float(valid_energy[i]),
                bass_change=float(valid_bass[i]),
                bass_removal=float(bass_removal[i]),
                spectral_shift=float(spectral_shifts[i]),
                filter_detected=bool(filter_detected[i]),
                filter_direction=(
                    'lowpass_close' if filter_detected[i] and centroid_velocity is not None and centroid_velocity[valid_peaks[i]] < 0
                    else 'highpass_open' if filter_detected[i]
                    else 'none'
                )
            )
            for i in range(n_valid)
        ]

        return mixouts

    def _merge_nearby_mixins(self, mixins: List[MixinEvent]) -> List[MixinEvent]:
        """
        Merge nearby mixin points into single transition entry points.

        For long transitions (like smooth techno), multiple novelty peaks may be detected
        within the same transition zone. We keep only the FIRST mixin in each group
        (when Track B starts entering).

        Groups are formed by points within transition_merge_window_sec of each other.
        """
        if len(mixins) <= 1:
            return mixins

        sorted_mixins = sorted(mixins, key=lambda m: m.time_sec)
        merged = []
        current_group = [sorted_mixins[0]]

        for mixin in sorted_mixins[1:]:
            # Check if this mixin is within merge window of the first in current group
            time_since_group_start = mixin.time_sec - current_group[0].time_sec

            if time_since_group_start <= self.transition_merge_window_sec:
                # Add to current group
                current_group.append(mixin)
            else:
                # Finalize current group - keep the first mixin (earliest entry)
                # Use highest confidence if there are multiple
                best = max(current_group, key=lambda m: m.confidence)
                # But use the time of the first one
                best_with_first_time = MixinEvent(
                    frame_idx=current_group[0].frame_idx,
                    time_sec=current_group[0].time_sec,
                    confidence=best.confidence,
                    energy_change=best.energy_change,
                    bass_change=best.bass_change,
                    transition_type=best.transition_type,
                    duration_sec=best.duration_sec,
                    bass_introduction=best.bass_introduction,
                    spectral_shift=best.spectral_shift,
                    brightness_change=best.brightness_change,
                    filter_detected=best.filter_detected,
                    filter_direction=best.filter_direction,
                )
                merged.append(best_with_first_time)
                current_group = [mixin]

        # Don't forget the last group
        if current_group:
            best = max(current_group, key=lambda m: m.confidence)
            best_with_first_time = MixinEvent(
                frame_idx=current_group[0].frame_idx,
                time_sec=current_group[0].time_sec,
                confidence=best.confidence,
                energy_change=best.energy_change,
                bass_change=best.bass_change,
                transition_type=best.transition_type,
                duration_sec=best.duration_sec,
                bass_introduction=best.bass_introduction,
                spectral_shift=best.spectral_shift,
                brightness_change=best.brightness_change,
                filter_detected=best.filter_detected,
                filter_direction=best.filter_direction,
            )
            merged.append(best_with_first_time)

        return merged

    def _merge_nearby_mixouts(self, mixouts: List[MixoutEvent]) -> List[MixoutEvent]:
        """
        Merge nearby mixout points into single transition exit points.

        For long transitions, multiple mixout candidates may be detected.
        We keep only the LAST mixout in each group (when Track A finishes leaving).
        """
        if len(mixouts) <= 1:
            return mixouts

        sorted_mixouts = sorted(mixouts, key=lambda m: m.time_sec)
        merged = []
        current_group = [sorted_mixouts[0]]

        for mixout in sorted_mixouts[1:]:
            # Check if this mixout is within merge window of the last in current group
            time_since_last = mixout.time_sec - current_group[-1].time_sec

            if time_since_last <= self.transition_merge_window_sec:
                # Add to current group
                current_group.append(mixout)
            else:
                # Finalize current group - keep the last mixout (latest exit)
                # Use highest confidence if there are multiple
                best = max(current_group, key=lambda m: m.confidence)
                last_in_group = current_group[-1]
                # But use the time of the last one
                best_with_last_time = MixoutEvent(
                    frame_idx=last_in_group.frame_idx,
                    time_sec=last_in_group.time_sec,
                    confidence=best.confidence,
                    energy_change=best.energy_change,
                    bass_change=best.bass_change,
                    transition_type=best.transition_type,
                    duration_sec=best.duration_sec,
                    bass_removal=best.bass_removal,
                    spectral_shift=best.spectral_shift,
                    brightness_change=best.brightness_change,
                    filter_detected=best.filter_detected,
                    filter_direction=best.filter_direction,
                )
                merged.append(best_with_last_time)
                current_group = [mixout]

        # Don't forget the last group
        if current_group:
            best = max(current_group, key=lambda m: m.confidence)
            last_in_group = current_group[-1]
            best_with_last_time = MixoutEvent(
                frame_idx=last_in_group.frame_idx,
                time_sec=last_in_group.time_sec,
                confidence=best.confidence,
                energy_change=best.energy_change,
                bass_change=best.bass_change,
                transition_type=best.transition_type,
                duration_sec=best.duration_sec,
                bass_removal=best.bass_removal,
                spectral_shift=best.spectral_shift,
                brightness_change=best.brightness_change,
                filter_detected=best.filter_detected,
                filter_direction=best.filter_direction,
            )
            merged.append(best_with_last_time)

        return merged

    def _pair_transitions(
        self,
        mixins: List[MixinEvent],
        mixouts: List[MixoutEvent],
        sr: int,
        hop_length: int
    ) -> List[TransitionPair]:
        """Pair mixins with corresponding mixouts using distance matrix. Vectorized."""
        if not mixins or not mixouts:
            return []

        # STEP 1: Merge nearby points into transition zones
        # This prevents long transitions from being split into multiple false detections
        mixins_merged = self._merge_nearby_mixins(mixins)
        mixouts_merged = self._merge_nearby_mixouts(mixouts)

        # Sort by time
        mixins_sorted = sorted(mixins_merged, key=lambda m: m.time_sec)
        mixouts_sorted = sorted(mixouts_merged, key=lambda m: m.time_sec)

        n_mixins = len(mixins_sorted)
        n_mixouts = len(mixouts_sorted)

        # Build time arrays
        mixin_times = np.array([m.time_sec for m in mixins_sorted])
        mixout_times = np.array([m.time_sec for m in mixouts_sorted])

        # Compute distance matrix: (n_mixins, n_mixouts)
        # Overlap = mixout_time - mixin_time (positive = mixout after mixin)
        overlap_matrix = mixout_times[np.newaxis, :] - mixin_times[:, np.newaxis]

        # Valid overlaps: mixout after mixin, within 30-180 sec range
        valid_mask = (overlap_matrix > 30) & (overlap_matrix < 180)

        # Set invalid overlaps to inf for argmin
        cost_matrix = np.where(valid_mask, overlap_matrix, np.inf)

        # Greedy assignment: for each mixin, find best unassigned mixout
        transitions = []
        used_mixouts = np.zeros(n_mixouts, dtype=bool)

        for i in range(n_mixins):
            # Mask already used mixouts
            row_costs = np.where(used_mixouts, np.inf, cost_matrix[i])

            if np.all(np.isinf(row_costs)):
                continue

            best_j = int(np.argmin(row_costs))
            best_overlap = float(row_costs[best_j])

            if not np.isinf(best_overlap):
                used_mixouts[best_j] = True

                mixin = mixins_sorted[i]
                mixout = mixouts_sorted[best_j]

                # Quality score (vectorized where possible)
                quality = (mixin.confidence + mixout.confidence) / 2
                if 60 < best_overlap < 120:
                    quality *= 1.2
                quality = min(1.0, quality)

                trans_type = self._classify_transition(mixin, mixout, best_overlap)

                transitions.append(TransitionPair(
                    mixin=mixin,
                    mixout=mixout,
                    overlap_sec=best_overlap,
                    quality_score=quality,
                    transition_type=trans_type
                ))

        return transitions

    def _classify_transition(
        self,
        mixin: MixinEvent,
        mixout: MixoutEvent,
        duration: float
    ) -> TransitionType:
        """
        Classify transition type based on characteristics.

        Classification rules:
        - CUT: duration < 1 second
        - EQ_FILTER: filter sweep detected in mixin or mixout
        - FADE: significant energy change, no filter
        - BLEND: gradual bass/energy change
        """
        # Hard cut (instant)
        if duration < 1.0:
            return TransitionType.CUT

        # Filter sweep detected
        if mixin.filter_detected or mixout.filter_detected:
            return TransitionType.EQ_FILTER

        # Significant energy change without filter = fade
        if abs(mixin.energy_change) > 0.15 or abs(mixout.energy_change) > 0.15:
            return TransitionType.FADE

        # Gradual bass/energy change = blend
        if abs(mixin.bass_change) > 0.05 or abs(mixout.bass_change) > 0.05:
            return TransitionType.BLEND

        return TransitionType.UNKNOWN

    def _compute_timbral_novelty_from_mfcc(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Compute timbral novelty using MFCC distance.

        Detects timbre changes even when energy stays constant.
        Critical for smooth mixing detection.

        Args:
            mfcc: Pre-computed MFCCs from STFTCache, shape (13, n_frames)
        """
        # Compute frame-to-frame distance in MFCC space
        mfcc_diff = np.diff(mfcc, axis=1)
        mfcc_dist = np.sqrt(np.sum(mfcc_diff ** 2, axis=0))

        # Pad to match original length
        mfcc_dist = np.concatenate([[0], mfcc_dist])

        # Normalize
        if np.max(mfcc_dist) > 0:
            mfcc_dist = mfcc_dist / np.max(mfcc_dist)

        return mfcc_dist

    def _compute_chroma_novelty(self, chroma: np.ndarray) -> np.ndarray:
        """
        Compute harmonic novelty from chroma features.

        Detects key/harmonic changes between tracks.
        """
        # Compute frame-to-frame chroma distance
        chroma_diff = np.diff(chroma, axis=1)
        chroma_dist = np.sqrt(np.sum(chroma_diff ** 2, axis=0))

        # Pad to match original length
        chroma_dist = np.concatenate([[0], chroma_dist])

        # Normalize
        if np.max(chroma_dist) > 0:
            chroma_dist = chroma_dist / np.max(chroma_dist)

        return chroma_dist

    def _filter_by_drop_context(
        self,
        mixins: List[MixinEvent],
        context: AudioContext
    ) -> List[MixinEvent]:
        """
        Filter mixins that coincide with internal track drops.

        Uses DropDetectionTask to identify drops, then removes/penalizes mixins
        that are likely detecting internal drops rather than track transitions.

        Logic:
        - Strong bass change + weak drop → keep (likely real transition during drop)
        - Weak bass change + strong drop → filter (likely internal drop)
        - Moderate case → reduce confidence

        Args:
            mixins: Detected mixin candidates
            context: AudioContext with STFT cache (reused by DropDetectionTask)

        Returns:
            Filtered list of mixins
        """
        try:
            # Use injected detector if available (DIP compliance)
            if self._injected_drop_detector is not None:
                drop_task = self._injected_drop_detector
            else:
                # Lazy import for default creation (backwards compatibility)
                from .drop_detection import DropDetectionTask
                from ..config import MixingStyle, DropDetectionConfig
                drop_config = DropDetectionConfig.for_style(MixingStyle.SMOOTH)
                drop_task = DropDetectionTask(
                    min_drop_magnitude=drop_config.min_drop_magnitude,
                    min_confidence=drop_config.min_confidence
                )

            # Get drop context (reuses STFT cache → M2-optimized, no recomputation)
            drop_result = drop_task.execute(context)

            if not drop_result.success or not drop_result.drops:
                # No drops detected or task failed → keep all mixins
                return mixins

            drops = drop_result.drops

            # M2-OPTIMIZED: Vectorized distance computation using numpy
            drop_times = np.array([d.time_sec for d in drops])
            drop_confs = np.array([d.confidence for d in drops])

            filtered = []
            drop_filtered_count = 0

            for mixin in mixins:
                # Vectorized distance computation (Apple Accelerate)
                distances = np.abs(drop_times - mixin.time_sec)
                near_mask = distances < self.drop_window_sec

                if not np.any(near_mask):
                    # Not near any drop → keep
                    filtered.append(mixin)
                    continue

                # Near drop - check if it's a transition or internal drop
                max_drop_conf = np.max(drop_confs[near_mask])

                # Decision logic with bass change analysis
                if mixin.bass_change > 0.3 and max_drop_conf < 0.5:
                    # Strong bass shift suggests track change
                    filtered.append(mixin)
                elif max_drop_conf > 0.7 and mixin.bass_change < 0.2:
                    # Very confident drop + no bass change → internal drop
                    drop_filtered_count += 1
                    continue  # Filter out
                else:
                    # Uncertain - reduce confidence but keep
                    mixin.confidence *= 0.6
                    filtered.append(mixin)

            if self.verbose and drop_filtered_count > 0:
                print(f"  Drop filter: removed {drop_filtered_count}/{len(mixins)} candidates")

            return filtered

        except Exception as e:
            # Graceful degradation: if drop detection fails, don't break transition detection
            if self.verbose:
                print(f"  Drop filter failed: {e}, keeping all mixins")
            return mixins

    def _filter_by_segmentation_context(
        self,
        mixins: List[MixinEvent],
        context: AudioContext
    ) -> List[MixinEvent]:
        """
        Adjust mixin confidence based on structural segment boundaries.

        Uses SegmentationTask to identify structural boundaries, then:
        - Boosts confidence for mixins ON segment boundaries (likely real transitions)
        - Penalizes mixins INSIDE segments (likely internal changes)

        Args:
            mixins: Detected mixin candidates
            context: AudioContext with STFT cache (reused by SegmentationTask)

        Returns:
            Mixins with adjusted confidence
        """
        try:
            # Use injected segmenter if available (DIP compliance)
            if self._injected_segmenter is not None:
                seg_task = self._injected_segmenter
            else:
                # Lazy import for default creation (backwards compatibility)
                from .segmentation import SegmentationTask
                seg_task = SegmentationTask(
                    min_segment_sec=60.0,  # Minimum track duration
                    bar_downsample=4       # Use downbeats only (faster)
                )

            # Run segmentation (reuses STFT cache → M2-optimized)
            seg_result = seg_task.execute(context)

            if not seg_result.success or not seg_result.boundaries:
                # No segmentation available → no adjustment
                return mixins

            boundaries = seg_result.boundaries

            # M2-OPTIMIZED: Matrix broadcasting for ALL mixins at once
            # Shape: (n_mixins, 1) - (1, n_boundaries) = (n_mixins, n_boundaries)
            mixin_times = np.array([m.time_sec for m in mixins])
            boundary_times = np.array([0.0] + [b.time_sec for b in boundaries] + [context.duration_sec])

            # Broadcasting distance computation (O(1) matrix operation)
            dist_matrix = np.abs(mixin_times[:, np.newaxis] - boundary_times[np.newaxis, :])
            min_distances = np.min(dist_matrix, axis=1)

            segment_boost_count = 0
            segment_penalty_count = 0

            # Apply adjustments
            for i, mixin in enumerate(mixins):
                if min_distances[i] < self.boundary_tolerance_sec:
                    # Near boundary → boost confidence
                    mixin.confidence = min(1.0, mixin.confidence * 1.25)
                    segment_boost_count += 1
                else:
                    # Inside segment → penalty
                    mixin.confidence *= 0.85
                    segment_penalty_count += 1

            if self.verbose:
                print(f"  Segmentation filter: {segment_boost_count} boosted, {segment_penalty_count} penalized")

            return mixins

        except Exception as e:
            # Graceful degradation: if segmentation fails, don't break transition detection
            if self.verbose:
                print(f"  Segmentation filter failed: {e}, keeping all mixins")
            return mixins

    def _apply_drop_filter_from_cache(self, mixins: List[MixinEvent], drops: List) -> List[MixinEvent]:
        """Apply drop filtering using cached drop results."""
        if not drops:
            return mixins

        drop_times = np.array([d.time_sec for d in drops])
        drop_confs = np.array([d.confidence for d in drops])

        filtered = []
        for mixin in mixins:
            distances = np.abs(drop_times - mixin.time_sec)
            near_mask = distances < self.drop_window_sec

            if not np.any(near_mask):
                filtered.append(mixin)
                continue

            max_drop_conf = np.max(drop_confs[near_mask])

            if mixin.bass_change > 0.3 and max_drop_conf < 0.5:
                filtered.append(mixin)
            elif max_drop_conf > 0.7 and mixin.bass_change < 0.2:
                continue
            else:
                mixin.confidence *= 0.6
                filtered.append(mixin)

        return filtered

    def _apply_segmentation_filter_from_cache(self, mixins: List[MixinEvent], seg_boundaries: List, duration_sec: float) -> List[MixinEvent]:
        """Apply segmentation filtering using cached boundary results."""
        if not seg_boundaries:
            return mixins

        mixin_times = np.array([m.time_sec for m in mixins])
        boundary_times = np.array([0.0] + [b.time_sec for b in seg_boundaries] + [duration_sec])

        dist_matrix = np.abs(mixin_times[:, np.newaxis] - boundary_times[np.newaxis, :])
        min_distances = np.min(dist_matrix, axis=1)

        for i, mixin in enumerate(mixins):
            if min_distances[i] < self.boundary_tolerance_sec:
                mixin.confidence = min(1.0, mixin.confidence * 1.25)
            else:
                mixin.confidence *= 0.85

        return mixins

    # ============== Fast execution for calibration ==============

    @staticmethod
    def compute_raw_features(context: 'AudioContext', include_context_filters: bool = True) -> Dict[str, Any]:
        """
        Compute raw features that don't depend on task parameters.

        Use this for calibration: compute once, cache, then call execute_from_features()
        with different parameters.

        Args:
            context: AudioContext with STFT cache
            include_context_filters: If True, also compute drop/segmentation results for filtering

        Returns:
            Dict with:
                - rms, bass_energy, novelty, timbral_novelty, chroma_novelty
                - centroid, rolloff, centroid_velocity
                - sr, hop_length, duration_sec (metadata)
                - drops, seg_boundaries (if include_context_filters=True)
        """
        S = context.stft_cache.S
        freqs = context.stft_cache.freqs
        sr = context.sr
        hop_length = context.hop_length

        # Compute features (use STFTCache for consistency)
        rms = context.stft_cache.get_rms()
        bands = compute_frequency_bands(S, freqs)
        bass_energy = bands.bass + bands.sub_bass
        novelty = compute_novelty(S, metric='cosine')

        # Timbral novelty (MFCC-based) - use STFTCache lazy computation
        mfcc = context.stft_cache.get_mfcc(n_mfcc=13)
        mfcc_diff = np.diff(mfcc, axis=1)
        mfcc_dist = np.sqrt(np.sum(mfcc_diff ** 2, axis=0))
        mfcc_dist = np.concatenate([[0], mfcc_dist])
        if np.max(mfcc_dist) > 0:
            mfcc_dist = mfcc_dist / np.max(mfcc_dist)
        timbral_novelty = mfcc_dist

        # Chroma novelty - use STFTCache lazy computation
        chroma = context.stft_cache.get_chroma()
        chroma_diff = np.diff(chroma, axis=1)
        chroma_dist = np.sqrt(np.sum(chroma_diff ** 2, axis=0))
        chroma_dist = np.concatenate([[0], chroma_dist])
        if np.max(chroma_dist) > 0:
            chroma_dist = chroma_dist / np.max(chroma_dist)
        chroma_novelty = chroma_dist

        # Spectral features for filter detection (use STFTCache for consistency)
        centroid = context.stft_cache.get_spectral_centroid()
        rolloff = context.stft_cache.get_spectral_rolloff()
        centroid_velocity = compute_spectral_velocity(centroid, sr, hop_length)

        result = {
            'rms': rms,
            'bass_energy': bass_energy,
            'novelty': novelty,
            'timbral_novelty': timbral_novelty,
            'chroma_novelty': chroma_novelty,
            'centroid': centroid,
            'rolloff': rolloff,
            'centroid_velocity': centroid_velocity,
            'sr': sr,
            'hop_length': hop_length,
            'duration_sec': context.duration_sec,
        }

        # CONTEXT FILTERS: Cache drop and segmentation results for filtering
        if include_context_filters:
            try:
                # Run DropDetectionTask (reuses STFT cache)
                from .drop_detection import DropDetectionTask
                from ..config import MixingStyle, DropDetectionConfig

                drop_config = DropDetectionConfig.for_style(MixingStyle.SMOOTH)
                drop_task = DropDetectionTask(
                    min_drop_magnitude=drop_config.min_drop_magnitude,
                    min_confidence=drop_config.min_confidence
                )
                drop_result = drop_task.execute(context)
                result['drops'] = drop_result.drops if drop_result.success else []
            except Exception as e:
                print(f"WARNING: DropDetection failed in compute_raw_features: {e}", flush=True)
                result['drops'] = []

            try:
                # Run SegmentationTask (reuses STFT cache)
                from .segmentation import SegmentationTask

                seg_task = SegmentationTask(
                    min_segment_sec=60.0,
                    bar_downsample=4
                )
                seg_result = seg_task.execute(context)
                result['seg_boundaries'] = seg_result.boundaries if seg_result.success else []
            except Exception as e:
                print(f"WARNING: Segmentation failed in compute_raw_features: {e}", flush=True)
                result['seg_boundaries'] = []

        return result

    def execute_from_features(
        self,
        rms: np.ndarray,
        bass_energy: np.ndarray,
        novelty: np.ndarray,
        timbral_novelty: np.ndarray,
        chroma_novelty: np.ndarray,
        centroid: np.ndarray,
        rolloff: np.ndarray,
        centroid_velocity: np.ndarray,
        sr: int,
        hop_length: int,
        duration_sec: float,
        drops: Optional[List] = None,
        seg_boundaries: Optional[List] = None
    ) -> TransitionDetectionResult:
        """
        Execute with pre-computed features (for calibration).

        Skips feature computation, only does smoothing + detection.
        ~30ms per call vs ~3 sec with full execute().

        Args:
            rms: Raw RMS energy (before smoothing)
            bass_energy: Bass + sub-bass energy
            novelty: Spectral novelty
            timbral_novelty: MFCC-based timbral novelty
            chroma_novelty: Chroma-based harmonic novelty
            centroid: Spectral centroid
            rolloff: Spectral rolloff
            centroid_velocity: Centroid velocity for filter detection
            sr: Sample rate
            hop_length: Hop length
            duration_sec: Total duration in seconds
            drops: Optional pre-computed drop events for filtering
            seg_boundaries: Optional pre-computed segmentation boundaries for filtering

        Returns:
            TransitionDetectionResult
        """
        start_time = time.time()

        try:
            # Apply smoothing (parameter-dependent)
            rms_smooth = smooth_gaussian(rms, sigma=self.smooth_sigma)
            bass_smooth = smooth_gaussian(bass_energy, sigma=self.smooth_sigma)
            novelty_smooth = smooth_gaussian(novelty, sigma=self.smooth_sigma)
            timbral_smooth = smooth_gaussian(timbral_novelty, sigma=self.smooth_sigma)
            chroma_smooth = smooth_gaussian(chroma_novelty, sigma=self.smooth_sigma)

            # Combined novelty
            combined_novelty = (
                novelty_smooth * 0.3 +
                timbral_smooth * 0.5 +
                chroma_smooth * 0.2
            )

            # Filter detection (parameter-dependent threshold)
            filter_pos = None
            sweep_mask = None
            if self.detect_filters:
                filter_pos = compute_filter_position(centroid, rolloff)
                sweep_mask = detect_filter_sweeps(
                    centroid_velocity, sr, hop_length,
                    velocity_threshold=self.filter_velocity_threshold
                )

            # Detect mixins and mixouts
            mixins = self._detect_mixins(
                rms_smooth, bass_smooth, combined_novelty,
                sr, hop_length,
                centroid=centroid,
                centroid_velocity=centroid_velocity,
                sweep_mask=sweep_mask
            )

            # CONTEXT FILTERING: Apply drop and segmentation filters if data provided
            if drops is not None and len(drops) > 0:
                mixins = self._apply_drop_filter_from_cache(mixins, drops)
            if seg_boundaries is not None and len(seg_boundaries) > 0:
                mixins = self._apply_segmentation_filter_from_cache(mixins, seg_boundaries, duration_sec)

            mixouts = self._detect_mixouts(
                rms_smooth, bass_smooth, combined_novelty,
                sr, hop_length,
                centroid=centroid,
                centroid_velocity=centroid_velocity,
                sweep_mask=sweep_mask
            )

            # Pair transitions
            transitions = self._pair_transitions(mixins, mixouts, sr, hop_length)

            # Calculate metrics
            transition_density = len(transitions) * 60.0 / duration_sec if duration_sec > 0 else 0.0
            avg_duration = float(np.mean([t.overlap_sec for t in transitions])) if transitions else 0.0

            type_dist = {}
            for t in transitions:
                type_name = t.transition_type.name
                type_dist[type_name] = type_dist.get(type_name, 0) + 1

            return TransitionDetectionResult(
                success=True,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                mixins=mixins,
                mixouts=mixouts,
                transitions=transitions,
                transition_density=transition_density,
                avg_transition_duration=avg_duration,
                transition_type_distribution=type_dist,
                energy_curve=rms_smooth,
                bass_curve=bass_smooth,
                filter_curve=filter_pos
            )

        except Exception as e:
            return TransitionDetectionResult(
                success=False,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                error=str(e)
            )
