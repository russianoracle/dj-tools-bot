"""
Drop Detection Task - Detect drops and buildups in tracks.

Drops are characterized by:
1. Energy buildup phase
2. Sudden energy increase
3. Sustained high energy after
4. (Optional) Multi-band energy analysis for better drop detection

Librosa is called here (Task layer) to compute mel spectrograms,
which are then passed to pure mathematical primitives.

Mode Selection:
- TRACK mode (default): For individual tracks with clear buildup→drop structure
  - Higher thresholds (min_drop_magnitude=0.3, min_confidence=0.5)

- DJ_SET mode: For continuous DJ mixes with sustained energy
  - Lower thresholds (min_drop_magnitude=0.05, min_confidence=0.2)
  - DJ sets maintain consistent energy, so drops are less pronounced
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time

from .base import AudioContext, TaskResult, BaseTask


class DropDetectionMode(Enum):
    """Mode for drop detection thresholds."""
    TRACK = "track"      # Individual track with clear buildup→drop
    DJ_SET = "dj_set"    # Continuous DJ mix with sustained energy


from ..primitives import (
    # NOTE: compute_rms is BLOCKED - use context.stft_cache.get_rms() instead
    detect_drop_candidates,
    compute_buildup_score,
    compute_novelty,
    detect_peaks,
    detect_valleys,
    smooth_gaussian,
    DropCandidate,
    # Multi-band analysis
    compute_mel_band_energies,
    compute_weighted_energy,
)


@dataclass
class BuildupEvent:
    """A buildup event leading to a drop."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    intensity: float
    duration_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'intensity': self.intensity,
            'duration_sec': self.duration_sec,
        }


@dataclass
class DropDetectionResult(TaskResult):
    """
    Result of drop detection with multi-band analysis.

    Attributes:
        drops: List of detected drops
        buildups: List of buildup events
        drop_density: Drops per minute
        max_drop_magnitude: Largest drop magnitude
        drop_timeline: Time series of drop likelihood

        # Aggregate metrics (from multi-band analysis)
        avg_drop_intensity: Average drop magnitude
        avg_buildup_duration: Average buildup duration (seconds)
        avg_recovery_rate: Average energy recovery rate
        avg_bass_prominence: Average bass prominence at drops

        # Temporal distribution metrics
        drops_in_first_half: Number of drops in first half
        drops_in_second_half: Number of drops in second half
        drop_temporal_distribution: 0=early-heavy, 0.5=balanced, 1=late-heavy

        # Energy metrics
        energy_variance: Overall energy variance
        energy_range: Peak-to-peak energy range
        bass_energy_mean: Mean bass band energy
        bass_energy_var: Bass energy variance
    """
    drops: List[DropCandidate] = field(default_factory=list)
    buildups: List[BuildupEvent] = field(default_factory=list)
    drop_density: float = 0.0
    max_drop_magnitude: float = 0.0
    drop_timeline: Optional[np.ndarray] = None

    # Aggregate metrics
    avg_drop_intensity: float = 0.0
    avg_buildup_duration: float = 0.0
    avg_recovery_rate: float = 0.0
    avg_bass_prominence: float = 0.0

    # Temporal distribution metrics
    drops_in_first_half: int = 0
    drops_in_second_half: int = 0
    drop_temporal_distribution: float = 0.5

    # Energy metrics
    energy_variance: float = 0.0
    energy_range: float = 0.0
    bass_energy_mean: float = 0.0
    bass_energy_var: float = 0.0

    @property
    def n_drops(self) -> int:
        return len(self.drops)

    @property
    def has_drops(self) -> bool:
        return len(self.drops) > 0

    def get_drop_times(self) -> List[float]:
        """Get drop times in seconds."""
        return [d.time_sec for d in self.drops]

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'drops': [d.to_dict() for d in self.drops],
            'buildups': [b.to_dict() for b in self.buildups],
            'drop_density': self.drop_density,
            'max_drop_magnitude': self.max_drop_magnitude,
            'n_drops': self.n_drops,
            'drop_times': self.get_drop_times(),
            # Aggregate metrics
            'avg_drop_intensity': self.avg_drop_intensity,
            'avg_buildup_duration': self.avg_buildup_duration,
            'avg_recovery_rate': self.avg_recovery_rate,
            'avg_bass_prominence': self.avg_bass_prominence,
            # Temporal
            'drops_in_first_half': self.drops_in_first_half,
            'drops_in_second_half': self.drops_in_second_half,
            'drop_temporal_distribution': self.drop_temporal_distribution,
            # Energy
            'energy_variance': self.energy_variance,
            'energy_range': self.energy_range,
            'bass_energy_mean': self.bass_energy_mean,
            'bass_energy_var': self.bass_energy_var,
        })
        return base


class DropDetectionTask(BaseTask):
    """
    Detect drops and buildups in a track.

    Uses energy analysis and novelty detection to find:
    - Buildup phases (energy accumulation)
    - Drop moments (sudden energy release)
    - Drop intensity and confidence

    Multi-band mode uses mel spectrogram for better drop detection
    in electronic music (bass-weighted energy analysis).

    Configuration:
        min_drop_magnitude: Minimum energy jump for a drop (0-1)
        min_confidence: Minimum confidence threshold
        buildup_window_sec: Lookback window for buildup detection
        use_multiband: Enable mel-band weighted energy analysis

    Usage:
        # For individual tracks (default)
        task = DropDetectionTask.for_track()

        # For DJ sets (lower thresholds)
        task = DropDetectionTask.for_dj_set()

        # Custom configuration
        task = DropDetectionTask(min_drop_magnitude=0.3, use_multiband=True)

        result = task.execute(context)
        for drop in result.drops:
            print(f"Drop at {drop.time_sec:.1f}s (confidence: {drop.confidence:.2f})")
    """

    # Mode-specific threshold presets
    MODE_PRESETS = {
        DropDetectionMode.TRACK: {
            'min_drop_magnitude': 0.3,
            'min_confidence': 0.5,
        },
        DropDetectionMode.DJ_SET: {
            # DJ sets have sustained energy - drops are less pronounced
            'min_drop_magnitude': 0.05,
            'min_confidence': 0.2,
        },
    }

    @classmethod
    def for_track(cls, **kwargs) -> 'DropDetectionTask':
        """
        Create task for individual track analysis.

        Uses higher thresholds for clear buildup→drop detection.
        """
        preset = cls.MODE_PRESETS[DropDetectionMode.TRACK].copy()
        preset.update(kwargs)
        return cls(**preset)

    @classmethod
    def for_dj_set(cls, **kwargs) -> 'DropDetectionTask':
        """
        Create task for DJ set analysis.

        Uses lower thresholds since DJ sets have:
        - Sustained high energy (no dramatic drops)
        - Gradual transitions (30+ second fades)
        - Lower energy variance

        Typically detects ~2 drops per minute in techno sets.
        """
        preset = cls.MODE_PRESETS[DropDetectionMode.DJ_SET].copy()
        preset.update(kwargs)
        return cls(**preset)

    def __init__(
        self,
        min_drop_magnitude: float = 0.3,
        min_confidence: float = 0.5,
        buildup_window_sec: float = 2.0,
        smooth_sigma: float = 3.0,
        use_multiband: bool = True,
        n_mels: int = 128,
        # Beat grid alignment parameters
        snap_to_grid: bool = True,
        grid_confidence_boost: float = 1.25,
        grid_confidence_penalty: float = 0.85,
        grid_snap_tolerance_beats: float = 2.0,
    ):
        """
        Initialize drop detection task.

        Args:
            min_drop_magnitude: Minimum energy jump for a drop
            min_confidence: Minimum confidence threshold
            buildup_window_sec: Lookback window in seconds
            smooth_sigma: Gaussian smoothing sigma
            use_multiband: Use mel-band weighted energy (recommended)
            n_mels: Number of mel bands for multi-band analysis
            snap_to_grid: Snap drops to phrase boundaries if beat_grid available
            grid_confidence_boost: Confidence multiplier for drops on phrase boundaries
            grid_confidence_penalty: Confidence multiplier for drops off phrase boundaries
            grid_snap_tolerance_beats: Max distance (in beats) to snap to boundary
        """
        self.min_drop_magnitude = min_drop_magnitude
        self.min_confidence = min_confidence
        self.buildup_window_sec = buildup_window_sec
        self.smooth_sigma = smooth_sigma
        self.use_multiband = use_multiband
        self.n_mels = n_mels
        # Beat grid alignment
        self.snap_to_grid = snap_to_grid
        self.grid_confidence_boost = grid_confidence_boost
        self.grid_confidence_penalty = grid_confidence_penalty
        self.grid_snap_tolerance_beats = grid_snap_tolerance_beats

    def execute(self, context: AudioContext) -> DropDetectionResult:
        """Detect drops in the track with optional multi-band analysis."""
        start_time = time.time()

        try:
            S = context.stft_cache.S
            sr = context.sr
            hop_length = context.hop_length
            duration = context.duration_sec
            y = context.y

            # Compute RMS energy (use STFTCache for consistency)
            rms = context.stft_cache.get_rms()
            rms_smooth = smooth_gaussian(rms, sigma=self.smooth_sigma)

            # Multi-band analysis (librosa call happens here in Task layer)
            bass_energy = None
            mel_bands = None
            combined_energy = rms_smooth

            if self.use_multiband and y is not None:
                # Compute mel spectrogram (Task layer - librosa)
                S_mel = librosa.feature.melspectrogram(
                    y=y, sr=sr, hop_length=hop_length, n_mels=self.n_mels
                )
                S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

                # Extract band energies (Primitive layer - pure math)
                mel_bands = compute_mel_band_energies(S_mel_db, n_mels=self.n_mels)
                bass_energy = mel_bands.bass

                # Compute weighted energy for drop detection
                combined_energy = compute_weighted_energy(mel_bands, rms_smooth)
                combined_energy = smooth_gaussian(combined_energy, sigma=self.smooth_sigma)

            # Compute buildup window in frames
            buildup_frames = int(self.buildup_window_sec * sr / hop_length)
            recovery_frames = int(1.0 * sr / hop_length)  # 1 second recovery window

            # Detect drops with extended metrics
            drops = detect_drop_candidates(
                energy=combined_energy,
                sr=sr,
                hop_length=hop_length,
                buildup_window=buildup_frames,
                min_drop_magnitude=self.min_drop_magnitude,
                min_confidence=self.min_confidence,
                bass_energy=bass_energy,
                recovery_window=recovery_frames
            )

            # Apply beat grid alignment if available
            beat_grid = context.beat_grid
            if beat_grid is not None and self.snap_to_grid and len(drops) > 0:
                drops = self._apply_grid_alignment(drops, beat_grid, sr, hop_length)

            # Detect buildups
            buildups = self._detect_buildups(
                rms_smooth, drops, sr, hop_length
            )

            # Compute basic statistics
            drop_density = len(drops) * 60.0 / duration if duration > 0 else 0.0
            max_magnitude = max((d.drop_magnitude for d in drops), default=0.0)

            # Compute aggregate metrics (vectorized)
            avg_drop_intensity = float(np.mean([d.drop_magnitude for d in drops])) if drops else 0.0
            avg_buildup_duration = float(np.mean([d.buildup_duration for d in drops])) if drops else 0.0
            avg_recovery_rate = float(np.mean([d.recovery_rate for d in drops])) if drops else 0.0
            avg_bass_prominence = float(np.mean([d.bass_prominence for d in drops])) if drops else 0.0

            # Compute temporal distribution (vectorized)
            mid_time = duration / 2
            drop_times = np.array([d.time_sec for d in drops]) if drops else np.array([])
            drops_first = int(np.sum(drop_times < mid_time)) if len(drop_times) > 0 else 0
            drops_second = len(drops) - drops_first
            temporal_dist = drops_second / len(drops) if drops else 0.5

            # Compute energy statistics (vectorized)
            energy_variance = float(np.var(combined_energy))
            energy_range = float(np.ptp(combined_energy))
            bass_mean = float(np.mean(bass_energy)) if bass_energy is not None else 0.0
            bass_var = float(np.var(bass_energy)) if bass_energy is not None else 0.0

            # Compute drop timeline
            drop_timeline = self._compute_drop_timeline(
                rms_smooth, drops, buildup_frames
            )

            return DropDetectionResult(
                success=True,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                drops=drops,
                buildups=buildups,
                drop_density=drop_density,
                max_drop_magnitude=max_magnitude,
                drop_timeline=drop_timeline,
                # Aggregate metrics
                avg_drop_intensity=avg_drop_intensity,
                avg_buildup_duration=avg_buildup_duration,
                avg_recovery_rate=avg_recovery_rate,
                avg_bass_prominence=avg_bass_prominence,
                # Temporal
                drops_in_first_half=drops_first,
                drops_in_second_half=drops_second,
                drop_temporal_distribution=temporal_dist,
                # Energy
                energy_variance=energy_variance,
                energy_range=energy_range,
                bass_energy_mean=bass_mean,
                bass_energy_var=bass_var,
            )

        except Exception as e:
            return DropDetectionResult(
                success=False,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                error=str(e)
            )

    def _detect_buildups(
        self,
        rms: np.ndarray,
        drops: List[DropCandidate],
        sr: int,
        hop_length: int
    ) -> List[BuildupEvent]:
        """Detect buildup phases before each drop."""
        buildups = []
        frame_to_time = hop_length / sr

        # Compute buildup score
        buildup_frames = int(self.buildup_window_sec * sr / hop_length)
        buildup_score = compute_buildup_score(rms, window_frames=buildup_frames)

        for drop in drops:
            # Look back from drop to find buildup start
            end_frame = drop.frame_idx
            start_frame = max(0, end_frame - buildup_frames * 2)

            # Find where buildup starts (buildup score rises above threshold)
            segment = buildup_score[start_frame:end_frame]
            if len(segment) == 0:
                continue

            threshold = 0.3 * np.max(segment)
            above_threshold = segment > threshold

            if np.any(above_threshold):
                first_above = np.argmax(above_threshold)
                actual_start = start_frame + first_above
            else:
                actual_start = start_frame

            # Calculate buildup metrics
            duration = (end_frame - actual_start) * frame_to_time
            intensity = float(np.mean(segment[first_above:] if np.any(above_threshold) else segment))

            # Check for REAL buildup: duration AND significant intensity increase
            # A real buildup should show rising energy, not just any energy before drop
            if duration > 0.5 and intensity > 0.2:  # Minimum 0.5s AND intensity > 0.2
                # Additional check: energy should actually RISE during buildup
                buildup_segment = rms[actual_start:end_frame]
                if len(buildup_segment) > 2:
                    # Check if energy is rising (end > start)
                    start_energy = np.mean(buildup_segment[:len(buildup_segment)//3])
                    end_energy = np.mean(buildup_segment[-len(buildup_segment)//3:])

                    if end_energy > start_energy * 1.1:  # At least 10% increase
                        buildups.append(BuildupEvent(
                            start_frame=actual_start,
                            end_frame=end_frame,
                            start_time=actual_start * frame_to_time,
                            end_time=end_frame * frame_to_time,
                            intensity=intensity,
                            duration_sec=duration
                        ))

        return buildups

    def _compute_drop_timeline(
        self,
        rms: np.ndarray,
        drops: List[DropCandidate],
        buildup_frames: int
    ) -> np.ndarray:
        """
        Compute drop likelihood per frame.

        Vectorized implementation using broadcasting for Gaussian kernels.
        ~5-10x faster than nested loop version.
        """
        n_frames = len(rms)

        if not drops:
            return np.zeros(n_frames)

        sigma = buildup_frames // 4
        if sigma == 0:
            sigma = 1  # Avoid division by zero

        # Create index array for all frames
        indices = np.arange(n_frames, dtype=np.float32)

        # Stack all Gaussian kernels at once
        # Shape: (n_drops, n_frames)
        gaussians = []
        for drop in drops:
            # Compute Gaussian for this drop across all frames
            center = drop.frame_idx
            gaussian = drop.confidence * np.exp(-0.5 * ((indices - center) / sigma) ** 2)
            gaussians.append(gaussian)

        # Stack and take element-wise maximum
        if gaussians:
            gaussians = np.stack(gaussians, axis=0)
            timeline = np.max(gaussians, axis=0)
        else:
            timeline = np.zeros(n_frames, dtype=np.float32)

        return timeline

    def _apply_grid_alignment(
        self,
        drops: List[DropCandidate],
        beat_grid,  # BeatGridResult
        sr: int,
        hop_length: int
    ) -> List[DropCandidate]:
        """
        Align drops to phrase boundaries using beat grid.

        In electronic music, drops ALWAYS happen on phrase boundaries
        (every 16 beats / 4 bars). This method:
        1. Checks if each drop is near a phrase boundary
        2. Snaps drops to the nearest boundary if within tolerance
        3. Boosts confidence for drops on boundaries
        4. Penalizes confidence for drops far from boundaries

        Args:
            drops: List of detected drops
            beat_grid: BeatGridResult with phrase boundaries
            sr: Sample rate
            hop_length: Hop length for frame conversion

        Returns:
            List of aligned drops with adjusted confidence
        """
        if not drops or beat_grid is None:
            return drops

        aligned_drops = []
        tolerance_sec = self.grid_snap_tolerance_beats * beat_grid.beat_duration_sec

        for drop in drops:
            original_time = drop.time_sec

            # Check if drop is near a phrase boundary
            is_on_boundary = beat_grid.is_on_phrase_boundary(
                original_time,
                tolerance_beats=self.grid_snap_tolerance_beats
            )

            if is_on_boundary:
                # Snap to exact boundary and boost confidence
                snapped_time = beat_grid.snap_to_phrase(original_time)
                new_confidence = min(1.0, drop.confidence * self.grid_confidence_boost)

                # Convert snapped time to frame
                snapped_frame = int(snapped_time * sr / hop_length)
            else:
                # Keep original time but penalize confidence
                snapped_time = original_time
                snapped_frame = drop.frame_idx
                new_confidence = drop.confidence * self.grid_confidence_penalty

            # Create aligned drop with adjusted values
            # DropCandidate is a dataclass, so we need to create a new instance
            aligned_drop = DropCandidate(
                frame_idx=snapped_frame,
                time_sec=snapped_time,
                buildup_score=drop.buildup_score,  # Note: field is buildup_score, not buildup_magnitude
                drop_magnitude=drop.drop_magnitude,
                confidence=new_confidence,
                bass_prominence=drop.bass_prominence,
                buildup_duration=drop.buildup_duration,
                recovery_rate=drop.recovery_rate,
            )
            aligned_drops.append(aligned_drop)

        return aligned_drops
