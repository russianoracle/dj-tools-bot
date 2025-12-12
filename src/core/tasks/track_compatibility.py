"""
Track Compatibility Tasks - Business logic for DJ track compatibility.

ARCHITECTURE: This is TASKS layer.
- Each Task has ONE responsibility
- Tasks do NOT call other Tasks (Pipeline orchestrates)
- Tasks call Primitives for mathematical operations
- NO file I/O (that's Pipeline's job)

Available Tasks:
- BpmDetectionTask: Detect BPM with octave correction
- MixPointDetectionTask: Find optimal mix-in/out points
- GridCalibrationTask: Calibrate beat grid using drops

For orchestration of all tasks, use TrackAnalysisPipeline.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

from .base import AudioContext, TaskResult, BaseTask
from ..primitives import (
    # NOTE: compute_rms, compute_centroid, compute_onset_strength are BLOCKED
    # Use context.stft_cache.get_rms(), get_spectral_centroid(), get_onset_strength() instead
    smooth_gaussian,
    normalize_minmax,
    compute_tempo_multi,
    calibrate_grid_phase,
    apply_phase_correction,
)
from ..primitives.transition_scoring import (
    TransitionScore,
    compute_transition_score,
)


# ============== Data Classes ==============

@dataclass
class TrackAnalysis:
    """
    Complete track analysis for DJ set generation.

    This is a DATA CLASS, not a Task.
    Contains all data needed to score transitions and place
    tracks in a set.

    Populated by TrackAnalysisPipeline, not by a single Task.
    """
    # File info (set by Pipeline)
    path: str = ""
    filename: str = ""
    duration_sec: float = 0.0

    # BPM & Key (from BpmDetectionTask, KeyAnalysisTask)
    bpm: float = 0.0
    key: str = ""
    camelot: str = ""

    # Genre (from GenreAnalysisTask)
    genre: str = "Unknown"
    genre_confidence: float = 0.0
    dj_category: str = "Unknown"

    # Energy (from EnergyArcAnalysisTask)
    energy_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    intro_energy: float = 0.5
    outro_energy: float = 0.5
    peak_energy: float = 0.5

    # Drops (from DropDetectionTask)
    drop_times: List[float] = field(default_factory=list)
    drop_count: int = 0

    # Mix points (from MixPointDetectionTask)
    best_mix_in: float = 0.0
    best_mix_out: float = 0.0

    # Spectral (from SpectralAnalysisTask or primitive)
    spectral_centroid_mean: float = 2000.0

    # Beat grid (from BeatGridTask + GridCalibrationTask)
    beat_times: List[float] = field(default_factory=list)
    bar_boundaries: List[float] = field(default_factory=list)
    phrase_boundaries: List[float] = field(default_factory=list)
    beat_duration_sec: float = 0.5
    bar_duration_sec: float = 2.0
    phrase_duration_sec: float = 8.0
    grid_calibrated: bool = False
    calibration_confidence: float = 0.0

    # Metadata
    analysis_time_sec: float = 0.0
    source: str = "dsp"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'path': self.path,
            'filename': self.filename,
            'duration_sec': float(self.duration_sec),
            'bpm': float(self.bpm),
            'key': self.key,
            'camelot': self.camelot,
            'genre': self.genre,
            'genre_confidence': float(self.genre_confidence),
            'dj_category': self.dj_category,
            'intro_energy': float(self.intro_energy),
            'outro_energy': float(self.outro_energy),
            'peak_energy': float(self.peak_energy),
            'drop_times': [float(t) for t in self.drop_times],
            'drop_count': self.drop_count,
            'best_mix_in': float(self.best_mix_in),
            'best_mix_out': float(self.best_mix_out),
            'spectral_centroid_mean': float(self.spectral_centroid_mean),
            'beat_times': [float(t) for t in self.beat_times],
            'bar_boundaries': [float(t) for t in self.bar_boundaries],
            'phrase_boundaries': [float(t) for t in self.phrase_boundaries],
            'beat_duration_sec': float(self.beat_duration_sec),
            'bar_duration_sec': float(self.bar_duration_sec),
            'phrase_duration_sec': float(self.phrase_duration_sec),
            'grid_calibrated': self.grid_calibrated,
            'calibration_confidence': float(self.calibration_confidence),
            'analysis_time_sec': float(self.analysis_time_sec),
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackAnalysis':
        """Create from dict (e.g., from cache)."""
        return cls(
            path=data.get('path', ''),
            filename=data.get('filename', ''),
            duration_sec=data.get('duration_sec', 0.0),
            bpm=data.get('bpm', 0.0),
            key=data.get('key', ''),
            camelot=data.get('camelot', ''),
            genre=data.get('genre', 'Unknown'),
            genre_confidence=data.get('genre_confidence', 0.0),
            dj_category=data.get('dj_category', 'Unknown'),
            energy_curve=np.array(data.get('energy_curve', [])),
            intro_energy=data.get('intro_energy', 0.5),
            outro_energy=data.get('outro_energy', 0.5),
            peak_energy=data.get('peak_energy', 0.5),
            drop_times=data.get('drop_times', []),
            drop_count=data.get('drop_count', 0),
            best_mix_in=data.get('best_mix_in', 0.0),
            best_mix_out=data.get('best_mix_out', 0.0),
            spectral_centroid_mean=data.get('spectral_centroid_mean', 2000.0),
            beat_times=data.get('beat_times', []),
            bar_boundaries=data.get('bar_boundaries', []),
            phrase_boundaries=data.get('phrase_boundaries', []),
            beat_duration_sec=data.get('beat_duration_sec', 0.5),
            bar_duration_sec=data.get('bar_duration_sec', 2.0),
            phrase_duration_sec=data.get('phrase_duration_sec', 8.0),
            grid_calibrated=data.get('grid_calibrated', False),
            calibration_confidence=data.get('calibration_confidence', 0.0),
            analysis_time_sec=data.get('analysis_time_sec', 0.0),
            source=data.get('source', 'dsp'),
        )


# ============== Task Results ==============

@dataclass
class BpmDetectionResult(TaskResult):
    """Result of BPM detection."""
    success: bool = True
    task_name: str = "BpmDetection"
    processing_time_sec: float = 0.0
    error: Optional[str] = None
    bpm: float = 0.0
    confidence: float = 0.0
    bpm_std: float = 0.0
    corrected: bool = False


@dataclass
class MixPointResult(TaskResult):
    """Result of mix point detection."""
    success: bool = True
    task_name: str = "MixPointDetection"
    processing_time_sec: float = 0.0
    error: Optional[str] = None
    best_mix_in: float = 0.0
    best_mix_out: float = 0.0


@dataclass
class GridCalibrationResult(TaskResult):
    """Result of beat grid calibration."""
    success: bool = True
    task_name: str = "GridCalibration"
    processing_time_sec: float = 0.0
    error: Optional[str] = None
    calibrated: bool = False
    confidence: float = 0.0
    phase_offset_sec: float = 0.0


@dataclass
class SpectralAnalysisResult(TaskResult):
    """Result of spectral analysis."""
    success: bool = True
    task_name: str = "SpectralAnalysis"
    processing_time_sec: float = 0.0
    error: Optional[str] = None
    centroid_mean: float = 0.0
    centroid_std: float = 0.0


# ============== Individual Tasks ==============

class BpmDetectionTask(BaseTask):
    """
    Detect BPM with octave error correction.

    Uses compute_tempo_multi() from primitives and applies
    correction for common detection errors (half/double tempo).

    M2 Optimized: Uses vectorized onset_strength computation.

    Does NOT call other Tasks - Pipeline orchestrates.
    """

    def __init__(
        self,
        target_range: tuple = (118, 145),
        correct_octave: bool = True,
    ):
        """
        Args:
            target_range: Expected BPM range for dance music
            correct_octave: Apply octave error correction
        """
        super().__init__()
        self.target_min, self.target_max = target_range
        self.correct_octave = correct_octave

    @property
    def name(self) -> str:
        return "BpmDetection"

    def execute(self, context: AudioContext) -> BpmDetectionResult:
        """Detect BPM from audio context."""
        start_time = time.time()

        try:
            # Get onset envelope from STFTCache (librosa-based, M2 optimized)
            onset_env = context.stft_cache.get_onset_strength(aggregate=True)

            # Multi-method tempo estimation (uses primitives)
            bpm, confidence, bpm_std = compute_tempo_multi(
                onset_env,
                context.sr,
                context.hop_length
            )

            # Octave error correction for dance music
            corrected = False
            if self.correct_octave:
                bpm, corrected = self._correct_octave_error(bpm)

            return BpmDetectionResult(
                success=True,
                processing_time_sec=time.time() - start_time,
                bpm=bpm,
                confidence=confidence,
                bpm_std=bpm_std,
                corrected=corrected,
            )

        except Exception as e:
            return BpmDetectionResult(
                success=False,
                processing_time_sec=time.time() - start_time,
                error=str(e),
            )

    def _correct_octave_error(self, bpm: float) -> tuple:
        """Correct common octave detection errors."""
        if bpm < 115:
            if self.target_min <= bpm * 1.5 <= self.target_max:
                return bpm * 1.5, True
            if self.target_min <= bpm * 2 <= 150:
                return bpm * 2, True
            if self.target_min <= bpm * 1.25 <= self.target_max:
                return bpm * 1.25, True
        elif bpm > 155:
            if self.target_min <= bpm / 2 <= self.target_max:
                return bpm / 2, True
        return bpm, False


class MixPointDetectionTask(BaseTask):
    """
    Detect optimal mix-in and mix-out points.

    Uses energy curve and drop positions to find safe mixing zones.
    Pure business logic, calls NO other Tasks.

    Input data (energy_curve, drop_times) must be provided
    by Pipeline via execute_with_data().
    """

    def __init__(self, mix_zone_sec: float = 32.0):
        super().__init__()
        self.mix_zone_sec = mix_zone_sec

    @property
    def name(self) -> str:
        return "MixPointDetection"

    def execute(self, context: AudioContext) -> MixPointResult:
        """
        Standard execute - requires energy_curve and drop_times in context.metadata.

        Prefer execute_with_data() for explicit data passing.
        """
        energy_curve = context.metadata.get('energy_curve', np.array([]))
        drop_times = context.metadata.get('drop_times', [])
        return self.execute_with_data(context.duration_sec, energy_curve, drop_times)

    def execute_with_data(
        self,
        duration_sec: float,
        energy_curve: np.ndarray,
        drop_times: List[float],
    ) -> MixPointResult:
        """
        Detect mix points from energy and drops.

        Called by Pipeline after EnergyArcAnalysisTask and DropDetectionTask.
        """
        start_time = time.time()

        try:
            best_mix_in = self._find_mix_in_point(
                energy_curve, duration_sec, drop_times
            )
            best_mix_out = self._find_mix_out_point(
                energy_curve, duration_sec, drop_times
            )

            return MixPointResult(
                success=True,
                processing_time_sec=time.time() - start_time,
                best_mix_in=best_mix_in,
                best_mix_out=best_mix_out,
            )

        except Exception as e:
            return MixPointResult(
                success=False,
                processing_time_sec=time.time() - start_time,
                error=str(e),
            )

    def _find_mix_in_point(
        self,
        energy_curve: np.ndarray,
        duration_sec: float,
        drop_times: List[float]
    ) -> float:
        """Find optimal mix-in point (vectorized)."""
        if len(energy_curve) == 0 or duration_sec <= 0:
            return 0.0

        # Search first 20% (vectorized)
        search_end_idx = max(2, int(len(energy_curve) * 0.2))
        search_region = energy_curve[:search_end_idx]
        min_idx = np.argmin(search_region)
        mix_in_sec = (min_idx / len(energy_curve)) * duration_sec

        # Avoid drops
        for drop_time in drop_times:
            if abs(drop_time - mix_in_sec) < self.mix_zone_sec / 2:
                mix_in_sec = max(0, drop_time - self.mix_zone_sec)
                break

        return max(0.0, mix_in_sec)

    def _find_mix_out_point(
        self,
        energy_curve: np.ndarray,
        duration_sec: float,
        drop_times: List[float]
    ) -> float:
        """Find optimal mix-out point (vectorized)."""
        if len(energy_curve) == 0 or duration_sec <= 0:
            return duration_sec

        # Search last 20% (vectorized)
        search_start_idx = min(len(energy_curve) - 2, int(len(energy_curve) * 0.8))
        search_region = energy_curve[search_start_idx:]
        min_idx = np.argmin(search_region) + search_start_idx
        mix_out_sec = (min_idx / len(energy_curve)) * duration_sec

        # Avoid drops
        for drop_time in drop_times:
            if abs(drop_time - mix_out_sec) < self.mix_zone_sec / 2:
                mix_out_sec = min(duration_sec, drop_time + self.mix_zone_sec)
                break

        return min(duration_sec, mix_out_sec)


class GridCalibrationTask(BaseTask):
    """
    Calibrate beat grid using anchor events (drops).

    Drops in electronic music ALWAYS start on phrase boundaries.
    Uses this to correct grid phase.

    Input data (beat_grid, anchor_events) must be provided
    by Pipeline via execute_with_grid().
    """

    def __init__(
        self,
        tolerance_beats: float = 2.0,
        min_events: int = 2,
        min_confidence: float = 0.3,
    ):
        super().__init__()
        self.tolerance_beats = tolerance_beats
        self.min_events = min_events
        self.min_confidence = min_confidence

    @property
    def name(self) -> str:
        return "GridCalibration"

    def execute(self, context: AudioContext) -> GridCalibrationResult:
        """
        Standard execute - requires beat_grid and drop_times in context.

        Prefer execute_with_grid() for explicit data passing.
        """
        beat_grid = context.beat_grid
        drop_times = context.metadata.get('drop_times', [])

        if beat_grid is None:
            return GridCalibrationResult(
                success=False,
                error="No beat_grid in context",
            )

        return self.execute_with_grid(beat_grid, drop_times)

    def execute_with_grid(
        self,
        beat_grid,  # BeatGridResult
        anchor_events: List[float],
    ) -> GridCalibrationResult:
        """
        Calibrate grid using anchor events.

        Called by Pipeline after BeatGridTask and DropDetectionTask.
        """
        start_time = time.time()

        try:
            if len(anchor_events) < self.min_events or len(beat_grid.phrases) == 0:
                return GridCalibrationResult(
                    success=True,
                    processing_time_sec=time.time() - start_time,
                    calibrated=False,
                    confidence=0.0,
                )

            # Use primitive for calibration
            anchor_arr = np.array(anchor_events, dtype=np.float32)
            calibration = calibrate_grid_phase(
                beat_grid=beat_grid,
                anchor_events=anchor_arr,
                tolerance_beats=self.tolerance_beats,
                min_events=self.min_events,
            )

            calibrated = calibration.calibration_confidence >= self.min_confidence

            return GridCalibrationResult(
                success=True,
                processing_time_sec=time.time() - start_time,
                calibrated=calibrated,
                confidence=calibration.calibration_confidence,
                phase_offset_sec=calibration.phase_offset_sec if calibrated else 0.0,
            )

        except Exception as e:
            return GridCalibrationResult(
                success=False,
                processing_time_sec=time.time() - start_time,
                error=str(e),
            )


class SpectralAnalysisTask(BaseTask):
    """
    Compute spectral characteristics (centroid, etc).

    Simple Task that calls primitives for spectral features.
    """

    @property
    def name(self) -> str:
        return "SpectralAnalysis"

    def execute(self, context: AudioContext) -> SpectralAnalysisResult:
        """Compute spectral features from STFT."""
        start_time = time.time()

        try:
            # Spectral centroid from STFTCache (librosa-based, consistent)
            centroid = context.stft_cache.get_spectral_centroid()

            return SpectralAnalysisResult(
                success=True,
                processing_time_sec=time.time() - start_time,
                centroid_mean=float(np.mean(centroid)),
                centroid_std=float(np.std(centroid)),
            )

        except Exception as e:
            return SpectralAnalysisResult(
                success=False,
                processing_time_sec=time.time() - start_time,
                error=str(e),
            )


# ============== Utility Functions ==============

def compute_track_transition_score(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    mix_zone_sec: float = 32.0,
) -> TransitionScore:
    """
    Score compatibility of track A outro → track B intro.

    Pure function - no state needed.
    Uses primitive compute_transition_score().

    This is NOT a Task - just a utility function.
    """
    return compute_transition_score(
        camelot_a=track_a.camelot,
        outro_energy_a=track_a.outro_energy,
        drop_times_a=track_a.drop_times,
        duration_a=track_a.duration_sec,
        spectral_centroid_a=track_a.spectral_centroid_mean,
        genre_a=track_a.dj_category,
        bpm_a=track_a.bpm,
        camelot_b=track_b.camelot,
        intro_energy_b=track_b.intro_energy,
        drop_times_b=track_b.drop_times,
        duration_b=track_b.duration_sec,
        spectral_centroid_b=track_b.spectral_centroid_mean,
        genre_b=track_b.dj_category,
        bpm_b=track_b.bpm,
        genre_confidence_a=track_a.genre_confidence,
        genre_confidence_b=track_b.genre_confidence,
        mix_zone_sec=mix_zone_sec,
    )


def get_energy_curve_normalized(
    trajectory: Optional[np.ndarray],
    stft_cache,
    n_points: int = 100,
) -> np.ndarray:
    """
    Get normalized energy curve (n_points points).

    Utility function for Pipeline to use.
    """
    if trajectory is not None and len(trajectory) > 0:
        return np.interp(
            np.linspace(0, 1, n_points),
            np.linspace(0, 1, len(trajectory)),
            trajectory
        )
    else:
        # Fallback: compute from RMS (use STFTCache for consistency)
        rms = stft_cache.get_rms()
        rms_smooth = smooth_gaussian(rms, sigma=10)
        rms_norm = normalize_minmax(rms_smooth)
        return np.interp(
            np.linspace(0, 1, n_points),
            np.linspace(0, 1, len(rms_norm)),
            rms_norm
        )
