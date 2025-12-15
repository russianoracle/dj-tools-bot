"""
Track Compatibility Pipeline - Analysis for DJ set generation.

Extracts features needed for track-to-track compatibility scoring:
- BPM (with correction for librosa errors)
- Key/Camelot
- Energy (intro/outro/peak)
- Drops
- Beat grid
- Mix points (best_mix_in/out)
- Genre (optional ML classification)

Uses Tasks layer for all analysis, following the architecture:
Primitives → Tasks → Pipelines
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

from .base import (
    Pipeline, PipelineContext, PipelineStage,
    LoadAudioStage, ComputeSTFTStage,
)
from app.core.connectors import CacheRepository
from app.modules.analysis.tasks import (
    # Existing Tasks
    KeyAnalysisTask,
    DropDetectionTask,
    EnergyArcAnalysisTask,
    BeatGridTask,
    GenreAnalysisTask,
    # New specialized Tasks (no inter-task dependencies)
    BpmDetectionTask,
    MixPointDetectionTask,
    GridCalibrationTask,
    SpectralAnalysisTask,
    # Utilities
    get_energy_curve_normalized,
)
from app.common.primitives import (
    apply_phase_correction,
)

logger = logging.getLogger(__name__)


# ============== Result Dataclass ==============

@dataclass
class TrackCompatibilityResult:
    """
    Result of track compatibility analysis.

    Contains all data needed for transition scoring and set building.
    """
    # Basic info
    path: str = ""
    filename: str = ""
    duration_sec: float = 0.0

    # BPM & Key
    bpm: float = 0.0
    key: str = ""
    camelot: str = ""

    # Energy
    intro_energy: float = 0.5
    outro_energy: float = 0.5
    peak_energy: float = 0.5

    # Drops
    drop_times: List[float] = field(default_factory=list)
    drop_count: int = 0

    # Mix points
    best_mix_in: float = 0.0
    best_mix_out: float = 0.0

    # Spectral
    spectral_centroid_mean: float = 2000.0

    # Beat grid
    beat_times: List[float] = field(default_factory=list)
    bar_boundaries: List[float] = field(default_factory=list)
    phrase_boundaries: List[float] = field(default_factory=list)
    beat_duration_sec: float = 0.5
    bar_duration_sec: float = 2.0
    phrase_duration_sec: float = 8.0
    grid_calibrated: bool = False
    calibration_confidence: float = 0.0

    # Genre (optional)
    dj_category: str = "Unknown"
    genre_confidence: float = 0.0

    # Metadata
    analysis_time_sec: float = 0.0
    source: str = "pipeline"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'path': self.path,
            'filename': self.filename,
            'duration_sec': float(self.duration_sec),
            'bpm': float(self.bpm),
            'key': self.key,
            'camelot': self.camelot,
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
            'dj_category': self.dj_category,
            'genre_confidence': float(self.genre_confidence),
            'analysis_time_sec': float(self.analysis_time_sec),
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackCompatibilityResult':
        """Create from dict (e.g., from cache)."""
        return cls(
            path=data.get('path', ''),
            filename=data.get('filename', ''),
            duration_sec=data.get('duration_sec', 0.0),
            bpm=data.get('bpm', 0.0),
            key=data.get('key', ''),
            camelot=data.get('camelot', ''),
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
            dj_category=data.get('dj_category', 'Unknown'),
            genre_confidence=data.get('genre_confidence', 0.0),
            analysis_time_sec=data.get('analysis_time_sec', 0.0),
            source=data.get('source', 'cache'),
        )


# ============== Pipeline Stages ==============

class BpmDetectionStage(PipelineStage):
    """
    Detect BPM using BpmDetectionTask.

    Delegates to Task layer for business logic.
    """

    def __init__(self, target_range: tuple = (118, 145)):
        self.task = BpmDetectionTask(target_range=target_range)

    def process(self, context: PipelineContext) -> PipelineContext:
        audio_ctx = context.audio_context
        if audio_ctx is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(audio_ctx)

        if result.corrected:
            logger.debug(f"BPM corrected to {result.bpm:.1f}")

        context.set_result('bpm', result.bpm)
        context.set_result('bpm_confidence', result.confidence)
        return context


class KeyAnalysisStage(PipelineStage):
    """Analyze musical key using KeyAnalysisTask."""

    def __init__(self, window_sec: float = 30.0, hop_sec: float = 15.0):
        self.task = KeyAnalysisTask(
            window_duration_sec=window_sec,
            hop_duration_sec=hop_sec,
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        audio_ctx = context.audio_context
        if audio_ctx is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(audio_ctx)
        context.set_result('key', result.dominant_key)
        context.set_result('camelot', result.dominant_camelot)
        context.set_result('key_result', result)
        return context


class EnergyAnalysisStage(PipelineStage):
    """Analyze energy arc using EnergyArcAnalysisTask."""

    def __init__(self, intro_sec: float = 30.0, outro_sec: float = 30.0):
        self.task = EnergyArcAnalysisTask(
            smooth_sigma_sec=5.0,
            opening_duration_sec=intro_sec,
            closing_duration_sec=outro_sec,
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        audio_ctx = context.audio_context
        if audio_ctx is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(audio_ctx)
        context.set_result('intro_energy', result.opening_energy)
        context.set_result('outro_energy', result.closing_energy)
        context.set_result('peak_energy', result.peak_energy)
        context.set_result('energy_trajectory', result.trajectory)
        context.set_result('energy_result', result)
        return context


class DropDetectionStage(PipelineStage):
    """Detect drops using DropDetectionTask."""

    def __init__(self, for_tracks: bool = True):
        """
        Args:
            for_tracks: Use track-optimized parameters (vs DJ set parameters)
        """
        if for_tracks:
            self.task = DropDetectionTask(
                min_drop_magnitude=0.15,
                min_confidence=0.3,
            )
        else:
            self.task = DropDetectionTask.for_dj_set()

    def process(self, context: PipelineContext) -> PipelineContext:
        audio_ctx = context.audio_context
        if audio_ctx is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(audio_ctx)
        drop_times = result.get_drop_times() if result.success else []

        context.set_result('drop_times', drop_times)
        context.set_result('drop_count', len(drop_times))
        context.set_result('drop_result', result)
        return context


class BeatGridStage(PipelineStage):
    """
    Compute beat grid using BeatGridTask + GridCalibrationTask.

    Delegates to Task layer for business logic.
    """

    def __init__(self):
        self.beat_grid_task = BeatGridTask(beats_per_bar=4, bars_per_phrase=4)
        self.calibration_task = GridCalibrationTask(
            tolerance_beats=2.0,
            min_events=2,
            min_confidence=0.3,
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        audio_ctx = context.audio_context
        if audio_ctx is None:
            raise ValueError("AudioContext required")

        # Step 1: Compute beat grid (uses BeatGridTask)
        grid_result = self.beat_grid_task.execute(audio_ctx)
        beat_grid = grid_result.beat_grid

        # Step 2: Calibrate using detected drops as anchors (uses GridCalibrationTask)
        drop_times = context.get_result('drop_times', [])
        cal_result = self.calibration_task.execute_with_grid(beat_grid, drop_times)

        if cal_result.calibrated:
            beat_grid = apply_phase_correction(beat_grid, cal_result.phase_offset_sec)

        # Store results
        context.set_result('beat_grid', beat_grid)
        context.set_result('beat_times', [b.time_sec for b in beat_grid.beats])
        context.set_result('bar_boundaries', beat_grid.get_bar_boundaries().tolist())
        context.set_result('phrase_boundaries', beat_grid.get_phrase_boundaries().tolist())
        context.set_result('beat_duration_sec', beat_grid.beat_duration_sec)
        context.set_result('bar_duration_sec', beat_grid.bar_duration_sec)
        context.set_result('phrase_duration_sec', beat_grid.phrase_duration_sec)
        context.set_result('grid_calibrated', cal_result.calibrated)
        context.set_result('calibration_confidence', cal_result.confidence)

        return context


class MixPointDetectionStage(PipelineStage):
    """
    Find optimal mix-in and mix-out points using MixPointDetectionTask.

    Delegates to Task layer for business logic.
    """

    def __init__(self, mix_zone_sec: float = 32.0):
        self.task = MixPointDetectionTask(mix_zone_sec=mix_zone_sec)

    def process(self, context: PipelineContext) -> PipelineContext:
        audio_ctx = context.audio_context
        if audio_ctx is None:
            raise ValueError("AudioContext required")

        duration_sec = audio_ctx.duration_sec
        energy_trajectory = context.get_result('energy_trajectory')
        drop_times = context.get_result('drop_times', [])

        # Get normalized energy curve (100 points)
        energy_curve = get_energy_curve_normalized(
            energy_trajectory,
            audio_ctx.stft_cache
        )

        # Use MixPointDetectionTask
        result = self.task.execute_with_data(duration_sec, energy_curve, drop_times)

        context.set_result('best_mix_in', result.best_mix_in)
        context.set_result('best_mix_out', result.best_mix_out)
        return context


class SpectralAnalysisStage(PipelineStage):
    """Compute spectral characteristics using SpectralAnalysisTask."""

    def __init__(self):
        self.task = SpectralAnalysisTask()

    def process(self, context: PipelineContext) -> PipelineContext:
        audio_ctx = context.audio_context
        if audio_ctx is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(audio_ctx)
        context.set_result('spectral_centroid_mean', result.centroid_mean)
        return context


class GenreAnalysisStage(PipelineStage):
    """Optional genre classification using ML."""

    def __init__(self):
        self.task = None
        try:
            self.task = GenreAnalysisTask()
        except Exception:
            logger.debug("GenreAnalysisTask not available")

    def should_skip(self, context: PipelineContext) -> bool:
        return self.task is None

    def process(self, context: PipelineContext) -> PipelineContext:
        if self.task is None:
            return context

        audio_ctx = context.audio_context
        if audio_ctx is None:
            return context

        try:
            result = self.task.execute(audio_ctx)
            if result.success:
                context.set_result('dj_category', result.dj_category)
                context.set_result('genre_confidence', result.confidence)
        except Exception as e:
            logger.debug(f"Genre analysis failed: {e}")

        return context


# ============== Main Pipeline ==============

class TrackCompatibilityPipeline(Pipeline):
    """
    Pipeline for track compatibility analysis.

    Extracts all features needed for DJ set generation:
    - BPM, Key, Energy, Drops, Beat grid, Mix points, Genre

    Follows architecture: Primitives → Tasks → Pipelines

    Usage:
        pipeline = TrackCompatibilityPipeline()
        result = pipeline.analyze("/path/to/track.mp3")
        print(f"BPM: {result.bpm}, Key: {result.camelot}")
    """

    def __init__(
        self,
        cache: Optional[CacheRepository] = None,
        include_genre: bool = True,
        sr: int = 22050,
        intro_sec: float = 30.0,
        outro_sec: float = 30.0,
        mix_zone_sec: float = 32.0,
        bpm_range: tuple = (118, 145),
    ):
        """
        Initialize track compatibility pipeline.

        Args:
            cache: CacheRepository for caching results
            include_genre: Whether to run ML genre classification
            sr: Sample rate for loading
            intro_sec: Duration to analyze for intro energy
            outro_sec: Duration to analyze for outro energy
            mix_zone_sec: Duration of mix zone for mix point detection
            bpm_range: Expected BPM range for tempo correction
        """
        stages = [
            LoadAudioStage(sr=sr),
            ComputeSTFTStage(),
            BpmDetectionStage(target_range=bpm_range),
            KeyAnalysisStage(),
            EnergyAnalysisStage(intro_sec=intro_sec, outro_sec=outro_sec),
            DropDetectionStage(for_tracks=True),
            BeatGridStage(),
            MixPointDetectionStage(mix_zone_sec=mix_zone_sec),
            SpectralAnalysisStage(),
        ]

        if include_genre:
            stages.append(GenreAnalysisStage())

        super().__init__(stages, name="TrackCompatibility")

        self.cache = cache or CacheRepository()
        self.include_genre = include_genre

    def analyze(
        self,
        path: str,
        use_cache: bool = True,
        stage_callback: Optional[callable] = None,
    ) -> TrackCompatibilityResult:
        """
        Analyze a single track.

        Args:
            path: Path to audio file
            use_cache: Whether to use cached results
            stage_callback: Optional callback(stage_num, total_stages, stage_name)
                           Called after each stage completes

        Returns:
            TrackCompatibilityResult with all analysis data
        """
        start_time = time.time()

        # Check cache
        if use_cache:
            cached = self.cache.get_track_analysis(path)
            if cached:
                logger.debug(f"Using cached analysis for {Path(path).name}")
                result = TrackCompatibilityResult.from_dict(cached)
                result.source = "cache"
                return result

        # Run pipeline
        try:
            context = PipelineContext(input_path=path)

            # Set up stage callback if provided
            if stage_callback:
                total_stages = len(self.stages)

                def on_stage_complete(stage_name: str, ctx: PipelineContext):
                    # Find stage index
                    stage_idx = next(
                        (i for i, s in enumerate(self.stages) if s.name == stage_name),
                        0
                    )
                    stage_callback(stage_idx + 1, total_stages, stage_name)

                self.on_stage_complete = on_stage_complete

            context = self.run(context)

            # Reset callback
            self.on_stage_complete = None

            # Build result from context
            result = TrackCompatibilityResult(
                path=path,
                filename=Path(path).name,
                duration_sec=context.results.get('_duration', 0.0),
                bpm=context.get_result('bpm', 0.0),
                key=context.get_result('key', ''),
                camelot=context.get_result('camelot', ''),
                intro_energy=context.get_result('intro_energy', 0.5),
                outro_energy=context.get_result('outro_energy', 0.5),
                peak_energy=context.get_result('peak_energy', 0.5),
                drop_times=context.get_result('drop_times', []),
                drop_count=context.get_result('drop_count', 0),
                best_mix_in=context.get_result('best_mix_in', 0.0),
                best_mix_out=context.get_result('best_mix_out', 0.0),
                spectral_centroid_mean=context.get_result('spectral_centroid_mean', 2000.0),
                beat_times=context.get_result('beat_times', []),
                bar_boundaries=context.get_result('bar_boundaries', []),
                phrase_boundaries=context.get_result('phrase_boundaries', []),
                beat_duration_sec=context.get_result('beat_duration_sec', 0.5),
                bar_duration_sec=context.get_result('bar_duration_sec', 2.0),
                phrase_duration_sec=context.get_result('phrase_duration_sec', 8.0),
                grid_calibrated=context.get_result('grid_calibrated', False),
                calibration_confidence=context.get_result('calibration_confidence', 0.0),
                dj_category=context.get_result('dj_category', 'Unknown'),
                genre_confidence=context.get_result('genre_confidence', 0.0),
                analysis_time_sec=time.time() - start_time,
                source="pipeline",
            )

            # Cache result
            if use_cache:
                self.cache.save_track_analysis(path, result.to_dict())

            return result

        except Exception as e:
            logger.error(f"Track analysis failed for {path}: {e}")
            return TrackCompatibilityResult(
                path=path,
                filename=Path(path).name,
                source="error",
            )

    def analyze_batch(
        self,
        paths: List[str],
        use_cache: bool = True,
        progress_callback: Optional[callable] = None,
        stage_callback: Optional[callable] = None,
    ) -> List[TrackCompatibilityResult]:
        """
        Analyze multiple tracks sequentially.

        Args:
            paths: List of audio file paths
            use_cache: Whether to use cached results
            progress_callback: Optional callback(current, total, path, status)
                              status: "analyzing", "cached", "done", "error"
            stage_callback: Optional callback(stage_num, total_stages, stage_name)
                           Called after each stage within track analysis

        Returns:
            List of TrackCompatibilityResult
        """
        results = []
        total = len(paths)

        for i, path in enumerate(paths):
            filename = Path(path).name

            if progress_callback:
                progress_callback(i + 1, total, filename, "analyzing")

            result = self.analyze(path, use_cache=use_cache, stage_callback=stage_callback)
            results.append(result)

            if progress_callback:
                status = "cached" if result.source == "cache" else (
                    "error" if result.source == "error" else "done"
                )
                progress_callback(i + 1, total, filename, status)

        return results
