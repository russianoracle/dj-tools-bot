"""
DJ Profiling Pipelines - Extract DJ style characteristics from sets.

Focuses on metrics that are ROBUST to imperfect transition detection:
- Energy arc (opening, peak, closing, shape)
- Drop patterns (frequency, intensity, buildups)
- Tempo distribution
- Genre diversity

These metrics work without precise track boundaries.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from .base import PipelineStage, PipelineContext
from app.modules.analysis.tasks import (
    DropDetectionTask,
    DropDetectorML,
    EnergyArcAnalysisTask,
    EnergyArcAnalysisResult,
    TempoDistributionAnalysisTask,
    TempoDistributionAnalysisResult,
    KeyAnalysisTask,
    KeyAnalysisResult,
    BeatGridTask,
    BeatGridAnalysisResult,
)

logger = logging.getLogger(__name__)


# ============== Metrics Dataclasses (for Pipeline results) ==============

# Type aliases for Task results (for clarity in Pipeline layer)
EnergyArcMetrics = EnergyArcAnalysisResult
TempoDistributionMetrics = TempoDistributionAnalysisResult


@dataclass
class DropPatternMetrics:
    """
    Drop pattern profiling metrics.

    Analyzes how DJ uses drops/buildups:
    - drops_per_hour: Frequency of energy drops
    - drop_magnitudes: Distribution of drop intensities
    - buildup_count: How many buildups before drops
    - drop_clustering: Style classification based on drop patterns
    """
    drops_per_hour: float
    drop_magnitudes: List[float]         # All drop magnitudes
    avg_drop_magnitude: float
    max_drop_magnitude: float
    buildup_count: int                   # Rising energy before drops
    drop_clustering: str                 # "technical", "festival", "minimal"
    drop_times: List[float]              # Timestamps of drops (for visualization)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'drops_per_hour': float(self.drops_per_hour),
            'drop_magnitudes': [float(x) for x in self.drop_magnitudes],
            'avg_drop_magnitude': float(self.avg_drop_magnitude),
            'max_drop_magnitude': float(self.max_drop_magnitude),
            'buildup_count': int(self.buildup_count),
            'drop_clustering': self.drop_clustering,
            'drop_times': [float(x) for x in self.drop_times],
        }


@dataclass
class TempoDistributionMetrics:
    """
    Tempo distribution profiling metrics.

    Analyzes tempo range and progression:
    - tempo_mean: Average BPM across set
    - tempo_std: Tempo variance (tight vs eclectic)
    - tempo_range: Min to max BPM
    - tempo_histogram: Distribution of BPM values
    """
    tempo_mean: float
    tempo_std: float
    tempo_min: float
    tempo_max: float
    tempo_range: float                   # max - min
    tempo_histogram: Dict[int, int]      # {BPM: count}
    dominant_tempo: int                  # Most frequent BPM

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'tempo_mean': float(self.tempo_mean),
            'tempo_std': float(self.tempo_std),
            'tempo_min': float(self.tempo_min),
            'tempo_max': float(self.tempo_max),
            'tempo_range': float(self.tempo_range),
            'tempo_histogram': {int(k): int(v) for k, v in self.tempo_histogram.items()},
            'dominant_tempo': int(self.dominant_tempo),
        }


@dataclass
class DJProfileMetrics:
    """
    Complete DJ profile combining all metrics.
    """
    energy_arc: Optional[EnergyArcMetrics] = None
    drop_pattern: Optional[DropPatternMetrics] = None
    tempo_distribution: Optional[TempoDistributionMetrics] = None
    key_analysis: Optional[KeyAnalysisResult] = None
    genre: Optional[Any] = None  # GenreAnalysisResult

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {}
        if self.energy_arc:
            result['energy_arc'] = self.energy_arc.to_dict()
        if self.drop_pattern:
            result['drop_pattern'] = self.drop_pattern.to_dict()
        if self.tempo_distribution:
            result['tempo_distribution'] = self.tempo_distribution.to_dict()
        if self.key_analysis:
            result['key_analysis'] = self.key_analysis.to_dict()
        if self.genre:
            result['genre'] = self.genre.to_dict() if hasattr(self.genre, 'to_dict') else self.genre
        return result


# ============== Energy Arc Pipeline Stage ==============

class EnergyArcProfilingStage(PipelineStage):
    """
    Extract energy arc metrics from a DJ set.

    Delegates to EnergyArcAnalysisTask (Tasks layer).
    Stage only orchestrates Task execution and stores results.
    """

    def __init__(
        self,
        smooth_sigma_sec: float = 30.0,  # Gaussian smoothing (seconds)
        opening_duration_sec: float = 300.0,  # First 5 minutes
        closing_duration_sec: float = 300.0,  # Last 5 minutes
    ):
        """
        Initialize energy arc profiling stage.

        Args:
            smooth_sigma_sec: Gaussian smoothing sigma (seconds)
            opening_duration_sec: Duration to consider as "opening"
            closing_duration_sec: Duration to consider as "closing"
        """
        # Create Task with configuration
        self.task = EnergyArcAnalysisTask(
            smooth_sigma_sec=smooth_sigma_sec,
            opening_duration_sec=opening_duration_sec,
            closing_duration_sec=closing_duration_sec,
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        """Execute energy arc analysis task."""
        if context.audio_context is None:
            raise ValueError("AudioContext not created. Run ComputeSTFTStage first.")

        logger.info("→ Analyzing energy arc...")

        # Execute Task
        result = self.task.execute(context.audio_context)

        # Store result in context
        context.set_result('energy_arc_metrics', result)

        logger.info(
            f"  Energy Arc: {result.arc_shape} | "
            f"Opening={result.opening_energy:.2f}, Peak={result.peak_energy:.2f}, "
            f"Closing={result.closing_energy:.2f}, Variance={result.energy_variance:.3f}"
        )

        return context


# ============== Drop Pattern Pipeline Stage ==============

class DropPatternProfilingStage(PipelineStage):
    """
    Extract drop pattern metrics from a DJ set.

    Delegates to DropDetectionTask (Tasks layer).
    Stage only orchestrates Task execution and aggregates results.
    """

    def __init__(
        self,
        min_drop_magnitude: float = 0.05,
        min_confidence: float = 0.10,
        use_multiband: bool = True,
        smooth_sigma: float = 1.5,
    ):
        """
        Initialize drop pattern profiling stage.

        Args:
            min_drop_magnitude: Minimum drop magnitude to consider (default: 0.05)
            min_confidence: Minimum drop confidence threshold (default: 0.10)
            use_multiband: Use mel-band weighted energy (recommended)
            smooth_sigma: Gaussian smoothing sigma (lower = more sensitive, default: 1.5)
        """
        # Create Task with configuration
        self.task = DropDetectionTask(
            min_drop_magnitude=min_drop_magnitude,
            min_confidence=min_confidence,
            use_multiband=use_multiband,
            smooth_sigma=smooth_sigma,
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        """Execute drop detection task and aggregate metrics."""
        if context.audio_context is None:
            raise ValueError("AudioContext not created. Run ComputeSTFTStage first.")

        logger.info("→ Analyzing drop patterns...")

        audio_ctx = context.audio_context

        # Execute Task
        drop_result = self.task.execute(audio_ctx)

        # Calculate duration for drops per hour
        duration_sec = len(audio_ctx.y) / audio_ctx.sr
        duration_hours = duration_sec / 3600.0

        # Aggregate metrics from Task result
        if len(drop_result.drops) == 0:
            # No drops detected
            metrics = DropPatternMetrics(
                drops_per_hour=0.0,
                drop_magnitudes=[],
                avg_drop_magnitude=0.0,
                max_drop_magnitude=0.0,
                buildup_count=0,
                drop_clustering="minimal",
                drop_times=[],
            )
        else:
            # Vectorized extraction of drop data
            drops_arr = drop_result.drops
            drop_magnitudes = np.array([d.drop_magnitude for d in drops_arr], dtype=np.float32)
            drop_times = np.array([d.time_sec for d in drops_arr], dtype=np.float32)

            drops_per_hour = len(drops_arr) / duration_hours
            avg_drop_magnitude = drop_result.avg_drop_intensity
            max_drop_magnitude = drop_result.max_drop_magnitude

            # Use buildups from Task result (already computed!)
            buildup_count = len(drop_result.buildups)

            # Simple classification based on aggregated metrics
            drop_clustering = self._classify_drop_pattern(
                drops_per_hour, avg_drop_magnitude
            )

            metrics = DropPatternMetrics(
                drops_per_hour=drops_per_hour,
                drop_magnitudes=drop_magnitudes.tolist(),  # Convert numpy to list
                avg_drop_magnitude=avg_drop_magnitude,
                max_drop_magnitude=max_drop_magnitude,
                buildup_count=buildup_count,
                drop_clustering=drop_clustering,
                drop_times=drop_times.tolist(),  # Convert numpy to list
            )

        # Store in context
        context.set_result('drop_pattern_metrics', metrics)

        logger.info(
            f"  Drop Pattern: {metrics.drop_clustering} | "
            f"{len(drop_result.drops)} drops ({metrics.drops_per_hour:.1f}/hour), "
            f"Avg magnitude={metrics.avg_drop_magnitude:.2f}, "
            f"{metrics.buildup_count} buildups"
        )

        return context

    @staticmethod
    def _classify_drop_pattern(drops_per_hour: float, avg_magnitude: float) -> str:
        """
        Simple classification based on frequency and magnitude.

        This is just aggregation logic, not business logic.
        """
        if drops_per_hour < 2:
            return "minimal"
        elif drops_per_hour > 8 and avg_magnitude < 0.25:
            return "technical"
        elif avg_magnitude > 0.3:
            return "festival"
        else:
            return "technical"


class DropPatternProfilingMLStage(PipelineStage):
    """
    Extract drop pattern metrics using XGBoost ML model (DropDetectorML).

    Requires beat grid for phrase boundaries.
    Falls back to rule-based if model not available.
    """

    def __init__(
        self,
        model_path: str = 'models/drop_detector_xgb.pkl',
        detection_threshold: float = 0.5,
        use_rule_fallback: bool = True,
    ):
        """
        Args:
            model_path: Path to trained XGBoost model
            detection_threshold: Minimum probability to classify as drop
            use_rule_fallback: Fall back to rule-based if model unavailable
        """
        self.detector = DropDetectorML(
            model_path=model_path,
            detection_threshold=detection_threshold,
            use_rule_fallback=use_rule_fallback,
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        """Execute ML drop detection and aggregate metrics."""
        if context.audio_context is None:
            raise ValueError("AudioContext not created. Run ComputeSTFTStage first.")

        logger.info("→ Analyzing drop patterns (ML)...")

        audio_ctx = context.audio_context

        # Get phrase boundaries from beat grid
        beat_grid = context.get_result('beat_grid')
        if beat_grid is None:
            logger.warning("DropPatternProfilingMLStage: No beat_grid, using fallback")
            # Create basic phrase boundaries
            duration_sec = len(audio_ctx.y) / audio_ctx.sr
            phrase_duration = 7.5  # ~16 beats at 128 BPM
            phrase_boundaries = np.arange(phrase_duration, duration_sec, phrase_duration)
        else:
            phrase_boundaries = beat_grid.get_phrase_boundaries()

        if len(phrase_boundaries) == 0:
            logger.warning("No phrase boundaries for ML drop detection")
            metrics = DropPatternMetrics(
                drops_per_hour=0.0,
                drop_magnitudes=[],
                avg_drop_magnitude=0.0,
                max_drop_magnitude=0.0,
                buildup_count=0,
                drop_clustering="minimal",
                drop_times=[],
            )
            context.set_result('drop_pattern_metrics', metrics)
            return context

        # Execute ML detection
        drop_result = self.detector.execute(audio_ctx, phrase_boundaries)

        # Calculate duration for drops per hour
        duration_sec = len(audio_ctx.y) / audio_ctx.sr
        duration_hours = duration_sec / 3600.0

        if len(drop_result.drops) == 0:
            metrics = DropPatternMetrics(
                drops_per_hour=0.0,
                drop_magnitudes=[],
                avg_drop_magnitude=0.0,
                max_drop_magnitude=0.0,
                buildup_count=0,
                drop_clustering="minimal",
                drop_times=[],
            )
        else:
            # Vectorized extraction of drop data from ML result
            # Note: ML drops have different fields than rule-based
            drop_times = np.array([d.time_sec for d in drop_result.drops], dtype=np.float32)
            confidences = np.array([d.confidence for d in drop_result.drops], dtype=np.float32)

            drops_per_hour = len(drop_result.drops) / duration_hours
            avg_confidence = float(np.mean(confidences))
            max_confidence = float(np.max(confidences))

            # Classify based on frequency
            drop_clustering = self._classify_drop_pattern(drops_per_hour, avg_confidence)

            metrics = DropPatternMetrics(
                drops_per_hour=drops_per_hour,
                drop_magnitudes=confidences.tolist(),  # Use confidence as magnitude proxy
                avg_drop_magnitude=avg_confidence,
                max_drop_magnitude=max_confidence,
                buildup_count=0,  # ML doesn't detect buildups separately
                drop_clustering=drop_clustering,
                drop_times=drop_times.tolist(),
            )

        context.set_result('drop_pattern_metrics', metrics)

        logger.info(
            f"  Drop Pattern (ML): {metrics.drop_clustering} | "
            f"{len(drop_result.drops)} drops ({metrics.drops_per_hour:.1f}/hour)"
        )

        return context

    @staticmethod
    def _classify_drop_pattern(drops_per_hour: float, avg_confidence: float) -> str:
        """Classify drop pattern based on frequency."""
        if drops_per_hour < 2:
            return "minimal"
        elif drops_per_hour > 8:
            return "technical"
        elif avg_confidence > 0.7:
            return "festival"
        else:
            return "technical"


# ============== Tempo Distribution Pipeline Stage ==============

class TempoDistributionProfilingStage(PipelineStage):
    """
    Extract tempo distribution metrics from a DJ set.

    Delegates to TempoDistributionAnalysisTask (Tasks layer).
    Stage only orchestrates Task execution and stores results.
    """

    def __init__(
        self,
        window_duration_sec: float = 60.0,  # 1 minute windows
        hop_duration_sec: float = 30.0,     # 50% overlap
    ):
        """
        Initialize tempo distribution profiling stage.

        Args:
            window_duration_sec: Window size for tempo estimation
            hop_duration_sec: Hop between windows
        """
        # Create Task with configuration
        self.task = TempoDistributionAnalysisTask(
            window_duration_sec=window_duration_sec,
            hop_duration_sec=hop_duration_sec,
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        """Execute tempo distribution analysis task."""
        if context.audio_context is None:
            raise ValueError("AudioContext not created. Run ComputeSTFTStage first.")

        logger.info("→ Analyzing tempo distribution...")

        # Execute Task
        result = self.task.execute(context.audio_context)

        # Store result in context
        context.set_result('tempo_distribution_metrics', result)

        logger.info(
            f"  Tempo Distribution: Mean={result.tempo_mean:.1f} BPM, "
            f"Range={result.tempo_min:.0f}-{result.tempo_max:.0f}, "
            f"Dominant={result.dominant_tempo} BPM, "
            f"Variance={result.tempo_std:.1f}"
        )

        return context


# ============== Key Analysis Pipeline Stage ==============

class KeyAnalysisProfilingStage(PipelineStage):
    """
    Extract key distribution metrics from a DJ set.

    Delegates to KeyAnalysisTask (Tasks layer).
    Stage only orchestrates Task execution and stores results.
    """

    def __init__(
        self,
        window_duration_sec: float = 60.0,  # 1 minute windows
        hop_duration_sec: float = 30.0,     # 50% overlap
    ):
        """
        Initialize key analysis profiling stage.

        Args:
            window_duration_sec: Window size for key estimation
            hop_duration_sec: Hop between windows
        """
        # Create Task with configuration
        self.task = KeyAnalysisTask(
            window_duration_sec=window_duration_sec,
            hop_duration_sec=hop_duration_sec,
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        """Execute key analysis task."""
        if context.audio_context is None:
            raise ValueError("AudioContext not created. Run ComputeSTFTStage first.")

        logger.info("→ Analyzing key distribution...")

        # Execute Task
        result = self.task.execute(context.audio_context)

        # Store result in context
        context.set_result('key_analysis_metrics', result)

        logger.info(
            f"  Key Analysis: Dominant={result.dominant_camelot} ({result.dominant_key}), "
            f"Changes={result.key_changes}, "
            f"Stability={result.key_stability:.2f}"
        )

        return context


# ============== Genre Profiling Stage ==============

class GenreProfilingStage(PipelineStage):
    """
    Genre classification stage using Essentia ML model.

    Optional - requires essentia-tensorflow installed.
    """

    def __init__(self):
        try:
            from app.modules.analysis.tasks.genre_analysis import GenreAnalysisTask
            self.task = GenreAnalysisTask(top_n=5, min_confidence=0.1)
            self.available = True
        except ImportError:
            self.task = None
            self.available = False

    def process(self, context: PipelineContext) -> PipelineContext:
        """Execute genre analysis if available."""
        if not self.available or context.audio_context is None:
            return context

        try:
            logger.info("→ Analyzing genre (ML model)...")
            result = self.task.execute(context.audio_context)
            context.set_result('genre_metrics', result)

            if result.success:
                logger.info(f"  Genre: {result.dj_category} - {result.genre} (confidence: {result.confidence:.1%})")
        except Exception as e:
            logger.debug(f"Genre analysis failed: {e}")

        return context


# ============== Beat Grid Pipeline Stage ==============

class BeatGridProfilingStage(PipelineStage):
    """
    Beat grid analysis stage.

    Computes musical structure (beats → bars → phrases) for accurate
    event alignment. This should run FIRST as other stages can use
    the beat grid for alignment.
    """

    def __init__(
        self,
        beats_per_bar: int = 4,
        bars_per_phrase: int = 4,
    ):
        """
        Initialize beat grid profiling stage.

        Args:
            beats_per_bar: Beats per bar (4 for 4/4)
            bars_per_phrase: Bars per phrase (4 = 16 beats standard)
        """
        self.task = BeatGridTask(
            beats_per_bar=beats_per_bar,
            bars_per_phrase=bars_per_phrase,
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        """Execute beat grid analysis task."""
        if context.audio_context is None:
            raise ValueError("AudioContext not created. Run ComputeSTFTStage first.")

        logger.info("→ Computing beat grid...")

        # Execute Task
        result = self.task.execute(context.audio_context)

        # Store result in context
        context.set_result('beat_grid_metrics', result)
        # Also store the raw BeatGridResult for other stages to use
        context.set_result('beat_grid', result.beat_grid)

        # IMPORTANT: Also set beat_grid in AudioContext for Tasks to use
        # This allows DropDetectionTask and other tasks to access beat_grid
        context.audio_context.beat_grid = result.beat_grid

        logger.info(
            f"  Beat Grid: {result.tempo:.1f} BPM | "
            f"{result.n_beats} beats, {result.n_bars} bars, {result.n_phrases} phrases | "
            f"regularity={result.beat_regularity:.2f}"
        )

        return context


# ============== Combined DJ Profiling Pipeline ==============

class DJProfilingStage(PipelineStage):
    """
    Combined DJ profiling stage.

    Runs all profiling stages and aggregates results into
    a single DJProfileMetrics object.
    """

    def __init__(
        self,
        include_energy_arc: bool = True,
        include_drop_pattern: bool = True,
        include_tempo_distribution: bool = True,
        include_key_analysis: bool = True,
        include_genre: bool = False,  # Optional - requires essentia
        include_beat_grid: bool = True,  # Beat grid for musical alignment
    ):
        """
        Initialize DJ profiling stage.

        Args:
            include_energy_arc: Include energy arc profiling
            include_drop_pattern: Include drop pattern profiling
            include_tempo_distribution: Include tempo distribution
            include_key_analysis: Include key analysis (Camelot wheel)
            include_genre: Include ML-based genre classification (requires essentia-tensorflow)
            include_beat_grid: Include beat grid analysis (recommended - enables accurate event alignment)
        """
        self.include_energy_arc = include_energy_arc
        self.include_drop_pattern = include_drop_pattern
        self.include_tempo_distribution = include_tempo_distribution
        self.include_key_analysis = include_key_analysis
        self.include_genre = include_genre
        self.include_beat_grid = include_beat_grid

        # Create sub-stages
        # Beat grid should run FIRST so other stages can use it
        self.beat_grid_stage = BeatGridProfilingStage() if include_beat_grid else None
        self.energy_arc_stage = EnergyArcProfilingStage() if include_energy_arc else None
        self.drop_pattern_stage = DropPatternProfilingStage() if include_drop_pattern else None
        self.tempo_distribution_stage = TempoDistributionProfilingStage() if include_tempo_distribution else None
        self.key_analysis_stage = KeyAnalysisProfilingStage() if include_key_analysis else None
        self.genre_stage = GenreProfilingStage() if include_genre else None

    def process(self, context: PipelineContext) -> PipelineContext:
        """Run all enabled profiling stages."""

        # Run beat grid FIRST (other stages can use it for alignment)
        if self.beat_grid_stage:
            context = self.beat_grid_stage.process(context)

        # Run energy arc profiling
        if self.energy_arc_stage:
            context = self.energy_arc_stage.process(context)

        # Run drop pattern profiling
        if self.drop_pattern_stage:
            context = self.drop_pattern_stage.process(context)

        # Run tempo distribution profiling
        if self.tempo_distribution_stage:
            context = self.tempo_distribution_stage.process(context)

        # Run key analysis
        if self.key_analysis_stage:
            context = self.key_analysis_stage.process(context)

        # Run genre analysis
        if self.genre_stage:
            context = self.genre_stage.process(context)

        # Aggregate results
        profile = DJProfileMetrics(
            energy_arc=context.get_result('energy_arc_metrics'),
            drop_pattern=context.get_result('drop_pattern_metrics'),
            tempo_distribution=context.get_result('tempo_distribution_metrics'),
            key_analysis=context.get_result('key_analysis_metrics'),
            genre=context.get_result('genre_metrics'),
        )

        context.set_result('dj_profile', profile)

        return context
