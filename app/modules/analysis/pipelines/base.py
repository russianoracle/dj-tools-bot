"""
Base classes for Pipelines layer.

Pipeline = sequence of PipelineStages that transform PipelineContext.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import time

from app.common.logging import get_logger
from app.common.primitives import STFTCache, compute_stft
from app.modules.analysis.tasks import AudioContext, create_audio_context
from app.core.adapters.loader import AudioLoader
from app.core.errors import (
    AudioLoadError,
    STFTError,
    TaskExecutionError,
    AnalysisError,
)

logger = get_logger(__name__)


@dataclass
class PipelineContext:
    """
    Context shared across pipeline stages.

    Contains:
    - Input/output paths
    - Configuration
    - Accumulated results from stages

    Attributes:
        input_path: Source audio file path
        output_dir: Output directory for results
        cache_dir: Cache directory
        config: Pipeline configuration
        results: Accumulated results from stages
        audio_context: AudioContext for task execution (set by LoadAudio/ComputeSTFT)
    """
    input_path: str
    output_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    audio_context: Optional[AudioContext] = None

    # Computed properties
    @property
    def file_name(self) -> str:
        return Path(self.input_path).name

    @property
    def file_stem(self) -> str:
        return Path(self.input_path).stem

    @property
    def is_loaded(self) -> bool:
        return self.audio_context is not None

    def get_result(self, key: str, default: Any = None) -> Any:
        """Get a result by key."""
        return self.results.get(key, default)

    def set_result(self, key: str, value: Any):
        """Set a result."""
        self.results[key] = value


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage transforms a PipelineContext and returns
    the (possibly modified) context.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return self.__class__.__name__

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process the context.

        Args:
            context: Current pipeline context

        Returns:
            Modified context
        """
        pass

    def should_skip(self, context: PipelineContext) -> bool:
        """
        Check if this stage should be skipped.

        Override in subclasses for conditional execution.
        """
        return False

    def __repr__(self) -> str:
        return f"{self.name}()"


class Pipeline:
    """
    Base pipeline with stage composition.

    Pipelines execute a sequence of stages in order,
    passing context between them.

    Usage:
        pipeline = Pipeline([
            LoadAudioStage(),
            ComputeSTFTStage(),
            MyCustomStage(),
        ])
        context = pipeline.run(PipelineContext(input_path="track.mp3"))
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        name: Optional[str] = None,
        on_stage_complete: Optional[Callable[[str, PipelineContext], None]] = None
    ):
        """
        Initialize pipeline.

        Args:
            stages: List of stages to execute
            name: Pipeline name (auto-generated if None)
            on_stage_complete: Callback after each stage
        """
        self.stages = stages
        self.name = name or self.__class__.__name__
        self.on_stage_complete = on_stage_complete

    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Run the pipeline.

        Args:
            context: Initial context

        Returns:
            Final context with all results
        """
        import psutil
        from app.common.monitoring import get_metrics_collector

        context.results['_pipeline_name'] = self.name
        context.results['_start_time'] = time.time()

        logger.info(f"[{self.name}] Starting pipeline for {context.file_name}")

        # Get metrics collector
        metrics = get_metrics_collector()
        process = psutil.Process()

        for i, stage in enumerate(self.stages):
            if stage.should_skip(context):
                logger.debug(f"[{self.name}] Skipping {stage.name}")
                continue

            logger.info(f"[{self.name}] Stage {i+1}/{len(self.stages)}: {stage.name}")

            # Track memory before stage
            mem_before_mb = process.memory_info().rss / 1024 / 1024

            stage_start = time.time()
            context = stage.process(context)
            elapsed = time.time() - stage_start

            # Track memory after stage
            mem_after_mb = process.memory_info().rss / 1024 / 1024
            mem_delta_mb = mem_after_mb - mem_before_mb

            context.results[f'_stage_{stage.name}_time'] = elapsed
            context.results[f'_stage_{stage.name}_memory_mb'] = mem_after_mb
            context.results[f'_stage_{stage.name}_memory_delta_mb'] = mem_delta_mb

            logger.info(f"[{self.name}] {stage.name} done in {elapsed:.1f}s", data={
                "stage": stage.name,
                "duration_sec": round(elapsed, 2),
                "memory_before_mb": round(mem_before_mb, 1),
                "memory_after_mb": round(mem_after_mb, 1),
                "memory_delta_mb": round(mem_delta_mb, 1),
            })

            # Send stage metrics to YC Monitoring
            stage_context = {
                "file_duration_sec": context.results.get('_duration', 0),
                "peak_memory_mb": mem_after_mb,
            }
            metrics.record_stage_metrics(
                stage_name=stage.name,
                duration_sec=elapsed,
                memory_mb=mem_after_mb,
                context=stage_context
            )

            if self.on_stage_complete:
                self.on_stage_complete(stage.name, context)

        total = time.time() - context.results['_start_time']
        context.results['_total_time'] = total
        logger.info(f"[{self.name}] Pipeline complete in {total:.1f}s")
        return context

    def run_from_path(self, path: str, **config) -> PipelineContext:
        """
        Convenience method to run pipeline from file path.

        Args:
            path: Input file path
            **config: Additional configuration

        Returns:
            Final context
        """
        context = PipelineContext(input_path=path, config=config)
        return self.run(context)

    def __repr__(self) -> str:
        stage_names = [s.name for s in self.stages]
        return f"{self.name}({' -> '.join(stage_names)})"


# ============== Common Pipeline Stages ==============

class LoadAudioStage(PipelineStage):
    """
    Load audio from file.

    Loads audio file and creates initial audio data in context.
    Does NOT compute STFT (that's ComputeSTFTStage's job).
    """

    def __init__(
        self,
        sr: int = 22050,
        mono: bool = True,
        duration: Optional[float] = None
    ):
        """
        Initialize load audio stage.

        Args:
            sr: Target sample rate
            mono: Convert to mono
            duration: Load only first N seconds (None = full track)
        """
        self.sr = sr
        self.mono = mono
        self.duration = duration

    def process(self, context: PipelineContext) -> PipelineContext:
        """Load audio file."""
        path = context.input_path

        # Load audio using AudioLoader (delegates to librosa internally)
        loader = AudioLoader(sample_rate=self.sr)
        y, sr = loader.load(path, duration=self.duration)

        # Store in context
        context.results['_audio'] = y
        context.results['_sr'] = sr
        context.results['_duration'] = len(y) / sr

        return context


class ComputeSTFTStage(PipelineStage):
    """
    Compute STFT and create AudioContext.

    This stage must come after LoadAudioStage.
    Creates the AudioContext that tasks will use.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize STFT computation stage.

        Args:
            n_fft: FFT size
            hop_length: Hop length
        """
        self.n_fft = n_fft
        self.hop_length = hop_length

    def process(self, context: PipelineContext) -> PipelineContext:
        """Compute STFT and create AudioContext."""
        import psutil

        y = context.results.get('_audio')
        sr = context.results.get('_sr')

        if y is None or sr is None:
            raise STFTError("Audio not loaded. Run LoadAudioStage first.", data={"file": context.input_path})

        # Track memory usage
        process = psutil.Process()
        mem_before_mb = process.memory_info().rss / 1024 / 1024

        # Create AudioContext with STFT
        audio_ctx = create_audio_context(
            y=y,
            sr=sr,
            file_path=context.input_path,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Track peak memory
        mem_after_mb = process.memory_info().rss / 1024 / 1024
        peak_memory_mb = max(mem_before_mb, mem_after_mb)

        # Store metrics in context
        context.set_result('_peak_memory_mb', peak_memory_mb)
        context.set_result('_stft_memory_delta_mb', mem_after_mb - mem_before_mb)

        logger.info("STFT computed", data={
            "memory_before_mb": round(mem_before_mb, 1),
            "memory_after_mb": round(mem_after_mb, 1),
            "memory_delta_mb": round(mem_after_mb - mem_before_mb, 1),
            "peak_memory_mb": round(peak_memory_mb, 1),
        })

        context.audio_context = audio_ctx

        return context


class TaskStage(PipelineStage):
    """
    Generic stage that runs a Task.

    Wraps any BaseTask for use in a pipeline.
    """

    def __init__(self, task, result_key: str):
        """
        Initialize task stage.

        Args:
            task: Task instance to run
            result_key: Key to store result in context.results
        """
        self.task = task
        self.result_key = result_key

    @property
    def name(self) -> str:
        return f"Task_{self.task.name}"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Run the task."""
        if context.audio_context is None:
            raise TaskExecutionError("AudioContext not created. Run ComputeSTFTStage first.")

        try:
            result = self.task.execute(context.audio_context)
            context.set_result(self.result_key, result)
        except Exception as e:
            raise TaskExecutionError(
                f"Task {self.task.__class__.__name__} failed",
                data={"task": self.task.__class__.__name__, "result_key": self.result_key},
                cause=e,
            )

        return context


class ConditionalStage(PipelineStage):
    """
    Stage that conditionally executes another stage.
    """

    def __init__(
        self,
        stage: PipelineStage,
        condition: Callable[[PipelineContext], bool]
    ):
        """
        Initialize conditional stage.

        Args:
            stage: Stage to execute if condition is true
            condition: Function that takes context and returns bool
        """
        self.stage = stage
        self.condition = condition

    @property
    def name(self) -> str:
        return f"Conditional_{self.stage.name}"

    def process(self, context: PipelineContext) -> PipelineContext:
        if self.condition(context):
            return self.stage.process(context)
        return context


class ParallelStage(PipelineStage):
    """
    Stage that runs multiple stages in parallel.

    Note: Due to GIL, this is only beneficial for I/O-bound stages.
    For CPU-bound work, use M2BatchProcessor instead.
    """

    def __init__(self, stages: List[PipelineStage]):
        """
        Initialize parallel stage.

        Args:
            stages: Stages to run in parallel
        """
        self.stages_to_run = stages

    @property
    def name(self) -> str:
        return f"Parallel({len(self.stages_to_run)} stages)"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Run stages in parallel using thread pool (OPTIMIZED).

        Uses ThreadPoolExecutor for I/O-bound stages.
        Result merging is done in batch after all stages complete.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if len(self.stages_to_run) <= 1:
            # Single stage, run directly - no threading overhead
            for stage in self.stages_to_run:
                context = stage.process(context)
            return context

        def run_stage(stage):
            return stage.name, stage.process(context)

        # Collect all results first, then merge in batch
        all_results = []

        with ThreadPoolExecutor(max_workers=len(self.stages_to_run)) as executor:
            futures = {executor.submit(run_stage, stage): stage for stage in self.stages_to_run}
            for future in as_completed(futures):
                stage = futures[future]
                try:
                    name, result_context = future.result()
                    all_results.append(result_context.results)
                except Exception as e:
                    raise AnalysisError(
                        f"Pipeline stage '{stage.name}' failed",
                        data={"stage": stage.name, "pipeline": self.__class__.__name__},
                        cause=e,
                    )

        # Batch merge all results (avoids repeated dict updates)
        for result_dict in all_results:
            for key, value in result_dict.items():
                if key not in context.results or key.startswith('_stage_'):
                    context.results[key] = value

        return context
