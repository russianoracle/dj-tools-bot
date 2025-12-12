"""
Base classes for Pipelines layer.

Pipeline = sequence of PipelineStages that transform PipelineContext.
"""

import logging
import numpy as np
import librosa
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import time

from ..primitives import STFTCache, compute_stft
from ..tasks import AudioContext, create_audio_context

logger = logging.getLogger(__name__)


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
        context.results['_pipeline_name'] = self.name
        context.results['_start_time'] = time.time()

        logger.info(f"[{self.name}] Starting pipeline for {context.file_name}")

        for i, stage in enumerate(self.stages):
            if stage.should_skip(context):
                logger.debug(f"[{self.name}] Skipping {stage.name}")
                continue

            logger.info(f"[{self.name}] Stage {i+1}/{len(self.stages)}: {stage.name}")
            stage_start = time.time()
            context = stage.process(context)
            elapsed = time.time() - stage_start
            context.results[f'_stage_{stage.name}_time'] = elapsed
            logger.info(f"[{self.name}] {stage.name} done in {elapsed:.1f}s")

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

        # Load audio
        y, sr = librosa.load(
            path,
            sr=self.sr,
            mono=self.mono,
            duration=self.duration
        )

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
        y = context.results.get('_audio')
        sr = context.results.get('_sr')

        if y is None or sr is None:
            raise ValueError("Audio not loaded. Run LoadAudioStage first.")

        # Create AudioContext with STFT
        audio_ctx = create_audio_context(
            y=y,
            sr=sr,
            file_path=context.input_path,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

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
            raise ValueError("AudioContext not created. Run ComputeSTFTStage first.")

        result = self.task.execute(context.audio_context)
        context.set_result(self.result_key, result)

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
        """Run stages (currently sequential, can be parallelized)."""
        # For now, run sequentially
        # TODO: Use threading for I/O-bound stages
        for stage in self.stages_to_run:
            context = stage.process(context)
        return context
