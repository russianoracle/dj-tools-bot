"""
Base classes for Tasks layer.

AudioContext holds all shared data for task execution.
TaskResult is the base class for all task outputs.
ProgressCallback enables status updates from lower layers.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
import time

from ..primitives import STFTCache, compute_stft
from ..primitives.beat_grid import BeatGridResult


# Type alias for progress callback
# Args: (stage: str, progress: float 0-1, message: str)
ProgressCallback = Callable[[str, float, str], None]


@dataclass
class AudioContext:
    """
    Shared context for all tasks.

    Contains the audio signal and pre-computed STFT cache.
    All tasks should use this shared cache instead of
    recomputing STFT.

    Attributes:
        y: Audio time series (mono, float32)
        sr: Sample rate
        stft_cache: Pre-computed STFT (the foundation)
        duration_sec: Track duration in seconds
        file_path: Optional source file path
        metadata: Optional additional metadata
        beat_grid: Optional beat grid for musical structure alignment
        progress_callback: Optional callback for progress updates
    """
    y: np.ndarray
    sr: int
    stft_cache: STFTCache
    duration_sec: float
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    beat_grid: Optional[BeatGridResult] = None  # For musical structure alignment
    progress_callback: Optional[ProgressCallback] = None  # For status bar updates

    def report_progress(self, stage: str, progress: float, message: str = ""):
        """Report progress to callback if set."""
        if self.progress_callback:
            self.progress_callback(stage, progress, message)

    @property
    def n_frames(self) -> int:
        """Number of STFT frames."""
        return self.stft_cache.n_frames

    @property
    def hop_length(self) -> int:
        """Hop length used for STFT."""
        return self.stft_cache.hop_length

    @property
    def n_fft(self) -> int:
        """FFT size used."""
        return self.stft_cache.n_fft


def create_audio_context(
    y: np.ndarray,
    sr: int,
    file_path: Optional[str] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    progress_callback: Optional[ProgressCallback] = None,
    **metadata
) -> AudioContext:
    """
    Create AudioContext from raw audio signal.

    This is the main entry point for processing a track.
    Computes STFT once, which is then shared by all tasks.

    Args:
        y: Audio signal (will be converted to mono float32)
        sr: Sample rate
        file_path: Optional source file path
        n_fft: FFT size (default: 2048)
        hop_length: Hop length (default: 512)
        progress_callback: Optional callback for progress updates (stage, progress, message)
        **metadata: Additional metadata to store

    Returns:
        AudioContext ready for task execution

    Example:
        >>> y, sr = librosa.load("track.mp3", sr=22050)
        >>> ctx = create_audio_context(y, sr, file_path="track.mp3")
        >>> features = FeatureExtractionTask().execute(ctx)

        # With progress callback:
        >>> def on_progress(stage, progress, msg):
        ...     print(f"{stage}: {progress*100:.0f}% - {msg}")
        >>> ctx = create_audio_context(y, sr, progress_callback=on_progress)
    """
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Ensure float32 contiguous
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Compute duration
    duration_sec = len(y) / sr

    # Compute STFT cache (the foundation for all analysis)
    stft_cache = compute_stft(
        y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )

    return AudioContext(
        y=y,
        sr=sr,
        stft_cache=stft_cache,
        duration_sec=duration_sec,
        file_path=file_path,
        metadata=metadata,
        progress_callback=progress_callback
    )


@dataclass
class TaskResult:
    """
    Base result for all tasks.

    All task results must inherit from this class
    and add their specific output fields.

    Attributes:
        success: Whether the task completed successfully
        task_name: Name of the task
        processing_time_sec: How long the task took
        error: Error message if success is False
    """
    success: bool
    task_name: str
    processing_time_sec: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'task_name': self.task_name,
            'processing_time_sec': self.processing_time_sec,
            'error': self.error
        }


class BaseTask(ABC):
    """
    Abstract base class for all tasks.

    Each task must implement the execute() method.

    Example:
        class MyTask(BaseTask):
            def execute(self, context: AudioContext) -> MyResult:
                # Use primitives to do work
                rms = compute_rms(context.stft_cache.S)
                return MyResult(success=True, rms=rms, ...)
    """

    @property
    def name(self) -> str:
        """Task name (class name by default)."""
        return self.__class__.__name__

    @abstractmethod
    def execute(self, context: AudioContext) -> TaskResult:
        """
        Execute the task on the given audio context.

        Args:
            context: AudioContext with pre-computed STFT

        Returns:
            TaskResult subclass with task-specific outputs
        """
        pass

    def execute_timed(self, context: AudioContext) -> TaskResult:
        """
        Execute the task and measure processing time.

        This is a convenience wrapper that sets processing_time_sec.
        """
        start = time.time()
        try:
            result = self.execute(context)
            result.processing_time_sec = time.time() - start
            return result
        except Exception as e:
            return TaskResult(
                success=False,
                task_name=self.name,
                processing_time_sec=time.time() - start,
                error=str(e)
            )

    def __repr__(self) -> str:
        return f"{self.name}()"
