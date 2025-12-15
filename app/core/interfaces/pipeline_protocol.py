"""
Pipeline Protocol - Interface for analysis pipelines.

Pipelines orchestrate multiple tasks in sequence.
They manage context, caching, and error handling.
"""

from typing import Protocol, Any, List, Optional, Callable, TypeVar, Generic, runtime_checkable
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PipelineResult:
    """Base result from pipeline execution."""
    success: bool
    data: Any = None
    error: str = None
    stages_completed: List[str] = None
    duration_seconds: float = 0.0


T = TypeVar('T', bound=PipelineResult)


@runtime_checkable
class PipelineProtocol(Protocol[T]):
    """Protocol for analysis pipelines (DI interface)."""

    def run(self, input_path: Path, **kwargs) -> T:
        """
        Run pipeline on input file.

        Args:
            input_path: Path to audio file
            **kwargs: Additional parameters

        Returns:
            PipelineResult with analysis output
        """
        ...

    @property
    def name(self) -> str:
        """Pipeline name for logging."""
        ...


@runtime_checkable
class ProgressivePipelineProtocol(PipelineProtocol[T], Protocol[T]):
    """Protocol for pipelines with progress callbacks."""

    def run_with_progress(
        self,
        input_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs
    ) -> T:
        """
        Run pipeline with progress updates.

        Args:
            input_path: Path to audio file
            progress_callback: Callback(progress_percent, stage_name)
            **kwargs: Additional parameters

        Returns:
            PipelineResult with analysis output
        """
        ...


@runtime_checkable
class StageProtocol(Protocol):
    """Protocol for pipeline stages."""

    def process(self, context: Any) -> Any:
        """Process context through this stage."""
        ...

    @property
    def name(self) -> str:
        """Stage name for logging."""
        ...

    @property
    def weight(self) -> float:
        """Stage weight for progress calculation (0.0-1.0)."""
        ...
