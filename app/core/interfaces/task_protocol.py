"""
Task Protocol - Interface for analysis tasks.

Tasks are units of work that operate on AudioContext.
They follow Single Responsibility Principle.
"""

from typing import Protocol, Any, TypeVar, Generic, runtime_checkable
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Base result from task execution."""
    success: bool
    data: Any = None
    error: str = None


T = TypeVar('T', bound=TaskResult)


@runtime_checkable
class TaskProtocol(Protocol[T]):
    """Protocol for analysis tasks (DI interface)."""

    def execute(self, context: Any) -> T:
        """
        Execute task on audio context.

        Args:
            context: AudioContext with audio data and STFT cache

        Returns:
            TaskResult with analysis output
        """
        ...

    @property
    def name(self) -> str:
        """Task name for logging."""
        ...


@runtime_checkable
class ConfigurableTaskProtocol(TaskProtocol[T], Protocol[T]):
    """Protocol for tasks with configuration."""

    def configure(self, **kwargs) -> None:
        """Configure task parameters."""
        ...

    def reset(self) -> None:
        """Reset to default configuration."""
        ...
