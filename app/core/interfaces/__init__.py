"""
Interfaces - Protocols for Dependency Injection.

These protocols define contracts that implementations must follow.
Use Protocol for type hints to enable loose coupling.

Example:
    def analyze(cache: CacheProtocol, pipeline: PipelineProtocol):
        # Works with any implementation
        result = pipeline.run(path)
        cache.save_set(result)
"""

from .cache_protocol import (
    CacheProtocol,
    SetCacheProtocol,
    DJProfileCacheProtocol,
)
from .task_protocol import (
    TaskProtocol,
    ConfigurableTaskProtocol,
    TaskResult,
)
from .pipeline_protocol import (
    PipelineProtocol,
    ProgressivePipelineProtocol,
    StageProtocol,
    PipelineResult,
)

__all__ = [
    # Cache
    'CacheProtocol',
    'SetCacheProtocol',
    'DJProfileCacheProtocol',
    # Task
    'TaskProtocol',
    'ConfigurableTaskProtocol',
    'TaskResult',
    # Pipeline
    'PipelineProtocol',
    'ProgressivePipelineProtocol',
    'StageProtocol',
    'PipelineResult',
]
