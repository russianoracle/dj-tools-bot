"""
Layer 3: PIPELINES - Orchestration

Apple Silicon M2 Optimized

Pipelines orchestrate tasks for complex workflows:
- File I/O
- Task composition
- Caching
- Parallelization
- Progress tracking

Usage:
    from src.core.pipelines import TrackAnalysisPipeline, M2BatchProcessor

    # Single track
    pipeline = TrackAnalysisPipeline()
    result = pipeline.run("track.mp3")

    # Batch processing
    processor = M2BatchProcessor(pipeline, workers=4)
    results = processor.process_directory("/music/folder")
"""

from .base import (
    Pipeline,
    PipelineContext,
    PipelineStage,
    LoadAudioStage,
    ComputeSTFTStage,
    IntegratedProgressTracker,
    IntegratedProgressCallback,
    make_cli_progress_bar,
)

from .track_analysis import (
    TrackAnalysisPipeline,
    TrackAnalysisResult,
)

from .set_analysis import (
    SetAnalysisPipeline,
    SetAnalysisResult,
    SegmentInfo,
    SegmentGenre,
    SetGenreDistribution,
    AnalyzeSegmentGenresStage,
    MixingStyle,
    SetBatchAnalyzer,
)

from .batch_processor import (
    M2BatchProcessor,
    BatchResult,
)

# NOTE: CacheManager is internal implementation.
# Use CacheRepository from src.core.cache for all cache operations.

# Training/Calibration pipelines moved to src/training/pipelines/
# Use: from src.training.pipelines import CalibrationPipeline, TrainingPipeline

__all__ = [
    # Base
    'Pipeline',
    'PipelineContext',
    'PipelineStage',
    'LoadAudioStage',
    'ComputeSTFTStage',
    'IntegratedProgressTracker',
    'IntegratedProgressCallback',
    'make_cli_progress_bar',
    # Track Analysis
    'TrackAnalysisPipeline',
    'TrackAnalysisResult',
    # Set Analysis
    'SetAnalysisPipeline',
    'SetAnalysisResult',
    'SegmentInfo',
    'SegmentGenre',
    'SetGenreDistribution',
    'AnalyzeSegmentGenresStage',
    'MixingStyle',
    'SetBatchAnalyzer',
    # Batch Processing
    'M2BatchProcessor',
    'BatchResult',
]
