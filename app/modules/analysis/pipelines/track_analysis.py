"""
Track Analysis Pipeline - Complete analysis of a single track.

Combines all analysis tasks into a unified workflow.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

from .base import (
    Pipeline, PipelineContext, PipelineStage,
    LoadAudioStage, ComputeSTFTStage, TaskStage
)
from app.modules.analysis.tasks import (
    FeatureExtractionTask,
    ZoneClassificationTask,
    DropDetectionTask,
)


@dataclass
class TrackAnalysisResult:
    """
    Complete analysis result for a track.

    Combines results from all analysis tasks.
    """
    file_path: str
    file_name: str
    duration_sec: float

    # Zone classification
    zone: str
    zone_confidence: float
    zone_scores: Dict[str, float]

    # Features
    features: Dict[str, float]
    feature_count: int

    # Drops
    drop_count: int
    drop_density: float
    drop_times: List[float]

    # Metadata
    processing_time_sec: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'duration_sec': self.duration_sec,
            'zone': self.zone,
            'zone_confidence': self.zone_confidence,
            'zone_scores': self.zone_scores,
            'features': self.features,
            'feature_count': self.feature_count,
            'drop_count': self.drop_count,
            'drop_density': self.drop_density,
            'drop_times': self.drop_times,
            'processing_time_sec': self.processing_time_sec,
            'success': self.success,
            'error': self.error,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_context(cls, context: PipelineContext) -> 'TrackAnalysisResult':
        """Create from pipeline context."""
        features_result = context.get_result('features')
        zone_result = context.get_result('zone')
        drops_result = context.get_result('drops')

        return cls(
            file_path=context.input_path,
            file_name=context.file_name,
            duration_sec=context.results.get('_duration', 0.0),
            zone=zone_result.zone if zone_result else 'unknown',
            zone_confidence=zone_result.confidence if zone_result else 0.0,
            zone_scores=zone_result.zone_scores if zone_result else {},
            features=features_result.features if features_result else {},
            feature_count=len(features_result.features) if features_result else 0,
            drop_count=drops_result.n_drops if drops_result else 0,
            drop_density=drops_result.drop_density if drops_result else 0.0,
            drop_times=drops_result.get_drop_times() if drops_result else [],
            processing_time_sec=context.results.get('_total_time', 0.0),
            success=True,
            error=None
        )


class ExtractFeaturesStage(PipelineStage):
    """Stage that extracts features."""

    def __init__(self):
        self.task = FeatureExtractionTask()

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.audio_context is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(context.audio_context)
        context.set_result('features', result)
        return context


class ClassifyZoneStage(PipelineStage):
    """Stage that classifies zone."""

    def __init__(self, model_path: Optional[str] = None):
        self.task = ZoneClassificationTask(model_path=model_path)

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.audio_context is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(context.audio_context)
        context.set_result('zone', result)
        return context


class DetectDropsStage(PipelineStage):
    """Stage that detects drops."""

    def __init__(self):
        self.task = DropDetectionTask()

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.audio_context is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(context.audio_context)
        context.set_result('drops', result)
        return context


class ExportResultsStage(PipelineStage):
    """Stage that exports results to file."""

    def __init__(self, export_format: str = 'json'):
        """
        Initialize export stage.

        Args:
            export_format: Output format ('json', 'csv')
        """
        self.export_format = export_format

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.output_dir is None:
            return context

        output_dir = Path(context.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = TrackAnalysisResult.from_context(context)

        if self.export_format == 'json':
            output_path = output_dir / f"{context.file_stem}_analysis.json"
            with open(output_path, 'w') as f:
                f.write(result.to_json())
        elif self.export_format == 'csv':
            # Single row CSV
            output_path = output_dir / f"{context.file_stem}_analysis.csv"
            data = result.to_dict()
            # Flatten for CSV
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    k for k in data.keys() if not isinstance(data[k], (dict, list))
                ])
                writer.writeheader()
                writer.writerow({
                    k: v for k, v in data.items()
                    if not isinstance(v, (dict, list))
                })

        context.set_result('_export_path', str(output_path))
        return context


class TrackAnalysisPipeline(Pipeline):
    """
    Complete track analysis pipeline.

    Stages:
    1. LoadAudio - Load audio file
    2. ComputeSTFT - Compute spectrogram (once)
    3. ExtractFeatures - 79 features
    4. ClassifyZone - yellow/green/purple
    5. DetectDrops - Find drops
    6. ExportResults - Optional export

    Usage:
        pipeline = TrackAnalysisPipeline()
        result = pipeline.analyze("track.mp3")
        print(f"Zone: {result.zone} ({result.zone_confidence:.0%})")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        export_format: Optional[str] = None,
        output_dir: Optional[str] = None,
        sr: int = 22050,
        include_drops: bool = True
    ):
        """
        Initialize track analysis pipeline.

        Args:
            model_path: Path to zone classification model
            export_format: Export format ('json', 'csv', None)
            output_dir: Output directory for exports
            sr: Sample rate for loading
            include_drops: Whether to run drop detection
        """
        stages = [
            LoadAudioStage(sr=sr),
            ComputeSTFTStage(),
            ExtractFeaturesStage(),
            ClassifyZoneStage(model_path=model_path),
        ]

        if include_drops:
            stages.append(DetectDropsStage())

        if export_format:
            stages.append(ExportResultsStage(export_format=export_format))

        super().__init__(stages, name="TrackAnalysis")

        self.model_path = model_path
        self.output_dir = output_dir

    def analyze(self, path: str, output_dir: Optional[str] = None) -> TrackAnalysisResult:
        """
        Analyze a single track.

        Args:
            path: Path to audio file
            output_dir: Override output directory

        Returns:
            TrackAnalysisResult with all analysis data
        """
        context = PipelineContext(
            input_path=path,
            output_dir=output_dir or self.output_dir
        )

        try:
            context = self.run(context)
            return TrackAnalysisResult.from_context(context)
        except Exception as e:
            return TrackAnalysisResult(
                file_path=path,
                file_name=Path(path).name,
                duration_sec=0.0,
                zone='unknown',
                zone_confidence=0.0,
                zone_scores={},
                features={},
                feature_count=0,
                drop_count=0,
                drop_density=0.0,
                drop_times=[],
                processing_time_sec=0.0,
                success=False,
                error=str(e)
            )

    def analyze_batch(self, paths: List[str]) -> List[TrackAnalysisResult]:
        """
        Analyze multiple tracks sequentially.

        For parallel processing, use M2BatchProcessor.

        Args:
            paths: List of audio file paths

        Returns:
            List of TrackAnalysisResult
        """
        return [self.analyze(path) for path in paths]
