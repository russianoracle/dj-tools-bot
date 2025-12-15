"""
Layer 2: TASKS - Application Tasks

Apple Silicon M2 Optimized

Tasks combine primitives to solve specific application problems.
Each task:
- Takes an AudioContext (with STFTCache)
- Returns a TaskResult subclass
- Contains business logic
- Is reusable and testable

Usage:
    from app.modules.analysis.tasks import FeatureExtractionTask, ZoneClassificationTask

    context = AudioContext(y=audio, sr=22050, stft_cache=cache)
    features = FeatureExtractionTask().execute(context)
    zone = ZoneClassificationTask(model_path="...").execute(context)
"""

from .base import (
    AudioContext,
    TaskResult,
    BaseTask,
    create_audio_context,
    ProgressCallback,
)

from .feature_extraction import (
    FeatureExtractionResult,
    FeatureExtractionTask,
    FEATURE_NAMES,
)

from .zone_classification import (
    ZoneClassificationResult,
    ZoneClassificationTask,
)

from .drop_detection import (
    DropDetectionResult,
    DropDetectionTask,
    DropDetectionMode,
)

from .transition_detection import (
    TransitionDetectionResult,
    TransitionDetectionTask,
    TransitionType,
    TransitionPair,
    MixinEvent,
    MixoutEvent,
)

from .genre_analysis import (
    GenreAnalysisResult,
    GenreAnalysisTask,
)

from .segmentation import (
    SegmentationTaskResult,
    SegmentationTask,
    SegmentBoundary,
)

from .energy_arc_analysis import (
    EnergyArcAnalysisResult,
    EnergyArcAnalysisTask,
)

from .tempo_distribution_analysis import (
    TempoDistributionAnalysisResult,
    TempoDistributionAnalysisTask,
)

from .key_analysis import (
    KeyAnalysisResult,
    KeyAnalysisTask,
)

from .track_compatibility import (
    # Data classes
    TrackAnalysis,
    # Task results
    BpmDetectionResult,
    MixPointResult,
    GridCalibrationResult,
    SpectralAnalysisResult,
    # Tasks (each has ONE responsibility)
    BpmDetectionTask,
    MixPointDetectionTask,
    GridCalibrationTask,
    SpectralAnalysisTask,
    # Utility functions
    compute_track_transition_score,
    get_energy_curve_normalized,
)

from .beat_grid import (
    BeatGridAnalysisResult,
    BeatGridTask,
    BeatGridMode,
)

from .local_tempo import (
    LocalTempoResult,
    LocalTempoAnalysisTask,
)

from .drop_detector_ml import (
    DropDetectorMLResult,
    DropDetectorML,
    DetectedDrop,
)

from .buildup_detector_ml import (
    BuildupDetectorResult,
    BuildupDetectorML,
    BuildupZone,
    BuildupPhase,
)

from .grid_aware_drop_detection import (
    GridAwareDropResult,
    GridAwareDropDetectionTask,
    GridAlignedDrop,
    DropType,
)

from .track_boundary_detection import (
    TrackBoundaryResult,
    TrackBoundaryDetectionTask,
    TrackBoundary,
    TrackInfo,
    TransitionStyle,
)

__all__ = [
    # Base
    'AudioContext',
    'TaskResult',
    'BaseTask',
    'create_audio_context',
    'ProgressCallback',
    # Feature Extraction
    'FeatureExtractionResult',
    'FeatureExtractionTask',
    'FEATURE_NAMES',
    # Zone Classification
    'ZoneClassificationResult',
    'ZoneClassificationTask',
    # Drop Detection
    'DropDetectionResult',
    'DropDetectionTask',
    'DropDetectionMode',
    # Transition Detection
    'TransitionDetectionResult',
    'TransitionDetectionTask',
    'TransitionType',
    'TransitionPair',
    'MixinEvent',
    'MixoutEvent',
    # Genre Analysis
    'GenreAnalysisResult',
    'GenreAnalysisTask',
    # Segmentation
    'SegmentationTaskResult',
    'SegmentationTask',
    'SegmentBoundary',
    # Energy Arc Analysis
    'EnergyArcAnalysisResult',
    'EnergyArcAnalysisTask',
    # Tempo Distribution Analysis
    'TempoDistributionAnalysisResult',
    'TempoDistributionAnalysisTask',
    # Key Analysis
    'KeyAnalysisResult',
    'KeyAnalysisTask',
    # Track Compatibility (individual Tasks)
    'TrackAnalysis',
    'BpmDetectionResult',
    'MixPointResult',
    'GridCalibrationResult',
    'SpectralAnalysisResult',
    'BpmDetectionTask',
    'MixPointDetectionTask',
    'GridCalibrationTask',
    'SpectralAnalysisTask',
    'compute_track_transition_score',
    'get_energy_curve_normalized',
    # Beat Grid
    'BeatGridAnalysisResult',
    'BeatGridTask',
    'BeatGridMode',
    # Local Tempo (PLP for DJ sets)
    'LocalTempoResult',
    'LocalTempoAnalysisTask',
    # ML Drop Detection
    'DropDetectorMLResult',
    'DropDetectorML',
    'DetectedDrop',
    # ML Buildup Detection
    'BuildupDetectorResult',
    'BuildupDetectorML',
    'BuildupZone',
    'BuildupPhase',
    # Grid-Aware Drop Detection (ML + beat grid + buildup)
    'GridAwareDropResult',
    'GridAwareDropDetectionTask',
    'GridAlignedDrop',
    'DropType',
    # Track Boundary Detection (timbral analysis for DJ sets)
    'TrackBoundaryResult',
    'TrackBoundaryDetectionTask',
    'TrackBoundary',
    'TrackInfo',
    'TransitionStyle',
]
