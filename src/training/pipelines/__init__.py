"""
Training Pipelines - ML training and calibration workflows.

These pipelines handle training-specific operations and are separate from
production pipelines in src/core/pipelines/.

Components:
- TrainingPipeline: Universal ML model training with caching
- CalibrationPipeline: Optimize detection parameters via differential_evolution
"""

from .training import (
    TrainingPipeline,
    TrainingResult,
    BaseGroundTruthLoader,
    DropGroundTruthLoader,
    FeatureDataset,
    LabelAssigner,
    BaseTrainer,
    XGBoostTrainer,
    Evaluator,
    nms_vectorized,
)

from .calibration import (
    CalibrationPipeline,
    CalibrationResult,
    CachedSetFeatures,
)

__all__ = [
    # Training
    'TrainingPipeline',
    'TrainingResult',
    'BaseGroundTruthLoader',
    'DropGroundTruthLoader',
    'FeatureDataset',
    'LabelAssigner',
    'BaseTrainer',
    'XGBoostTrainer',
    'Evaluator',
    'nms_vectorized',
    # Calibration
    'CalibrationPipeline',
    'CalibrationResult',
    'CachedSetFeatures',
]
