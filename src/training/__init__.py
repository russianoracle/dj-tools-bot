"""
Training module for zone classification using machine learning.

Primary Components:
- ZoneFeatureExtractor: Full feature extraction (51 features)
- FastZoneFeatureExtractor: Quick extraction (10 features)
- ProductionPipeline: Recommended training pipeline
- ZoneTrainer: GUI-oriented trainer with checkpointing

Training Pipelines (moved from src/core/pipelines/):
- TrainingPipeline: Universal ML model training with caching
- CalibrationPipeline: Optimize detection parameters

Legacy modules moved to _legacy/ subdirectory.
"""

# Zone classification (primary)
from .zone_features import ZoneFeatureExtractor, ZoneFeatures
from .fast_features import FastZoneFeatureExtractor, FastZoneFeatures
from .zone_models import XGBoostZoneClassifier, NeuralZoneClassifier, EnsembleZoneClassifier
from .zone_trainer import ZoneTrainer
from .production_pipeline import ProductionPipeline, PipelineConfig

# Checkpointing (shared utility)
from .checkpoint_manager import CheckpointManager, TrainingResumer

# Training pipelines (moved from core)
from .pipelines import (
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
    CalibrationPipeline,
    CalibrationResult,
    CachedSetFeatures,
)

__all__ = [
    # Zone classification
    'ZoneFeatureExtractor',
    'ZoneFeatures',
    'FastZoneFeatureExtractor',
    'FastZoneFeatures',
    'XGBoostZoneClassifier',
    'NeuralZoneClassifier',
    'EnsembleZoneClassifier',
    'ZoneTrainer',
    'ProductionPipeline',
    'PipelineConfig',
    # Checkpointing
    'CheckpointManager',
    'TrainingResumer',
    # Training Pipelines
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
    'CalibrationPipeline',
    'CalibrationResult',
    'CachedSetFeatures',
]
