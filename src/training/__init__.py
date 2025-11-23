"""
Training module for zone classification using machine learning.
"""

# Zone classification (primary)
from .zone_features import ZoneFeatureExtractor, ZoneFeatures
from .fast_features import FastZoneFeatureExtractor, FastZoneFeatures
from .zone_models import XGBoostZoneClassifier, NeuralZoneClassifier, EnsembleZoneClassifier
from .zone_trainer import ZoneTrainer

# BPM correction (legacy/experimental)
from .tempo_features import TempoFeatureExtractor
from .models import XGBoostBPMModel, NeuralBPMModel, EnsembleBPMModel
from .bpm_trainer import BPMTrainer

# Checkpointing (shared)
from .checkpoint_manager import CheckpointManager, TrainingResumer

__all__ = [
    # Zone classification
    'ZoneFeatureExtractor',
    'ZoneFeatures',
    'XGBoostZoneClassifier',
    'NeuralZoneClassifier',
    'EnsembleZoneClassifier',
    'ZoneTrainer',
    # BPM correction (legacy)
    'TempoFeatureExtractor',
    'XGBoostBPMModel',
    'NeuralBPMModel',
    'EnsembleBPMModel',
    'BPMTrainer',
    # Checkpointing
    'CheckpointManager',
    'TrainingResumer'
]
