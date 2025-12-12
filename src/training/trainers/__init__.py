"""
Trainers - ML model training extracted from core tasks.

These trainers contain training logic that was previously embedded in
production tasks (SRP violation). Now training is separated from inference.

Trainers:
- DropDetectorTrainer: Train XGBoost drop detector
- BuildupDetectorTrainer: Train XGBoost buildup phase classifier
"""

from .drop_trainer import DropDetectorTrainer
from .buildup_trainer import BuildupDetectorTrainer

__all__ = [
    'DropDetectorTrainer',
    'BuildupDetectorTrainer',
]
