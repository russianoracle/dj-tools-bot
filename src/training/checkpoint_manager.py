"""Checkpoint management for training resumption."""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..utils import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages training checkpoints and resumption."""

    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.checkpoint_dir / "training_history.json"
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"

    def save_checkpoint(self, model: Any, epoch: int, metrics: Dict[str, float],
                       algorithm: str, features_df: Optional[Any] = None):
        """
        Save training checkpoint.

        Args:
            model: Model to save
            epoch: Current epoch/iteration
            metrics: Current metrics
            algorithm: Algorithm name
            features_df: Optional features DataFrame (saved only once)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{algorithm}_epoch_{epoch}_{timestamp}"

        # Save model (if provided)
        model_path = None
        if model is not None:
            model_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
            model.save(str(model_path))

        # Save metadata
        metadata = {
            'epoch': epoch,
            'algorithm': algorithm,
            'metrics': metrics,
            'timestamp': timestamp,
            'model_path': str(model_path) if model_path else None
        }

        # Update checkpoint metadata
        self._update_metadata(checkpoint_name, metadata)

        # Update training history
        self._update_history(algorithm, epoch, metrics)

        # Save features - ALWAYS overwrite to keep cache up-to-date
        if features_df is not None:
            features_path = self.checkpoint_dir / "features.pkl"
            # Remove old cache if exists
            if features_path.exists():
                features_path.unlink()
                logger.info(f"Removed old features cache: {features_path}")
            # Save new cache
            features_df.to_pickle(str(features_path))
            logger.info(f"Saved features cache to {features_path}")

        logger.info(f"Checkpoint saved: {checkpoint_name}")

    def _update_metadata(self, checkpoint_name: str, metadata: Dict[str, Any]):
        """Update checkpoint metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}

        all_metadata[checkpoint_name] = metadata

        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)

    def _update_history(self, algorithm: str, epoch: int, metrics: Dict[str, float]):
        """Update training history."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {}

        if algorithm not in history:
            history[algorithm] = {
                'epochs': [],
                'metrics': {}
            }

        history[algorithm]['epochs'].append(epoch)

        for metric_name, metric_value in metrics.items():
            # Skip non-numeric metrics (like confusion_matrix, classification_report)
            if isinstance(metric_value, (list, dict)):
                continue

            if metric_name not in history[algorithm]['metrics']:
                history[algorithm]['metrics'][metric_name] = []

            try:
                history[algorithm]['metrics'][metric_name].append(float(metric_value))
            except (ValueError, TypeError):
                # Skip values that can't be converted to float
                pass

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def get_latest_checkpoint(self, algorithm: str) -> Optional[Dict[str, Any]]:
        """
        Get latest checkpoint for algorithm.

        Returns:
            Checkpoint metadata or None if not found
        """
        if not self.metadata_file.exists():
            return None

        with open(self.metadata_file, 'r') as f:
            all_metadata = json.load(f)

        # Filter by algorithm
        algo_checkpoints = {k: v for k, v in all_metadata.items() if v['algorithm'] == algorithm}

        if not algo_checkpoints:
            return None

        # Get latest by epoch
        latest = max(algo_checkpoints.values(), key=lambda x: x['epoch'])
        return latest

    def load_checkpoint(self, checkpoint_path: str, model: Any) -> Dict[str, Any]:
        """
        Load checkpoint into model.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model instance to load into

        Returns:
            Checkpoint metadata
        """
        model.load(checkpoint_path)

        # Find metadata for this checkpoint
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)

            for checkpoint_name, metadata in all_metadata.items():
                if metadata['model_path'] == checkpoint_path:
                    logger.info(f"Loaded checkpoint from epoch {metadata['epoch']}")
                    return metadata

        return {}

    def load_features(self) -> Optional[Any]:
        """Load cached features DataFrame."""
        features_path = self.checkpoint_dir / "features.pkl"
        if features_path.exists():
            import pandas as pd
            logger.info(f"Loading cached features from {features_path}")
            return pd.read_pickle(str(features_path))
        return None

    def get_training_history(self, algorithm: str) -> Optional[Dict[str, Any]]:
        """Get training history for algorithm."""
        if not self.history_file.exists():
            return None

        with open(self.history_file, 'r') as f:
            history = json.load(f)

        return history.get(algorithm)

    def cleanup_old_checkpoints(self, algorithm: str, keep_last: int = 3):
        """
        Remove old checkpoints, keeping only the last N.

        Args:
            algorithm: Algorithm name
            keep_last: Number of recent checkpoints to keep
        """
        if not self.metadata_file.exists():
            return

        with open(self.metadata_file, 'r') as f:
            all_metadata = json.load(f)

        # Filter by algorithm and sort by epoch
        algo_checkpoints = [(k, v) for k, v in all_metadata.items() if v['algorithm'] == algorithm]
        algo_checkpoints.sort(key=lambda x: x[1]['epoch'])

        # Remove old checkpoints
        to_remove = algo_checkpoints[:-keep_last] if len(algo_checkpoints) > keep_last else []

        for checkpoint_name, metadata in to_remove:
            model_path = Path(metadata['model_path'])
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_name}")

            del all_metadata[checkpoint_name]

        # Update metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)

    def get_best_checkpoint(self, algorithm: str, metric: str = 'val_mae') -> Optional[Dict[str, Any]]:
        """
        Get checkpoint with best metric value.

        Args:
            algorithm: Algorithm name
            metric: Metric to optimize (lower is better)

        Returns:
            Best checkpoint metadata
        """
        if not self.metadata_file.exists():
            return None

        with open(self.metadata_file, 'r') as f:
            all_metadata = json.load(f)

        algo_checkpoints = {k: v for k, v in all_metadata.items() if v['algorithm'] == algorithm}

        if not algo_checkpoints:
            return None

        # Find best by metric
        valid_checkpoints = [(k, v) for k, v in algo_checkpoints.items()
                            if metric in v.get('metrics', {})]

        if not valid_checkpoints:
            return None

        best_name, best_metadata = min(valid_checkpoints,
                                       key=lambda x: x[1]['metrics'][metric])

        logger.info(f"Best checkpoint: {best_name} ({metric}={best_metadata['metrics'][metric]:.3f})")
        return best_metadata


class TrainingResumer:
    """Helper for resuming interrupted training."""

    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager

    def can_resume(self, algorithm: str) -> bool:
        """Check if training can be resumed for algorithm."""
        return self.checkpoint_manager.get_latest_checkpoint(algorithm) is not None

    def resume_training(self, algorithm: str, model: Any,
                       features_df: Any = None) -> tuple:
        """
        Resume training from latest checkpoint.

        Args:
            algorithm: Algorithm name
            model: Model instance
            features_df: Features DataFrame (will try to load cached if None)

        Returns:
            (loaded_model, start_epoch, features_df)
        """
        # Load latest checkpoint
        checkpoint = self.checkpoint_manager.get_latest_checkpoint(algorithm)

        if checkpoint is None:
            logger.warning(f"No checkpoint found for {algorithm}, starting fresh")
            return model, 0, features_df

        # Load model
        self.checkpoint_manager.load_checkpoint(checkpoint['model_path'], model)

        start_epoch = checkpoint['epoch'] + 1

        # Load features if not provided
        if features_df is None:
            features_df = self.checkpoint_manager.load_features()

        logger.info(f"Resuming {algorithm} training from epoch {start_epoch}")

        return model, start_epoch, features_df
