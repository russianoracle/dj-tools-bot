"""Main training coordinator for zone classification."""

import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from .zone_features import ZoneFeatureExtractor, ZoneFeatures
from .fast_features import FastZoneFeatureExtractor, FastZoneFeatures
from .zone_models import (
    XGBoostZoneClassifier,
    NeuralZoneClassifier,
    EnsembleZoneClassifier,
    ZONE_LABELS,
    ZONE_NAMES
)
from .checkpoint_manager import CheckpointManager, TrainingResumer
from ..utils import get_logger

logger = get_logger(__name__)


class ZoneTrainer:
    """Coordinates zone classification training pipeline."""

    def __init__(self, test_data_path: str, use_gpu: bool = True, use_embeddings: bool = False, use_fast_mode: bool = True):
        """
        Initialize zone trainer.

        Args:
            test_data_path: Path to TSV file with zone labels
            use_gpu: Use GPU for feature extraction
            use_embeddings: Use deep learning embeddings (wav2vec2) - SLOW but more accurate
            use_fast_mode: Use fast feature extraction (10 features vs 30+) - 10x faster, good accuracy
        """
        self.test_data_path = Path(test_data_path)
        self.use_gpu = use_gpu
        self.use_embeddings = use_embeddings
        self.use_fast_mode = use_fast_mode
        self._should_stop = False  # Graceful stop flag

        # Feature extractor (choose fast or full version)
        if use_fast_mode:
            self.feature_extractor = FastZoneFeatureExtractor()
            logger.info("Using FAST feature extraction (10 features, ~3s/track)")
        else:
            self.feature_extractor = ZoneFeatureExtractor(use_gpu=use_gpu, use_embeddings=use_embeddings)
            logger.info("Using FULL feature extraction (30+ features, ~30s/track)")

        # Data storage
        self.audio_paths = []
        self.zone_labels = []
        self.features_list = []

        # Training data
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_training_data(self, progress_callback: Optional[Callable] = None,
                          log_callback: Optional[Callable] = None) -> int:
        """
        Load training data from TSV file.

        Args:
            progress_callback: Callback for progress updates
            log_callback: Callback for log messages

        Returns:
            Number of tracks loaded
        """
        self._log(log_callback, "INFO", f"⚙️  Opening training data file: {self.test_data_path}")

        # Read TSV file (UTF-16 encoding)
        with open(self.test_data_path, 'r', encoding='utf-16') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)

        total = len(rows)
        self._log(log_callback, "INFO", f"✓ File loaded: {total} rows found")
        self._log(log_callback, "INFO", f"⚙️  Validating entries...")

        loaded = 0
        skipped_no_zone = 0
        skipped_invalid_zone = 0
        skipped_not_found = 0

        for i, row in enumerate(rows):
            # Get file path
            file_path = (
                row.get('Location', '').strip() or
                row.get('Path', '').strip() or
                row.get('путь', '').strip()
            )

            # Get zone label
            zone = (
                row.get('Zone', '').strip().lower() or
                row.get('zone', '').strip().lower() or
                row.get('зона', '').strip().lower()
            )

            if not file_path or not zone:
                skipped_no_zone += 1
                continue

            # Validate zone
            if zone not in ZONE_LABELS:
                self._log(log_callback, "WARNING", f"⚠️  Unknown zone '{zone}' for {file_path}")
                skipped_invalid_zone += 1
                continue

            # Validate file exists
            path_obj = Path(file_path)
            if not path_obj.exists():
                self._log(log_callback, "WARNING", f"⚠️  File not found: {file_path}")
                skipped_not_found += 1
                continue

            self.audio_paths.append(str(path_obj))
            self.zone_labels.append(ZONE_LABELS[zone])
            loaded += 1

            if progress_callback:
                progress_callback(i + 1, total, f"Validated {i+1}/{total} entries")

        # Summary
        self._log(log_callback, "INFO", f"\n{'='*60}")
        self._log(log_callback, "INFO", f"✓ Data validation complete:")
        self._log(log_callback, "INFO", f"  • Loaded: {loaded} tracks")
        if skipped_no_zone > 0:
            self._log(log_callback, "WARNING", f"  • Skipped (no zone): {skipped_no_zone}")
        if skipped_invalid_zone > 0:
            self._log(log_callback, "WARNING", f"  • Skipped (invalid zone): {skipped_invalid_zone}")
        if skipped_not_found > 0:
            self._log(log_callback, "WARNING", f"  • Skipped (not found): {skipped_not_found}")
        self._log(log_callback, "INFO", f"{'='*60}\n")

        return loaded

    def request_stop(self):
        """Request graceful stop of training process."""
        self._should_stop = True
        logger.info("Graceful stop requested")

    def extract_features(self, use_cache: bool = True,
                        checkpoint_manager: Optional[CheckpointManager] = None,
                        progress_callback: Optional[Callable] = None,
                        log_callback: Optional[Callable] = None,
                        checkpoint_interval: int = 5) -> pd.DataFrame:
        """
        Extract features from all audio files.

        Args:
            use_cache: Try to load cached features
            checkpoint_manager: Checkpoint manager for caching
            progress_callback: Progress callback
            log_callback: Log callback
            checkpoint_interval: Save progress every N tracks (default: 10)

        Returns:
            Features DataFrame
        """
        # Try to load from cache
        if use_cache and checkpoint_manager:
            cached_features = checkpoint_manager.load_features()
            if cached_features is not None:
                self._log(log_callback, "INFO", "Loaded features from cache")
                self.features_list = cached_features['features_list'].tolist()
                self._log(log_callback, "INFO", f"Resuming from {len(self.features_list)} cached features")
                # Start from where we left off
                start_index = len(self.features_list)
            else:
                start_index = 0
                self.features_list = []
        else:
            start_index = 0
            self.features_list = []

        self._log(log_callback, "INFO", f"Extracting features from {len(self.audio_paths)} tracks (starting at {start_index})...")

        total = len(self.audio_paths)

        for i in range(start_index, total):
            # Check for graceful stop
            if self._should_stop:
                self._log(log_callback, "WARNING", f"⏸️  Graceful stop requested at {i}/{total} tracks")
                # Save checkpoint before stopping
                if checkpoint_manager and len(self.features_list) > 0:
                    self._save_incremental_checkpoint(checkpoint_manager, i, total, log_callback)
                    self._log(log_callback, "INFO", "💾 Final checkpoint saved before stopping")
                break

            audio_path = self.audio_paths[i]

            try:
                self._log(log_callback, "INFO", f"Processing [{i+1}/{total}]: {Path(audio_path).name}")

                features = self.feature_extractor.extract(audio_path)
                self.features_list.append(features)

                if progress_callback:
                    progress_callback(i + 1, total, f"Extracted features from {Path(audio_path).name}")

                self._log(log_callback, "INFO", f"✓ Completed {Path(audio_path).name}")

                # Incremental checkpoint every N tracks
                if checkpoint_manager and (i + 1) % checkpoint_interval == 0:
                    self._save_incremental_checkpoint(checkpoint_manager, i + 1, total, log_callback)

            except Exception as e:
                self._log(log_callback, "ERROR", f"Failed to extract from {audio_path}: {e}")
                self.features_list.append(None)
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                self._log(log_callback, "WARNING", "⏸️  Keyboard interrupt detected")
                self._should_stop = True
                if checkpoint_manager and len(self.features_list) > 0:
                    self._save_incremental_checkpoint(checkpoint_manager, i, total, log_callback)
                    self._log(log_callback, "INFO", "💾 Checkpoint saved after interrupt")
                raise

        # Filter out None values
        valid_indices = [i for i, f in enumerate(self.features_list) if f is not None]
        self.features_list = [self.features_list[i] for i in valid_indices]
        self.audio_paths = [self.audio_paths[i] for i in valid_indices]
        self.zone_labels = [self.zone_labels[i] for i in valid_indices]

        self._log(log_callback, "INFO", f"Successfully extracted {len(self.features_list)} feature sets")

        # Check if we have any valid features
        if len(self.features_list) == 0:
            self._log(log_callback, "ERROR", "❌ No features were successfully extracted!")
            self._log(log_callback, "ERROR", "All audio files failed during feature extraction.")
            self._log(log_callback, "ERROR", "Please check the error messages above for details.")
            raise RuntimeError("No features were successfully extracted from any audio files")

        # Create DataFrame and cache
        features_df = pd.DataFrame({
            'audio_path': self.audio_paths,
            'zone_label': self.zone_labels,
            'features_list': self.features_list
        })

        if checkpoint_manager:
            checkpoint_manager.save_checkpoint(
                model=None,
                epoch=0,
                metrics={},
                algorithm='features',
                features_df=features_df
            )

        return features_df

    def prepare_datasets(self, test_size: float = 0.15, val_size: float = 0.15,
                        include_embeddings: bool = True,
                        log_callback: Optional[Callable] = None):
        """
        Prepare train/val/test splits.

        Args:
            test_size: Test set proportion
            val_size: Validation set proportion
            include_embeddings: Include torchaudio embeddings
            log_callback: Log callback
        """
        # Convert features to vectors
        X = np.array([f.to_vector(include_embeddings=include_embeddings)
                     for f in self.features_list])
        y = np.array(self.zone_labels)

        self._log(log_callback, "INFO", f"Feature vectors shape: {X.shape}")

        # Check if we have enough samples per class for stratified splitting
        # Need at least 2 samples per class for stratification
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)

        # Determine if stratification is possible
        use_stratify = min_class_count >= 2

        if not use_stratify:
            self._log(log_callback, "WARNING",
                     f"Dataset too small for stratified splitting (min class count: {min_class_count}). "
                     f"Using random split instead.")

        # Split: train/temp, then temp -> val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=42,
            stratify=y if use_stratify else None
        )

        # Split temp into val and test
        val_proportion = val_size / (test_size + val_size)

        # Check again for temp split
        if use_stratify:
            unique_temp, temp_counts = np.unique(y_temp, return_counts=True)
            min_temp_count = np.min(temp_counts)
            use_stratify_temp = min_temp_count >= 2
        else:
            use_stratify_temp = False

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_proportion), random_state=42,
            stratify=y_temp if use_stratify_temp else None
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self._log(log_callback, "INFO", f"\n{'='*60}")
        self._log(log_callback, "INFO", f"✓ Dataset split complete:")
        self._log(log_callback, "INFO", f"  • Train: {len(y_train)} samples ({len(y_train)/len(y)*100:.1f}%)")
        self._log(log_callback, "INFO", f"  • Val:   {len(y_val)} samples ({len(y_val)/len(y)*100:.1f}%)")
        self._log(log_callback, "INFO", f"  • Test:  {len(y_test)} samples ({len(y_test)/len(y)*100:.1f}%)")

        # Show class distribution
        self._log(log_callback, "INFO", f"\nClass distribution by zone:")
        for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            dist = {ZONE_NAMES[i]: np.sum(y_split == i) for i in range(3)}
            yellow_pct = dist['yellow'] / len(y_split) * 100 if len(y_split) > 0 else 0
            green_pct = dist['green'] / len(y_split) * 100 if len(y_split) > 0 else 0
            purple_pct = dist['purple'] / len(y_split) * 100 if len(y_split) > 0 else 0
            self._log(log_callback, "INFO",
                     f"  {split_name:5s}: 🟨 {dist['yellow']:2d} ({yellow_pct:4.1f}%), "
                     f"🟩 {dist['green']:2d} ({green_pct:4.1f}%), "
                     f"🟪 {dist['purple']:2d} ({purple_pct:4.1f}%)")
        self._log(log_callback, "INFO", f"{'='*60}\n")

    def train_model(self, algorithm: str,
                   checkpoint_manager: Optional[CheckpointManager] = None,
                   resume: bool = False,
                   progress_callback: Optional[Callable] = None,
                   log_callback: Optional[Callable] = None,
                   **kwargs) -> Tuple[object, Dict]:
        """
        Train a specific model.

        Args:
            algorithm: 'xgboost', 'neural_network', or 'ensemble'
            checkpoint_manager: Checkpoint manager
            resume: Resume from checkpoint
            progress_callback: Progress callback
            log_callback: Log callback
            **kwargs: Algorithm-specific parameters

        Returns:
            (trained_model, metrics)
        """
        self._log(log_callback, "INFO", f"Training {algorithm} classifier...")

        # Check for resumption
        if resume and checkpoint_manager:
            resumer = TrainingResumer(checkpoint_manager)
            if resumer.can_resume(algorithm):
                self._log(log_callback, "INFO", f"Resuming {algorithm} from checkpoint")

        # Create model
        if algorithm == 'xgboost':
            model = XGBoostZoneClassifier()
        elif algorithm == 'neural_network':
            model = NeuralZoneClassifier(input_dim=self.X_train.shape[1])
        elif algorithm == 'ensemble':
            model = EnsembleZoneClassifier()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Filter kwargs based on algorithm
        train_kwargs = self._filter_train_kwargs(algorithm, kwargs)

        self._log(log_callback, "INFO", f"Starting {algorithm} training with params: {train_kwargs}")
        self._log(log_callback, "INFO", f"Training set size: {len(self.y_train)}, Validation set size: {len(self.y_val)}")

        # Train
        metrics = model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            **train_kwargs
        )

        self._log(log_callback, "INFO", f"{algorithm} training finished. Evaluating on test set...")

        # Evaluate on test set
        test_metrics = self._evaluate_model(model, algorithm, log_callback)
        metrics.update(test_metrics)

        # Save final checkpoint
        if checkpoint_manager:
            checkpoint_manager.save_checkpoint(
                model=model,
                epoch=metrics.get('epochs_trained', metrics.get('best_iteration', 0)),
                metrics=metrics,
                algorithm=algorithm
            )

        return model, metrics

    def _evaluate_model(self, model, algorithm: str,
                       log_callback: Optional[Callable] = None) -> Dict:
        """Evaluate model on test set."""
        self._log(log_callback, "INFO", f"⚙️  Predicting on test set ({len(self.X_test)} samples)...")
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)
        self._log(log_callback, "INFO", f"✓ Predictions complete")

        # Accuracy
        self._log(log_callback, "INFO", f"⚙️  Computing test metrics...")
        test_acc = accuracy_score(self.y_test, y_pred)

        # Per-class accuracy
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

        # Classification report
        report = classification_report(
            self.y_test, y_pred,
            target_names=['yellow', 'green', 'purple'],
            output_dict=True
        )

        self._log(log_callback, "INFO", f"\n{'='*60}")
        self._log(log_callback, "INFO", f"📊 {algorithm.upper()} TEST RESULTS")
        self._log(log_callback, "INFO", f"{'='*60}")
        self._log(log_callback, "INFO", f"Overall Accuracy: {test_acc:.1%}")
        self._log(log_callback, "INFO", f"\nPer-zone metrics:")
        for i, zone_name in enumerate(['yellow', 'green', 'purple']):
            emoji = "🟨" if zone_name == "yellow" else ("🟩" if zone_name == "green" else "🟪")
            self._log(log_callback, "INFO",
                     f"  {emoji} {zone_name.upper()}: "
                     f"acc={per_class_acc[i]:.1%}, "
                     f"precision={report[zone_name]['precision']:.1%}, "
                     f"recall={report[zone_name]['recall']:.1%}, "
                     f"f1={report[zone_name]['f1-score']:.3f}")

        self._log(log_callback, "INFO", f"\nConfusion Matrix:")
        self._log(log_callback, "INFO", f"              Predicted")
        self._log(log_callback, "INFO", f"           Y    G    P")
        self._log(log_callback, "INFO", f"Actual Y [{conf_matrix[0,0]:3d} {conf_matrix[0,1]:3d} {conf_matrix[0,2]:3d}]")
        self._log(log_callback, "INFO", f"       G [{conf_matrix[1,0]:3d} {conf_matrix[1,1]:3d} {conf_matrix[1,2]:3d}]")
        self._log(log_callback, "INFO", f"       P [{conf_matrix[2,0]:3d} {conf_matrix[2,1]:3d} {conf_matrix[2,2]:3d}]")
        self._log(log_callback, "INFO", f"{'='*60}\n")

        return {
            'test_accuracy': test_acc,
            'test_accuracy_yellow': per_class_acc[0],
            'test_accuracy_green': per_class_acc[1],
            'test_accuracy_purple': per_class_acc[2],
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report
        }

    def run_full_training_pipeline(
        self,
        algorithms: List[str] = None,
        save_dir: str = 'models/zone_classifiers',
        checkpoint_dir: str = None,
        include_embeddings: bool = True,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict:
        """
        Run complete training pipeline.

        Args:
            algorithms: List of algorithms to train
            save_dir: Directory to save models
            checkpoint_dir: Checkpoint directory
            include_embeddings: Use torchaudio embeddings
            progress_callback: Progress callback
            log_callback: Log callback
            **kwargs: Training parameters

        Returns:
            Results dictionary
        """
        algorithms = algorithms or ['xgboost', 'neural_network', 'ensemble']
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Setup checkpoints
        checkpoint_manager = None
        if checkpoint_dir:
            checkpoint_manager = CheckpointManager(checkpoint_dir)

        results = {}

        try:
            # 1. Load data
            self._log(log_callback, "INFO", "=" * 70)
            self._log(log_callback, "INFO", "ZONE CLASSIFICATION TRAINING PIPELINE")
            self._log(log_callback, "INFO", "=" * 70)

            num_tracks = self.load_training_data(progress_callback, log_callback)
            results['num_tracks'] = num_tracks

            # 2. Extract features
            self.extract_features(
                use_cache=True,
                checkpoint_manager=checkpoint_manager,
                progress_callback=progress_callback,
                log_callback=log_callback
            )

            # 3. Prepare datasets
            self.prepare_datasets(
                include_embeddings=include_embeddings,
                log_callback=log_callback
            )

            # 4. Train models
            for algo in algorithms:
                self._log(log_callback, "INFO", "\n" + "=" * 70)
                self._log(log_callback, "INFO", f"Training {algo.upper()}")
                self._log(log_callback, "INFO", "=" * 70)

                model, metrics = self.train_model(
                    algo,
                    checkpoint_manager=checkpoint_manager,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                    **kwargs
                )

                # Save model
                model_path = save_dir / f"{algo}_final.pkl"
                model.save(str(model_path))

                results[algo] = {
                    **metrics,
                    'model_path': str(model_path)
                }

            # 5. Determine best model
            best_model = max(
                algorithms,
                key=lambda a: results[a]['test_accuracy']
            )
            results['best_model'] = best_model
            results['best_test_accuracy'] = results[best_model]['test_accuracy']

            # Final summary
            self._log(log_callback, "SUCCESS", "\n" + "=" * 70)
            self._log(log_callback, "SUCCESS", "🎉 TRAINING PIPELINE COMPLETE 🎉")
            self._log(log_callback, "SUCCESS", "=" * 70)

            self._log(log_callback, "INFO", f"\n📊 Results Summary:")
            self._log(log_callback, "INFO", f"  Total tracks processed: {results['num_tracks']}")

            self._log(log_callback, "INFO", f"\n🏆 Model Rankings:")
            # Sort by test accuracy
            sorted_algos = sorted(algorithms, key=lambda a: results[a]['test_accuracy'], reverse=True)
            for rank, algo in enumerate(sorted_algos, 1):
                medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
                acc = results[algo]['test_accuracy']
                self._log(log_callback, "INFO",
                         f"  {medal} {rank}. {algo.upper()}: {acc:.1%}")

            # Best model details
            best_results = results[best_model]
            self._log(log_callback, "INFO", f"\n🌟 Best Model: {best_model.upper()}")
            self._log(log_callback, "INFO", f"  Overall Test Accuracy: {best_results['test_accuracy']:.1%}")
            self._log(log_callback, "INFO", f"  Per-zone accuracy:")
            self._log(log_callback, "INFO", f"    🟨 Yellow: {best_results['test_accuracy_yellow']:.1%}")
            self._log(log_callback, "INFO", f"    🟩 Green:  {best_results['test_accuracy_green']:.1%}")
            self._log(log_callback, "INFO", f"    🟪 Purple: {best_results['test_accuracy_purple']:.1%}")

            self._log(log_callback, "INFO", f"\n💾 Saved Models:")
            for algo in algorithms:
                self._log(log_callback, "INFO", f"  • {algo}: {results[algo]['model_path']}")

            self._log(log_callback, "SUCCESS", "=" * 70 + "\n")

            return results

        except Exception as e:
            self._log(log_callback, "ERROR", f"Training failed: {e}")
            raise

    def _save_incremental_checkpoint(self, checkpoint_manager: CheckpointManager,
                                     current: int, total: int,
                                     log_callback: Optional[Callable] = None):
        """
        Save incremental checkpoint during feature extraction.

        Args:
            checkpoint_manager: Checkpoint manager
            current: Current track number
            total: Total tracks
            log_callback: Log callback
        """
        try:
            # Create temporary DataFrame
            temp_df = pd.DataFrame({
                'audio_path': self.audio_paths[:current],
                'zone_label': self.zone_labels[:current],
                'features_list': self.features_list
            })

            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                model=None,
                epoch=current,
                metrics={'progress': current / total},
                algorithm='features_incremental',
                features_df=temp_df
            )

            self._log(log_callback, "INFO", f"💾 Checkpoint saved: {current}/{total} tracks")

        except Exception as e:
            self._log(log_callback, "WARNING", f"Failed to save checkpoint: {e}")

    def _filter_train_kwargs(self, algorithm: str, kwargs: Dict) -> Dict:
        """
        Filter training kwargs based on algorithm type.

        Args:
            algorithm: Algorithm name
            kwargs: All kwargs

        Returns:
            Filtered kwargs for the specific algorithm
        """
        if algorithm == 'xgboost':
            # XGBoost accepts: grid_search, early_stopping_rounds
            allowed = {'grid_search', 'early_stopping_rounds'}
        elif algorithm == 'neural_network':
            # Neural network accepts: epochs, batch_size, patience
            allowed = {'epochs', 'batch_size', 'patience'}
        elif algorithm == 'ensemble':
            # Ensemble accepts all parameters (it will distribute them)
            allowed = {'grid_search', 'early_stopping_rounds', 'epochs', 'batch_size', 'patience'}
        else:
            allowed = set()

        return {k: v for k, v in kwargs.items() if k in allowed}

    def _log(self, callback: Optional[Callable], level: str, message: str):
        """Send log message."""
        # ВСЕГДА выводим в консоль для отладки
        import sys
        print(f"[{level}] {message}", file=sys.stdout, flush=True)

        # И также отправляем в callback для GUI
        if callback:
            callback(level, message)
        else:
            # Use the appropriate logger method directly
            log_method = getattr(logger, level.lower(), logger.info)
            log_method(message)
