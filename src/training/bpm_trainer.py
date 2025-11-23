"""Main BPM correction model trainer."""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from sklearn.model_selection import train_test_split

from ..audio import AudioLoader
from ..utils import get_logger, get_config
from .tempo_features import TempoFeatureExtractor
from .models import XGBoostBPMModel, NeuralBPMModel, EnsembleBPMModel

logger = get_logger(__name__)


class BPMTrainer:
    """Coordinates BPM correction model training pipeline."""

    def __init__(self, test_data_path: str, config: Any = None):
        """
        Initialize BPM trainer.

        Args:
            test_data_path: Path to labeled data file (TSV)
            config: Configuration object
        """
        self.test_data_path = Path(test_data_path)
        self.config = config or get_config()

        sample_rate = self.config.get('audio.sample_rate', 22050)
        self.audio_loader = AudioLoader(sample_rate=sample_rate)
        self.feature_extractor = TempoFeatureExtractor(sr=sample_rate)

        logger.info(f"BPM Trainer initialized with data: {self.test_data_path}")

    def load_training_data(self) -> List[Dict[str, Any]]:
        """Load labeled training data from TSV file."""
        import csv

        logger.info(f"Loading training data from {self.test_data_path}")

        tracks = []
        with open(self.test_data_path, 'r', encoding='utf-16') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for row in reader:
                try:
                    bpm = float(row.get('BPM', '').strip())
                    file_path = (row.get('Location', '').strip() or
                                row.get('Path', '').strip() or
                                row.get('путь', '').strip())

                    if not file_path or not Path(file_path).exists():
                        continue

                    tracks.append({
                        'path': file_path,
                        'bpm': bpm,
                        'track_title': row.get('Track Title', '').strip(),
                        'artist': row.get('Artist', '').strip()
                    })
                except (ValueError, KeyError):
                    continue

        logger.info(f"Loaded {len(tracks)} valid tracks")
        return tracks

    def extract_training_features(self, tracks: List[Dict[str, Any]],
                                  progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Extract features from all tracks.

        Args:
            tracks: List of track dictionaries
            progress_callback: Optional callback(current, total, track_name)

        Returns:
            DataFrame with features and target BPM
        """
        logger.info(f"Extracting features from {len(tracks)} tracks...")

        features_list = []
        total = len(tracks)

        for idx, track in enumerate(tracks, 1):
            if progress_callback:
                progress_callback(idx, total, Path(track['path']).name)

            try:
                # Load audio
                y, sr = self.audio_loader.load(track['path'])

                # Extract tempo features
                tempo_features = self.feature_extractor.extract(y, sr)

                # Add target BPM
                feature_dict = tempo_features.to_dict()
                feature_dict['target_bpm'] = track['bpm']
                feature_dict['file_path'] = track['path']

                features_list.append(feature_dict)

            except Exception as e:
                logger.error(f"Failed to extract features from {track['path']}: {e}")
                continue

        df = pd.DataFrame(features_list)
        logger.info(f"Extracted features from {len(df)} tracks successfully")

        return df

    def prepare_train_val_test_split(self, features_df: pd.DataFrame,
                                     train_size: float = 0.7,
                                     val_size: float = 0.15,
                                     test_size: float = 0.15,
                                     random_state: int = 42) -> Tuple[pd.DataFrame, ...]:
        """
        Split data into train/val/test sets.

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separate features and target
        feature_cols = [col for col in features_df.columns if col not in ['target_bpm', 'file_path']]
        X = features_df[feature_cols]
        y = features_df['target_bpm']

        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), random_state=random_state
        )

        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state
        )

        logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, algorithm: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   progress_callback: Optional[Callable] = None,
                   **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a specific model.

        Args:
            algorithm: 'xgboost', 'neural_network', or 'ensemble'
            X_train, y_train: Training data
            X_val, y_val: Validation data
            progress_callback: Optional callback for progress updates
            **kwargs: Additional training parameters

        Returns:
            (trained_model, training_results)
        """
        logger.info(f"Training {algorithm} model...")

        if algorithm == 'xgboost':
            model = XGBoostBPMModel()
        elif algorithm == 'neural_network':
            model = NeuralBPMModel()
        elif algorithm == 'ensemble':
            model = EnsembleBPMModel()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        results = model.train(X_train, y_train, X_val, y_val, **kwargs)

        return model, results

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X_test)

        # Calculate metrics
        errors = np.abs(y_test.values - y_pred)
        mae = np.mean(errors)
        median_error = np.median(errors)

        # Accuracy within thresholds
        accuracy_2bpm = np.mean(errors <= 2.0) * 100
        accuracy_5bpm = np.mean(errors <= 5.0) * 100

        metrics = {
            'test_mae': mae,
            'test_median_error': median_error,
            'accuracy_within_2bpm': accuracy_2bpm,
            'accuracy_within_5bpm': accuracy_5bpm
        }

        logger.info(f"Test MAE: {mae:.2f} BPM, Accuracy (±2 BPM): {accuracy_2bpm:.1f}%")

        return metrics

    def run_full_training_pipeline(self,
                                   algorithms: List[str] = ['xgboost', 'neural_network', 'ensemble'],
                                   save_dir: str = 'models/bpm_correctors',
                                   progress_callback: Optional[Callable] = None,
                                   log_callback: Optional[Callable] = None,
                                   **kwargs) -> Dict[str, Any]:
        """
        Run complete training pipeline.

        Args:
            algorithms: List of algorithms to train
            save_dir: Directory to save trained models
            progress_callback: Callback for progress updates
            log_callback: Callback for log messages

        Returns:
            Dictionary with all results
        """
        results = {}

        try:
            # Step 1: Load data
            if log_callback:
                log_callback("INFO", "Loading training data...")
            tracks = self.load_training_data()

            # Step 2: Extract features
            if log_callback:
                log_callback("INFO", f"Extracting features from {len(tracks)} tracks...")
            features_df = self.extract_training_features(tracks, progress_callback)

            # Step 3: Split data
            if log_callback:
                log_callback("INFO", "Splitting into train/val/test sets...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_train_val_test_split(features_df)

            # Baseline evaluation (librosa without correction)
            baseline_mae = np.mean(np.abs(features_df['target_bpm'] - features_df['detected_bpm']))
            results['baseline_mae'] = baseline_mae
            if log_callback:
                log_callback("INFO", f"Baseline MAE (librosa): {baseline_mae:.2f} BPM")

            # Step 4: Train each algorithm
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            for algo in algorithms:
                if log_callback:
                    log_callback("INFO", f"Training {algo} model...")

                model, train_results = self.train_model(algo, X_train, y_train, X_val, y_val, **kwargs)

                # Evaluate
                test_metrics = self.evaluate_model(model, X_test, y_test)

                # Save model
                model_path = Path(save_dir) / f"{algo}_v1.pkl"
                model.save(str(model_path))

                results[algo] = {
                    **train_results,
                    **test_metrics,
                    'model_path': str(model_path)
                }

                if log_callback:
                    log_callback("SUCCESS", f"{algo}: Test MAE = {test_metrics['test_mae']:.2f} BPM, "
                                           f"Accuracy = {test_metrics['accuracy_within_2bpm']:.1f}%")

            # Determine best model
            best_algo = min(algorithms, key=lambda a: results[a]['test_mae'])
            results['best_model'] = best_algo

            if log_callback:
                log_callback("SUCCESS", f"Training complete! Best model: {best_algo}")

        except Exception as e:
            logger.exception("Training pipeline failed")
            if log_callback:
                log_callback("ERROR", f"Training failed: {str(e)}")
            raise

        return results

    def save_features_to_csv(self, features_df: pd.DataFrame, output_path: str):
        """Save extracted features to CSV for analysis."""
        features_df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
