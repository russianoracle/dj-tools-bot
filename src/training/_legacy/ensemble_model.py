#!/usr/bin/env python3
"""
Ensemble model combining:
1. Frame-based XGBoost (79 features per frame → track prediction)
2. Spectrogram CNN (visual energy/drops pattern recognition)

Strategies:
- voting: majority vote from both models
- weighted: weighted average of probabilities
- stacking: meta-classifier on top of both predictions
- cascade: use CNN only when XGBoost is uncertain
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Literal
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from collections import Counter

logger = logging.getLogger(__name__)

ZONES = ['YELLOW', 'GREEN', 'PURPLE']
ZONE_TO_IDX = {z: i for i, z in enumerate(ZONES)}
IDX_TO_ZONE = {i: z for i, z in enumerate(ZONES)}


class EnsembleZoneClassifier:
    """
    Ensemble classifier combining frame-based and spectrogram-based models.
    """

    def __init__(self,
                 frame_model_path: Optional[str] = None,
                 cnn_model_path: Optional[str] = None,
                 strategy: Literal['voting', 'weighted', 'stacking', 'cascade'] = 'weighted',
                 frame_weight: float = 0.6,
                 cnn_weight: float = 0.4,
                 confidence_threshold: float = 0.6):
        """
        Args:
            frame_model_path: Path to trained XGBoost frame model
            cnn_model_path: Path to trained CNN model
            strategy: Ensemble strategy
            frame_weight: Weight for frame model (for 'weighted' strategy)
            cnn_weight: Weight for CNN model (for 'weighted' strategy)
            confidence_threshold: Threshold for 'cascade' strategy
        """
        self.strategy = strategy
        self.frame_weight = frame_weight
        self.cnn_weight = cnn_weight
        self.confidence_threshold = confidence_threshold

        self.frame_model = None
        self.cnn_model = None
        self.meta_classifier = None  # For stacking

        if frame_model_path:
            self.load_frame_model(frame_model_path)
        if cnn_model_path:
            self.load_cnn_model(cnn_model_path)

    def load_frame_model(self, path: str):
        """Load XGBoost frame-based model."""
        with open(path, 'rb') as f:
            self.frame_model = pickle.load(f)
        logger.info(f"Loaded frame model: {path}")

    def load_cnn_model(self, path: str):
        """Load CNN spectrogram model."""
        from src.training.spectrogram_cnn import SpectrogramCNN
        self.cnn_model = SpectrogramCNN(model_path=path)
        logger.info(f"Loaded CNN model: {path}")

    def _predict_frame_model(self, frame_features: np.ndarray,
                            track_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with frame model, aggregating to track level.

        Args:
            frame_features: (n_frames, n_features)
            track_ids: (n_frames,) track ID for each frame

        Returns:
            predictions: (n_tracks,) predicted zone indices
            probabilities: (n_tracks, 3) class probabilities
        """
        if self.frame_model is None:
            raise ValueError("Frame model not loaded")

        # Get frame-level predictions
        frame_probs = self.frame_model.predict_proba(frame_features)

        # Aggregate by track
        unique_tracks = np.unique(track_ids)
        track_probs = []
        track_preds = []

        for tid in unique_tracks:
            mask = track_ids == tid
            # Average probabilities for this track
            avg_probs = np.mean(frame_probs[mask], axis=0)
            track_probs.append(avg_probs)
            track_preds.append(np.argmax(avg_probs))

        return np.array(track_preds), np.array(track_probs)

    def _predict_cnn_model(self, spectrograms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with CNN model.

        Args:
            spectrograms: (n_tracks, channels, height, width)

        Returns:
            predictions: (n_tracks,) predicted zone indices
            probabilities: (n_tracks, 3) class probabilities
        """
        if self.cnn_model is None:
            raise ValueError("CNN model not loaded")

        probs = self.cnn_model.predict_proba(spectrograms)
        preds = np.argmax(probs, axis=1)

        return preds, probs

    def predict(self,
               frame_features: Optional[np.ndarray] = None,
               track_ids: Optional[np.ndarray] = None,
               spectrograms: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict zone for tracks using ensemble.

        Args:
            frame_features: Frame features (n_frames, n_features)
            track_ids: Track ID for each frame
            spectrograms: Spectrogram tensors (n_tracks, C, H, W)

        Returns:
            Zone predictions as indices
        """
        probs = self.predict_proba(frame_features, track_ids, spectrograms)
        return np.argmax(probs, axis=1)

    def predict_proba(self,
                     frame_features: Optional[np.ndarray] = None,
                     track_ids: Optional[np.ndarray] = None,
                     spectrograms: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict probabilities using ensemble.

        Returns:
            (n_tracks, 3) probability array
        """
        frame_probs = None
        cnn_probs = None

        # Get predictions from each model
        if frame_features is not None and track_ids is not None and self.frame_model:
            _, frame_probs = self._predict_frame_model(frame_features, track_ids)

        if spectrograms is not None and self.cnn_model:
            _, cnn_probs = self._predict_cnn_model(spectrograms)

        # Handle missing models
        if frame_probs is None and cnn_probs is None:
            raise ValueError("At least one model must be loaded and have input data")

        if frame_probs is None:
            return cnn_probs
        if cnn_probs is None:
            return frame_probs

        # Combine predictions based on strategy
        if self.strategy == 'voting':
            return self._voting_combine(frame_probs, cnn_probs)

        elif self.strategy == 'weighted':
            return self._weighted_combine(frame_probs, cnn_probs)

        elif self.strategy == 'stacking':
            return self._stacking_combine(frame_probs, cnn_probs)

        elif self.strategy == 'cascade':
            return self._cascade_combine(frame_probs, cnn_probs)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _voting_combine(self, frame_probs: np.ndarray,
                       cnn_probs: np.ndarray) -> np.ndarray:
        """Majority voting - each model gets one vote."""
        frame_preds = np.argmax(frame_probs, axis=1)
        cnn_preds = np.argmax(cnn_probs, axis=1)

        # Create pseudo-probabilities from votes
        n_tracks = len(frame_preds)
        combined = np.zeros((n_tracks, 3))

        for i in range(n_tracks):
            votes = [frame_preds[i], cnn_preds[i]]
            for v in votes:
                combined[i, v] += 0.5

        return combined

    def _weighted_combine(self, frame_probs: np.ndarray,
                         cnn_probs: np.ndarray) -> np.ndarray:
        """Weighted average of probabilities."""
        total_weight = self.frame_weight + self.cnn_weight
        return (self.frame_weight * frame_probs + self.cnn_weight * cnn_probs) / total_weight

    def _stacking_combine(self, frame_probs: np.ndarray,
                         cnn_probs: np.ndarray) -> np.ndarray:
        """Use meta-classifier on concatenated probabilities."""
        if self.meta_classifier is None:
            # Fallback to weighted if meta not trained
            logger.warning("Meta-classifier not trained, using weighted average")
            return self._weighted_combine(frame_probs, cnn_probs)

        # Concatenate probabilities as features
        meta_features = np.concatenate([frame_probs, cnn_probs], axis=1)
        return self.meta_classifier.predict_proba(meta_features)

    def _cascade_combine(self, frame_probs: np.ndarray,
                        cnn_probs: np.ndarray) -> np.ndarray:
        """Use CNN only when frame model is uncertain."""
        n_tracks = len(frame_probs)
        combined = np.zeros((n_tracks, 3))

        for i in range(n_tracks):
            frame_conf = np.max(frame_probs[i])

            if frame_conf >= self.confidence_threshold:
                # Frame model is confident
                combined[i] = frame_probs[i]
            else:
                # Use CNN for uncertain cases
                combined[i] = cnn_probs[i]

        return combined

    def fit_stacking(self,
                    frame_features: np.ndarray,
                    track_ids: np.ndarray,
                    spectrograms: np.ndarray,
                    y: np.ndarray):
        """
        Train meta-classifier for stacking strategy.

        Args:
            frame_features: Frame features
            track_ids: Track IDs
            spectrograms: Spectrogram tensors
            y: True labels (zone indices)
        """
        _, frame_probs = self._predict_frame_model(frame_features, track_ids)
        _, cnn_probs = self._predict_cnn_model(spectrograms)

        # Concatenate as meta-features
        meta_features = np.concatenate([frame_probs, cnn_probs], axis=1)

        # Train logistic regression as meta-classifier
        self.meta_classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.meta_classifier.fit(meta_features, y)

        # Evaluate
        preds = self.meta_classifier.predict(meta_features)
        acc = accuracy_score(y, preds)
        logger.info(f"Meta-classifier trained, accuracy: {acc:.4f}")

    def evaluate(self,
                frame_features: Optional[np.ndarray] = None,
                track_ids: Optional[np.ndarray] = None,
                spectrograms: Optional[np.ndarray] = None,
                y: np.ndarray = None) -> Dict:
        """Evaluate ensemble performance."""
        preds = self.predict(frame_features, track_ids, spectrograms)

        results = {
            'accuracy': accuracy_score(y, preds),
            'confusion_matrix': confusion_matrix(y, preds),
            'classification_report': classification_report(
                y, preds,
                target_names=ZONES,
                output_dict=True
            )
        }

        # Per-model performance if both available
        if frame_features is not None and self.frame_model:
            frame_preds, _ = self._predict_frame_model(frame_features, track_ids)
            results['frame_accuracy'] = accuracy_score(y, frame_preds)

        if spectrograms is not None and self.cnn_model:
            cnn_preds, _ = self._predict_cnn_model(spectrograms)
            results['cnn_accuracy'] = accuracy_score(y, cnn_preds)

        return results

    def save(self, path: str):
        """Save ensemble model."""
        save_dict = {
            'strategy': self.strategy,
            'frame_weight': self.frame_weight,
            'cnn_weight': self.cnn_weight,
            'confidence_threshold': self.confidence_threshold,
            'meta_classifier': self.meta_classifier,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        logger.info(f"Ensemble saved: {path}")

    def load(self, path: str):
        """Load ensemble settings (models loaded separately)."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.strategy = save_dict['strategy']
        self.frame_weight = save_dict['frame_weight']
        self.cnn_weight = save_dict['cnn_weight']
        self.confidence_threshold = save_dict['confidence_threshold']
        self.meta_classifier = save_dict.get('meta_classifier')

        logger.info(f"Ensemble loaded: {path}")


def compare_strategies(frame_features: np.ndarray,
                      track_ids: np.ndarray,
                      spectrograms: np.ndarray,
                      y: np.ndarray,
                      frame_model_path: str,
                      cnn_model_path: str) -> Dict:
    """
    Compare all ensemble strategies.

    Returns:
        Dict with accuracy for each strategy
    """
    results = {}

    strategies = ['voting', 'weighted', 'cascade']

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")

        ensemble = EnsembleZoneClassifier(
            frame_model_path=frame_model_path,
            cnn_model_path=cnn_model_path,
            strategy=strategy
        )

        eval_results = ensemble.evaluate(
            frame_features, track_ids, spectrograms, y
        )

        results[strategy] = {
            'accuracy': eval_results['accuracy'],
            'frame_accuracy': eval_results.get('frame_accuracy'),
            'cnn_accuracy': eval_results.get('cnn_accuracy'),
        }

        print(f"Ensemble accuracy: {eval_results['accuracy']:.4f}")
        if 'frame_accuracy' in eval_results:
            print(f"Frame model only: {eval_results['frame_accuracy']:.4f}")
        if 'cnn_accuracy' in eval_results:
            print(f"CNN model only: {eval_results['cnn_accuracy']:.4f}")

    # Stacking (needs training)
    print(f"\n--- Strategy: stacking ---")
    ensemble = EnsembleZoneClassifier(
        frame_model_path=frame_model_path,
        cnn_model_path=cnn_model_path,
        strategy='stacking'
    )
    ensemble.fit_stacking(frame_features, track_ids, spectrograms, y)
    eval_results = ensemble.evaluate(frame_features, track_ids, spectrograms, y)
    results['stacking'] = {'accuracy': eval_results['accuracy']}
    print(f"Ensemble accuracy: {eval_results['accuracy']:.4f}")

    # Find best
    best_strategy = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\n{'='*50}")
    print(f"Best strategy: {best_strategy} ({results[best_strategy]['accuracy']:.4f})")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble zone classifier")
    parser.add_argument("--frame-model", required=True, help="Path to frame model")
    parser.add_argument("--cnn-model", required=True, help="Path to CNN model")
    parser.add_argument("--frame-data", required=True, help="Path to frame features pkl")
    parser.add_argument("--spec-data", required=True, help="Path to spectrogram dataset pkl")
    parser.add_argument("--strategy", default="weighted",
                       choices=['voting', 'weighted', 'stacking', 'cascade'],
                       help="Ensemble strategy")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all strategies")

    args = parser.parse_args()

    # Load data
    import pandas as pd

    print("Loading frame features...")
    frame_df = pd.read_pickle(args.frame_data)

    exclude_cols = ['zone', 'track_id', 'frame_idx', 'source', 'path', 'file', 'filename', 'source_dataset']
    feature_cols = [c for c in frame_df.select_dtypes(include=[np.number]).columns
                   if c not in exclude_cols]

    frame_features = frame_df[feature_cols].values
    track_ids = frame_df['track_id'].values
    frame_zones = frame_df.groupby('track_id')['zone'].first()

    print("Loading spectrogram data...")
    with open(args.spec_data, 'rb') as f:
        spec_dataset = pickle.load(f)

    spectrograms = np.array([d['tensor'] for d in spec_dataset])
    spec_zones = [d['zone'] for d in spec_dataset]

    # Align data (ensure same tracks)
    # This is simplified - in practice need to match by path
    y = np.array([ZONE_TO_IDX[z] for z in frame_zones.values])

    if args.compare:
        compare_strategies(
            frame_features, track_ids, spectrograms, y,
            args.frame_model, args.cnn_model
        )
    else:
        ensemble = EnsembleZoneClassifier(
            frame_model_path=args.frame_model,
            cnn_model_path=args.cnn_model,
            strategy=args.strategy
        )

        results = ensemble.evaluate(frame_features, track_ids, spectrograms, y)
        print(f"\nEnsemble accuracy ({args.strategy}): {results['accuracy']:.4f}")
        print("\nClassification Report:")
        for zone, metrics in results['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"  {zone}: precision={metrics['precision']:.2f}, "
                      f"recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}")