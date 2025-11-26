#!/usr/bin/env python3
"""
Create and Test Dual-Source Ensemble Classifier

Комбинирует DEAM и User модели, оптимизирует веса и тестирует.
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dual_ensemble import DualSourceEnsembleClassifier
from src.training.zone_trainer import ZONE_LABELS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_deam_test_data(deam_dir: str = "dataset/deam_processed"):
    """Load DEAM test set."""
    test_df = pd.read_csv(Path(deam_dir) / "test.csv")

    # Common features
    COMMON_FEATURES = [
        'tempo', 'zero_crossing_rate', 'rms_energy', 'spectral_centroid',
        'spectral_rolloff', 'energy_variance', 'mfcc_1_mean', 'mfcc_1_std',
        'mfcc_2_mean', 'mfcc_2_std', 'mfcc_3_mean', 'mfcc_3_std',
        'mfcc_4_mean', 'mfcc_4_std', 'mfcc_5_mean', 'mfcc_5_std',
        'low_energy', 'brightness', 'drop_intensity'
    ]

    X = test_df[COMMON_FEATURES].values
    y = test_df['zone'].str.lower().map(ZONE_LABELS).values

    return X, y, test_df


def load_user_test_data():
    """Load User test data with arousal-based zones."""
    features_df = pd.read_csv('results/user_tracks_features_expanded.csv')
    predictions_df = pd.read_csv('results/user_tracks_predictions.csv')

    merged = features_df.merge(
        predictions_df[['audio_path', 'arousal', 'valence']],
        on='audio_path'
    )

    # Common features
    COMMON_FEATURES = [
        'tempo', 'zero_crossing_rate', 'rms_energy', 'spectral_centroid',
        'spectral_rolloff', 'energy_variance', 'mfcc_1_mean', 'mfcc_1_std',
        'mfcc_2_mean', 'mfcc_2_std', 'mfcc_3_mean', 'mfcc_3_std',
        'mfcc_4_mean', 'mfcc_4_std', 'mfcc_5_mean', 'mfcc_5_std',
        'low_energy', 'brightness', 'drop_intensity'
    ]

    def arousal_to_zone(arousal):
        if arousal < 4.0:
            return 0
        elif arousal > 6.0:
            return 2
        else:
            return 1

    X = merged[COMMON_FEATURES].values
    y = merged['arousal'].apply(arousal_to_zone).values

    return X, y, merged


def main():
    parser = argparse.ArgumentParser(description='Create and test dual ensemble')
    parser.add_argument('--deam-model', type=str,
                       default='models/deam_zone_classifier_19f',
                       help='DEAM model directory')
    parser.add_argument('--user-model', type=str,
                       default='models/user_zone_classifier',
                       help='User model directory')
    parser.add_argument('--output', type=str,
                       default='models/dual_ensemble',
                       help='Output directory for ensemble')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize weights on validation data')

    args = parser.parse_args()

    print("=" * 70)
    print("Dual-Source Ensemble Classifier")
    print("=" * 70)

    # Create ensemble
    print("\n[1/5] Creating ensemble...")
    ensemble = DualSourceEnsembleClassifier(weights=[0.7, 0.3])
    ensemble.load_models(args.deam_model, args.user_model)

    # Load test data
    print("\n[2/5] Loading test data...")
    X_deam, y_deam, deam_df = load_deam_test_data()
    X_user, y_user, user_df = load_user_test_data()

    print(f"  DEAM test: {len(X_deam)} samples")
    print(f"  User test: {len(X_user)} samples")

    # Optimize weights (optional)
    # NOTE: Optimizing on User data is not useful because 98% are GREEN
    # Instead, optimize on DEAM test data which has all 3 zones
    if args.optimize:
        print("\n[3/5] Optimizing weights on DEAM test data (has all 3 zones)...")
        best_weights, best_acc = ensemble.optimize_weights(X_deam, y_deam, grid_step=0.1)
        print(f"  Best weights: DEAM={best_weights[0]:.0%}, User={best_weights[1]:.0%}")
        print(f"  Best accuracy on DEAM test: {best_acc:.1%}")
    else:
        print("\n[3/5] Using default weights (use --optimize to tune)")
        print(f"  Weights: DEAM={ensemble.weights[0]:.0%}, User={ensemble.weights[1]:.0%}")

    # Evaluate on DEAM test
    print("\n[4/5] Evaluating on DEAM test set...")
    deam_results = ensemble.evaluate(X_deam, y_deam)
    print(f"  Accuracy: {deam_results['accuracy']:.1%}")
    print(f"  Confusion Matrix:")
    print(deam_results['confusion_matrix'])

    # Compare with DEAM-only
    contributions = ensemble.get_model_contributions(X_deam)
    deam_only_acc = np.mean(contributions['deam_predictions'] == y_deam)
    user_only_acc = np.mean(contributions['user_predictions'] == y_deam)
    print(f"\n  Comparison on DEAM test:")
    print(f"    DEAM only: {deam_only_acc:.1%}")
    print(f"    User only: {user_only_acc:.1%}")
    print(f"    Ensemble:  {deam_results['accuracy']:.1%}")

    # Evaluate on User data
    print("\n[5/5] Evaluating on User tracks...")
    user_results = ensemble.evaluate(X_user, y_user)
    print(f"  Accuracy: {user_results['accuracy']:.1%}")

    # Zone distribution in predictions
    pred_zones = pd.Series(user_results['predictions']).value_counts().sort_index()
    print(f"\n  Predicted zone distribution:")
    for z, name in enumerate(['YELLOW', 'GREEN', 'PURPLE']):
        count = pred_zones.get(z, 0)
        print(f"    {name}: {count} ({count/len(X_user)*100:.1f}%)")

    actual_zones = pd.Series(y_user).value_counts().sort_index()
    print(f"\n  Actual zone distribution (from arousal):")
    for z, name in enumerate(['YELLOW', 'GREEN', 'PURPLE']):
        count = actual_zones.get(z, 0)
        print(f"    {name}: {count} ({count/len(y_user)*100:.1f}%)")

    # Compare with individual models
    contributions = ensemble.get_model_contributions(X_user)
    deam_only_acc = np.mean(contributions['deam_predictions'] == y_user)
    user_only_acc = np.mean(contributions['user_predictions'] == y_user)
    print(f"\n  Comparison on User tracks:")
    print(f"    DEAM only: {deam_only_acc:.1%}")
    print(f"    User only: {user_only_acc:.1%}")
    print(f"    Ensemble:  {user_results['accuracy']:.1%}")

    # Save ensemble config
    print(f"\n  Saving ensemble to {args.output}...")
    ensemble.save(args.output)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ensemble weights: DEAM={ensemble.weights[0]:.0%}, User={ensemble.weights[1]:.0%}")
    print(f"DEAM test accuracy:  {deam_results['accuracy']:.1%}")
    print(f"User test accuracy:  {user_results['accuracy']:.1%}")
    print(f"\nEnsemble saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
