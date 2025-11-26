#!/usr/bin/env python3
"""
Train Zone Classifier on Frame-Level Features

Supports two approaches:
1. Frame-level: Classify individual frames (majority vote for track)
2. Aggregated: Aggregate frame features per track, then classify

Usage:
    python scripts/train_on_frames.py \
        --user results/user_50_frames.pkl \
        --deam results/deam_50_frames.pkl \
        --output models/frame_model.pkl \
        --approach aggregated
"""

import argparse
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.unified_features import FRAME_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Features to use (exclude metadata columns)
FEATURE_COLS = [f for f in FRAME_FEATURES if f != 'frameTime']


def load_and_merge_datasets(
    user_path: Path,
    deam_path: Path,
    annotations_path: Optional[Path] = None
) -> pd.DataFrame:
    """Load and merge user and DEAM frame datasets."""
    logger.info(f"Loading user frames from: {user_path}")
    user_df = pd.read_pickle(user_path)
    logger.info(f"  {len(user_df)} frames, {user_df['track_id'].nunique()} tracks")

    logger.info(f"Loading DEAM frames from: {deam_path}")
    deam_df = pd.read_pickle(deam_path)
    logger.info(f"  {len(deam_df)} frames, {deam_df['track_id'].nunique()} tracks")

    # Add zones to DEAM if needed
    if 'zone' not in deam_df.columns or deam_df['zone'].isna().all():
        if annotations_path and annotations_path.exists():
            logger.info(f"Loading annotations from: {annotations_path}")
            annotations = pd.read_csv(annotations_path)
            annotations.columns = [c.strip() for c in annotations.columns]

            # Map arousal to zone
            def arousal_to_zone(arousal):
                if pd.isna(arousal):
                    return 'GREEN'
                if arousal < 4.0:
                    return 'YELLOW'
                elif arousal > 6.0:
                    return 'PURPLE'
                return 'GREEN'

            track_zones = {}
            for _, row in annotations.iterrows():
                track_id = int(row['song_id'])
                arousal = row.get('arousal_mean', 5.0)
                track_zones[track_id] = arousal_to_zone(arousal)

            deam_df['zone'] = deam_df['track_id'].map(track_zones)
        else:
            logger.warning("No annotations provided, DEAM zones will be None")

    # Combine
    combined = pd.concat([user_df, deam_df], ignore_index=True)
    logger.info(f"Combined: {len(combined)} frames, {combined['track_id'].nunique()} tracks")

    return combined


def aggregate_frames_per_track(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate frame features to track level.

    For each feature, compute:
    - mean, std, min, max
    - percentiles (10, 25, 50, 75, 90)
    - Special: drop_candidate_sum, energy_peak_sum, energy_valley_sum
    """
    logger.info("Aggregating frames per track...")

    # Group by track
    grouped = df.groupby(['track_id', 'source', 'zone'])

    records = []

    for (track_id, source, zone), group in grouped:
        record = {
            'track_id': track_id,
            'source': source,
            'zone': zone,
            'n_frames': len(group)
        }

        for feat in FEATURE_COLS:
            if feat not in group.columns:
                continue

            values = group[feat].dropna().values
            if len(values) == 0:
                continue

            # Basic stats
            record[f'{feat}_mean'] = np.mean(values)
            record[f'{feat}_std'] = np.std(values)
            record[f'{feat}_min'] = np.min(values)
            record[f'{feat}_max'] = np.max(values)

            # Percentiles
            record[f'{feat}_p10'] = np.percentile(values, 10)
            record[f'{feat}_p50'] = np.percentile(values, 50)
            record[f'{feat}_p90'] = np.percentile(values, 90)

            # For binary features, sum is more meaningful
            if feat in ['drop_candidate', 'energy_peak', 'energy_valley', 'beat_sync', 'low_energy_flag']:
                record[f'{feat}_sum'] = np.sum(values)
                record[f'{feat}_rate'] = np.mean(values)  # Fraction of frames

        records.append(record)

    agg_df = pd.DataFrame(records)
    logger.info(f"Aggregated to {len(agg_df)} tracks")

    return agg_df


def prepare_features(
    df: pd.DataFrame,
    approach: str = 'aggregated'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix and labels.

    Args:
        df: DataFrame with features
        approach: 'frame' or 'aggregated'

    Returns:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
    """
    if approach == 'aggregated':
        # Use all numeric columns except metadata
        meta_cols = ['track_id', 'source', 'zone', 'n_frames', 'path']
        feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in ['float64', 'int64']]
    else:
        # Frame-level: use FEATURE_COLS
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    X = df[feature_cols].values
    y = df['zone'].values

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_cols


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5
) -> Dict:
    """
    Train and evaluate classifier with cross-validation.

    Returns:
        Dictionary with results
    """
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rf_scores = cross_val_score(rf, X_scaled, y_encoded, cv=skf, scoring='accuracy')

    logger.info(f"RF CV Accuracy: {rf_scores.mean():.2%} (+/- {rf_scores.std():.2%})")

    # Train on full data for feature importance
    rf.fit(X_scaled, y_encoded)

    # Feature importance
    fi = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Full training predictions for confusion matrix
    y_pred = rf.predict(X_scaled)

    results = {
        'model': rf,
        'scaler': scaler,
        'label_encoder': le,
        'cv_scores': rf_scores,
        'cv_mean': rf_scores.mean(),
        'cv_std': rf_scores.std(),
        'feature_importance': fi,
        'classification_report': classification_report(y_encoded, y_pred, target_names=le.classes_),
        'confusion_matrix': confusion_matrix(y_encoded, y_pred),
        'classes': le.classes_,
        'feature_names': feature_names
    }

    return results


def print_results(results: Dict, approach: str):
    """Print training results."""
    print("\n" + "=" * 60)
    print(f"TRAINING RESULTS ({approach.upper()} APPROACH)")
    print("=" * 60)

    print(f"\nCross-validation accuracy: {results['cv_mean']:.2%} (+/- {results['cv_std']:.2%})")
    print(f"Number of features: {len(results['feature_names'])}")

    print("\nTop 15 features:")
    for _, row in results['feature_importance'].head(15).iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.4f}")

    print("\nClassification Report:")
    print(results['classification_report'])

    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"         {' '.join(f'{c:>8s}' for c in results['classes'])}")
    for i, cls in enumerate(results['classes']):
        print(f"{cls:8s} {' '.join(f'{cm[i,j]:8d}' for j in range(len(results['classes'])))}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Train zone classifier on frame-level features'
    )

    parser.add_argument(
        '--user', '-u',
        type=Path,
        required=True,
        help='Path to user frames pickle file'
    )
    parser.add_argument(
        '--deam', '-d',
        type=Path,
        required=True,
        help='Path to DEAM frames pickle file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output path for trained model'
    )
    parser.add_argument(
        '--annotations', '-a',
        type=Path,
        default=None,
        help='Path to DEAM annotations CSV'
    )
    parser.add_argument(
        '--approach',
        choices=['frame', 'aggregated'],
        default='aggregated',
        help='Training approach: frame (per-frame) or aggregated (per-track)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )

    args = parser.parse_args()

    # Load data
    combined = load_and_merge_datasets(
        args.user,
        args.deam,
        args.annotations
    )

    # Filter tracks with valid zones
    valid_zones = ['GREEN', 'PURPLE', 'YELLOW']
    combined = combined[combined['zone'].isin(valid_zones)]

    if args.approach == 'aggregated':
        # Aggregate to track level
        df = aggregate_frames_per_track(combined)
    else:
        # Use frame-level data directly
        df = combined

    # Prepare features
    X, y, feature_names = prepare_features(df, args.approach)

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

    # Train and evaluate
    results = train_and_evaluate(X, y, feature_names, args.cv_folds)

    # Print results
    print_results(results, args.approach)

    # Save model
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump({
                'model': results['model'],
                'scaler': results['scaler'],
                'label_encoder': results['label_encoder'],
                'feature_names': results['feature_names'],
                'approach': args.approach,
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std']
            }, f)
        logger.info(f"Model saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
