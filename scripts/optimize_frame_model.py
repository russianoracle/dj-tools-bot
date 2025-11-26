#!/usr/bin/env python3
"""
Optimize Frame-Based Zone Classifier

Approaches:
1. Feature selection (top N features)
2. Class balancing (SMOTE, class weights)
3. Multiple algorithms (RF, XGBoost, GradientBoosting)
4. Hyperparameter tuning (GridSearchCV)
5. Source normalization (User vs DEAM)

Usage:
    python scripts/optimize_frame_model.py \
        --user results/user_200_frames.pkl \
        --deam results/deam_200_frames.pkl \
        --output models/optimized_model.pkl
"""

import argparse
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, RFE
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

# Try to import optional libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.warning("imbalanced-learn not available, class balancing limited")


def load_and_merge(user_path: Path, deam_path: Path) -> pd.DataFrame:
    """Load and merge user and DEAM frames."""
    logger.info(f"Loading user frames: {user_path}")
    user_df = pd.read_pickle(user_path)

    logger.info(f"Loading DEAM frames: {deam_path}")
    deam_df = pd.read_pickle(deam_path)

    combined = pd.concat([user_df, deam_df], ignore_index=True)

    # Filter valid zones
    valid_zones = ['GREEN', 'PURPLE', 'YELLOW']
    combined = combined[combined['zone'].isin(valid_zones)]

    logger.info(f"Combined: {len(combined)} frames, {combined['track_id'].nunique()} tracks")
    return combined


def aggregate_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frame features to track level."""
    logger.info("Aggregating frames per track...")

    feature_cols = [f for f in FRAME_FEATURES if f != 'frameTime' and f in df.columns]

    grouped = df.groupby(['track_id', 'source', 'zone'])
    records = []

    for (track_id, source, zone), group in grouped:
        record = {
            'track_id': track_id,
            'source': source,
            'zone': zone,
            'n_frames': len(group)
        }

        for feat in feature_cols:
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

            # Range and IQR
            record[f'{feat}_range'] = np.max(values) - np.min(values)

            # For binary features
            if feat in ['drop_candidate', 'energy_peak', 'energy_valley', 'beat_sync', 'low_energy_flag']:
                record[f'{feat}_sum'] = np.sum(values)
                record[f'{feat}_rate'] = np.mean(values)

        records.append(record)

    agg_df = pd.DataFrame(records)
    agg_df = add_derived_drop_features(agg_df)
    logger.info(f"Aggregated to {len(agg_df)} tracks with derived features")
    return agg_df


def add_derived_drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized drop features for better zone discrimination."""
    df = df.copy()
    n_added = 0

    # === Only normalized features (key insight from analysis) ===

    # 1. Normalized buildup max: buildup relative to track's energy level
    if 'energy_buildup_score_max' in df.columns and 'rms_energy_mean' in df.columns:
        df['normalized_buildup_max'] = df['energy_buildup_score_max'] / (df['rms_energy_mean'] + 1e-6)
        n_added += 1

    # 2. Normalized buildup mean
    if 'energy_buildup_score_mean' in df.columns and 'rms_energy_mean' in df.columns:
        df['normalized_buildup_mean'] = df['energy_buildup_score_mean'] / (df['rms_energy_mean'] + 1e-6)
        n_added += 1

    # 3. Relative drop intensity: drop strength relative to energy variance
    if all(c in df.columns for c in ['energy_buildup_score_max', 'energy_buildup_score_mean', 'rms_energy_std']):
        drop_intensity = df['energy_buildup_score_max'] - df['energy_buildup_score_mean']
        df['relative_drop_intensity'] = drop_intensity / (df['rms_energy_std'] + 1e-6)
        n_added += 1

    # 4. Drop prominence: buildup range relative to energy range
    if all(c in df.columns for c in ['energy_buildup_score_p90', 'energy_buildup_score_p10', 'rms_energy_range']):
        buildup_range = df['energy_buildup_score_p90'] - df['energy_buildup_score_p10']
        df['drop_prominence'] = buildup_range / (df['rms_energy_range'] + 1e-6)
        n_added += 1

    # 5. Energy dynamics score: combined metric for PURPLE detection
    if all(c in df.columns for c in ['drop_candidate_rate', 'rms_energy_std', 'energy_buildup_score_std']):
        df['energy_dynamics_score'] = (
            df['drop_candidate_rate'] * 10 +
            df['rms_energy_std'] * 5 +
            df['energy_buildup_score_std']
        )
        n_added += 1

    # 6. Drop frequency (normalized by track length)
    if 'drop_candidate_sum' in df.columns and 'n_frames' in df.columns:
        df['drop_frequency'] = df['drop_candidate_sum'] / df['n_frames']
        n_added += 1

    logger.info(f"Added {n_added} normalized drop features")
    return df


def normalize_by_source(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Normalize features separately for each source to reduce domain shift."""
    logger.info("Normalizing by source...")

    df = df.copy()

    for source in df['source'].unique():
        mask = df['source'] == source
        scaler = StandardScaler()
        df.loc[mask, feature_cols] = scaler.fit_transform(df.loc[mask, feature_cols])

    return df


def prepare_features(
    df: pd.DataFrame,
    normalize_source: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare feature matrix and labels."""
    meta_cols = ['track_id', 'source', 'zone', 'n_frames', 'path']
    feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    if normalize_source:
        df = normalize_by_source(df, feature_cols)

    X = df[feature_cols].values
    y = df['zone'].values

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_cols


def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features: int = 50
) -> Tuple[np.ndarray, List[str]]:
    """Select top N features using Random Forest importance."""
    logger.info(f"Selecting top {n_features} features...")

    # Train RF to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Get importance ranking
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]

    selected_features = [feature_names[i] for i in indices]
    X_selected = X[:, indices]

    logger.info(f"Top 10 features: {selected_features[:10]}")

    return X_selected, selected_features


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'smote'
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance classes using SMOTE or undersampling."""
    if not HAS_IMBLEARN:
        logger.warning("imbalanced-learn not available, skipping balancing")
        return X, y

    logger.info(f"Balancing classes using {method}...")
    logger.info(f"Before: {pd.Series(y).value_counts().to_dict()}")

    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        return X, y

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    logger.info(f"After: {pd.Series(y_resampled).value_counts().to_dict()}")

    return X_resampled, y_resampled


def train_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5
) -> Dict:
    """Train and compare multiple models."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}

    # 1. Random Forest (baseline)
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_scores = cross_val_score(rf, X_scaled, y_encoded, cv=skf, scoring='accuracy')
    results['RandomForest'] = {
        'model': rf,
        'cv_mean': rf_scores.mean(),
        'cv_std': rf_scores.std()
    }
    logger.info(f"  RF: {rf_scores.mean():.2%} (+/- {rf_scores.std():.2%})")

    # 2. Gradient Boosting
    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42
    )
    gb_scores = cross_val_score(gb, X_scaled, y_encoded, cv=skf, scoring='accuracy')
    results['GradientBoosting'] = {
        'model': gb,
        'cv_mean': gb_scores.mean(),
        'cv_std': gb_scores.std()
    }
    logger.info(f"  GB: {gb_scores.mean():.2%} (+/- {gb_scores.std():.2%})")

    # 3. XGBoost (if available)
    if HAS_XGBOOST:
        logger.info("Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        xgb_scores = cross_val_score(xgb, X_scaled, y_encoded, cv=skf, scoring='accuracy')
        results['XGBoost'] = {
            'model': xgb,
            'cv_mean': xgb_scores.mean(),
            'cv_std': xgb_scores.std()
        }
        logger.info(f"  XGB: {xgb_scores.mean():.2%} (+/- {xgb_scores.std():.2%})")

    # Find best model
    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    best_model = results[best_name]['model']

    logger.info(f"\nBest model: {best_name}")

    # Train best model on full data
    best_model.fit(X_scaled, y_encoded)

    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        fi = None

    return {
        'best_model': best_model,
        'best_name': best_name,
        'scaler': scaler,
        'label_encoder': le,
        'all_results': results,
        'feature_importance': fi,
        'feature_names': feature_names
    }


def hyperparameter_tuning(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'xgboost'
) -> Dict:
    """Hyperparameter tuning with GridSearchCV."""
    logger.info(f"Hyperparameter tuning for {model_type}...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == 'xgboost' and HAS_XGBOOST:
        model = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
        }
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
    else:
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
        }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_scaled, y_encoded)

    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best score: {grid_search.best_score_:.2%}")

    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'scaler': scaler,
        'label_encoder': le
    }


def print_results(results: Dict):
    """Print optimization results."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print("\nModel Comparison:")
    for name, res in results['all_results'].items():
        print(f"  {name:20s}: {res['cv_mean']:.2%} (+/- {res['cv_std']:.2%})")

    print(f"\nBest Model: {results['best_name']}")
    print(f"Best CV Accuracy: {results['all_results'][results['best_name']]['cv_mean']:.2%}")

    if results['feature_importance'] is not None:
        print("\nTop 15 Features:")
        for _, row in results['feature_importance'].head(15).iterrows():
            print(f"  {row['feature']:40s}: {row['importance']:.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Optimize frame-based zone classifier')

    parser.add_argument('--user', '-u', type=Path, required=True, help='User frames pickle')
    parser.add_argument('--deam', '-d', type=Path, required=True, help='DEAM frames pickle')
    parser.add_argument('--output', '-o', type=Path, default=None, help='Output model path')

    # Optimization options
    parser.add_argument('--top-features', type=int, default=None, help='Select top N features')
    parser.add_argument('--balance', choices=['none', 'smote', 'undersample'], default='none')
    parser.add_argument('--normalize-source', action='store_true', help='Normalize by source')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--tune-model', choices=['rf', 'gb', 'xgboost'], default='xgboost')

    args = parser.parse_args()

    # Load and prepare data
    combined = load_and_merge(args.user, args.deam)
    agg_df = aggregate_frames(combined)

    X, y, feature_names = prepare_features(agg_df, normalize_source=args.normalize_source)

    logger.info(f"Initial features: {len(feature_names)}")
    logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

    # Feature selection
    if args.top_features:
        X, feature_names = select_top_features(X, y, feature_names, args.top_features)

    # Class balancing
    if args.balance != 'none':
        X, y = balance_classes(X, y, args.balance)

    # Training
    if args.tune:
        results = hyperparameter_tuning(X, y, args.tune_model)
        print(f"\nBest params: {results['best_params']}")
        print(f"Best score: {results['best_score']:.2%}")
    else:
        results = train_models(X, y, feature_names)
        print_results(results)

    # Save model
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump({
                'model': results.get('best_model'),
                'scaler': results['scaler'],
                'label_encoder': results['label_encoder'],
                'feature_names': feature_names,
                'cv_score': results.get('best_score') or results['all_results'][results['best_name']]['cv_mean']
            }, f)
        logger.info(f"Model saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
