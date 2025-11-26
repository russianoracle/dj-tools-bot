#!/usr/bin/env python3
"""
Analyze Drop Detection Features for Zone Classification

This script investigates the hypothesis that drop detection features
(energy_buildup_score, drop_candidate, etc.) are highly informative
for zone classification, especially for PURPLE zone.

Analyses:
1. Feature importance ranking for all drop-related features
2. Drop-only model accuracy vs full model
3. Distribution visualization per zone
4. User-only model (better drop detection from audio)
5. Derived drop features engineering

Usage:
    python scripts/analyze_drop_features.py \
        --user results/user_200_frames.pkl \
        --deam results/deam_200_frames.pkl
"""

import argparse
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
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

# Try matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, skipping visualizations")

# Drop-related frame features
DROP_FRAME_FEATURES = [
    'energy_buildup_score',
    'drop_candidate',
    'energy_peak',
    'energy_valley',
    'energy_delta',
    'rms_energy',  # Base for drop detection
]

# All aggregation suffixes we create
AGG_SUFFIXES = ['_mean', '_std', '_min', '_max', '_p10', '_p50', '_p90', '_range', '_sum', '_rate']


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

            # Range
            record[f'{feat}_range'] = np.max(values) - np.min(values)

            # For binary features
            if feat in ['drop_candidate', 'energy_peak', 'energy_valley', 'beat_sync', 'low_energy_flag']:
                record[f'{feat}_sum'] = np.sum(values)
                record[f'{feat}_rate'] = np.mean(values)

        records.append(record)

    agg_df = pd.DataFrame(records)
    logger.info(f"Aggregated to {len(agg_df)} tracks")
    return agg_df


def add_derived_drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived drop features."""
    df = df.copy()

    # Drop intensity: how strong are the strongest drops
    if 'energy_buildup_score_max' in df.columns and 'energy_buildup_score_mean' in df.columns:
        df['drop_intensity'] = df['energy_buildup_score_max'] - df['energy_buildup_score_mean']

    # Drop consistency: how variable are drops
    if 'energy_buildup_score_std' in df.columns and 'energy_buildup_score_mean' in df.columns:
        df['drop_consistency'] = df['energy_buildup_score_std'] / (df['energy_buildup_score_mean'] + 1e-6)

    # Drop frequency: rate of drop candidates
    if 'drop_candidate_sum' in df.columns and 'n_frames' in df.columns:
        df['drop_frequency'] = df['drop_candidate_sum'] / df['n_frames']

    # Buildup range: dynamic range of buildups
    if 'energy_buildup_score_p90' in df.columns and 'energy_buildup_score_p10' in df.columns:
        df['buildup_range'] = df['energy_buildup_score_p90'] - df['energy_buildup_score_p10']

    # Energy contrast: peaks vs valleys
    if 'energy_peak_rate' in df.columns and 'energy_valley_rate' in df.columns:
        df['energy_contrast'] = df['energy_peak_rate'] - df['energy_valley_rate']

    # Peak-to-valley ratio
    if 'energy_peak_sum' in df.columns and 'energy_valley_sum' in df.columns:
        df['peak_valley_ratio'] = df['energy_peak_sum'] / (df['energy_valley_sum'] + 1)

    # === NEW: Normalized buildup scores (relative to track's baseline energy) ===
    # This addresses the paradox where YELLOW has higher absolute buildup scores

    # Normalized buildup max: buildup relative to track's energy level
    if 'energy_buildup_score_max' in df.columns and 'rms_energy_mean' in df.columns:
        df['normalized_buildup_max'] = df['energy_buildup_score_max'] / (df['rms_energy_mean'] + 1e-6)

    # Normalized buildup mean
    if 'energy_buildup_score_mean' in df.columns and 'rms_energy_mean' in df.columns:
        df['normalized_buildup_mean'] = df['energy_buildup_score_mean'] / (df['rms_energy_mean'] + 1e-6)

    # Relative drop intensity: how dramatic are drops relative to track energy
    if 'drop_intensity' in df.columns and 'rms_energy_std' in df.columns:
        df['relative_drop_intensity'] = df['drop_intensity'] / (df['rms_energy_std'] + 1e-6)

    # Drop prominence: buildup range relative to energy variance
    if 'buildup_range' in df.columns and 'rms_energy_range' in df.columns:
        df['drop_prominence'] = df['buildup_range'] / (df['rms_energy_range'] + 1e-6)

    # Energy dynamics score: combined metric for dynamic tracks (PURPLE indicator)
    if all(col in df.columns for col in ['drop_candidate_rate', 'rms_energy_std', 'energy_buildup_score_std']):
        df['energy_dynamics_score'] = (
            df['drop_candidate_rate'] * 10 +  # Weight drop frequency
            df['rms_energy_std'] * 5 +         # Weight energy variance
            df['energy_buildup_score_std']     # Weight buildup variance
        )

    logger.info(f"Added derived features. Total columns: {len(df.columns)}")
    return df


def get_drop_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all drop-related feature columns."""
    drop_cols = []

    for base_feat in DROP_FRAME_FEATURES:
        for suffix in AGG_SUFFIXES:
            col = f'{base_feat}{suffix}'
            if col in df.columns:
                drop_cols.append(col)

    # Add derived features (including new normalized ones)
    derived = ['drop_intensity', 'drop_consistency', 'drop_frequency',
               'buildup_range', 'energy_contrast', 'peak_valley_ratio',
               'normalized_buildup_max', 'normalized_buildup_mean',
               'relative_drop_intensity', 'drop_prominence', 'energy_dynamics_score']
    for d in derived:
        if d in df.columns:
            drop_cols.append(d)

    return drop_cols


def analyze_feature_importance(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Analyze feature importance using Random Forest."""
    logger.info("Analyzing feature importance...")

    X = df[feature_cols].values
    y = df['zone'].values

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y_encoded)

    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return fi


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: List[str],
    name: str = "Model"
) -> Dict:
    """Train model and return CV scores."""
    X = df[feature_cols].values
    y = df['zone'].values

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_scaled, y_encoded, cv=skf, scoring='accuracy')

    logger.info(f"{name}: {scores.mean():.2%} (+/- {scores.std():.2%})")

    return {
        'name': name,
        'cv_mean': scores.mean(),
        'cv_std': scores.std(),
        'n_features': len(feature_cols)
    }


def visualize_drop_features_by_zone(df: pd.DataFrame, output_dir: Path):
    """Create visualizations of drop features by zone."""
    if not HAS_MATPLOTLIB:
        logger.warning("Skipping visualization - matplotlib not available")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Key drop features to visualize (original + normalized)
    key_features = [
        'energy_buildup_score_max',
        'normalized_buildup_max',  # NEW: should show better PURPLE separation
        'drop_candidate_rate',
        'energy_dynamics_score',   # NEW: combined metric
        'relative_drop_intensity', # NEW: normalized
        'drop_prominence'          # NEW: normalized
    ]

    available_features = [f for f in key_features if f in df.columns]

    if not available_features:
        logger.warning("No drop features available for visualization")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    zone_colors = {'YELLOW': '#FFD700', 'GREEN': '#32CD32', 'PURPLE': '#9370DB'}
    zone_order = ['YELLOW', 'GREEN', 'PURPLE']

    for idx, feat in enumerate(available_features[:6]):
        ax = axes[idx]

        data_by_zone = [df[df['zone'] == z][feat].dropna().values for z in zone_order]

        bp = ax.boxplot(data_by_zone, patch_artist=True, labels=zone_order)

        for patch, zone in zip(bp['boxes'], zone_order):
            patch.set_facecolor(zone_colors[zone])
            patch.set_alpha(0.7)

        ax.set_title(feat)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(available_features), 6):
        axes[idx].set_visible(False)

    plt.suptitle('Drop Detection Features by Zone', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'drop_features_by_zone.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved visualization to: {output_path}")


def print_results(results: List[Dict], fi_all: pd.DataFrame, fi_drop: pd.DataFrame):
    """Print analysis results."""
    print("\n" + "=" * 70)
    print("DROP FEATURE ANALYSIS RESULTS")
    print("=" * 70)

    print("\n📊 MODEL COMPARISON:")
    print("-" * 50)
    for r in results:
        print(f"  {r['name']:30s}: {r['cv_mean']:.2%} (+/- {r['cv_std']:.2%}) [{r['n_features']} features]")

    print("\n🎯 DROP FEATURES IN TOP-20 (Full Model):")
    print("-" * 50)
    drop_keywords = ['buildup', 'drop', 'peak', 'valley', 'delta', 'intensity', 'contrast']
    top20 = fi_all.head(20)
    for _, row in top20.iterrows():
        is_drop = any(kw in row['feature'].lower() for kw in drop_keywords)
        marker = "🔥" if is_drop else "  "
        print(f"  {marker} {row['feature']:45s}: {row['importance']:.4f}")

    print("\n📈 TOP-15 DROP-ONLY FEATURES:")
    print("-" * 50)
    for _, row in fi_drop.head(15).iterrows():
        print(f"  {row['feature']:45s}: {row['importance']:.4f}")

    # Count drop features in top positions
    drop_in_top10 = sum(1 for _, row in fi_all.head(10).iterrows()
                       if any(kw in row['feature'].lower() for kw in drop_keywords))
    drop_in_top20 = sum(1 for _, row in fi_all.head(20).iterrows()
                       if any(kw in row['feature'].lower() for kw in drop_keywords))

    print(f"\n📍 Drop features in top-10: {drop_in_top10}/10")
    print(f"📍 Drop features in top-20: {drop_in_top20}/20")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Analyze drop detection features')

    parser.add_argument('--user', '-u', type=Path, required=True, help='User frames pickle')
    parser.add_argument('--deam', '-d', type=Path, required=True, help='DEAM frames pickle')
    parser.add_argument('--output-dir', '-o', type=Path, default=Path('results/drop_analysis'))

    args = parser.parse_args()

    # Load and prepare data
    combined = load_and_merge(args.user, args.deam)
    agg_df = aggregate_frames(combined)
    agg_df = add_derived_drop_features(agg_df)

    # Get feature columns
    meta_cols = ['track_id', 'source', 'zone', 'n_frames', 'path']
    all_feature_cols = [c for c in agg_df.columns if c not in meta_cols
                        and agg_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    drop_feature_cols = get_drop_feature_columns(agg_df)

    logger.info(f"All features: {len(all_feature_cols)}")
    logger.info(f"Drop features: {len(drop_feature_cols)}")
    logger.info(f"Drop features list: {drop_feature_cols[:10]}...")

    # Analysis 1: Feature importance on full model
    fi_all = analyze_feature_importance(agg_df, all_feature_cols)

    # Analysis 2: Feature importance on drop-only model
    fi_drop = analyze_feature_importance(agg_df, drop_feature_cols)

    results = []

    # Model 1: Full features
    results.append(train_and_evaluate(agg_df, all_feature_cols, "Full Model (all features)"))

    # Model 2: Drop-only features
    results.append(train_and_evaluate(agg_df, drop_feature_cols, "Drop-Only Model"))

    # Model 3: User-only (better drop detection)
    user_df = agg_df[agg_df['source'] == 'user']
    if len(user_df) > 20:
        results.append(train_and_evaluate(user_df, all_feature_cols, "User-Only (all features)"))
        results.append(train_and_evaluate(user_df, drop_feature_cols, "User-Only (drop features)"))

    # Model 4: Drop + tempo/energy features
    tempo_energy_cols = [c for c in all_feature_cols if any(kw in c.lower()
                         for kw in ['tempo', 'bpm', 'rms', 'energy', 'buildup', 'drop', 'peak', 'valley', 'delta'])]
    if tempo_energy_cols:
        results.append(train_and_evaluate(agg_df, tempo_energy_cols, "Tempo+Energy+Drop"))

    # Visualization
    visualize_drop_features_by_zone(agg_df, args.output_dir)

    # Print results
    print_results(results, fi_all, fi_drop)

    # Save feature importance
    fi_all.to_csv(args.output_dir / 'feature_importance_all.csv', index=False)
    fi_drop.to_csv(args.output_dir / 'feature_importance_drop.csv', index=False)
    logger.info(f"Saved feature importance to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
