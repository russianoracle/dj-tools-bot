#!/usr/bin/env python3
"""
Retrain zone classifier on updated labeled data.

Uses test_data_2.txt (484 labeled tracks) instead of test_data.txt (415 tracks).
Supports wav2vec2 embeddings (768-dim) with optional PCA reduction.

Models used:
- torchaudio.pipelines.WAV2VEC2_BASE (768-dim embeddings)
- torchaudio.pipelines.HUBERT_BASE (768-dim embeddings)
"""

import sys
import os
import argparse
from pathlib import Path

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.training.zone_features import ZoneFeatureExtractor
from src.training.zone_trainer import ZONE_LABELS


def parse_test_data(filepath):
    """Parse test_data TSV and return list of (path, zone) tuples."""
    tracks = []

    with open(filepath, 'r', encoding='utf-16') as f:
        lines = f.readlines()

    if not lines:
        return tracks

    header = lines[0].strip().split('\t')
    zone_idx = None
    location_idx = None

    for i, col in enumerate(header):
        col_lower = col.strip().lower()
        if col_lower in ['zone', 'my tag']:
            zone_idx = i
        elif col_lower == 'location':
            location_idx = i

    if zone_idx is None or location_idx is None:
        print(f"Warning: Could not detect columns in {filepath}")
        return tracks

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) <= max(zone_idx, location_idx):
            continue

        zone = parts[zone_idx].strip().upper()
        path = parts[location_idx].strip().strip('"')

        if zone in ['YELLOW', 'GREEN', 'PURPLE'] and path:
            tracks.append((path, zone))

    return tracks


def load_cached_features():
    """Load existing cached features (DataFrame format)."""
    cache_path = Path('models/checkpoints/features.pkl')
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
            # Convert to dict: path -> features
            if isinstance(df, pd.DataFrame) and 'audio_path' in df.columns:
                cache = {}
                for _, row in df.iterrows():
                    cache[row['audio_path']] = row['features_list']
                return cache
            elif isinstance(df, dict):
                return df
    return {}


def save_cached_features(cache, original_df=None):
    """Save features cache."""
    cache_path = Path('models/checkpoints/features.pkl')
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame format for consistency
    rows = []
    for path, features in cache.items():
        rows.append({
            'audio_path': path,
            'zone_label': '',
            'features_list': features
        })
    df = pd.DataFrame(rows)

    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)


def main():
    parser = argparse.ArgumentParser(description='Retrain zone classifier on new labels')
    parser.add_argument('--use-embeddings', action='store_true',
                        help='Use wav2vec2 embeddings (768-dim)')
    parser.add_argument('--pca-dim', type=int, default=64,
                        help='PCA dimensions for embeddings (default: 64)')
    parser.add_argument('--no-pca', action='store_true',
                        help='Use full embeddings without PCA')
    parser.add_argument('--model-dir', type=str, default='models/user_zone_v2',
                        help='Output model directory')
    args = parser.parse_args()

    print("=" * 70)
    print("Retrain Zone Classifier on Updated Labels")
    if args.use_embeddings:
        if args.no_pca:
            print(f"  Mode: 35 base features + 768-dim embeddings = 803 features")
        else:
            print(f"  Mode: 35 base features + PCA({args.pca_dim}) embeddings = {35 + args.pca_dim} features")
    else:
        print(f"  Mode: 35 base features only")
    print("=" * 70)

    # Base features (19 common + 16 DJ-specific = 35)
    BASE_FEATURES = [
        'tempo', 'zero_crossing_rate', 'rms_energy', 'spectral_centroid',
        'spectral_rolloff', 'energy_variance', 'mfcc_1_mean', 'mfcc_1_std',
        'mfcc_2_mean', 'mfcc_2_std', 'mfcc_3_mean', 'mfcc_3_std',
        'mfcc_4_mean', 'mfcc_4_std', 'mfcc_5_mean', 'mfcc_5_std',
        'low_energy', 'brightness', 'drop_strength'
    ]

    DJ_FEATURES = [
        'onset_strength_mean', 'onset_strength_std', 'beat_strength',
        'tempo_stability', 'spectral_contrast_mean', 'spectral_contrast_std',
        'harmonic_ratio', 'percussive_ratio', 'energy_slope',
        'energy_buildup_ratio', 'onset_acceleration', 'drop_frequency',
        'peak_energy_ratio', 'rhythmic_regularity', 'harmonic_complexity',
        'dynamic_range',
        # New drop detection features
        'drop_contrast_mean', 'drop_contrast_max', 'drop_count', 'drop_intensity'
    ]

    ALL_BASE_FEATURES = BASE_FEATURES + DJ_FEATURES

    # Step 1: Load labeled tracks
    print("\n[1/5] Loading labeled tracks from test_data_2.txt...")
    labeled_tracks = parse_test_data('tests/test_data_2.txt')
    print(f"  Total labeled tracks: {len(labeled_tracks)}")

    zone_counts = {}
    for _, zone in labeled_tracks:
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
    print(f"  YELLOW: {zone_counts.get('YELLOW', 0)}")
    print(f"  GREEN: {zone_counts.get('GREEN', 0)}")
    print(f"  PURPLE: {zone_counts.get('PURPLE', 0)}")

    # Step 2: Load cache
    print("\n[2/5] Loading feature cache...")
    cache = load_cached_features()
    print(f"  Cached: {len(cache)} tracks")

    # Find missing tracks
    missing_paths = []
    for path, zone in labeled_tracks:
        if path not in cache:
            if os.path.exists(path):
                missing_paths.append(path)
            else:
                print(f"  Warning: File not found: {Path(path).name[:50]}...")

    print(f"  Missing: {len(missing_paths)} tracks")

    # Step 3: Extract features for missing tracks
    if missing_paths:
        print("\n[3/5] Extracting features for new tracks...")
        extractor = ZoneFeatureExtractor(use_gpu=True)

        for i, path in enumerate(missing_paths):
            try:
                features = extractor.extract(path)
                if features is not None:
                    cache[path] = features
                    if (i + 1) % 10 == 0 or (i + 1) == len(missing_paths):
                        print(f"  Extracted {i + 1}/{len(missing_paths)}...")
            except Exception as e:
                print(f"  Error: {Path(path).name}: {e}")

        save_cached_features(cache)
        print(f"  Cache updated: {len(cache)} total tracks")
    else:
        print("\n[3/5] All features cached!")

    # Step 4: Prepare training data
    print("\n[4/5] Preparing training data...")

    X_base_list = []
    X_emb_list = []
    y_list = []
    paths_used = []

    for path, zone in labeled_tracks:
        if path not in cache:
            continue

        features = cache[path]
        try:
            if hasattr(features, '__dict__'):
                feature_dict = features.__dict__
            elif isinstance(features, dict):
                feature_dict = features
            else:
                continue

            # Extract base features
            row = []
            for fname in ALL_BASE_FEATURES:
                val = feature_dict.get(fname, 0.0)
                # Handle arrays - take mean if array
                if isinstance(val, np.ndarray):
                    val = float(np.nanmean(val))
                elif isinstance(val, (list, tuple)):
                    val = float(np.nanmean(val)) if val else 0.0
                elif val is None:
                    val = 0.0
                else:
                    try:
                        val = float(val)
                        if np.isnan(val):
                            val = 0.0
                    except (TypeError, ValueError):
                        val = 0.0
                row.append(val)

            # Extract embeddings if requested
            embedding = None
            if args.use_embeddings:
                emb = feature_dict.get('wav2vec2_embedding')
                if emb is not None and hasattr(emb, '__len__') and len(emb) == 768:
                    embedding = np.array(emb)
                else:
                    continue  # Skip tracks without embeddings

            # Verify row length
            if len(row) != len(ALL_BASE_FEATURES):
                print(f"  Warning: {Path(path).name} has {len(row)} features (expected {len(ALL_BASE_FEATURES)})")
                continue

            X_base_list.append(row)
            if embedding is not None:
                X_emb_list.append(embedding)
            y_list.append(ZONE_LABELS[zone.lower()])
            paths_used.append(path)

        except Exception as e:
            print(f"  Error: {Path(path).name}: {e}")

    X_base = np.array(X_base_list)
    y = np.array(y_list)

    print(f"  Samples: {len(X_base)}")
    print(f"  Base features: {X_base.shape[1]}")

    # Handle embeddings
    if args.use_embeddings and X_emb_list:
        X_emb = np.array(X_emb_list)
        print(f"  Embedding features: {X_emb.shape[1]}")

        if args.no_pca:
            X = np.hstack([X_base, X_emb])
            pca = None
        else:
            # Apply PCA to embeddings
            print(f"  Applying PCA: {X_emb.shape[1]} -> {args.pca_dim}")
            pca = PCA(n_components=args.pca_dim, random_state=42)
            X_emb_pca = pca.fit_transform(X_emb)
            explained_var = sum(pca.explained_variance_ratio_) * 100
            print(f"  PCA explained variance: {explained_var:.1f}%")
            X = np.hstack([X_base, X_emb_pca])
    else:
        X = X_base
        pca = None

    print(f"  Total features: {X.shape[1]}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Train model
    print("\n[5/5] Training XGBoost model...")

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = len(y_train) / (3 * class_counts + 1e-6)
    sample_weights = np.array([class_weights[label] for label in y_train])

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }

    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params, dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # Evaluate
    y_pred = np.argmax(model.predict(dtest), axis=1)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n  Test Accuracy: {test_acc:.1%}")
    print(f"\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['YELLOW', 'GREEN', 'PURPLE'],
                                zero_division=0))

    # Save model
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_dir / 'xgboost_model.json'))

    with open(model_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open(model_dir / 'feature_names.pkl', 'wb') as f:
        pickle.dump(ALL_BASE_FEATURES, f)

    if pca is not None:
        with open(model_dir / 'pca.pkl', 'wb') as f:
            pickle.dump(pca, f)

    # Save metadata
    metadata = {
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'test_accuracy': test_acc,
        'n_features': X.shape[1],
        'use_embeddings': args.use_embeddings,
        'pca_dim': args.pca_dim if (args.use_embeddings and not args.no_pca) else None,
        'zone_distribution': {
            'YELLOW': int(zone_counts.get('YELLOW', 0)),
            'GREEN': int(zone_counts.get('GREEN', 0)),
            'PURPLE': int(zone_counts.get('PURPLE', 0))
        }
    }
    with open(model_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n  Model saved to: {model_dir}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Labeled tracks: {len(labeled_tracks)}")
    print(f"Used for training: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Test accuracy: {test_acc:.1%}")
    print(f"Model: {model_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
