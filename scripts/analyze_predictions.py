#!/usr/bin/env python3
"""
Analyze model predictions vs actual labels.
Visualize tracks and highlight disagreements for review.
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

ZONE_COLORS = {'YELLOW': '#FFD700', 'GREEN': '#32CD32', 'PURPLE': '#9370DB'}
ZONE_NAMES = {0: 'YELLOW', 1: 'GREEN', 2: 'PURPLE'}


def parse_test_data(filepath):
    """Parse test_data TSV."""
    tracks = {}
    with open(filepath, 'r', encoding='utf-16') as f:
        lines = f.readlines()

    header = lines[0].strip().split('\t')
    zone_idx = location_idx = None

    for i, col in enumerate(header):
        col_lower = col.strip().lower()
        if col_lower in ['zone', 'my tag']:
            zone_idx = i
        elif col_lower == 'location':
            location_idx = i

    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) <= max(zone_idx, location_idx):
            continue
        zone = parts[zone_idx].strip().upper()
        path = parts[location_idx].strip().strip('"')
        if zone in ['YELLOW', 'GREEN', 'PURPLE'] and path:
            tracks[path] = zone

    return tracks


def load_features_and_predict(model_dir, cache_path, tracks):
    """Load model, features and make predictions."""
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    # Load model
    model = xgb.Booster()
    model.load_model(str(Path(model_dir) / 'xgboost_model.json'))

    with open(Path(model_dir) / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(Path(model_dir) / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    # Load cache
    with open(cache_path, 'rb') as f:
        cache_df = pickle.load(f)

    cache = {}
    for _, row in cache_df.iterrows():
        cache[row['audio_path']] = row['features_list']

    # Extract features
    results = []
    for path, actual_zone in tracks.items():
        if path not in cache:
            continue

        features = cache[path]
        if hasattr(features, '__dict__'):
            feature_dict = features.__dict__
        elif isinstance(features, dict):
            feature_dict = features
        else:
            continue

        row = []
        for fname in feature_names:
            val = feature_dict.get(fname, 0.0)
            if isinstance(val, np.ndarray):
                val = float(np.nanmean(val))
            elif isinstance(val, (list, tuple)):
                val = float(np.nanmean(val)) if val else 0.0
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            row.append(float(val))

        results.append({
            'path': path,
            'filename': Path(path).name,
            'actual_zone': actual_zone,
            'features': row
        })

    # Predict
    X = np.array([r['features'] for r in results])
    X_scaled = scaler.transform(X)

    dmatrix = xgb.DMatrix(X_scaled)
    proba = model.predict(dmatrix)
    predictions = np.argmax(proba, axis=1)
    confidences = np.max(proba, axis=1)

    for i, r in enumerate(results):
        r['predicted_zone'] = ZONE_NAMES[predictions[i]]
        r['confidence'] = confidences[i]
        r['proba'] = proba[i]
        r['is_correct'] = r['actual_zone'] == r['predicted_zone']

    return results, X_scaled, feature_names


def main():
    print("=" * 70)
    print("Prediction Analysis")
    print("=" * 70)

    # Load data
    tracks = parse_test_data('tests/test_data_2.txt')
    print(f"\nLoaded {len(tracks)} labeled tracks")

    results, X_scaled, feature_names = load_features_and_predict(
        'models/user_zone_v2',
        'models/checkpoints/features.pkl',
        tracks
    )

    print(f"Analyzed {len(results)} tracks")

    # Statistics
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / len(results)
    print(f"\nAccuracy: {accuracy:.1%}")

    # Errors by zone
    errors = [r for r in results if not r['is_correct']]
    print(f"\nErrors: {len(errors)} tracks")

    error_types = {}
    for r in errors:
        key = f"{r['actual_zone']} -> {r['predicted_zone']}"
        error_types[key] = error_types.get(key, 0) + 1

    print("\nError breakdown:")
    for k, v in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"\nPCA variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Actual zones
    ax1 = axes[0]
    for zone in ['YELLOW', 'GREEN', 'PURPLE']:
        mask = [r['actual_zone'] == zone for r in results]
        x = X_pca[mask, 0]
        y = X_pca[mask, 1]
        ax1.scatter(x, y, c=ZONE_COLORS[zone], label=zone, alpha=0.6, s=50)
    ax1.set_title('Actual Labels (Your Markup)', fontsize=14)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predicted zones
    ax2 = axes[1]
    for zone in ['YELLOW', 'GREEN', 'PURPLE']:
        mask = [r['predicted_zone'] == zone for r in results]
        x = X_pca[mask, 0]
        y = X_pca[mask, 1]
        ax2.scatter(x, y, c=ZONE_COLORS[zone], label=zone, alpha=0.6, s=50)
    ax2.set_title('Model Predictions', fontsize=14)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Errors highlighted
    ax3 = axes[2]
    correct_mask = [r['is_correct'] for r in results]
    error_mask = [not r['is_correct'] for r in results]

    ax3.scatter(X_pca[correct_mask, 0], X_pca[correct_mask, 1],
                c='lightgray', alpha=0.4, s=30, label='Correct')
    ax3.scatter(X_pca[error_mask, 0], X_pca[error_mask, 1],
                c='red', alpha=0.8, s=80, marker='x', linewidths=2, label='Errors')
    ax3.set_title(f'Errors Highlighted ({len(errors)} tracks)', fontsize=14)
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/prediction_analysis.png', dpi=150)
    print("\nSaved: results/prediction_analysis.png")

    # Save errors for review
    error_df = pd.DataFrame([{
        'filename': r['filename'],
        'actual': r['actual_zone'],
        'predicted': r['predicted_zone'],
        'confidence': f"{r['confidence']:.0%}",
        'path': r['path']
    } for r in errors])

    error_df = error_df.sort_values(['actual', 'predicted', 'confidence'])
    error_df.to_csv('results/prediction_errors.csv', index=False)
    print("Saved: results/prediction_errors.csv")

    # Print low confidence predictions (uncertain)
    uncertain = [r for r in results if r['confidence'] < 0.5]
    print(f"\nUncertain predictions (confidence < 50%): {len(uncertain)}")

    # Print some errors for quick review
    print("\n" + "=" * 70)
    print("ERRORS FOR REVIEW (sorted by confidence)")
    print("=" * 70)

    errors_sorted = sorted(errors, key=lambda x: -x['confidence'])
    for r in errors_sorted[:30]:
        print(f"\n{r['actual_zone']:6} -> {r['predicted_zone']:6} ({r['confidence']:.0%})")
        print(f"  {r['filename'][:60]}")

    if len(errors) > 30:
        print(f"\n... and {len(errors) - 30} more errors in results/prediction_errors.csv")

    return 0


if __name__ == '__main__':
    sys.exit(main())
