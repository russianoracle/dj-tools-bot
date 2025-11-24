"""Analyze feature importance from trained XGBoost model."""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model_path = "models/with_dj_features/xgboost_final.pkl"
print(f"Loading model from: {model_path}")

with open(model_path, 'rb') as f:
    saved_data = pickle.load(f)

# Check structure
print(f"Model type: {type(saved_data)}")
if isinstance(saved_data, dict):
    print(f"Keys: {saved_data.keys()}")
    model = saved_data.get('model', saved_data)
else:
    model = saved_data

# Get feature importance
importance = model.feature_importances_

# Feature names (51 total)
feature_names = [
    # Basic librosa (19)
    'tempo', 'zero_crossing_rate', 'low_energy', 'rms_energy',
    'spectral_rolloff', 'brightness', 'spectral_centroid',
    'mfcc_1_mean', 'mfcc_1_std', 'mfcc_2_mean', 'mfcc_2_std',
    'mfcc_3_mean', 'mfcc_3_std', 'mfcc_4_mean', 'mfcc_4_std',
    'mfcc_5_mean', 'mfcc_5_std', 'energy_variance', 'drop_strength',

    # Emotion (2)
    'arousal', 'valence',

    # Temporal (4)
    'onset_strength_mean', 'onset_strength_std', 'beat_strength', 'tempo_stability',

    # Spectral contrast (14)
    'spectral_contrast_mean_0', 'spectral_contrast_mean_1', 'spectral_contrast_mean_2',
    'spectral_contrast_mean_3', 'spectral_contrast_mean_4', 'spectral_contrast_mean_5',
    'spectral_contrast_mean_6',
    'spectral_contrast_std_0', 'spectral_contrast_std_1', 'spectral_contrast_std_2',
    'spectral_contrast_std_3', 'spectral_contrast_std_4', 'spectral_contrast_std_5',
    'spectral_contrast_std_6',

    # Harmonic-percussive (2)
    'harmonic_ratio', 'percussive_ratio',

    # Build-up (3) - NEW
    'energy_slope', 'energy_buildup_ratio', 'onset_acceleration',

    # Drive (2) - NEW
    'drop_frequency', 'peak_energy_ratio',

    # Euphoria (2) - NEW
    'rhythmic_regularity', 'harmonic_complexity',

    # Climax (3) - NEW
    'has_climax', 'climax_position', 'dynamic_range'
]

# Sort by importance
indices = np.argsort(importance)[::-1]

print("\n" + "="*80)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*80)
for i, idx in enumerate(indices[:20], 1):
    marker = "NEW!" if idx >= 41 else ""
    print(f"{i:2d}. {feature_names[idx]:30s} {importance[idx]:.4f}  {marker}")

print(f"\nTotal features in model: {len(importance)}")
print(f"Total feature names: {len(feature_names)}")

if len(importance) >= 41:
    print("\n" + "="*80)
    print("DJ-SPECIFIC FEATURES IMPORTANCE (indices 41-50)")
    print("="*80)
    dj_features_start = 41
    for i in range(dj_features_start, min(len(feature_names), len(importance))):
        if i < len(indices):
            rank = list(indices).index(i) + 1
            print(f"Rank {rank:2d}: {feature_names[i]:30s} {importance[i]:.4f}")
else:
    print("\n⚠️  WARNING: Model has fewer features than expected!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
dj_importance = np.sum(importance[41:])
total_importance = np.sum(importance)
print(f"DJ-specific features total importance: {dj_importance:.4f} ({dj_importance/total_importance*100:.1f}%)")
print(f"Traditional features total importance: {np.sum(importance[:41]):.4f} ({np.sum(importance[:41])/total_importance*100:.1f}%)")
