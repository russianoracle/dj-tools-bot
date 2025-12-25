#!/usr/bin/env python3
"""
🔍 Find Noisy Labels using Confident Learning

Uses cleanlab to identify potentially mislabeled tracks.
Exports a list of tracks that should be reviewed and potentially relabeled.

Usage:
    # From CSV with audio paths (extracts features on the fly)
    python scripts/find_noisy_labels.py --input results/user_tracks.csv --mode csv

    # From PKL with pre-extracted features
    python scripts/find_noisy_labels.py --input results/frames.pkl --mode pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import json
import pickle
from collections import Counter
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores

# Zone order
ZONES = ['YELLOW', 'GREEN', 'PURPLE']


def extract_track_features(audio_path: str, sr: int = 22050) -> np.ndarray:
    """Extract features from a single audio file using librosa."""
    import librosa

    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=60)  # First 60 sec

        # Basic features
        features = []

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.extend([np.mean(centroid), np.std(centroid)])

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.extend([np.mean(rolloff), np.std(rolloff)])

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([np.mean(zcr), np.std(zcr)])

        # Spectral flux
        S = np.abs(librosa.stft(y))
        flux = np.sqrt(np.mean(np.diff(S, axis=1)**2, axis=0))
        features.extend([np.mean(flux), np.std(flux)])

        # MFCCs (first 5)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        for i in range(5):
            features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo))

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend([np.mean(contrast), np.std(contrast)])

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend([np.mean(chroma), np.std(chroma)])

        # Energy variance (important for drops)
        energy_var = np.var(rms)
        features.append(energy_var)

        # Onset strength
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        features.extend([np.mean(onset), np.std(onset), np.max(onset)])

        return np.array(features)

    except Exception as e:
        print(f"   Error extracting {Path(audio_path).name}: {e}")
        return None


def extract_features_sequential(tracks_df: pd.DataFrame, n_workers: int = 4) -> tuple:
    """Extract features from all tracks sequentially (more reliable)."""
    import os

    X_list = []
    y_list = []
    track_ids = []
    track_paths = {}
    failed = []

    total = len(tracks_df)
    print(f"\n🎵 Extracting features from {total} tracks...", flush=True)

    for idx, (_, row) in enumerate(tracks_df.iterrows()):
        path = row['path']
        zone = row['zone']
        track_id = Path(path).stem[:30]

        if idx > 0 and idx % 10 == 0:
            pct = idx / total * 100
            print(f"   [{idx}/{total}] {pct:.0f}% ({len(X_list)} success, {len(failed)} failed)", flush=True)

        if not os.path.exists(path):
            failed.append((track_id, "File not found"))
            continue

        features = extract_track_features(path)
        if features is None:
            failed.append((track_id, "Extraction failed"))
            continue

        X_list.append(features)
        y_list.append(zone)
        track_ids.append(track_id)
        track_paths[track_id] = path

    print(f"\n   ✅ Successfully extracted: {len(X_list)}")
    print(f"   ❌ Failed: {len(failed)}")

    if failed and len(failed) <= 10:
        print(f"   Failed tracks: {[f[0] for f in failed[:10]]}")

    return np.array(X_list), np.array(y_list), track_ids, track_paths


def aggregate_to_track_level(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Aggregate frame-level data to track-level for cleanlab analysis.

    Returns:
        X_tracks: (n_tracks, n_features) - mean features per track
        y_tracks: (n_tracks,) - zone labels
        track_ids: list of track IDs
        track_paths: dict mapping track_id -> file path
    """
    # Get unique tracks
    track_groups = df.groupby('track_id')

    X_list = []
    y_list = []
    track_ids = []
    track_paths = {}

    for track_id, group in track_groups:
        # Aggregate features (mean across frames)
        X_track = group[feature_cols].mean().values

        # Get zone label (should be same for all frames)
        zone = group['zone'].iloc[0]

        # Get path if available
        if 'path' in group.columns:
            track_paths[track_id] = group['path'].iloc[0]
        elif 'file' in group.columns:
            track_paths[track_id] = group['file'].iloc[0]

        X_list.append(X_track)
        y_list.append(zone)
        track_ids.append(track_id)

    return np.array(X_list), np.array(y_list), track_ids, track_paths


def find_noisy_labels(input_path: str, output_dir: str, n_splits: int = 5,
                       mode: str = 'csv', n_workers: int = 4):
    """
    Find potentially mislabeled tracks using Confident Learning.
    """
    print("\n" + "=" * 60)
    print("🔍 Finding Noisy Labels with Confident Learning")
    print("=" * 60)

    # Load data based on mode
    print(f"\n📂 Loading {input_path} (mode={mode})...")

    if mode == 'csv':
        # Load CSV with paths and labels
        df = pd.read_csv(input_path)
        print(f"   Tracks in CSV: {len(df)}")

        # Filter valid zones
        valid_mask = df['zone'].notna() & df['zone'].isin(ZONES)
        df = df[valid_mask].copy()
        print(f"   Valid tracks: {len(df)}")

        # Extract features from audio files
        X, y, track_ids, track_paths = extract_features_sequential(df, n_workers)

    else:  # pkl mode
        df = pd.read_pickle(input_path)

        print(f"   Frames: {len(df):,}")
        print(f"   Tracks: {df['track_id'].nunique()}")

        # Select features
        exclude_cols = ['zone', 'track_id', 'frame_idx', 'source', 'path', 'file',
                        'filename', 'source_dataset', 'frameTime']
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in exclude_cols]

        print(f"   Features: {len(feature_cols)}")

        # Filter valid zones
        valid_mask = df['zone'].notna() & df['zone'].isin(ZONES)
        df = df[valid_mask].copy()

        # Aggregate to track level
        print("\n📊 Aggregating to track level...")
        X, y, track_ids, track_paths = aggregate_to_track_level(df, feature_cols)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Encode labels
    le = LabelEncoder()
    le.classes_ = np.array(ZONES)
    y_encoded = le.transform(y)

    print(f"   Tracks for analysis: {len(X)}")
    print(f"   Zone distribution: {Counter(y)}")

    # Get out-of-fold predicted probabilities
    print(f"\n🔄 Getting {n_splits}-fold cross-validated predictions...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric='mlogloss'
    )

    # Use StratifiedKFold for track-level (no groups needed here)
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pred_probs = cross_val_predict(
        model, X, y_encoded,
        cv=cv,
        method='predict_proba'
    )

    # Find label issues using cleanlab
    print("\n🧹 Running Confident Learning...")

    # Method 1: find_label_issues
    label_issues_mask = find_label_issues(
        labels=y_encoded,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )

    # Method 2: get quality scores for all samples
    quality_scores = get_label_quality_scores(
        labels=y_encoded,
        pred_probs=pred_probs
    )

    # Combine results
    n_issues = len(label_issues_mask)
    print(f"\n📋 Found {n_issues} potential label issues ({n_issues/len(X)*100:.1f}% of tracks)")

    # Create detailed report
    results = []
    for i, (track_id, true_label, probs, quality) in enumerate(
        zip(track_ids, y, pred_probs, quality_scores)
    ):
        pred_label = ZONES[np.argmax(probs)]
        max_prob = np.max(probs)

        is_issue = i in label_issues_mask

        results.append({
            'track_id': track_id,
            'path': track_paths.get(track_id, ''),
            'current_label': true_label,
            'predicted_label': pred_label,
            'confidence': float(max_prob),
            'quality_score': float(quality),
            'is_potential_issue': is_issue,
            'prob_YELLOW': float(probs[0]),
            'prob_GREEN': float(probs[1]),
            'prob_PURPLE': float(probs[2]),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('quality_score', ascending=True)

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save full report
    full_report_path = output_dir / f'label_quality_report_{timestamp}.csv'
    results_df.to_csv(full_report_path, index=False)
    print(f"\n💾 Full report saved: {full_report_path}")

    # Save issues only
    issues_df = results_df[results_df['is_potential_issue']]
    issues_path = output_dir / f'noisy_labels_{timestamp}.csv'
    issues_df.to_csv(issues_path, index=False)
    print(f"💾 Issues report saved: {issues_path}")

    # Print top issues for review
    print("\n" + "=" * 60)
    print("🚨 TOP 30 TRACKS TO REVIEW")
    print("=" * 60)
    print(f"{'Track ID':<12} {'Current':<8} {'Predicted':<10} {'Conf':<6} {'Quality':<8} {'Path'}")
    print("-" * 100)

    for _, row in issues_df.head(30).iterrows():
        path_short = str(row['path'])[-50:] if row['path'] else ''
        mismatch = "⚠️" if row['current_label'] != row['predicted_label'] else "  "
        print(f"{str(row['track_id']):<12} {row['current_label']:<8} {row['predicted_label']:<10} "
              f"{row['confidence']:.2f}  {row['quality_score']:.3f}   {mismatch} {path_short}")

    # Statistics by zone
    print("\n" + "=" * 60)
    print("📊 ISSUES BY ZONE")
    print("=" * 60)

    for zone in ZONES:
        zone_issues = issues_df[issues_df['current_label'] == zone]
        zone_total = len(results_df[results_df['current_label'] == zone])
        pct = len(zone_issues) / zone_total * 100 if zone_total > 0 else 0

        print(f"   {zone}: {len(zone_issues)} issues / {zone_total} total ({pct:.1f}%)")

        # What are they being confused with?
        if len(zone_issues) > 0:
            confusion = zone_issues['predicted_label'].value_counts()
            for pred_zone, count in confusion.items():
                if pred_zone != zone:
                    print(f"      → confused with {pred_zone}: {count}")

    # Export for labeling app
    labeling_export = {
        'generated': timestamp,
        'total_tracks': len(results_df),
        'issues_found': len(issues_df),
        'tracks_to_review': []
    }

    for _, row in issues_df.head(50).iterrows():
        labeling_export['tracks_to_review'].append({
            'track_id': str(row['track_id']),
            'path': row['path'],
            'current_label': row['current_label'],
            'suggested_label': row['predicted_label'],
            'confidence': round(row['confidence'], 3),
            'probabilities': {
                'YELLOW': round(row['prob_YELLOW'], 3),
                'GREEN': round(row['prob_GREEN'], 3),
                'PURPLE': round(row['prob_PURPLE'], 3),
            }
        })

    labeling_json_path = output_dir / f'tracks_to_review_{timestamp}.json'
    with open(labeling_json_path, 'w', encoding='utf-8') as f:
        json.dump(labeling_export, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Labeling export saved: {labeling_json_path}")

    return results_df, issues_df


def main():
    parser = argparse.ArgumentParser(
        description="🔍 Find Noisy Labels using Confident Learning"
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Input CSV or PKL file")
    parser.add_argument("--output-dir", "-o", default="results/label_analysis",
                        help="Output directory for reports")
    parser.add_argument("--mode", "-m", choices=['csv', 'pkl'], default='csv',
                        help="Input mode: csv (extracts features) or pkl (pre-extracted)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of CV splits")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of parallel workers for feature extraction")

    args = parser.parse_args()

    find_noisy_labels(args.input, args.output_dir, args.n_splits,
                      mode=args.mode, n_workers=args.workers)

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the tracks in 'tracks_to_review_*.json'")
    print("2. Update labels in data/labels.json or labeling app")
    print("3. Re-extract features and retrain the model")


if __name__ == "__main__":
    main()
