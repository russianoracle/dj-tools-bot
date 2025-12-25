#!/usr/bin/env python3
"""
🎯 Train Zone Classifier on User Data Only

No DEAM required - uses your labeled data directly.
Optimized for M2 with class balancing for imbalanced zones.

Usage:
    python scripts/train_user_only.py --input results/serato_frames_m2.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from datetime import datetime

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score
)
from xgboost import XGBClassifier

# Zone order
ZONES = ['YELLOW', 'GREEN', 'PURPLE']


def train_model(input_path: str, output_path: str, top_n: int = 50):
    """
    Train XGBoost classifier on user frames data.

    Uses StratifiedGroupKFold to ensure:
    - Tracks don't leak between train/test
    - Zone distribution is preserved in each fold
    """
    print("\n" + "=" * 60)
    print("🎯 Training Zone Classifier (User Data Only)")
    print("=" * 60)

    # Load data
    print(f"\n📂 Loading {input_path}...")
    df = pd.read_pickle(input_path)

    print(f"   Frames: {len(df):,}")
    print(f"   Tracks: {df['track_id'].nunique()}")
    print(f"   Columns: {len(df.columns)}")

    # Zone distribution
    track_zones = df.groupby('track_id')['zone'].first()
    zone_counts = Counter(track_zones)
    print(f"\n📊 Zone distribution:")
    for zone in ZONES:
        count = zone_counts.get(zone, 0)
        pct = count / len(track_zones) * 100
        bar = "█" * int(pct / 2)
        print(f"   {zone:8s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Select features (exclude metadata columns)
    exclude_cols = ['zone', 'track_id', 'frame_idx', 'source', 'path', 'file',
                    'filename', 'source_dataset', 'frameTime']
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_cols]

    print(f"\n🔧 Features: {len(feature_cols)}")

    # Filter out rows with NaN zones
    valid_mask = df['zone'].notna() & df['zone'].isin(ZONES)
    df = df[valid_mask].copy()
    print(f"   After filtering NaN zones: {len(df):,} frames")

    # Prepare data
    X = df[feature_cols].values
    y = df['zone'].values
    groups = df['track_id'].astype(str).values

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Encode labels
    le = LabelEncoder()
    le.classes_ = np.array(ZONES)
    y_encoded = le.transform(y)

    # Calculate class weights for imbalanced data
    counts = Counter(y_encoded)
    total = len(y_encoded)
    n_classes = len(counts)
    class_weights = {c: total / (n_classes * count) for c, count in counts.items()}

    print(f"\n⚖️  Class weights (for imbalance):")
    for zone, weight in zip(ZONES, [class_weights.get(i, 1.0) for i in range(3)]):
        print(f"   {zone}: {weight:.2f}x")

    # Cross-validation
    print(f"\n🔄 5-Fold Stratified Group CV...")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    all_preds = {}  # track_id -> predictions

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded, groups)):
        print(f"   Fold {fold + 1}/5...", end=" ", flush=True)

        # Create model
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

        # Train with sample weights
        sample_weights = np.array([class_weights[yi] for yi in y_encoded[train_idx]])
        model.fit(X[train_idx], y_encoded[train_idx], sample_weight=sample_weights)

        # Predict
        preds = model.predict(X[test_idx])
        test_groups = groups[test_idx]

        # Store predictions by track
        for pred, tid, true_label in zip(preds, test_groups, y[test_idx]):
            if tid not in all_preds:
                all_preds[tid] = {'true': true_label, 'frame_preds': []}
            all_preds[tid]['frame_preds'].append(le.inverse_transform([pred])[0])

        # Fold accuracy (frame-level)
        fold_acc = accuracy_score(y_encoded[test_idx], preds)
        fold_results.append(fold_acc)
        print(f"accuracy: {fold_acc:.3f}")

    # Aggregate to track level (majority vote)
    track_true = []
    track_pred = []
    for tid, data in all_preds.items():
        track_true.append(data['true'])
        # Majority vote
        vote_counts = Counter(data['frame_preds'])
        track_pred.append(vote_counts.most_common(1)[0][0])

    # Track-level metrics
    track_acc = accuracy_score(track_true, track_pred)
    track_balanced_acc = balanced_accuracy_score(track_true, track_pred)
    track_f1_macro = f1_score(track_true, track_pred, average='macro')

    print("\n" + "=" * 60)
    print("📊 CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"\n   Frame-level CV accuracy: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"   Track-level accuracy:    {track_acc:.4f}")
    print(f"   Track-level balanced:    {track_balanced_acc:.4f}")
    print(f"   Track-level F1 (macro):  {track_f1_macro:.4f}")

    # Confusion matrix
    print(f"\n📋 Confusion Matrix (track-level):")
    cm = confusion_matrix(track_true, track_pred, labels=ZONES)
    print(f"              Predicted")
    print(f"              {'  '.join([z[:3] for z in ZONES])}")
    for i, zone in enumerate(ZONES):
        row = "  ".join([f"{cm[i, j]:3d}" for j in range(3)])
        print(f"   {zone:8s}  {row}")

    # Per-class metrics
    print(f"\n📈 Per-class metrics:")
    report = classification_report(track_true, track_pred, target_names=ZONES, output_dict=True)
    print(f"   {'Zone':10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    for zone in ZONES:
        m = report[zone]
        print(f"   {zone:10s} {m['precision']:10.2f} {m['recall']:10.2f} {m['f1-score']:10.2f} {m['support']:10.0f}")

    # Train final model on all data
    print(f"\n🏋️  Training final model on all data...")
    final_model = XGBClassifier(
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

    sample_weights = np.array([class_weights[yi] for yi in y_encoded])
    final_model.fit(X, y_encoded, sample_weight=sample_weights)

    # Feature importance
    print(f"\n🔝 Top {top_n} most important features:")
    importance = final_model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    for rank, idx in enumerate(indices[:15], 1):
        print(f"   {rank:2d}. {feature_cols[idx]:30s} {importance[idx]:.4f}")
    if top_n > 15:
        print(f"   ... and {top_n - 15} more")

    # Save model
    save_data = {
        'model': final_model,
        'feature_cols': feature_cols,
        'label_encoder': le,
        'class_weights': class_weights,
        'cv_accuracy': np.mean(fold_results),
        'track_accuracy': track_acc,
        'track_balanced_accuracy': track_balanced_acc,
        'training_date': datetime.now().isoformat(),
        'training_data': input_path,
        'n_tracks': len(track_zones),
        'n_frames': len(df),
        'zone_distribution': dict(zone_counts),
    }

    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\n💾 Model saved: {output_path}")
    print("=" * 60)

    return save_data


def main():
    parser = argparse.ArgumentParser(
        description="🎯 Train Zone Classifier on User Data Only"
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Input PKL file with frames")
    parser.add_argument("--output", "-o", default="models/production/zone_classifier_user.pkl",
                        help="Output model path")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of top features to show")

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    train_model(args.input, args.output, args.top_n)

    print("\n✅ Done!")
    print(f"\n📌 To use the model:")
    print(f"   from src.training.zone_predictor import ZonePredictor")
    print(f"   predictor = ZonePredictor('{args.output}')")
    print(f"   zone = predictor.predict('path/to/track.mp3')")


if __name__ == "__main__":
    main()
