#!/usr/bin/env python3
"""
Predict Arousal-Valence for User Tracks

Применяет обученный arousal-valence регрессор к пользовательским трекам.
Создаёт предсказания arousal/valence и автоматически маппит на зоны.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.arousal_valence_regressor import ArousalValenceRegressor


def predict_arousal_valence(features_path: str = "results/user_tracks_features.csv",
                           model_dir: str = "models/arousal_valence",
                           output_path: str = "results/user_tracks_predictions.csv",
                           yellow_arousal: float = 4.0,
                           purple_arousal: float = 6.0,
                           valence_threshold: float = 4.5):
    """
    Предсказывает arousal и valence для пользовательских треков.

    Args:
        features_path: Путь к CSV с извлечёнными фичами
        model_dir: Директория с обученной моделью
        output_path: Путь для сохранения предсказаний
        yellow_arousal: Порог для Yellow зоны
        purple_arousal: Порог для Purple зоны
        valence_threshold: Порог negative/positive

    Returns:
        DataFrame с предсказаниями
    """
    print("=" * 80)
    print("🎯 AROUSAL-VALENCE PREDICTION FOR USER TRACKS")
    print("=" * 80)

    # Загрузка фичей
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}\n"
            f"Please run: python scripts/extract_user_track_features.py"
        )

    print(f"\n📂 Loading features from: {features_path}")
    features_df = pd.read_csv(features_path)

    print(f"✅ Loaded features for {len(features_df)} tracks")
    print(f"   Feature columns: {features_df.shape[1]}")

    # Определяем feature columns (исключаем метаданные)
    meta_columns = ['audio_path', 'zone', 'duration', 'sr']
    feature_columns = [col for col in features_df.columns if col not in meta_columns]

    print(f"\n📊 Feature columns ({len(feature_columns)}):")
    print(f"   {', '.join(feature_columns[:5])}...")

    # Extract X
    X = features_df[feature_columns].values

    # Check for NaNs
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"\n⚠️  Warning: Found {nan_mask.sum()} tracks with NaN features")
        print(f"   These tracks will be skipped")

        # Remove NaN rows
        valid_mask = ~nan_mask
        X = X[valid_mask]
        features_df = features_df[valid_mask].reset_index(drop=True)

        print(f"   Remaining tracks: {len(features_df)}")

    # Load regressor
    print(f"\n🔧 Loading arousal-valence regressor from: {model_dir}")

    regressor = ArousalValenceRegressor(model_dir=model_dir)
    regressor.load()

    print(f"✅ Regressor loaded successfully")

    # Predict with zones
    print(f"\n🎯 Predicting arousal, valence and zones...")
    print(f"   Zone thresholds:")
    print(f"     Yellow arousal < {yellow_arousal}")
    print(f"     Purple arousal > {purple_arousal}")
    print(f"     Valence threshold = {valence_threshold}")

    predictions_df = regressor.predict_with_zones(
        X,
        yellow_arousal=yellow_arousal,
        purple_arousal=purple_arousal,
        valence_threshold=valence_threshold
    )

    # Combine with original data
    result_df = pd.concat([
        features_df[['audio_path']].reset_index(drop=True),
        predictions_df
    ], axis=1)

    # Статистика
    print(f"\n✅ Predictions completed!")
    print(f"\n📈 Arousal statistics:")
    print(f"   Range: [{predictions_df['arousal'].min():.2f}, {predictions_df['arousal'].max():.2f}]")
    print(f"   Mean:  {predictions_df['arousal'].mean():.2f}")
    print(f"   Std:   {predictions_df['arousal'].std():.2f}")

    print(f"\n📈 Valence statistics:")
    print(f"   Range: [{predictions_df['valence'].min():.2f}, {predictions_df['valence'].max():.2f}]")
    print(f"   Mean:  {predictions_df['valence'].mean():.2f}")
    print(f"   Std:   {predictions_df['valence'].std():.2f}")

    # Zone distribution
    zone_counts = predictions_df['zone'].value_counts()
    total = len(predictions_df)

    print(f"\n🎨 Zone distribution:")
    for zone in sorted(zone_counts.index):
        count = zone_counts[zone]
        pct = (count / total) * 100
        print(f"   {zone:20s}: {count:4d} ({pct:5.1f}%)")

    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_path, index=False)
    print(f"\n💾 Predictions saved to: {output_path}")

    # Sample output
    print(f"\n📋 Sample predictions (first 5 tracks):")
    print(result_df.head(5).to_string(index=False))

    print(f"\n" + "=" * 80)
    print(f"✅ PREDICTION COMPLETED!")
    print(f"=" * 80)
    print(f"\nPredictions saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Visualize distribution in arousal-valence space")
    print(f"  2. Compare with DEAM dataset distribution")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Predict arousal-valence for user tracks'
    )
    parser.add_argument(
        '--features', type=str, default='results/user_tracks_features.csv',
        help='Features CSV path (default: results/user_tracks_features.csv)'
    )
    parser.add_argument(
        '--model-dir', type=str, default='models/arousal_valence',
        help='Model directory (default: models/arousal_valence)'
    )
    parser.add_argument(
        '--output', type=str, default='results/user_tracks_predictions.csv',
        help='Output CSV path (default: results/user_tracks_predictions.csv)'
    )
    parser.add_argument(
        '--yellow-arousal', type=float, default=4.0,
        help='Yellow arousal threshold (default: 4.0)'
    )
    parser.add_argument(
        '--purple-arousal', type=float, default=6.0,
        help='Purple arousal threshold (default: 6.0)'
    )
    parser.add_argument(
        '--valence-threshold', type=float, default=4.5,
        help='Valence threshold negative/positive (default: 4.5)'
    )

    args = parser.parse_args()

    try:
        predict_arousal_valence(
            features_path=args.features,
            model_dir=args.model_dir,
            output_path=args.output,
            yellow_arousal=args.yellow_arousal,
            purple_arousal=args.purple_arousal,
            valence_threshold=args.valence_threshold
        )
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
