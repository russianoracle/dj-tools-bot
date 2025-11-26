#!/usr/bin/env python3
"""
Convert Features Format

Конвертирует сохранённые фичи из агрегированного формата
(ZoneFeatures objects) в развёрнутый формат (отдельные колонки).
"""

import sys
import pickle
import pandas as pd
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.zone_features import ZoneFeatures


def convert_features(
    input_pkl: str = "models/checkpoints/features.pkl",
    output_csv: str = "results/user_tracks_features_expanded.csv"
):
    """
    Конвертирует features.pkl в развёрнутый CSV формат.

    Args:
        input_pkl: Путь к сохранённым фичам
        output_csv: Путь для сохранения развёрнутого CSV
    """
    print("=" * 80)
    print("🔄 CONVERTING FEATURES FORMAT")
    print("=" * 80)

    input_pkl = Path(input_pkl)
    if not input_pkl.exists():
        raise FileNotFoundError(f"Features not found: {input_pkl}")

    print(f"\n📂 Loading features from: {input_pkl}")

    # Загружаем features
    with open(input_pkl, 'rb') as f:
        features_df = pickle.load(f)

    print(f"✅ Loaded {len(features_df)} tracks")
    print(f"   Columns: {list(features_df.columns)}")

    # Проверяем формат
    if 'features_list' not in features_df.columns:
        print(f"\n❌ Error: 'features_list' column not found")
        print(f"   Available columns: {list(features_df.columns)}")
        return None

    print(f"\n🔧 Expanding ZoneFeatures objects...")

    # Развёрнутые фичи
    expanded_rows = []

    for idx, row in features_df.iterrows():
        zone_features = row['features_list']

        # Если это ZoneFeatures объект, извлекаем фичи
        if isinstance(zone_features, ZoneFeatures):
            # Используем to_vector() для получения всех фичей
            feature_vector = zone_features.to_vector()

            # Названия фичей (соответствуют порядку в to_vector)
            feature_names = [
                'tempo',
                'zero_crossing_rate',
                'rms_energy',
                'spectral_centroid',
                'spectral_rolloff',
                'energy_variance',
                'mfcc_1_mean', 'mfcc_1_std',
                'mfcc_2_mean', 'mfcc_2_std',
                'mfcc_3_mean', 'mfcc_3_std',
                'mfcc_4_mean', 'mfcc_4_std',
                'mfcc_5_mean', 'mfcc_5_std',
                'low_energy',
                'brightness',
                'drop_intensity'
            ]

            # Создаём словарь с фичами
            feature_dict = {name: value for name, value in zip(feature_names, feature_vector)}

            # Добавляем audio_path
            feature_dict['audio_path'] = row['audio_path']

            expanded_rows.append(feature_dict)
        else:
            print(f"⚠️  Warning: Row {idx} has unexpected type: {type(zone_features)}")

    # Создаём DataFrame с развёрнутыми фичами
    expanded_df = pd.DataFrame(expanded_rows)

    # Переставляем audio_path в начало
    cols = ['audio_path'] + [col for col in expanded_df.columns if col != 'audio_path']
    expanded_df = expanded_df[cols]

    print(f"\n✅ Expansion completed!")
    print(f"   Original shape: {features_df.shape}")
    print(f"   Expanded shape: {expanded_df.shape}")
    print(f"   Feature columns: {expanded_df.shape[1] - 1}")

    # Сохраняем в CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    expanded_df.to_csv(output_csv, index=False)
    print(f"\n💾 Saved to: {output_csv}")

    # Статистика
    print(f"\n📊 Feature statistics:")
    print(f"   Columns: {', '.join(expanded_df.columns[:6].tolist())}...")
    print(f"\n   Sample (first 3 rows):")
    print(expanded_df.head(3).iloc[:, :6].to_string())

    # Проверка на NaN
    nan_count = expanded_df.isna().sum().sum()
    if nan_count > 0:
        print(f"\n⚠️  Warning: Found {nan_count} NaN values")
        nan_cols = expanded_df.columns[expanded_df.isna().any()].tolist()
        for col in nan_cols:
            nan_count_col = expanded_df[col].isna().sum()
            print(f"     - {col}: {nan_count_col} NaNs")
    else:
        print(f"\n✅ No NaN values found")

    print(f"\n" + "=" * 80)
    print(f"✅ CONVERSION COMPLETED!")
    print(f"=" * 80)
    print(f"\nNext step:")
    print(f"  python scripts/predict_user_tracks.py \\")
    print(f"    --features {output_csv} \\")
    print(f"    --model-dir models/arousal_valence \\")
    print(f"    --output results/user_tracks_predictions.csv")

    return expanded_df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert features from aggregated to expanded format'
    )
    parser.add_argument(
        '--input', type=str, default='models/checkpoints/features.pkl',
        help='Input features pickle (default: models/checkpoints/features.pkl)'
    )
    parser.add_argument(
        '--output', type=str, default='results/user_tracks_features_expanded.csv',
        help='Output CSV (default: results/user_tracks_features_expanded.csv)'
    )

    args = parser.parse_args()

    try:
        convert_features(
            input_pkl=args.input,
            output_csv=args.output
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
