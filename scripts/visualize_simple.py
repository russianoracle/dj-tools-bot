#!/usr/bin/env python3
"""
Simple Arousal-Valence Distribution Visualization

Визуализирует распределение пользовательских треков vs DEAM датасет
в arousal-valence пространстве без использования DEAMLoader.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def visualize_distribution(
    user_predictions_path: str = "results/user_tracks_predictions.csv",
    deam_data_path: str = "dataset/deam_processed/deam_complete.csv",
    output_path: str = "results/arousal_valence_distribution.png",
    yellow_arousal: float = 4.0,
    purple_arousal: float = 6.0,
    valence_threshold: float = 4.5
):
    """
    Визуализирует распределение треков в arousal-valence пространстве.

    Args:
        user_predictions_path: Путь к предсказаниям пользовательских треков
        deam_data_path: Путь к DEAM complete CSV
        output_path: Путь для сохранения графика
        yellow_arousal: Порог для YELLOW зоны
        purple_arousal: Порог для PURPLE зоны
        valence_threshold: Порог для позитив/негатив
    """
    print("=" * 80)
    print("📊 AROUSAL-VALENCE DISTRIBUTION VISUALIZATION")
    print("=" * 80)

    # Загрузка данных
    print(f"\n📂 Loading user predictions: {user_predictions_path}")
    user_df = pd.read_csv(user_predictions_path)
    print(f"✅ Loaded {len(user_df)} user tracks")

    print(f"\n📂 Loading DEAM data: {deam_data_path}")
    deam_df = pd.read_csv(deam_data_path)
    print(f"✅ Loaded {len(deam_df)} DEAM tracks")

    # Определяем диапазоны
    arousal_min = min(deam_df['arousal'].min(), user_df['arousal'].min())
    arousal_max = max(deam_df['arousal'].max(), user_df['arousal'].max())
    valence_min = min(deam_df['valence'].min(), user_df['valence'].min())
    valence_max = max(deam_df['valence'].max(), user_df['valence'].max())

    # Создаём фигуру
    fig, ax = plt.subplots(figsize=(16, 12))

    # Рисуем зоны
    print(f"\n🎨 Drawing emotion zones...")

    # YELLOW CHILL (низкая энергия + позитив)
    yellow_chill = patches.Rectangle(
        (valence_threshold, arousal_min),
        valence_max - valence_threshold,
        yellow_arousal - arousal_min,
        linewidth=2,
        edgecolor='gold',
        facecolor='lightyellow',
        alpha=0.15,
    )
    ax.add_patch(yellow_chill)

    # YELLOW DARK (низкая энергия + негатив)
    yellow_dark = patches.Rectangle(
        (valence_min, arousal_min),
        valence_threshold - valence_min,
        yellow_arousal - arousal_min,
        linewidth=2,
        edgecolor='orange',
        facecolor='wheat',
        alpha=0.15,
    )
    ax.add_patch(yellow_dark)

    # GREEN POSITIVE (средняя энергия + позитив)
    green_positive = patches.Rectangle(
        (valence_threshold, yellow_arousal),
        valence_max - valence_threshold,
        purple_arousal - yellow_arousal,
        linewidth=2,
        edgecolor='green',
        facecolor='lightgreen',
        alpha=0.15,
    )
    ax.add_patch(green_positive)

    # GREEN NEGATIVE (средняя энергия + негатив)
    green_negative = patches.Rectangle(
        (valence_min, yellow_arousal),
        valence_threshold - valence_min,
        purple_arousal - yellow_arousal,
        linewidth=2,
        edgecolor='darkgreen',
        facecolor='palegreen',
        alpha=0.15,
    )
    ax.add_patch(green_negative)

    # PURPLE EUPHORIC (высокая энергия + позитив)
    purple_euphoric = patches.Rectangle(
        (valence_threshold, purple_arousal),
        valence_max - valence_threshold,
        arousal_max - purple_arousal,
        linewidth=2,
        edgecolor='purple',
        facecolor='violet',
        alpha=0.15,
    )
    ax.add_patch(purple_euphoric)

    # PURPLE AGGRESSIVE (высокая энергия + негатив)
    purple_aggressive = patches.Rectangle(
        (valence_min, purple_arousal),
        valence_threshold - valence_min,
        arousal_max - purple_arousal,
        linewidth=2,
        edgecolor='darkred',
        facecolor='plum',
        alpha=0.15,
    )
    ax.add_patch(purple_aggressive)

    # Границы зон
    ax.axhline(y=yellow_arousal, color='gold', linestyle='--', linewidth=2, alpha=0.6)
    ax.axhline(y=purple_arousal, color='purple', linestyle='--', linewidth=2, alpha=0.6)
    ax.axvline(x=valence_threshold, color='gray', linestyle='-', linewidth=2, alpha=0.5)

    # DEAM density heatmap
    print(f"🔥 Creating DEAM density heatmap...")
    h_deam, xedges, yedges = np.histogram2d(
        deam_df['valence'],
        deam_df['arousal'],
        bins=40,
        range=[[valence_min, valence_max], [arousal_min, arousal_max]]
    )

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        h_deam.T,
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap='YlOrRd',
        alpha=0.4,
        interpolation='gaussian'
    )

    # Colorbar для DEAM
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('DEAM Track Density', fontsize=12, weight='bold')

    # User tracks scatter
    print(f"📍 Plotting user tracks...")
    ax.scatter(
        user_df['valence'],
        user_df['arousal'],
        c='blue',
        s=100,
        alpha=0.7,
        edgecolors='navy',
        linewidths=1.5,
        label='User Tracks (414)',
        zorder=10
    )

    # Настройка осей
    ax.set_xlim(valence_min - 0.2, valence_max + 0.2)
    ax.set_ylim(arousal_min - 0.2, arousal_max + 0.2)

    ax.set_xlabel('Valence (Emotional Tone: Negative ← → Positive)', fontsize=14, weight='bold')
    ax.set_ylabel('Arousal (Energy Level: Calm ← → Energetic)', fontsize=14, weight='bold')

    ax.set_title(
        'User Tracks vs DEAM Dataset: Arousal-Valence Distribution\n'
        f'DEAM: {len(deam_df)} tracks (heatmap) | User: {len(user_df)} tracks (blue dots)',
        fontsize=16,
        weight='bold',
        pad=20
    )

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    # Zone labels
    label_offset = 0.3
    ax.text(
        (valence_threshold + valence_max) / 2,
        arousal_min + label_offset,
        'YELLOW CHILL',
        ha='center',
        va='center',
        fontsize=10,
        weight='bold',
        color='darkgoldenrod',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7)
    )

    ax.text(
        (valence_min + valence_threshold) / 2,
        arousal_min + label_offset,
        'YELLOW DARK',
        ha='center',
        va='center',
        fontsize=10,
        weight='bold',
        color='darkorange',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
    )

    ax.text(
        (valence_threshold + valence_max) / 2,
        (yellow_arousal + purple_arousal) / 2,
        'GREEN\nPOSITIVE',
        ha='center',
        va='center',
        fontsize=10,
        weight='bold',
        color='darkgreen',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7)
    )

    ax.text(
        (valence_min + valence_threshold) / 2,
        (yellow_arousal + purple_arousal) / 2,
        'GREEN\nNEGATIVE',
        ha='center',
        va='center',
        fontsize=10,
        weight='bold',
        color='darkgreen',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='palegreen', alpha=0.7)
    )

    ax.text(
        (valence_threshold + valence_max) / 2,
        purple_arousal + label_offset,
        'PURPLE\nEUPHORIC',
        ha='center',
        va='center',
        fontsize=10,
        weight='bold',
        color='purple',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='violet', alpha=0.7)
    )

    ax.text(
        (valence_min + valence_threshold) / 2,
        purple_arousal + label_offset,
        'PURPLE\nAGGRESSIVE',
        ha='center',
        va='center',
        fontsize=10,
        weight='bold',
        color='darkred',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='plum', alpha=0.7)
    )

    # Сохранение
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualization saved to: {output_path}")

    # Статистика
    print(f"\n📊 Distribution statistics:")
    print(f"\nDEAM Dataset ({len(deam_df)} tracks):")
    print(f"  Arousal: [{deam_df['arousal'].min():.2f}, {deam_df['arousal'].max():.2f}], mean={deam_df['arousal'].mean():.2f}")
    print(f"  Valence: [{deam_df['valence'].min():.2f}, {deam_df['valence'].max():.2f}], mean={deam_df['valence'].mean():.2f}")

    print(f"\nUser Tracks ({len(user_df)} tracks):")
    print(f"  Arousal: [{user_df['arousal'].min():.2f}, {user_df['arousal'].max():.2f}], mean={user_df['arousal'].mean():.2f}")
    print(f"  Valence: [{user_df['valence'].min():.2f}, {user_df['valence'].max():.2f}], mean={user_df['valence'].mean():.2f}")

    # Zone distribution
    user_zones = user_df['zone'].value_counts()
    print(f"\nUser Track Zone Distribution:")
    for zone, count in user_zones.items():
        pct = (count / len(user_df)) * 100
        print(f"  {zone:<20}: {count:3d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize user tracks vs DEAM distribution'
    )
    parser.add_argument(
        '--user-predictions', type=str,
        default='results/user_tracks_predictions.csv',
        help='User predictions CSV'
    )
    parser.add_argument(
        '--deam-data', type=str,
        default='dataset/deam_processed/deam_complete.csv',
        help='DEAM complete data CSV'
    )
    parser.add_argument(
        '--output', type=str,
        default='results/arousal_valence_distribution.png',
        help='Output visualization path'
    )
    parser.add_argument(
        '--yellow-arousal', type=float, default=4.0,
        help='Yellow arousal threshold'
    )
    parser.add_argument(
        '--purple-arousal', type=float, default=6.0,
        help='Purple arousal threshold'
    )
    parser.add_argument(
        '--valence-threshold', type=float, default=4.5,
        help='Valence threshold'
    )

    args = parser.parse_args()

    visualize_distribution(
        user_predictions_path=args.user_predictions,
        deam_data_path=args.deam_data,
        output_path=args.output,
        yellow_arousal=args.yellow_arousal,
        purple_arousal=args.purple_arousal,
        valence_threshold=args.valence_threshold
    )
