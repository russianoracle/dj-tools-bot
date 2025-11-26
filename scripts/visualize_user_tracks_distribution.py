#!/usr/bin/env python3
"""
Visualize User Tracks vs DEAM Distribution

Визуализирует распределение пользовательских треков в arousal-valence пространстве
и сравнивает с распределением DEAM датасета.
"""

import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# Настройка кириллицы для matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.deam_loader import DEAMLoader


def visualize_user_tracks_vs_deam(
    user_predictions_path: str = "results/user_tracks_predictions.csv",
    deam_dir: str = "dataset",
    yellow_arousal: float = 4.0,
    purple_arousal: float = 6.0,
    valence_threshold: float = 4.5,
    output_path: str = "results/user_tracks_vs_deam_distribution.png"
):
    """
    Визуализирует распределение пользовательских треков vs DEAM.

    Args:
        user_predictions_path: Путь к CSV с предсказаниями для пользовательских треков
        deam_dir: Директория с DEAM данными
        yellow_arousal: Порог для Yellow зоны
        purple_arousal: Порог для Purple зоны
        valence_threshold: Порог negative/positive
        output_path: Путь для сохранения визуализации
    """

    print("=" * 80)
    print("📊 USER TRACKS VS DEAM DISTRIBUTION VISUALIZATION")
    print("=" * 80)

    # === LOAD USER PREDICTIONS ===

    user_predictions_path = Path(user_predictions_path)
    if not user_predictions_path.exists():
        raise FileNotFoundError(
            f"User predictions not found: {user_predictions_path}\n"
            f"Please run: python scripts/predict_user_tracks.py"
        )

    print(f"\n📂 Loading user track predictions from: {user_predictions_path}")
    user_df = pd.read_csv(user_predictions_path)

    print(f"✅ Loaded {len(user_df)} user tracks")
    print(f"   Arousal range: [{user_df['arousal'].min():.2f}, {user_df['arousal'].max():.2f}]")
    print(f"   Valence range: [{user_df['valence'].min():.2f}, {user_df['valence'].max():.2f}]")

    # === LOAD DEAM ANNOTATIONS ===

    print(f"\n📂 Loading DEAM annotations from: {deam_dir}")
    loader = DEAMLoader(dataset_root=deam_dir)
    deam_df = loader.load_annotations()

    print(f"✅ Loaded {len(deam_df)} DEAM tracks")
    print(f"   Arousal range: [{deam_df['arousal'].min():.2f}, {deam_df['arousal'].max():.2f}]")
    print(f"   Valence range: [{deam_df['valence'].min():.2f}, {deam_df['valence'].max():.2f}]")

    # === DETERMINE PLOT RANGES ===

    arousal_min = min(deam_df['arousal'].min(), user_df['arousal'].min())
    arousal_max = max(deam_df['arousal'].max(), user_df['arousal'].max())
    valence_min = min(deam_df['valence'].min(), user_df['valence'].min())
    valence_max = max(deam_df['valence'].max(), user_df['valence'].max())

    # === CREATE FIGURE ===

    fig, ax = plt.subplots(figsize=(16, 12))

    # === DRAW ZONES (полупрозрачные фоны) ===

    print(f"\n🎨 Drawing zones...")

    # 1. YELLOW CHILL (низкая энергия + позитив)
    yellow_chill = patches.Rectangle(
        (valence_threshold, arousal_min),
        valence_max - valence_threshold,
        yellow_arousal - arousal_min,
        linewidth=2,
        edgecolor='gold',
        facecolor='yellow',
        alpha=0.15,
    )
    ax.add_patch(yellow_chill)

    # 2. YELLOW DARK (низкая энергия + негатив)
    yellow_dark = patches.Rectangle(
        (valence_min, arousal_min),
        valence_threshold - valence_min,
        yellow_arousal - arousal_min,
        linewidth=2,
        edgecolor='darkgoldenrod',
        facecolor='wheat',
        alpha=0.15,
    )
    ax.add_patch(yellow_dark)

    # 3. GREEN (средняя энергия)
    green_zone = patches.Rectangle(
        (valence_min, yellow_arousal),
        valence_max - valence_min,
        purple_arousal - yellow_arousal,
        linewidth=2,
        edgecolor='darkgreen',
        facecolor='lightgreen',
        alpha=0.15,
    )
    ax.add_patch(green_zone)

    # 4. PURPLE EUPHORIC (высокая энергия + позитив)
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

    # 5. PURPLE AGGRESSIVE (высокая энергия + негатив)
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

    # === ZONE BOUNDARIES ===

    ax.axhline(y=yellow_arousal, color='gold', linestyle='--', linewidth=2, alpha=0.6)
    ax.axhline(y=purple_arousal, color='purple', linestyle='--', linewidth=2, alpha=0.6)
    ax.axvline(x=valence_threshold, color='gray', linestyle='-', linewidth=2, alpha=0.5)

    # === DEAM DENSITY HEATMAP ===

    print(f"🔥 Creating DEAM density heatmap...")

    h_deam, xedges, yedges = np.histogram2d(
        deam_df['valence'],
        deam_df['arousal'],
        bins=[40, 40],
        range=[[valence_min, valence_max], [arousal_min, arousal_max]]
    )

    h_deam = h_deam.T

    # Custom colormap (от прозрачного к красному)
    colors_deam = ['#ffffff00', '#ff000020', '#ff000040', '#ff0000', '#8b0000']
    cmap_deam = LinearSegmentedColormap.from_list('deam_density', colors_deam, N=100)

    im_deam = ax.imshow(
        h_deam,
        extent=[valence_min, valence_max, arousal_min, arousal_max],
        origin='lower',
        cmap=cmap_deam,
        alpha=0.4,
        aspect='auto',
        zorder=1
    )

    # === USER TRACKS SCATTER ===

    print(f"📍 Plotting user tracks...")

    ax.scatter(
        user_df['valence'],
        user_df['arousal'],
        c='blue',
        s=40,
        alpha=0.7,
        edgecolors='darkblue',
        linewidths=1,
        label=f'User Tracks (n={len(user_df)})',
        zorder=3
    )

    # === DEAM SCATTER (smaller, behind) ===

    ax.scatter(
        deam_df['valence'],
        deam_df['arousal'],
        c='red',
        s=8,
        alpha=0.2,
        edgecolors='none',
        label=f'DEAM Tracks (n={len(deam_df)})',
        zorder=2
    )

    # === ZONE STATISTICS ===

    print(f"\n📊 Calculating zone statistics...")

    def classify_zone(row):
        a = row['arousal']
        v = row['valence']

        if a < yellow_arousal:
            return 'YELLOW_CHILL' if v >= valence_threshold else 'YELLOW_DARK'
        elif a > purple_arousal:
            return 'PURPLE_EUPHORIC' if v >= valence_threshold else 'PURPLE_AGGRESSIVE'
        else:
            return 'GREEN'

    # User zones
    user_df['zone_computed'] = user_df.apply(classify_zone, axis=1)
    user_zone_counts = user_df['zone_computed'].value_counts()
    user_total = len(user_df)

    # DEAM zones
    deam_df['zone'] = deam_df.apply(classify_zone, axis=1)
    deam_zone_counts = deam_df['zone'].value_counts()
    deam_total = len(deam_df)

    print(f"\n📈 User tracks distribution:")
    for zone, count in user_zone_counts.items():
        pct = count / user_total * 100
        print(f"  {zone}: {count} ({pct:.1f}%)")

    print(f"\n📈 DEAM distribution:")
    for zone, count in deam_zone_counts.items():
        pct = count / deam_total * 100
        print(f"  {zone}: {count} ({pct:.1f}%)")

    # === ZONE LABELS WITH STATISTICS ===

    label_fontsize = 11
    zone_info_fontsize = 9

    def get_zone_label(zone_name, user_count, deam_count):
        user_pct = (user_count / user_total) * 100 if user_total > 0 else 0
        deam_pct = (deam_count / deam_total) * 100 if deam_total > 0 else 0
        return (f"{zone_name}\n\n"
                f"User: {user_count} ({user_pct:.1f}%)\n"
                f"DEAM: {deam_count} ({deam_pct:.1f}%)")

    # YELLOW CHILL
    yc_user = user_zone_counts.get('YELLOW_CHILL', 0)
    yc_deam = deam_zone_counts.get('YELLOW_CHILL', 0)
    ax.text(
        valence_threshold + (valence_max - valence_threshold) / 2,
        arousal_min + (yellow_arousal - arousal_min) / 2,
        get_zone_label('YELLOW CHILL', yc_user, yc_deam),
        fontsize=label_fontsize,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='gold', linewidth=2)
    )

    # YELLOW DARK
    yd_user = user_zone_counts.get('YELLOW_DARK', 0)
    yd_deam = deam_zone_counts.get('YELLOW_DARK', 0)
    ax.text(
        valence_min + (valence_threshold - valence_min) / 2,
        arousal_min + (yellow_arousal - arousal_min) / 2,
        get_zone_label('YELLOW DARK', yd_user, yd_deam),
        fontsize=label_fontsize,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='darkgoldenrod', linewidth=2)
    )

    # GREEN
    g_user = user_zone_counts.get('GREEN', 0)
    g_deam = deam_zone_counts.get('GREEN', 0)
    ax.text(
        (valence_min + valence_max) / 2,
        (yellow_arousal + purple_arousal) / 2,
        get_zone_label('GREEN', g_user, g_deam),
        fontsize=label_fontsize,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=2)
    )

    # PURPLE EUPHORIC
    pe_user = user_zone_counts.get('PURPLE_EUPHORIC', 0)
    pe_deam = deam_zone_counts.get('PURPLE_EUPHORIC', 0)
    ax.text(
        valence_threshold + (valence_max - valence_threshold) / 2,
        purple_arousal + (arousal_max - purple_arousal) / 2,
        get_zone_label('PURPLE\nEUPHORIC', pe_user, pe_deam),
        fontsize=label_fontsize,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='violet', alpha=0.7, edgecolor='purple', linewidth=2)
    )

    # PURPLE AGGRESSIVE
    pa_user = user_zone_counts.get('PURPLE_AGGRESSIVE', 0)
    pa_deam = deam_zone_counts.get('PURPLE_AGGRESSIVE', 0)
    ax.text(
        valence_min + (valence_threshold - valence_min) / 2,
        purple_arousal + (arousal_max - purple_arousal) / 2,
        get_zone_label('PURPLE\nAGGRESSIVE', pa_user, pa_deam),
        fontsize=label_fontsize,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7, edgecolor='darkred', linewidth=2)
    )

    # === AXES & LABELS ===

    ax.set_xlim(valence_min - 0.2, valence_max + 0.2)
    ax.set_ylim(arousal_min - 0.2, arousal_max + 0.2)

    ax.set_xlabel('Valence (Настроение: Негативное ← → Позитивное)', fontsize=14, weight='bold')
    ax.set_ylabel('Arousal (Энергия: Низкая ↑ Высокая)', fontsize=14, weight='bold')

    ax.set_title(
        f'User Tracks ({len(user_df)}) vs DEAM ({len(deam_df)}) Distribution\n'
        f'Blue = User Tracks | Red heatmap = DEAM density | '
        f'Thresholds: Arousal ({yellow_arousal}/{purple_arousal}), Valence ({valence_threshold})',
        fontsize=14,
        weight='bold',
        pad=20
    )

    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # === INFO BOX ===

    info_text = (
        f"User Tracks: {len(user_df)}\n"
        f"DEAM Reference: {len(deam_df)}\n"
        f"\n"
        f"User Arousal: [{user_df['arousal'].min():.1f}, {user_df['arousal'].max():.1f}] (μ={user_df['arousal'].mean():.1f})\n"
        f"User Valence: [{user_df['valence'].min():.1f}, {user_df['valence'].max():.1f}] (μ={user_df['valence'].mean():.1f})\n"
        f"\n"
        f"Zone Distribution Comparison:\n"
        f"• YELLOW: User {yc_user+yd_user} vs DEAM {yc_deam+yd_deam}\n"
        f"• GREEN: User {g_user} vs DEAM {g_deam}\n"
        f"• PURPLE: User {pe_user+pa_user} vs DEAM {pe_deam+pa_deam}"
    )

    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray', linewidth=1)
    )

    # === SAVE ===

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    plt.close()

    print(f"\n" + "=" * 80)
    print(f"✅ VISUALIZATION COMPLETED!")
    print(f"=" * 80)

    return user_df, deam_df


def main():
    parser = argparse.ArgumentParser(
        description='Visualize user tracks vs DEAM distribution'
    )
    parser.add_argument(
        '--user-predictions', type=str, default='results/user_tracks_predictions.csv',
        help='User predictions CSV (default: results/user_tracks_predictions.csv)'
    )
    parser.add_argument(
        '--deam-dir', type=str, default='dataset',
        help='DEAM dataset directory (default: dataset)'
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
    parser.add_argument(
        '--output', type=str, default='results/user_tracks_vs_deam_distribution.png',
        help='Output path (default: results/user_tracks_vs_deam_distribution.png)'
    )

    args = parser.parse_args()

    try:
        visualize_user_tracks_vs_deam(
            user_predictions_path=args.user_predictions,
            deam_dir=args.deam_dir,
            yellow_arousal=args.yellow_arousal,
            purple_arousal=args.purple_arousal,
            valence_threshold=args.valence_threshold,
            output_path=args.output
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
