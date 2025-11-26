#!/usr/bin/env python3
"""
DEAM Distribution Visualization with Zone Mapping

Показывает реальное распределение треков DEAM в 2D пространстве arousal-valence
с наложением зон и статистикой по плотности.
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


def visualize_deam_distribution_2d(
    deam_dir: str = "dataset",
    yellow_arousal: float = 4.0,
    purple_arousal: float = 6.0,
    valence_threshold: float = 4.5,
    output_path: str = "results/deam_distribution_2d.png"
):
    """
    Визуализирует реальное распределение DEAM треков с зонами

    Args:
        deam_dir: Директория с DEAM данными
        yellow_arousal: Порог для Yellow зоны
        purple_arousal: Порог для Purple зоны
        valence_threshold: Порог разделения negative/positive
        output_path: Путь сохранения
    """

    # Загружаем DEAM аннотации
    print("📂 Загружаем DEAM аннотации...")
    loader = DEAMLoader(dataset_root=deam_dir)
    annotations = loader.load_annotations()

    print(f"✅ Загружено {len(annotations)} треков")
    print(f"   Arousal range: [{annotations['arousal'].min():.2f}, {annotations['arousal'].max():.2f}]")
    print(f"   Valence range: [{annotations['valence'].min():.2f}, {annotations['valence'].max():.2f}]")

    arousal_min = annotations['arousal'].min()
    arousal_max = annotations['arousal'].max()
    valence_min = annotations['valence'].min()
    valence_max = annotations['valence'].max()

    # Создаём фигуру
    fig, ax = plt.subplots(figsize=(16, 12))

    # === ЗОНЫ (полупрозрачные фоны) ===

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

    # === ГРАНИЦЫ ЗОН ===

    ax.axhline(y=yellow_arousal, color='gold', linestyle='--', linewidth=2, alpha=0.6)
    ax.axhline(y=purple_arousal, color='purple', linestyle='--', linewidth=2, alpha=0.6)
    ax.axvline(x=valence_threshold, color='gray', linestyle='-', linewidth=2, alpha=0.5)

    # === HEAT MAP (2D HISTOGRAM) ===

    # Создаём 2D гистограмму для плотности
    print("🔥 Создаём heat map плотности...")

    h, xedges, yedges = np.histogram2d(
        annotations['valence'],
        annotations['arousal'],
        bins=[40, 40],
        range=[[valence_min, valence_max], [arousal_min, arousal_max]]
    )

    # Транспонируем для корректного отображения
    h = h.T

    # Кастомная colormap (от прозрачного к насыщенному)
    colors = ['#ffffff00', '#ff000020', '#ff000040', '#ff0000', '#8b0000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('density', colors, N=n_bins)

    # Отображаем heat map
    im = ax.imshow(
        h,
        extent=[valence_min, valence_max, arousal_min, arousal_max],
        origin='lower',
        cmap=cmap,
        alpha=0.6,
        aspect='auto'
    )

    # Добавляем colorbar
    cbar = plt.colorbar(im, ax=ax, label='Плотность треков', pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    # === SCATTER PLOT (точки треков) ===

    # Добавляем точки для наглядности (с прозрачностью)
    ax.scatter(
        annotations['valence'],
        annotations['arousal'],
        c='black',
        s=8,
        alpha=0.3,
        edgecolors='none',
        label=f'DEAM треки (n={len(annotations)})'
    )

    # === СТАТИСТИКА ПО ЗОНАМ ===

    # Классифицируем треки по зонам
    def classify_zone(row):
        a = row['arousal']
        v = row['valence']

        if a < yellow_arousal:
            return 'YELLOW_CHILL' if v >= valence_threshold else 'YELLOW_DARK'
        elif a > purple_arousal:
            return 'PURPLE_EUPHORIC' if v >= valence_threshold else 'PURPLE_AGGRESSIVE'
        else:
            return 'GREEN'

    annotations['zone'] = annotations.apply(classify_zone, axis=1)
    zone_counts = annotations['zone'].value_counts()
    total = len(annotations)

    print("\n📊 Распределение по зонам:")
    for zone, count in zone_counts.items():
        pct = count / total * 100
        print(f"  {zone}: {count} ({pct:.1f}%)")

    # === МЕТКИ С КОЛИЧЕСТВОМ ТРЕКОВ ===

    label_fontsize = 12
    count_fontsize = 16

    # YELLOW CHILL
    yc_count = zone_counts.get('YELLOW_CHILL', 0)
    yc_pct = (yc_count / total) * 100
    ax.text(
        valence_threshold + (valence_max - valence_threshold) / 2,
        arousal_min + (yellow_arousal - arousal_min) / 2,
        f'YELLOW CHILL\n\n{yc_count} треков\n({yc_pct:.1f}%)',
        fontsize=label_fontsize,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='gold', linewidth=2)
    )

    # YELLOW DARK
    yd_count = zone_counts.get('YELLOW_DARK', 0)
    yd_pct = (yd_count / total) * 100
    ax.text(
        valence_min + (valence_threshold - valence_min) / 2,
        arousal_min + (yellow_arousal - arousal_min) / 2,
        f'YELLOW DARK\n\n{yd_count} треков\n({yd_pct:.1f}%)',
        fontsize=label_fontsize,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='darkgoldenrod', linewidth=2)
    )

    # GREEN
    g_count = zone_counts.get('GREEN', 0)
    g_pct = (g_count / total) * 100
    ax.text(
        (valence_min + valence_max) / 2,
        (yellow_arousal + purple_arousal) / 2,
        f'GREEN\n\n{g_count} треков\n({g_pct:.1f}%)',
        fontsize=label_fontsize,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=2)
    )

    # PURPLE EUPHORIC
    pe_count = zone_counts.get('PURPLE_EUPHORIC', 0)
    pe_pct = (pe_count / total) * 100
    ax.text(
        valence_threshold + (valence_max - valence_threshold) / 2,
        purple_arousal + (arousal_max - purple_arousal) / 2,
        f'PURPLE\nEUPHORIC\n\n{pe_count} треков\n({pe_pct:.1f}%)',
        fontsize=label_fontsize,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='violet', alpha=0.7, edgecolor='purple', linewidth=2)
    )

    # PURPLE AGGRESSIVE
    pa_count = zone_counts.get('PURPLE_AGGRESSIVE', 0)
    pa_pct = (pa_count / total) * 100
    ax.text(
        valence_min + (valence_threshold - valence_min) / 2,
        purple_arousal + (arousal_max - purple_arousal) / 2,
        f'PURPLE\nAGGRESSIVE\n\n{pa_count} треков\n({pa_pct:.1f}%)',
        fontsize=label_fontsize,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7, edgecolor='darkred', linewidth=2)
    )

    # === НАСТРОЙКИ ОСЕЙ ===

    ax.set_xlim(valence_min - 0.2, valence_max + 0.2)
    ax.set_ylim(arousal_min - 0.2, arousal_max + 0.2)

    ax.set_xlabel('Valence (Настроение: Негативное ← → Позитивное)', fontsize=14, weight='bold')
    ax.set_ylabel('Arousal (Энергия: Низкая ↑ Высокая)', fontsize=14, weight='bold')

    ax.set_title(
        f'Распределение DEAM треков ({len(annotations)}) по энергетическим зонам\n'
        f'Heat map показывает плотность треков | '
        f'Пороги: Arousal ({yellow_arousal}/{purple_arousal}), Valence ({valence_threshold})',
        fontsize=14,
        weight='bold',
        pad=20
    )

    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # === АННОТАЦИЯ ===

    info_text = (
        f"Dataset: DEAM ({len(annotations)} треков)\n"
        f"Arousal: [{arousal_min:.1f}, {arousal_max:.1f}]\n"
        f"Valence: [{valence_min:.1f}, {valence_max:.1f}]\n"
        "\n"
        "Распределение по зонам:\n"
        f"• YELLOW: {yc_count + yd_count} ({(yc_count+yd_count)/total*100:.1f}%)\n"
        f"• GREEN: {g_count} ({g_pct:.1f}%)\n"
        f"• PURPLE: {pe_count + pa_count} ({(pe_count+pa_count)/total*100:.1f}%)"
    )

    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray', linewidth=1)
    )

    # === СОХРАНЕНИЕ ===

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    plt.close()

    return annotations, zone_counts


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DEAM distribution with zone mapping'
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
        '--output', type=str, default='results/deam_distribution_2d.png',
        help='Output path (default: results/deam_distribution_2d.png)'
    )

    args = parser.parse_args()

    try:
        annotations, zone_counts = visualize_deam_distribution_2d(
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
