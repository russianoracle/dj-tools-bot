#!/usr/bin/env python3
"""
DJ Flow Map: Emotional Transitions Visualization

Показывает как DJ использует треки для управления настроением аудитории.
Визуализирует типичные переходы между эмоциональными зонами с учётом
плотности треков-мостов для плавных переходов.
"""

import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from pathlib import Path

# Настройка кириллицы для matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.deam_loader import DEAMLoader


def visualize_dj_flow(
    deam_dir: str = "dataset",
    yellow_arousal: float = 4.0,
    purple_arousal: float = 6.0,
    valence_threshold: float = 4.5,
    output_path: str = "results/dj_flow_map.png"
):
    """
    Визуализирует DJ flow: пути управления настроением аудитории

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

    arousal_min = annotations['arousal'].min()
    arousal_max = annotations['arousal'].max()
    valence_min = annotations['valence'].min()
    valence_max = annotations['valence'].max()

    # Классифицируем треки
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

    # Создаём фигуру
    fig, ax = plt.subplots(figsize=(18, 14))

    # === ЗОНЫ (полупрозрачные фоны) ===

    zones = [
        # (x, y, width, height, color, edge, label)
        (valence_threshold, arousal_min, valence_max - valence_threshold,
         yellow_arousal - arousal_min, 'yellow', 'gold', 'YELLOW\nCHILL'),

        (valence_min, arousal_min, valence_threshold - valence_min,
         yellow_arousal - arousal_min, 'wheat', 'darkgoldenrod', 'YELLOW\nDARK'),

        (valence_min, yellow_arousal, valence_max - valence_min,
         purple_arousal - yellow_arousal, 'lightgreen', 'darkgreen', 'GREEN'),

        (valence_threshold, purple_arousal, valence_max - valence_threshold,
         arousal_max - purple_arousal, 'violet', 'purple', 'PURPLE\nEUPHORIC'),

        (valence_min, purple_arousal, valence_threshold - valence_min,
         arousal_max - purple_arousal, 'plum', 'darkred', 'PURPLE\nAGGRESSIVE'),
    ]

    for x, y, w, h, color, edge, label in zones:
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                 edgecolor=edge, facecolor=color, alpha=0.12)
        ax.add_patch(rect)

    # Границы
    ax.axhline(y=yellow_arousal, color='gold', linestyle='--', linewidth=2, alpha=0.4)
    ax.axhline(y=purple_arousal, color='purple', linestyle='--', linewidth=2, alpha=0.4)
    ax.axvline(x=valence_threshold, color='gray', linestyle='-', linewidth=2, alpha=0.3)

    # === ЦЕНТРЫ ЗОН для стрелок ===

    zone_centers = {
        'YELLOW_CHILL': (
            valence_threshold + (valence_max - valence_threshold) / 2,
            arousal_min + (yellow_arousal - arousal_min) / 2
        ),
        'YELLOW_DARK': (
            valence_min + (valence_threshold - valence_min) / 2,
            arousal_min + (yellow_arousal - arousal_min) / 2
        ),
        'GREEN': (
            (valence_min + valence_max) / 2,
            (yellow_arousal + purple_arousal) / 2
        ),
        'PURPLE_EUPHORIC': (
            valence_threshold + (valence_max - valence_threshold) / 2,
            purple_arousal + (arousal_max - purple_arousal) / 2
        ),
        'PURPLE_AGGRESSIVE': (
            valence_min + (valence_threshold - valence_min) / 2,
            purple_arousal + (arousal_max - purple_arousal) / 2
        ),
    }

    # === ТИПИЧНЫЕ DJ FLOW ПАТТЕРНЫ ===

    # Паттерн 1: Classic Warm-Up → Peak → Cool-Down (EUPHORIC PATH)
    flow_patterns = [
        {
            'name': 'Classic Euphoric Flow',
            'path': ['YELLOW_CHILL', 'GREEN', 'PURPLE_EUPHORIC', 'GREEN', 'YELLOW_CHILL'],
            'color': 'green',
            'description': 'Разогрев → Build-up → Кульминация → Cool-down',
            'weight': 5,
            'alpha': 0.7
        },

        # Паттерн 2: Underground Dark Flow
        {
            'name': 'Dark Underground Flow',
            'path': ['YELLOW_DARK', 'GREEN', 'PURPLE_AGGRESSIVE'],
            'color': 'darkred',
            'description': 'Тёмный разогрев → Агрессивная кульминация',
            'weight': 3,
            'alpha': 0.6
        },

        # Паттерн 3: Quick Energy Boost
        {
            'name': 'Energy Boost',
            'path': ['YELLOW_CHILL', 'PURPLE_EUPHORIC'],
            'color': 'orange',
            'description': 'Резкий подъём энергии',
            'weight': 2,
            'alpha': 0.5
        },

        # Паттерн 4: Mood Shifter (Dark → Light)
        {
            'name': 'Mood Shifter',
            'path': ['YELLOW_DARK', 'GREEN', 'PURPLE_EUPHORIC'],
            'color': 'blue',
            'description': 'Смена настроения: темное → светлое',
            'weight': 3,
            'alpha': 0.5
        },

        # Паттерн 5: Peak-to-Peak (Aggressive → Euphoric)
        {
            'name': 'Peak Transition',
            'path': ['PURPLE_AGGRESSIVE', 'GREEN', 'PURPLE_EUPHORIC'],
            'color': 'purple',
            'description': 'Переход между пиками',
            'weight': 2,
            'alpha': 0.4
        },
    ]

    # === РИСУЕМ FLOW СТРЕЛКИ ===

    print("🔄 Создаём flow patterns...")

    # Сначала рисуем стрелки
    arrow_offset = 0.15  # Смещение для множественных стрелок

    for pattern_idx, pattern in enumerate(flow_patterns):
        path = pattern['path']
        color = pattern['color']
        weight = pattern['weight']
        alpha = pattern['alpha']

        for i in range(len(path) - 1):
            from_zone = path[i]
            to_zone = path[i + 1]

            if from_zone not in zone_centers or to_zone not in zone_centers:
                continue

            x1, y1 = zone_centers[from_zone]
            x2, y2 = zone_centers[to_zone]

            # Добавляем небольшое смещение для множественных стрелок
            offset_x = (pattern_idx - len(flow_patterns)/2) * arrow_offset * 0.3
            offset_y = (pattern_idx - len(flow_patterns)/2) * arrow_offset * 0.2

            # Стрелка с изгибом
            arrow = FancyArrowPatch(
                (x1 + offset_x, y1 + offset_y),
                (x2 + offset_x, y2 + offset_y),
                arrowstyle='->, head_width=0.6, head_length=0.8',
                connectionstyle=f"arc3,rad=0.2",
                color=color,
                linewidth=weight,
                alpha=alpha,
                zorder=10
            )
            ax.add_patch(arrow)

    # === HEAT MAP треков ===

    print("🔥 Добавляем heat map плотности...")

    from matplotlib.colors import LinearSegmentedColormap

    h, xedges, yedges = np.histogram2d(
        annotations['valence'],
        annotations['arousal'],
        bins=[40, 40],
        range=[[valence_min, valence_max], [arousal_min, arousal_max]]
    )
    h = h.T

    colors = ['#ffffff00', '#00000010', '#00000020', '#000000', '#000000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('density', colors, N=n_bins)

    im = ax.imshow(
        h,
        extent=[valence_min, valence_max, arousal_min, arousal_max],
        origin='lower',
        cmap=cmap,
        alpha=0.3,
        aspect='auto',
        zorder=1
    )

    # === МЕТКИ ЗОН с количеством треков ===

    zone_counts = annotations['zone'].value_counts()
    total = len(annotations)

    labels = {
        'YELLOW_CHILL': f"YELLOW CHILL\n\nОтдых + Позитив\n{zone_counts.get('YELLOW_CHILL', 0)} треков",
        'YELLOW_DARK': f"YELLOW DARK\n\nОтдых + Меланхолия\n{zone_counts.get('YELLOW_DARK', 0)} треков",
        'GREEN': f"GREEN\n\nПереходная зона\n(Build-up мосты)\n{zone_counts.get('GREEN', 0)} треков",
        'PURPLE_EUPHORIC': f"PURPLE\nEUPHORIC\n\nКульминация + Эйфория\n{zone_counts.get('PURPLE_EUPHORIC', 0)} треков",
        'PURPLE_AGGRESSIVE': f"PURPLE\nAGGRESSIVE\n\nЭнергия + Агрессия\n{zone_counts.get('PURPLE_AGGRESSIVE', 0)} треков",
    }

    colors_map = {
        'YELLOW_CHILL': ('yellow', 'gold'),
        'YELLOW_DARK': ('wheat', 'darkgoldenrod'),
        'GREEN': ('lightgreen', 'darkgreen'),
        'PURPLE_EUPHORIC': ('violet', 'purple'),
        'PURPLE_AGGRESSIVE': ('plum', 'darkred'),
    }

    for zone, (x, y) in zone_centers.items():
        bg_color, edge_color = colors_map[zone]
        ax.text(
            x, y,
            labels[zone],
            fontsize=10,
            weight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor=bg_color,
                     alpha=0.8, edgecolor=edge_color, linewidth=2),
            zorder=15
        )

    # === ЛЕГЕНДА FLOW PATTERNS ===

    legend_elements = []
    for pattern in flow_patterns:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], color=pattern['color'], linewidth=pattern['weight'],
                   label=f"{pattern['name']}\n{pattern['description']}", alpha=pattern['alpha'])
        )

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             framealpha=0.95, title='DJ Flow Patterns', title_fontsize=10)

    # === НАСТРОЙКИ ОСЕЙ ===

    ax.set_xlim(valence_min - 0.3, valence_max + 0.3)
    ax.set_ylim(arousal_min - 0.3, arousal_max + 0.3)

    ax.set_xlabel('Valence (Настроение: Негативное ← → Позитивное)',
                 fontsize=14, weight='bold')
    ax.set_ylabel('Arousal (Энергия: Низкая ↑ Высокая)',
                 fontsize=14, weight='bold')

    ax.set_title(
        'DJ Flow Map: Управление настроением аудитории через эмоциональные переходы\n'
        f'Heat map = плотность треков-мостов | Стрелки = типичные DJ flow паттерны',
        fontsize=14,
        weight='bold',
        pad=20
    )

    ax.grid(True, alpha=0.2, linestyle='--', zorder=0)

    # === АННОТАЦИЯ ===

    info_text = (
        f"Dataset: {len(annotations)} DEAM треков\n"
        "\n"
        "Ключевые концепции:\n"
        "• Стрелки = пути управления энергией\n"
        "• Толщина стрелки = частота использования\n"
        "• Heat map = плотность треков-мостов\n"
        "• GREEN зона = критична для плавных переходов\n"
        "\n"
        f"Мостовых треков (GREEN): {zone_counts.get('GREEN', 0)} ({zone_counts.get('GREEN', 0)/total*100:.1f}%)"
    )

    ax.text(
        0.98, 0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                 alpha=0.95, edgecolor='gray', linewidth=1)
    )

    # === СОХРАНЕНИЕ ===

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ DJ Flow Map saved to: {output_path}")
    plt.close()

    return annotations


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DJ flow patterns for emotional transitions'
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
        help='Valence threshold (default: 4.5)'
    )
    parser.add_argument(
        '--output', type=str, default='results/dj_flow_map.png',
        help='Output path (default: results/dj_flow_map.png)'
    )

    args = parser.parse_args()

    try:
        visualize_dj_flow(
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
