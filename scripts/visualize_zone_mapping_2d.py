#!/usr/bin/env python3
"""
2D Zone Mapping Visualization (Arousal + Valence)

Визуализация зон с учётом обеих осей: arousal И valence.
Показывает как эмоциональная окраска влияет на классификацию.
"""

import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Настройка кириллицы для matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_2d_zone_mapping(
    arousal_min: float = 1.6,
    arousal_max: float = 8.1,
    valence_min: float = 1.6,
    valence_max: float = 8.4,
    output_path: str = "results/zone_mapping_2d.png"
):
    """
    2D маппинг зон с учётом arousal И valence

    Зоны:
    - YELLOW_CHILL: низкая энергия + позитив (правый нижний)
    - YELLOW_MELANCHOLY: низкая энергия + негатив (левый нижний)
    - GREEN: средняя энергия (любое настроение)
    - PURPLE_EUPHORIC: высокая энергия + позитив (правый верхний)
    - PURPLE_AGGRESSIVE: высокая энергия + негатив (левый верхний)
    """

    fig, ax = plt.subplots(figsize=(14, 11))

    # Пороги
    arousal_mid = (arousal_min + arousal_max) / 2
    valence_mid = (valence_min + valence_max) / 2

    # Пороги для зон (можно настроить)
    yellow_arousal = 4.0
    purple_arousal = 6.0
    valence_threshold = 4.5  # Разделитель negative/positive

    # === ЗОНЫ ===

    # 1. YELLOW_CHILL (низкая энергия + позитив)
    yellow_chill = patches.Rectangle(
        (valence_threshold, arousal_min),
        valence_max - valence_threshold,
        yellow_arousal - arousal_min,
        linewidth=2,
        edgecolor='gold',
        facecolor='yellow',
        alpha=0.4,
        label='YELLOW CHILL (отдых, позитив)'
    )
    ax.add_patch(yellow_chill)

    # 2. YELLOW_MELANCHOLY (низкая энергия + негатив)
    yellow_dark = patches.Rectangle(
        (valence_min, arousal_min),
        valence_threshold - valence_min,
        yellow_arousal - arousal_min,
        linewidth=2,
        edgecolor='darkgoldenrod',
        facecolor='wheat',
        alpha=0.4,
        label='YELLOW DARK (отдых, меланхолия)'
    )
    ax.add_patch(yellow_dark)

    # 3. GREEN (средняя энергия - весь диапазон valence)
    green_zone = patches.Rectangle(
        (valence_min, yellow_arousal),
        valence_max - valence_min,
        purple_arousal - yellow_arousal,
        linewidth=2,
        edgecolor='darkgreen',
        facecolor='lightgreen',
        alpha=0.4,
        label='GREEN (переход, любое настроение)'
    )
    ax.add_patch(green_zone)

    # 4. PURPLE_EUPHORIC (высокая энергия + позитив)
    purple_euphoric = patches.Rectangle(
        (valence_threshold, purple_arousal),
        valence_max - valence_threshold,
        arousal_max - purple_arousal,
        linewidth=2,
        edgecolor='purple',
        facecolor='violet',
        alpha=0.4,
        label='PURPLE EUPHORIC (кульминация, эйфория)'
    )
    ax.add_patch(purple_euphoric)

    # 5. PURPLE_AGGRESSIVE (высокая энергия + негатив)
    purple_aggressive = patches.Rectangle(
        (valence_min, purple_arousal),
        valence_threshold - valence_min,
        arousal_max - purple_arousal,
        linewidth=2,
        edgecolor='darkred',
        facecolor='plum',
        alpha=0.4,
        label='PURPLE AGGRESSIVE (энергия, агрессия)'
    )
    ax.add_patch(purple_aggressive)

    # === ГРАНИЦЫ ===

    # Горизонтальные (arousal thresholds)
    ax.axhline(y=yellow_arousal, color='gold', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Yellow порог (arousal={yellow_arousal})')
    ax.axhline(y=purple_arousal, color='purple', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Purple порог (arousal={purple_arousal})')

    # Вертикальная (valence threshold)
    ax.axvline(x=valence_threshold, color='gray', linestyle='-', linewidth=2, alpha=0.6,
               label=f'Valence порог (negative/positive={valence_threshold})')

    # === ТЕКСТОВЫЕ МЕТКИ ЗОН ===

    # YELLOW CHILL
    ax.text(
        valence_threshold + (valence_max - valence_threshold) / 2,
        arousal_min + (yellow_arousal - arousal_min) / 2,
        'YELLOW\nCHILL\n(Отдых + Позитив)\n\nLounge, Chill-out',
        fontsize=11,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='gold', linewidth=2)
    )

    # YELLOW DARK
    ax.text(
        valence_min + (valence_threshold - valence_min) / 2,
        arousal_min + (yellow_arousal - arousal_min) / 2,
        'YELLOW\nDARK\n(Отдых + Меланхолия)\n\nSad ballad, Ambient',
        fontsize=11,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='darkgoldenrod', linewidth=2)
    )

    # GREEN
    ax.text(
        (valence_min + valence_max) / 2,
        (yellow_arousal + purple_arousal) / 2,
        'GREEN\nTRANSITION\n(Переход)\n\nBuild-up треки',
        fontsize=12,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=2)
    )

    # PURPLE EUPHORIC
    ax.text(
        valence_threshold + (valence_max - valence_threshold) / 2,
        purple_arousal + (arousal_max - purple_arousal) / 2,
        'PURPLE\nEUPHORIC\n(Кульминация + Эйфория)\n\nUplifting, Happy Hardcore',
        fontsize=11,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='violet', alpha=0.7, edgecolor='purple', linewidth=2)
    )

    # PURPLE AGGRESSIVE
    ax.text(
        valence_min + (valence_threshold - valence_min) / 2,
        purple_arousal + (arousal_max - purple_arousal) / 2,
        'PURPLE\nAGGRESSIVE\n(Энергия + Агрессия)\n\nIndustrial, Hard Techno',
        fontsize=11,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7, edgecolor='darkred', linewidth=2)
    )

    # === ЭМОЦИОНАЛЬНЫЕ КВАДРАНТЫ (Russell's Model) ===

    quadrant_fontsize = 9
    quadrant_alpha = 0.5

    # Q1: High Arousal + Positive Valence
    ax.text(
        valence_threshold + (valence_max - valence_threshold) * 0.7,
        purple_arousal + (arousal_max - purple_arousal) * 0.7,
        'Возбуждённый\nРадостный',
        fontsize=quadrant_fontsize,
        ha='center',
        va='center',
        style='italic',
        alpha=quadrant_alpha,
        color='darkgreen'
    )

    # Q2: High Arousal + Negative Valence
    ax.text(
        valence_min + (valence_threshold - valence_min) * 0.3,
        purple_arousal + (arousal_max - purple_arousal) * 0.7,
        'Злой\nНапряжённый',
        fontsize=quadrant_fontsize,
        ha='center',
        va='center',
        style='italic',
        alpha=quadrant_alpha,
        color='darkred'
    )

    # Q3: Low Arousal + Negative Valence
    ax.text(
        valence_min + (valence_threshold - valence_min) * 0.3,
        arousal_min + (yellow_arousal - arousal_min) * 0.3,
        'Грустный\nПодавленный',
        fontsize=quadrant_fontsize,
        ha='center',
        va='center',
        style='italic',
        alpha=quadrant_alpha,
        color='darkblue'
    )

    # Q4: Low Arousal + Positive Valence
    ax.text(
        valence_threshold + (valence_max - valence_threshold) * 0.7,
        arousal_min + (yellow_arousal - arousal_min) * 0.3,
        'Спокойный\nРасслабленный',
        fontsize=quadrant_fontsize,
        ha='center',
        va='center',
        style='italic',
        alpha=quadrant_alpha,
        color='darkgreen'
    )

    # === НАСТРОЙКИ ОСЕЙ ===

    ax.set_xlim(valence_min - 0.2, valence_max + 0.2)
    ax.set_ylim(arousal_min - 0.2, arousal_max + 0.2)

    ax.set_xlabel('Valence (Настроение: Негативное ← → Позитивное)', fontsize=14, weight='bold')
    ax.set_ylabel('Arousal (Энергия: Низкая ↑ Высокая)', fontsize=14, weight='bold')

    ax.set_title(
        '2D Маппинг: Arousal × Valence → Энергетические зоны для DJ\n'
        f'Arousal пороги: Yellow<{yellow_arousal}, Green={yellow_arousal}-{purple_arousal}, Purple>{purple_arousal} | '
        f'Valence порог: {valence_threshold}',
        fontsize=15,
        weight='bold',
        pad=20
    )

    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--')

    # Легенда
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=1)

    # === АННОТАЦИИ ===

    info_text = (
        "Стратегия: Advanced 2D (Arousal + Valence)\n"
        f"Диапазон Arousal: [{arousal_min:.1f}, {arousal_max:.1f}]\n"
        f"Диапазон Valence: [{valence_min:.1f}, {valence_max:.1f}]\n"
        "\n"
        "5 зон с учётом эмоций:\n"
        "• YELLOW CHILL: низкая энергия + позитив\n"
        "• YELLOW DARK: низкая энергия + негатив\n"
        "• GREEN: средняя энергия (любое настроение)\n"
        "• PURPLE EUPHORIC: высокая энергия + позитив\n"
        "• PURPLE AGGRESSIVE: высокая энергия + негатив"
    )

    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray', linewidth=1)
    )

    # === СОХРАНЕНИЕ ===

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 2D zone mapping visualization saved to: {output_path}")
    plt.close()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize 2D arousal-valence zone mapping'
    )
    parser.add_argument(
        '--arousal-min', type=float, default=1.6,
        help='Minimum arousal (default: 1.6)'
    )
    parser.add_argument(
        '--arousal-max', type=float, default=8.1,
        help='Maximum arousal (default: 8.1)'
    )
    parser.add_argument(
        '--valence-min', type=float, default=1.6,
        help='Minimum valence (default: 1.6)'
    )
    parser.add_argument(
        '--valence-max', type=float, default=8.4,
        help='Maximum valence (default: 8.4)'
    )
    parser.add_argument(
        '--output', type=str, default='results/zone_mapping_2d.png',
        help='Output path (default: results/zone_mapping_2d.png)'
    )

    args = parser.parse_args()

    try:
        visualize_2d_zone_mapping(
            arousal_min=args.arousal_min,
            arousal_max=args.arousal_max,
            valence_min=args.valence_min,
            valence_max=args.valence_max,
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
