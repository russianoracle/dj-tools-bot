#!/usr/bin/env python3
"""
Visualize Arousal-Valence Zone Mapping

Создаёт 2D визуализацию arousal-valence пространства с зонами.
Помогает подтвердить или скорректировать критерии отнесения к зонам.
"""

import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Настройка кириллицы для matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # Чтобы минус отображался корректно

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_zone_mapping(
    yellow_threshold: float = 4.0,
    purple_threshold: float = 6.0,
    arousal_min: float = 1.6,
    arousal_max: float = 8.1,
    valence_min: float = 1.6,
    valence_max: float = 8.4,
    output_path: str = "results/zone_mapping.png"
):
    """
    Визуализирует arousal-valence пространство с зонами

    Args:
        yellow_threshold: Порог arousal для Yellow зоны (arousal < threshold)
        purple_threshold: Порог arousal для Purple зоны (arousal > threshold)
        arousal_min: Минимальное значение arousal в датасете
        arousal_max: Максимальное значение arousal в датасете
        valence_min: Минимальное значение valence в датасете
        valence_max: Максимальное значение valence в датасете
        output_path: Путь для сохранения изображения
    """

    # Создаём figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # === РИСУЕМ ЗОНЫ ===

    # YELLOW зона (низкая энергия, arousal < yellow_threshold)
    yellow_rect = patches.Rectangle(
        (valence_min, arousal_min),
        valence_max - valence_min,
        yellow_threshold - arousal_min,
        linewidth=2,
        edgecolor='gold',
        facecolor='yellow',
        alpha=0.3,
        label='YELLOW (зона отдыха)'
    )
    ax.add_patch(yellow_rect)

    # GREEN зона (средняя энергия, yellow_threshold <= arousal <= purple_threshold)
    green_rect = patches.Rectangle(
        (valence_min, yellow_threshold),
        valence_max - valence_min,
        purple_threshold - yellow_threshold,
        linewidth=2,
        edgecolor='darkgreen',
        facecolor='green',
        alpha=0.3,
        label='GREEN (переходная)'
    )
    ax.add_patch(green_rect)

    # PURPLE зона (высокая энергия, arousal > purple_threshold)
    purple_rect = patches.Rectangle(
        (valence_min, purple_threshold),
        valence_max - valence_min,
        arousal_max - purple_threshold,
        linewidth=2,
        edgecolor='purple',
        facecolor='purple',
        alpha=0.3,
        label='PURPLE (энергия/хиты)'
    )
    ax.add_patch(purple_rect)

    # === ГРАНИЦЫ ЗᲝᲜХ ===

    # Линия порога Yellow
    ax.axhline(y=yellow_threshold, color='gold', linestyle='--', linewidth=2,
               label=f'Порог Yellow (arousal={yellow_threshold})')

    # Линия порога Purple
    ax.axhline(y=purple_threshold, color='purple', linestyle='--', linewidth=2,
               label=f'Порог Purple (arousal={purple_threshold})')

    # === ДОБАВЛЯЕМ ТЕКСТОВЫЕ МЕТКИ ===

    # YELLOW зона
    ax.text(
        (valence_min + valence_max) / 2,
        (arousal_min + yellow_threshold) / 2,
        'YELLOW\nЗона отдыха\n(Низкая энергия)',
        fontsize=14,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6)
    )

    # GREEN зона
    ax.text(
        (valence_min + valence_max) / 2,
        (yellow_threshold + purple_threshold) / 2,
        'GREEN\nПереходная\n(Средняя энергия)',
        fontsize=14,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6)
    )

    # PURPLE зона
    ax.text(
        (valence_min + valence_max) / 2,
        (purple_threshold + arousal_max) / 2,
        'PURPLE\nЭнергия/Хиты\n(Высокая энергия)',
        fontsize=14,
        weight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.6)
    )

    # === QUADRANTS (Russell's Circumplex Model) ===

    # Средние точки
    valence_mid = (valence_min + valence_max) / 2
    arousal_mid = (arousal_min + arousal_max) / 2

    # Вертикальная линия (разделяет negative/positive valence)
    ax.axvline(x=valence_mid, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Горизонтальная линия (разделяет low/high arousal)
    ax.axhline(y=arousal_mid, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Подписи квадрантов (Russell's model)
    quadrant_fontsize = 10
    quadrant_alpha = 0.7

    # Q1: High Arousal + Positive Valence (Excited, Happy)
    ax.text(
        valence_mid + (valence_max - valence_mid) * 0.5,
        arousal_mid + (arousal_max - arousal_mid) * 0.7,
        'Возбуждённый\nРадостный\nЭнергичный',
        fontsize=quadrant_fontsize,
        ha='center',
        va='center',
        style='italic',
        alpha=quadrant_alpha,
        color='darkgreen'
    )

    # Q2: High Arousal + Negative Valence (Angry, Tense)
    ax.text(
        valence_min + (valence_mid - valence_min) * 0.5,
        arousal_mid + (arousal_max - arousal_mid) * 0.7,
        'Злой\nНапряжённый\nНервный',
        fontsize=quadrant_fontsize,
        ha='center',
        va='center',
        style='italic',
        alpha=quadrant_alpha,
        color='darkred'
    )

    # Q3: Low Arousal + Negative Valence (Sad, Depressed)
    ax.text(
        valence_min + (valence_mid - valence_min) * 0.5,
        arousal_min + (arousal_mid - arousal_min) * 0.5,
        'Грустный\nПодавленный\nСкучающий',
        fontsize=quadrant_fontsize,
        ha='center',
        va='center',
        style='italic',
        alpha=quadrant_alpha,
        color='darkblue'
    )

    # Q4: Low Arousal + Positive Valence (Calm, Relaxed)
    ax.text(
        valence_mid + (valence_max - valence_mid) * 0.5,
        arousal_min + (arousal_mid - arousal_min) * 0.5,
        'Спокойный\nРасслабленный\nУмиротворённый',
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
        'Маппинг Arousal-Valence в энергетические зоны для DJ\n'
        f'Пороги: Yellow < {yellow_threshold} | {yellow_threshold} ≤ Green ≤ {purple_threshold} | Purple > {purple_threshold}',
        fontsize=16,
        weight='bold',
        pad=20
    )

    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--')

    # Легенда
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # === АННОТАЦИИ ===

    # Добавляем информацию о стратегии
    info_text = (
        "Стратегия маппинга: Простая (только Arousal)\n"
        f"Диапазон Arousal: [{arousal_min:.1f}, {arousal_max:.1f}]\n"
        f"Диапазон Valence: [{valence_min:.1f}, {valence_max:.1f}]\n"
        "\n"
        "Характеристики DJ-зон:\n"
        "• YELLOW: Низкоэнергичные треки для отдыха\n"
        "• GREEN: Переходные треки с постепенным нарастанием\n"
        "• PURPLE: Высокоэнергичные треки с дропами/кульминацией"
    )

    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # === СОХРАНЕНИЕ ===

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Zone mapping visualization saved to: {output_path}")
    plt.close()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize arousal-valence zone mapping'
    )
    parser.add_argument(
        '--yellow-threshold', type=float, default=4.0,
        help='Arousal threshold for Yellow zone (default: 4.0)'
    )
    parser.add_argument(
        '--purple-threshold', type=float, default=6.0,
        help='Arousal threshold for Purple zone (default: 6.0)'
    )
    parser.add_argument(
        '--arousal-min', type=float, default=1.6,
        help='Minimum arousal in dataset (default: 1.6)'
    )
    parser.add_argument(
        '--arousal-max', type=float, default=8.1,
        help='Maximum arousal in dataset (default: 8.1)'
    )
    parser.add_argument(
        '--valence-min', type=float, default=1.6,
        help='Minimum valence in dataset (default: 1.6)'
    )
    parser.add_argument(
        '--valence-max', type=float, default=8.4,
        help='Maximum valence in dataset (default: 8.4)'
    )
    parser.add_argument(
        '--output', type=str, default='results/zone_mapping.png',
        help='Output path for visualization (default: results/zone_mapping.png)'
    )

    args = parser.parse_args()

    try:
        visualize_zone_mapping(
            yellow_threshold=args.yellow_threshold,
            purple_threshold=args.purple_threshold,
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
