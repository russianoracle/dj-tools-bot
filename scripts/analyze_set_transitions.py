#!/usr/bin/env python3
"""
DJ Set Transition Analyzer

Apple Silicon M2 Optimized script for analyzing mixin/mixout
points in DJ sets.

Features:
- Batch processing of multiple sets
- Parallel analysis using M2 performance cores
- JSON/CSV export
- Optional visualization
- Integration with existing mood-classifier pipeline

Usage:
    # Single file
    python scripts/analyze_set_transitions.py path/to/mix.mp3

    # Batch processing
    python scripts/analyze_set_transitions.py --batch path/to/sets/ --workers 4

    # With visualization
    python scripts/analyze_set_transitions.py path/to/mix.mp3 --visualize

    # Export to JSON
    python scripts/analyze_set_transitions.py path/to/mix.mp3 --output results.json

Author: Optimized for Apple Silicon M2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Force NumPy to use Apple Accelerate
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import pandas as pd

from src.audio.mixin_mixout import (
    M2MixinMixoutDetector,
    M2DetectorConfig,
    TransitionAnalysis,
    detect_mixin_mixout
)


def analyze_single_file(args: Tuple) -> Optional[Tuple[str, TransitionAnalysis]]:
    """Analyze single file (for multiprocessing)."""
    audio_path, config_dict = args

    try:
        config = M2DetectorConfig(**config_dict) if config_dict else None
        analysis = detect_mixin_mixout(audio_path, config)
        return (audio_path, analysis)
    except Exception as e:
        print(f"Error analyzing {Path(audio_path).name}: {e}")
        return None


def find_audio_files(path: str, extensions: List[str] = None) -> List[str]:
    """Find all audio files in path."""
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg']

    path = Path(path)

    if path.is_file():
        return [str(path)]

    files = []
    for ext in extensions:
        files.extend(path.rglob(f'*{ext}'))

    return sorted([str(f) for f in files])


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def create_timeline_report(analysis: TransitionAnalysis) -> str:
    """Create a detailed timeline report."""
    lines = [
        "=" * 70,
        f"TRANSITION TIMELINE: {Path(analysis.file_path).name}",
        "=" * 70,
        f"Duration: {format_time(analysis.duration_sec)} ({analysis.duration_sec/60:.1f} min)",
        f"Total Transitions: {len(analysis.mixins) + len(analysis.mixouts)}",
        "",
    ]

    # Combine and sort all events by time
    events = []

    for m in analysis.mixins:
        events.append({
            'time': m.time_sec,
            'type': 'MIXIN',
            'event': m,
        })

    for m in analysis.mixouts:
        events.append({
            'time': m.time_sec,
            'type': 'MIXOUT',
            'event': m,
        })

    events.sort(key=lambda x: x['time'])

    if not events:
        lines.append("No transitions detected.")
        return '\n'.join(lines)

    lines.append("Timeline:")
    lines.append("-" * 70)

    for i, e in enumerate(events, 1):
        event = e['event']
        time_str = format_time(e['time'])
        event_type = e['type']

        # Icon based on transition type
        icon = {
            'CUT': '  ',
            'FADE': '  ',
            'BLEND': '  ',
            'EQ_FILTER': '  ',
            'UNKNOWN': '  ',
        }.get(event.transition_type.name, '  ')

        # Confidence bar
        conf_bar = '' * int(event.confidence * 10) + '' * (10 - int(event.confidence * 10))

        lines.append(
            f"  [{time_str}] {icon} {event_type:7} | "
            f"{event.transition_type.name:10} | "
            f"dur={event.duration_sec:4.1f}s | "
            f"[{conf_bar}] {event.confidence:.0%}"
        )

        # Additional details
        if hasattr(event, 'bass_introduction') and event.bass_introduction > 0.2:
            lines.append(f"             Bass introduction: +{event.bass_introduction:.0%}")
        if hasattr(event, 'bass_removal') and event.bass_removal > 0.2:
            lines.append(f"             Bass removal: -{event.bass_removal:.0%}")
        if event.filter_detected:
            lines.append(f"             Filter: {event.filter_direction}")

    lines.append("-" * 70)

    # Statistics
    lines.append("")
    lines.append("Statistics:")
    lines.append(f"  Average transition duration: {analysis.avg_transition_duration:.1f}s")

    if analysis.transition_type_distribution:
        lines.append("  Transition types:")
        for t_type, count in sorted(analysis.transition_type_distribution.items()):
            lines.append(f"    {t_type}: {count}")

    # Paired transitions
    if analysis.transitions:
        lines.append("")
        lines.append(f"Paired Transitions ({len(analysis.transitions)}):")
        for i, t in enumerate(analysis.transitions, 1):
            lines.append(
                f"  {i}. {format_time(t.mixout.time_sec)} -> {format_time(t.mixin.time_sec)} "
                f"(overlap: {t.overlap_sec:.1f}s, beatmatch: {t.beatmatch_quality:.0%})"
            )

    return '\n'.join(lines)


def export_to_csv(analyses: List[TransitionAnalysis], output_path: str):
    """Export all transitions to CSV."""
    rows = []

    for analysis in analyses:
        file_name = Path(analysis.file_path).name

        # Export mixins
        for m in analysis.mixins:
            rows.append({
                'file': file_name,
                'event_type': 'mixin',
                'time_sec': m.time_sec,
                'time_formatted': format_time(m.time_sec),
                'transition_type': m.transition_type.name,
                'duration_sec': m.duration_sec,
                'confidence': m.confidence,
                'energy_slope': m.energy_slope,
                'bass_change': m.bass_introduction,
                'spectral_shift': m.spectral_shift,
                'filter_detected': m.filter_detected,
                'filter_direction': m.filter_direction,
            })

        # Export mixouts
        for m in analysis.mixouts:
            rows.append({
                'file': file_name,
                'event_type': 'mixout',
                'time_sec': m.time_sec,
                'time_formatted': format_time(m.time_sec),
                'transition_type': m.transition_type.name,
                'duration_sec': m.duration_sec,
                'confidence': m.confidence,
                'energy_slope': m.energy_slope,
                'bass_change': m.bass_removal,
                'spectral_shift': m.spectral_shift,
                'filter_detected': m.filter_detected,
                'filter_direction': m.filter_direction,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(rows)} events to {output_path}")


def visualize_analysis(analysis: TransitionAnalysis, output_path: str = None):
    """Create visualization of the analysis."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    time_axis = analysis.time_axis / 60  # Convert to minutes

    # 1. Energy curve
    ax1 = axes[0]
    ax1.plot(time_axis, analysis.energy_curve, 'b-', linewidth=1, alpha=0.8)
    ax1.fill_between(time_axis, 0, analysis.energy_curve, alpha=0.3)
    ax1.set_ylabel('Energy')
    ax1.set_title(f'Transition Analysis: {Path(analysis.file_path).name}')

    # Mark mixins (green) and mixouts (red)
    for m in analysis.mixins:
        ax1.axvline(x=m.time_sec/60, color='green', alpha=0.7, linestyle='--', linewidth=1.5)
    for m in analysis.mixouts:
        ax1.axvline(x=m.time_sec/60, color='red', alpha=0.7, linestyle='--', linewidth=1.5)

    # 2. Bass curve
    ax2 = axes[1]
    ax2.plot(time_axis, analysis.bass_curve, 'orange', linewidth=1, alpha=0.8)
    ax2.fill_between(time_axis, 0, analysis.bass_curve, alpha=0.3, color='orange')
    ax2.set_ylabel('Bass')

    # 3. Spectral/Filter curve
    ax3 = axes[2]
    ax3.plot(time_axis, analysis.filter_curve, 'purple', linewidth=1, alpha=0.8)
    ax3.fill_between(time_axis, 0, analysis.filter_curve, alpha=0.3, color='purple')
    ax3.set_ylabel('Filter Position')

    # 4. Transition events
    ax4 = axes[3]
    ax4.set_ylabel('Events')
    ax4.set_xlabel('Time (minutes)')

    # Plot transition regions
    colors = {
        'CUT': 'red',
        'FADE': 'blue',
        'BLEND': 'green',
        'EQ_FILTER': 'purple',
        'UNKNOWN': 'gray',
    }

    legend_handles = []
    plotted_types = set()

    for m in analysis.mixins:
        color = colors.get(m.transition_type.name, 'gray')
        start = m.time_sec / 60
        duration = m.duration_sec / 60
        ax4.axvspan(start, start + duration, alpha=0.4, color=color)
        ax4.annotate('IN', (start, 0.7), fontsize=8, ha='center')

        if m.transition_type.name not in plotted_types:
            legend_handles.append(mpatches.Patch(color=color, label=m.transition_type.name))
            plotted_types.add(m.transition_type.name)

    for m in analysis.mixouts:
        color = colors.get(m.transition_type.name, 'gray')
        start = m.time_sec / 60
        duration = m.duration_sec / 60
        ax4.axvspan(start, start + duration, alpha=0.2, color=color, hatch='//')
        ax4.annotate('OUT', (start, 0.3), fontsize=8, ha='center')

    ax4.set_ylim(0, 1)
    ax4.legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="DJ Set Transition Analyzer (Apple Silicon M2 Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  python scripts/analyze_set_transitions.py mix.mp3

  # Batch analyze folder
  python scripts/analyze_set_transitions.py --batch sets/ --workers 4

  # With visualization
  python scripts/analyze_set_transitions.py mix.mp3 --visualize

  # Export results
  python scripts/analyze_set_transitions.py mix.mp3 --output results.json
  python scripts/analyze_set_transitions.py --batch sets/ --csv transitions.csv
        """
    )

    parser.add_argument("input", nargs='?',
                       help="Audio file or directory to analyze")
    parser.add_argument("--batch", metavar="DIR",
                       help="Batch process directory")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--output", "-o", metavar="FILE",
                       help="Output JSON file")
    parser.add_argument("--csv", metavar="FILE",
                       help="Export transitions to CSV")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Create visualization")
    parser.add_argument("--viz-output", metavar="FILE",
                       help="Save visualization to file")
    parser.add_argument("--min-transition", type=float, default=2.0,
                       help="Minimum transition duration (seconds)")
    parser.add_argument("--max-transition", type=float, default=32.0,
                       help="Maximum transition duration (seconds)")
    parser.add_argument("--json", action="store_true",
                       help="Output raw JSON (for piping)")

    args = parser.parse_args()

    if not args.input and not args.batch:
        parser.print_help()
        sys.exit(1)

    # Configuration
    config_dict = {
        'min_transition_sec': args.min_transition,
        'max_transition_sec': args.max_transition,
    }

    print("\n" + "=" * 60)
    print("DJ Set Transition Analyzer")
    print("Apple Silicon M2 Optimized")
    print("=" * 60)

    # Collect files
    if args.batch:
        files = find_audio_files(args.batch)
        print(f"Found {len(files)} audio files")
    else:
        files = find_audio_files(args.input)

    if not files:
        print("No audio files found!")
        sys.exit(1)

    # Analyze
    analyses = []

    if len(files) == 1:
        # Single file - direct analysis
        print(f"\nAnalyzing: {files[0]}")
        analysis = detect_mixin_mixout(files[0], M2DetectorConfig(**config_dict))
        analyses.append(analysis)
    else:
        # Batch processing
        print(f"\nProcessing with {args.workers} workers...")
        print("-" * 60)

        tasks = [(f, config_dict) for f in files]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(analyze_single_file, task): task[0] for task in tasks}

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    audio_path, analysis = result
                    analyses.append(analysis)
                    completed += 1
                    print(f"  [{completed}/{len(files)}] {Path(audio_path).name}: "
                          f"{len(analysis.mixins)} mixins, {len(analysis.mixouts)} mixouts")

    print("-" * 60)

    # Output results
    if args.json:
        # Raw JSON output
        if len(analyses) == 1:
            print(json.dumps(analyses[0].to_dict(), indent=2))
        else:
            print(json.dumps([a.to_dict() for a in analyses], indent=2))

    elif args.output:
        # Save to JSON file
        if len(analyses) == 1:
            data = analyses[0].to_dict()
        else:
            data = [a.to_dict() for a in analyses]

        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to {args.output}")

    else:
        # Print reports
        for analysis in analyses:
            print("\n" + create_timeline_report(analysis))

    # CSV export
    if args.csv:
        export_to_csv(analyses, args.csv)

    # Visualization
    if args.visualize or args.viz_output:
        for analysis in analyses:
            viz_path = args.viz_output
            if not viz_path and len(analyses) > 1:
                viz_path = str(Path(analysis.file_path).with_suffix('.png'))
            visualize_analysis(analysis, viz_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_mixins = sum(len(a.mixins) for a in analyses)
    total_mixouts = sum(len(a.mixouts) for a in analyses)
    print(f"Files analyzed: {len(analyses)}")
    print(f"Total mixins detected: {total_mixins}")
    print(f"Total mixouts detected: {total_mixouts}")
    print(f"Average per file: {(total_mixins + total_mixouts) / len(analyses):.1f} transitions")
    print("=" * 60)


if __name__ == "__main__":
    main()