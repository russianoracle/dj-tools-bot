#!/usr/bin/env python3
"""
DJ Set Analysis Script - Analyze transitions and structure in DJ mixes.

Optimized for long mixes (30min - 2hr+):
- Detects track boundaries (mixin/mixout points)
- Classifies transition types (CUT, FADE, BLEND, EQ_FILTER)
- Analyzes energy flow and drop distribution
- Generates timeline visualization data

Usage:
    python scripts/analyze_dj_set.py path/to/set.mp3
    python scripts/analyze_dj_set.py path/to/set.mp3 --json --plot
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np

from src.core.tasks import (
    create_audio_context,
    DropDetectionTask,
    TransitionDetectionTask,
    TransitionType,
)


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def estimate_track_count(transitions: List, duration: float) -> int:
    """Estimate number of tracks in the set."""
    if not transitions:
        return 1
    # Each transition = 1 track change, plus initial track
    return len(transitions) + 1


def analyze_energy_flow(
    drop_result,
    duration: float,
    segment_minutes: float = 5.0
) -> List[dict]:
    """Analyze energy flow across the set in segments."""
    segments = []
    segment_sec = segment_minutes * 60

    n_segments = int(np.ceil(duration / segment_sec))

    for i in range(n_segments):
        start = i * segment_sec
        end = min((i + 1) * segment_sec, duration)

        # Count drops in this segment
        drops_in_segment = [
            d for d in drop_result.drops
            if start <= d.time_sec < end
        ]

        # Avg intensity
        avg_intensity = np.mean([d.drop_magnitude for d in drops_in_segment]) if drops_in_segment else 0

        segments.append({
            'segment': i + 1,
            'start': format_time(start),
            'end': format_time(end),
            'drops': len(drops_in_segment),
            'avg_intensity': float(avg_intensity),
            'energy_level': 'HIGH' if avg_intensity > 0.5 else 'MEDIUM' if avg_intensity > 0.3 else 'LOW'
        })

    return segments


def find_peak_moments(drop_result, top_n: int = 5) -> List[dict]:
    """Find the peak energy moments in the set."""
    if not drop_result.drops:
        return []

    # Sort by magnitude * confidence
    scored_drops = sorted(
        drop_result.drops,
        key=lambda d: d.drop_magnitude * d.confidence,
        reverse=True
    )

    return [
        {
            'time': format_time(d.time_sec),
            'time_sec': d.time_sec,
            'magnitude': d.drop_magnitude,
            'confidence': d.confidence,
            'bass_prominence': d.bass_prominence
        }
        for d in scored_drops[:top_n]
    ]


def analyze_dj_set(
    file_path: str,
    output_json: bool = False,
    generate_plot: bool = False,
    segment_minutes: float = 5.0
):
    """Analyze a DJ set for transitions and structure."""

    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Loading DJ set: {path.name}")
    print("This may take a while for long sets...")
    print()

    # Load audio
    y, sr = librosa.load(str(path), sr=22050)
    duration = len(y) / sr

    print(f"Duration: {format_time(duration)} ({duration / 60:.1f} minutes)")
    print(f"Sample rate: {sr} Hz")
    print()

    # Create context
    ctx = create_audio_context(y, sr, file_path=str(path))

    results = {
        'file': str(path),
        'duration_sec': duration,
        'duration_formatted': format_time(duration),
    }

    # ========================================
    # TRANSITION DETECTION
    # ========================================
    print("=" * 60)
    print("TRANSITION ANALYSIS")
    print("=" * 60)

    trans_task = TransitionDetectionTask(
        detect_filters=True,
        min_transition_gap_sec=45.0,  # DJ sets have longer gaps between transitions
        energy_threshold=0.2,
        filter_velocity_threshold=400.0
    )
    trans_result = trans_task.execute(ctx)

    if trans_result.success:
        estimated_tracks = estimate_track_count(trans_result.transitions, duration)

        print(f"Estimated track count: ~{estimated_tracks} tracks")
        print(f"Transitions detected: {trans_result.n_transitions}")
        print(f"Transition density: {trans_result.transition_density:.2f} per minute")
        print(f"Avg transition duration: {trans_result.avg_transition_duration:.0f}s")
        print()

        # Transition type breakdown
        if trans_result.transition_type_distribution:
            print("Transition types:")
            total = sum(trans_result.transition_type_distribution.values())
            for type_name, count in sorted(
                trans_result.transition_type_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                pct = count / total * 100 if total > 0 else 0
                print(f"  {type_name}: {count} ({pct:.0f}%)")
            print()

        # Timeline of transitions
        if trans_result.transitions:
            print("Transition timeline:")
            print("-" * 60)
            for i, t in enumerate(trans_result.transitions, 1):
                # Determine filter info
                filter_info = ""
                if t.mixin.filter_detected:
                    filter_info = f" [{t.mixin.filter_direction}]"
                elif t.mixout.filter_detected:
                    filter_info = f" [{t.mixout.filter_direction}]"

                print(f"  {i:2d}. {format_time(t.mixin.time_sec):>8s} -> "
                      f"{format_time(t.mixout.time_sec):>8s}  "
                      f"[{t.overlap_sec:3.0f}s]  "
                      f"{t.transition_type.name:<10s}{filter_info}")
            print()

        results['transitions'] = {
            'estimated_tracks': estimated_tracks,
            'count': trans_result.n_transitions,
            'density_per_min': trans_result.transition_density,
            'avg_duration_sec': trans_result.avg_transition_duration,
            'type_distribution': trans_result.transition_type_distribution,
            'timeline': [
                {
                    'mixin_time': t.mixin.time_sec,
                    'mixout_time': t.mixout.time_sec,
                    'duration_sec': t.overlap_sec,
                    'type': t.transition_type.name,
                    'quality': t.quality_score,
                    'mixin_filter': t.mixin.filter_direction if t.mixin.filter_detected else None,
                    'mixout_filter': t.mixout.filter_direction if t.mixout.filter_detected else None,
                }
                for t in trans_result.transitions
            ]
        }
    else:
        print(f"Error: {trans_result.error}")

    # ========================================
    # DROP DETECTION
    # ========================================
    print("=" * 60)
    print("DROP & ENERGY ANALYSIS")
    print("=" * 60)

    drop_task = DropDetectionTask(
        use_multiband=True,
        min_drop_magnitude=0.2,
        min_confidence=0.35
    )
    drop_result = drop_task.execute(ctx)

    if drop_result.success:
        print(f"Drops detected: {drop_result.n_drops}")
        print(f"Drop density: {drop_result.drop_density:.2f} per minute")
        print(f"Max drop magnitude: {drop_result.max_drop_magnitude:.2f}")
        print()

        # Energy flow analysis
        energy_segments = analyze_energy_flow(drop_result, duration, segment_minutes)

        print(f"Energy flow ({segment_minutes:.0f}-minute segments):")
        print("-" * 60)
        for seg in energy_segments:
            bar_len = int(seg['avg_intensity'] * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"  {seg['start']:>8s} - {seg['end']:>8s}  [{bar}]  "
                  f"{seg['drops']} drops  {seg['energy_level']}")
        print()

        # Peak moments
        peaks = find_peak_moments(drop_result, top_n=5)
        if peaks:
            print("Peak energy moments (top 5):")
            for i, peak in enumerate(peaks, 1):
                print(f"  {i}. {peak['time']} - magnitude: {peak['magnitude']:.2f}, "
                      f"bass: {peak['bass_prominence']:.2f}")
            print()

        # Temporal distribution
        dist = drop_result.drop_temporal_distribution
        if dist < 0.4:
            dist_desc = "Front-loaded (drops early)"
        elif dist > 0.6:
            dist_desc = "Back-loaded (builds to end)"
        else:
            dist_desc = "Balanced throughout"

        print(f"Set structure: {dist_desc}")
        print(f"  First half drops: {drop_result.drops_in_first_half}")
        print(f"  Second half drops: {drop_result.drops_in_second_half}")
        print()

        results['drops'] = {
            'count': drop_result.n_drops,
            'density_per_min': drop_result.drop_density,
            'max_magnitude': drop_result.max_drop_magnitude,
            'avg_intensity': drop_result.avg_drop_intensity,
            'avg_buildup_duration': drop_result.avg_buildup_duration,
            'temporal_distribution': drop_result.drop_temporal_distribution,
            'first_half_drops': drop_result.drops_in_first_half,
            'second_half_drops': drop_result.drops_in_second_half,
            'energy_segments': energy_segments,
            'peak_moments': peaks,
        }
    else:
        print(f"Error: {drop_result.error}")

    # ========================================
    # SET SUMMARY
    # ========================================
    print("=" * 60)
    print("SET SUMMARY")
    print("=" * 60)

    if trans_result.success and drop_result.success:
        # Determine mixing style
        if trans_result.transition_type_distribution:
            dominant_type = max(
                trans_result.transition_type_distribution.items(),
                key=lambda x: x[1]
            )[0]
        else:
            dominant_type = "UNKNOWN"

        # Determine energy style
        if drop_result.drop_density > 1.5:
            energy_style = "High-energy/Peak-time"
        elif drop_result.drop_density > 0.8:
            energy_style = "Main room/Festival"
        elif drop_result.drop_density > 0.3:
            energy_style = "Progressive/Journey"
        else:
            energy_style = "Deep/Minimal"

        print(f"Duration: {format_time(duration)}")
        print(f"Estimated tracks: ~{estimated_tracks}")
        print(f"Dominant transition style: {dominant_type}")
        print(f"Energy profile: {energy_style}")
        print(f"Avg drops per track: {drop_result.n_drops / max(1, estimated_tracks):.1f}")

        results['summary'] = {
            'duration_formatted': format_time(duration),
            'estimated_tracks': estimated_tracks,
            'dominant_transition_style': dominant_type,
            'energy_profile': energy_style,
            'avg_drops_per_track': drop_result.n_drops / max(1, estimated_tracks),
        }

    print()

    # ========================================
    # OUTPUT
    # ========================================
    if output_json:
        json_path = path.with_suffix('.set_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {json_path}")

    if generate_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

            time_axis = np.linspace(0, duration, len(trans_result.energy_curve))

            # Energy curve
            axes[0].plot(time_axis, trans_result.energy_curve, 'b-', alpha=0.7, linewidth=0.5)
            axes[0].set_ylabel('Energy')
            axes[0].set_title(f'DJ Set Analysis: {path.name}')

            # Mark drops
            for drop in drop_result.drops:
                axes[0].axvline(drop.time_sec, color='red', alpha=0.5, linewidth=1)

            # Bass curve
            axes[1].plot(time_axis, trans_result.bass_curve, 'g-', alpha=0.7, linewidth=0.5)
            axes[1].set_ylabel('Bass Energy')

            # Mark transitions
            for t in trans_result.transitions:
                axes[1].axvspan(t.mixin.time_sec, t.mixout.time_sec,
                               alpha=0.3, color='orange')

            # Filter position
            if trans_result.filter_curve is not None:
                axes[2].plot(time_axis, trans_result.filter_curve, 'purple', alpha=0.7, linewidth=0.5)
            axes[2].set_ylabel('Filter Position')
            axes[2].set_xlabel('Time (seconds)')

            plt.tight_layout()

            plot_path = path.with_suffix('.set_analysis.png')
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved to: {plot_path}")
            plt.close()

        except ImportError:
            print("Plot generation requires matplotlib (pip install matplotlib)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DJ set for transitions and structure"
    )
    parser.add_argument("file", help="Path to DJ set audio file")
    parser.add_argument("--json", action="store_true", help="Save results as JSON")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plot")
    parser.add_argument("--segment-minutes", type=float, default=5.0,
                        help="Segment length for energy analysis (default: 5)")

    args = parser.parse_args()

    analyze_dj_set(
        args.file,
        output_json=args.json,
        generate_plot=args.plot,
        segment_minutes=args.segment_minutes
    )


if __name__ == "__main__":
    main()