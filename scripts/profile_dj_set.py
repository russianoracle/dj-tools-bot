#!/usr/bin/env python3
"""
DJ Set Profiling Script - Extract DJ style characteristics from sets.

Focuses on metrics robust to imperfect transition detection:
- Energy arc: How DJ manages energy over the set
- Drop patterns: Frequency and intensity of drops
- Tempo distribution: BPM range and progression (TODO)

Usage:
    python scripts/profile_dj_set.py path/to/set.mp3
    python scripts/profile_dj_set.py path/to/set.mp3 --json output.json
    python scripts/profile_dj_set.py path/to/set.mp3 --visualize
"""

import argparse
import json
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pipelines import (
    Pipeline,
    PipelineContext,
    LoadAudioStage,
    ComputeSTFTStage,
)
from src.core.pipelines.dj_profiling import (
    DJProfilingStage,
    EnergyArcMetrics,
    DropPatternMetrics,
)
from src.core.cache import CacheRepository

# Configure logging for detailed progress
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,  # Use stdout instead of stderr
    force=True
)
# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def print_energy_arc_metrics(metrics: EnergyArcMetrics):
    """Pretty-print energy arc metrics."""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                   ENERGY ARC PROFILE                      ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║  Arc Shape:           {metrics.arc_shape.upper():>35} ║")
    print(f"║  Opening Energy:      {metrics.opening_energy:>35.2f} ║")
    print(f"║  Peak Energy:         {metrics.peak_energy:>35.2f} ║")
    print(f"║  Closing Energy:      {metrics.closing_energy:>35.2f} ║")
    print(f"║  Energy Variance:     {metrics.energy_variance:>35.3f} ║")
    print(f"║  Opening/Peak Ratio:  {metrics.opening_to_peak_ratio:>35.2f} ║")
    print(f"║  Peak Timing:         {metrics.peak_timing_normalized*100:>34.1f}% ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Interpretation
    print("Interpretation:")
    if metrics.arc_shape == "crescendo":
        print("  → DJ builds energy throughout the set, peaking near the end")
    elif metrics.arc_shape == "peak_and_fade":
        print("  → DJ reaches peak energy in the middle, then eases down")
    elif metrics.arc_shape == "plateau":
        print("  → DJ maintains sustained high energy throughout")
    elif metrics.arc_shape == "chaotic":
        print("  → Unpredictable energy changes, dynamic mixing style")

    if metrics.opening_to_peak_ratio < 0.4:
        print("  → Strong opening → peak buildup (large dynamic range)")
    elif metrics.opening_to_peak_ratio > 0.7:
        print("  → Minimal buildup (starts near peak energy)")

    if metrics.energy_variance < 0.15:
        print("  → Very stable energy flow (smooth DJ)")
    elif metrics.energy_variance > 0.25:
        print("  → High variance (energetic, dynamic style)")
    print()


def print_drop_pattern_metrics(metrics: DropPatternMetrics):
    """Pretty-print drop pattern metrics."""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                  DROP PATTERN PROFILE                     ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║  Style:               {metrics.drop_clustering.upper():>35} ║")
    print(f"║  Drops per Hour:      {metrics.drops_per_hour:>35.1f} ║")
    print(f"║  Total Drops:         {len(metrics.drop_magnitudes):>35} ║")
    print(f"║  Avg Magnitude:       {metrics.avg_drop_magnitude:>35.2f} ║")
    print(f"║  Max Magnitude:       {metrics.max_drop_magnitude:>35.2f} ║")
    print(f"║  Buildups Detected:   {metrics.buildup_count:>35} ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Interpretation
    print("Interpretation:")
    if metrics.drop_clustering == "technical":
        print("  → Technical DJ: Frequent drops, medium intensity")
    elif metrics.drop_clustering == "festival":
        print("  → Festival DJ: Few but massive drops for crowd impact")
    elif metrics.drop_clustering == "minimal":
        print("  → Minimal DJ: Very few drops, smooth sustained flow")

    if metrics.buildup_count > 0:
        buildup_ratio = metrics.buildup_count / max(len(metrics.drop_magnitudes), 1)
        print(f"  → {buildup_ratio*100:.0f}% of drops have buildups (intentional energy management)")

    if len(metrics.drop_magnitudes) > 0:
        print(f"\n  Top 5 Drops:")
        sorted_drops = sorted(
            zip(metrics.drop_times, metrics.drop_magnitudes),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for i, (time, mag) in enumerate(sorted_drops, 1):
            print(f"    {i}. {format_time(time)} - Magnitude: {mag:.2f}")
    print()


def print_tempo_distribution_metrics(metrics):
    """Pretty-print tempo distribution metrics."""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║              TEMPO DISTRIBUTION PROFILE                   ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║  Mean Tempo:          {metrics.tempo_mean:>35.1f} BPM ║")
    print(f"║  Tempo Range:         {metrics.tempo_min:>22.0f} - {metrics.tempo_max:.0f} BPM ║")
    print(f"║  Dominant Tempo:      {metrics.dominant_tempo:>35} BPM ║")
    print(f"║  Tempo Variance:      {metrics.tempo_std:>35.1f} BPM ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Interpretation (variance only)
    print("Interpretation:")
    if metrics.tempo_std < 3.0:
        print("  → Very tight tempo range (consistent)")
    elif metrics.tempo_std < 8.0:
        print("  → Moderate tempo variation")
    else:
        print("  → Wide tempo range (eclectic)")

    # Show top 3 tempos from histogram
    if metrics.tempo_histogram:
        top_tempos = sorted(metrics.tempo_histogram.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  Top 3 Tempo Zones:")
        for i, (bpm, count) in enumerate(top_tempos, 1):
            print(f"    {i}. {bpm} BPM ({count} windows)")
    print()


def print_genre_metrics(genre):
    """Pretty-print genre classification results."""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                   GENRE CLASSIFICATION                    ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║  DJ Category:         {genre.dj_category:<36} ║")
    print(f"║  Primary Genre:       {genre.genre:<36} ║")
    if genre.subgenre:
        print(f"║  Subgenre:            {genre.subgenre:<36} ║")
    print(f"║  Confidence:          {genre.confidence*100:>34.1f}% ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    if genre.all_styles:
        print("Top 5 Detected Styles:")
        for i, (style, conf) in enumerate(genre.all_styles[:5], 1):
            print(f"  {i}. {style:<30} {conf*100:>5.1f}%")
        print()


def print_key_analysis_metrics(metrics):
    """Pretty-print key analysis results."""
    from src.core.tasks.key_analysis import KeyAnalysisResult, key_to_camelot

    if not isinstance(metrics, KeyAnalysisResult):
        return

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║              KEY ANALYSIS PROFILE (CAMELOT)               ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║  Dominant Key:         {metrics.dominant_key:>8} ({metrics.dominant_camelot:>4})                    ║")
    print(f"║  Key Changes:                                       {metrics.key_changes:>5} ║")
    print(f"║  Key Stability:                                     {metrics.key_stability:>5.2f} ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # Interpret stability
    print("Interpretation:")
    if metrics.key_stability > 0.8:
        print("  → Very stable (single key or minimal changes)")
    elif metrics.key_stability > 0.5:
        print("  → Moderate key changes (harmonically mixed)")
    else:
        print("  → Frequent key changes (eclectic mixing)")

    # Show top keys if histogram available
    if metrics.key_histogram:
        sorted_keys = sorted(metrics.key_histogram.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 Keys:")
        for i, (key, count) in enumerate(sorted_keys, 1):
            camelot = key_to_camelot(key)
            print(f"    {i}. {key:>4} ({camelot:>4}) - {count:>3} windows")

    print()


def visualize_energy_arc(
    metrics: EnergyArcMetrics,
    tempo_trajectory=None,
    camelot_trajectory=None,
    output_path: str = None
):
    """
    Create energy arc visualization with BPM and Camelot key overlay.

    Args:
        metrics: Energy arc metrics
        tempo_trajectory: Optional list of BPM values over time
        camelot_trajectory: Optional list of Camelot keys over time
        output_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping visualization.")
        return

    trajectory = metrics.trajectory
    time_points = np.linspace(0, len(trajectory) * 5 / 60, len(trajectory))  # Convert to minutes

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot energy arc on primary axis
    color_energy = '#1f77b4'
    ax1.plot(time_points, trajectory, linewidth=2, color=color_energy, label='Energy')
    ax1.fill_between(time_points, trajectory, alpha=0.3, color=color_energy)

    # Mark opening, peak, closing
    opening_idx = int(len(trajectory) * 0.1)
    peak_idx = int(metrics.peak_timing_normalized * len(trajectory))
    closing_idx = int(len(trajectory) * 0.9)

    ax1.scatter([time_points[opening_idx]], [trajectory[opening_idx]],
                color='green', s=100, label='Opening', zorder=5)
    ax1.scatter([time_points[peak_idx]], [trajectory[peak_idx]],
                color='red', s=100, label='Peak', zorder=5)
    ax1.scatter([time_points[closing_idx]], [trajectory[closing_idx]],
                color='orange', s=100, label='Closing', zorder=5)

    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Normalized Energy', fontsize=12, color=color_energy)
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis='y', labelcolor=color_energy)
    ax1.grid(True, alpha=0.3)

    # Add BPM trajectory on secondary y-axis
    if tempo_trajectory is not None and len(tempo_trajectory) > 0:
        ax2 = ax1.twinx()

        # Align tempo trajectory with energy trajectory time points
        tempo_time_points = np.linspace(0, time_points[-1], len(tempo_trajectory))

        color_bpm = '#ff7f0e'
        ax2.plot(tempo_time_points, tempo_trajectory,
                linewidth=1.5, color=color_bpm, alpha=0.7,
                linestyle='--', label='BPM')
        ax2.set_ylabel('BPM', fontsize=12, color=color_bpm)
        ax2.tick_params(axis='y', labelcolor=color_bpm)

        # Set BPM y-axis limits with some padding
        bpm_min = min(tempo_trajectory) - 5
        bpm_max = max(tempo_trajectory) + 5
        ax2.set_ylim(bpm_min, bpm_max)

    # Add Camelot key annotations
    if camelot_trajectory is not None and len(camelot_trajectory) > 0:
        # Show keys at regular intervals (every 10 minutes or fewer if shorter)
        n_annotations = min(len(camelot_trajectory), int(time_points[-1] / 10) + 1)
        if n_annotations > 0:
            annotation_indices = np.linspace(0, len(camelot_trajectory) - 1, n_annotations, dtype=int)
            camelot_time_points = np.linspace(0, time_points[-1], len(camelot_trajectory))

            for idx in annotation_indices:
                key = camelot_trajectory[idx]
                time_min = camelot_time_points[idx]

                # Place annotation at top of plot
                ax1.annotate(
                    key,
                    xy=(time_min, 1.02),
                    xycoords=('data', 'axes fraction'),
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='black'),
                    zorder=10
                )

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    if tempo_trajectory is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    plt.title(f'DJ Set Profile: {metrics.arc_shape.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Energy arc visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_drop_pattern(metrics: DropPatternMetrics, output_path: str = None):
    """Create drop pattern visualization."""
    if len(metrics.drop_magnitudes) == 0:
        print("No drops detected - skipping visualization")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping visualization.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of drop magnitudes
    ax1.hist(metrics.drop_magnitudes, bins=20, color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax1.axvline(metrics.avg_drop_magnitude, color='red', linestyle='--',
                linewidth=2, label=f'Average: {metrics.avg_drop_magnitude:.2f}')
    ax1.set_title('Drop Magnitude Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Drop Magnitude', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Timeline of drops
    drop_times_min = [t / 60 for t in metrics.drop_times]
    ax2.scatter(drop_times_min, metrics.drop_magnitudes,
                s=100, alpha=0.6, color='#ff7f0e', edgecolors='black', linewidths=0.5)
    ax2.set_title(f'Drop Timeline - {metrics.drop_clustering.upper()} Style',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (minutes)', fontsize=10)
    ax2.set_ylabel('Drop Magnitude', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Drop pattern visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Profile DJ set characteristics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic profiling
  python scripts/profile_dj_set.py mix.mp3

  # Export to JSON
  python scripts/profile_dj_set.py mix.mp3 --json profile.json

  # Generate visualizations
  python scripts/profile_dj_set.py mix.mp3 --visualize

  # Full analysis with export
  python scripts/profile_dj_set.py mix.mp3 --json profile.json --visualize --output-dir ./viz
        """
    )
    parser.add_argument('input', help='Path to DJ set audio file')
    parser.add_argument('--json', dest='json_output',
                        help='Export profile to JSON file')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations (requires matplotlib)')
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for visualizations')
    parser.add_argument('--no-energy-arc', action='store_true',
                        help='Skip energy arc profiling')
    parser.add_argument('--no-drops', action='store_true',
                        help='Skip drop pattern profiling')
    parser.add_argument('--genre', action='store_true',
                        help='Enable ML-based genre classification (requires essentia-tensorflow)')
    parser.add_argument('--cache-dir', default='~/.mood-classifier/cache',
                        help='Cache directory (default: ~/.mood-classifier/cache)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: use hop_length=2048 (~4x faster, sufficient for DJ profiling)')
    parser.add_argument('--hop-length', type=int, default=None,
                        help='Custom hop_length (default: 512, fast mode: 2048)')
    parser.add_argument('--dj', type=str, default=None,
                        help='DJ name for profile database')
    parser.add_argument('--save-to-db', action='store_true',
                        help='Save analysis to DJ profile database')
    parser.add_argument('--venue', type=str, default=None,
                        help='Venue name (metadata)')
    parser.add_argument('--date', type=str, default=None,
                        help='Event date (metadata, format: YYYY-MM-DD)')
    parser.add_argument('--event', type=str, default=None,
                        help='Event name (metadata)')

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    print(f"╔═══════════════════════════════════════════════════════════╗")
    print(f"║              DJ SET PROFILING                             ║")
    print(f"╠═══════════════════════════════════════════════════════════╣")
    print(f"║  File: {input_path.name:<50} ║")
    print(f"╚═══════════════════════════════════════════════════════════╝")
    print()

    # Create cache repository
    cache_repo = CacheRepository(cache_dir=args.cache_dir)

    # Determine hop_length
    if args.hop_length:
        hop_length = args.hop_length
    elif args.fast:
        hop_length = 2048
        print("Fast mode enabled: using hop_length=2048 (~4x faster)")
    else:
        hop_length = 512

    # Create pipeline
    pipeline = Pipeline([
        LoadAudioStage(sr=22050, mono=True),
        ComputeSTFTStage(hop_length=hop_length),
        DJProfilingStage(
            include_energy_arc=not args.no_energy_arc,
            include_drop_pattern=not args.no_drops,
            include_genre=args.genre,
        ),
    ], name="DJProfilingPipeline")

    # Run pipeline with progress info
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                   PROCESSING PIPELINE                     ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    import time
    start_time = time.time()

    # Create context with cache_dir
    context = PipelineContext(
        input_path=str(input_path),
        cache_dir=str(Path(args.cache_dir).expanduser())
    )
    context = pipeline.run(context)
    elapsed = time.time() - start_time

    print()
    print(f"✓ Analysis complete in {elapsed:.1f}s")
    print()

    # Get results
    profile = context.get_result('dj_profile')

    if profile is None:
        print("Error: Profiling failed")
        sys.exit(1)

    # Print results
    if profile.genre and hasattr(profile.genre, 'success') and profile.genre.success:
        print_genre_metrics(profile.genre)

    if profile.energy_arc:
        print_energy_arc_metrics(profile.energy_arc)

    if profile.drop_pattern:
        print_drop_pattern_metrics(profile.drop_pattern)

    if profile.tempo_distribution:
        print_tempo_distribution_metrics(profile.tempo_distribution)

    if profile.key_analysis:
        print_key_analysis_metrics(profile.key_analysis)

    # Export to JSON
    if args.json_output:
        output_data = profile.to_dict()
        output_data['file'] = str(input_path)
        output_data['duration_sec'] = context.get_result('_duration')

        with open(args.json_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Profile exported to: {args.json_output}")
        print()

    # Generate visualizations
    if args.visualize:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        stem = input_path.stem

        if profile.energy_arc:
            viz_path = output_dir / f"{stem}_energy_arc.png"
            # Pass tempo and camelot trajectories if available
            tempo_traj = profile.tempo_distribution.tempo_trajectory if profile.tempo_distribution else None
            camelot_traj = profile.key_analysis.camelot_trajectory if profile.key_analysis else None
            visualize_energy_arc(
                profile.energy_arc,
                tempo_trajectory=tempo_traj,
                camelot_trajectory=camelot_traj,
                output_path=str(viz_path)
            )

        if profile.drop_pattern:
            viz_path = output_dir / f"{stem}_drop_pattern.png"
            visualize_drop_pattern(profile.drop_pattern, str(viz_path))

    # Save to DJ profile database
    if args.save_to_db:
        if not args.dj:
            print("Warning: --save-to-db requires --dj <name> parameter. Skipping database save.")
        else:
            print()
            print("╔═══════════════════════════════════════════════════════════╗")
            print("║              SAVING TO DJ PROFILE DATABASE                ║")
            print("╚═══════════════════════════════════════════════════════════╝")
            print()

            # Prepare set metadata
            set_data = profile.to_dict()
            set_data.update({
                'file_path': str(input_path.absolute()),
                'file_name': input_path.name,
                'duration_sec': context.get_result('_duration'),
                'analyzed_at': time.time(),
                'venue': args.venue,
                'event': args.event,
                'date': args.date,
            })

            # Save to database
            try:
                cache_repo.save_dj_profile_dict(
                    dj_name=args.dj,
                    profile_dict=set_data,
                    set_paths=[str(input_path.absolute())]
                )

                print(f"✓ Saved set analysis for: {args.dj}")
                print(f"  File: {input_path.name}")
                print(f"  Duration: {context.get_result('_duration')/60:.1f} min")

                # Show aggregate stats
                all_profiles = cache_repo.get_all_dj_profiles_info()
                dj_stats = next((p for p in all_profiles if p['dj_name'] == args.dj), None)

                if dj_stats:
                    print()
                    print(f"DJ Profile: {args.dj}")
                    print(f"  Total sets analyzed: {dj_stats['n_sets']}")
                    print(f"  Total hours: {dj_stats['total_hours']:.1f}h")
                    print(f"  Last updated: {time.strftime('%Y-%m-%d %H:%M', time.localtime(dj_stats['updated_at']))}")

            except Exception as e:
                print(f"Error saving to database: {e}")

    print()
    print("Done!")


if __name__ == '__main__':
    # Fix for numpy on macOS
    import numpy as np
    main()
