#!/usr/bin/env python3
"""
Track Analysis Script - Analyze drops, transitions, and genre.

Usage:
    python scripts/analyze_track.py path/to/track.mp3
    python scripts/analyze_track.py path/to/track.mp3 --json
    python scripts/analyze_track.py path/to/track.mp3 --drops-only
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Suppress TensorFlow noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np

from src.core.tasks import (
    create_audio_context,
    DropDetectionTask,
    TransitionDetectionTask,
    GenreAnalysisTask,
)


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def analyze_track(
    file_path: str,
    analyze_drops: bool = True,
    analyze_transitions: bool = True,
    analyze_genre: bool = True,
    output_json: bool = False
):
    """Analyze a track and print results."""

    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Loading: {path.name}")

    # Load audio
    y, sr = librosa.load(str(path), sr=22050)
    duration = len(y) / sr

    print(f"Duration: {format_time(duration)} ({duration:.1f}s)")
    print(f"Sample rate: {sr} Hz")
    print()

    # Create context
    ctx = create_audio_context(y, sr, file_path=str(path))

    results = {
        'file': str(path),
        'duration_sec': duration,
    }

    # Drop Detection
    if analyze_drops:
        print("=" * 50)
        print("DROP DETECTION (multi-band)")
        print("=" * 50)

        drop_task = DropDetectionTask(
            use_multiband=True,
            min_drop_magnitude=0.25,
            min_confidence=0.4
        )
        drop_result = drop_task.execute(ctx)

        if drop_result.success:
            print(f"Drops found: {drop_result.n_drops}")
            print(f"Drop density: {drop_result.drop_density:.2f} per minute")
            print(f"Max magnitude: {drop_result.max_drop_magnitude:.2f}")
            print()

            if drop_result.drops:
                print("Drops:")
                for i, drop in enumerate(drop_result.drops, 1):
                    print(f"  {i}. {format_time(drop.time_sec)} "
                          f"(conf: {drop.confidence:.2f}, "
                          f"mag: {drop.drop_magnitude:.2f}, "
                          f"bass: {drop.bass_prominence:.2f})")
                print()

                print("Aggregate metrics:")
                print(f"  Avg intensity: {drop_result.avg_drop_intensity:.2f}")
                print(f"  Avg buildup duration: {drop_result.avg_buildup_duration:.1f}s")
                print(f"  Avg bass prominence: {drop_result.avg_bass_prominence:.2f}")
                print(f"  Temporal distribution: {drop_result.drop_temporal_distribution:.2f} "
                      f"(0=early, 1=late)")
                print()

                print("Energy metrics:")
                print(f"  Energy variance: {drop_result.energy_variance:.4f}")
                print(f"  Energy range: {drop_result.energy_range:.2f}")
                print(f"  Bass energy mean: {drop_result.bass_energy_mean:.2f}")

            results['drops'] = drop_result.to_dict()
        else:
            print(f"Error: {drop_result.error}")

        print()

    # Transition Detection
    if analyze_transitions:
        print("=" * 50)
        print("TRANSITION DETECTION (filter sweep)")
        print("=" * 50)

        trans_task = TransitionDetectionTask(
            detect_filters=True,
            min_transition_gap_sec=20.0,
            energy_threshold=0.25
        )
        trans_result = trans_task.execute(ctx)

        if trans_result.success:
            print(f"Mixins found: {len(trans_result.mixins)}")
            print(f"Mixouts found: {len(trans_result.mixouts)}")
            print(f"Paired transitions: {trans_result.n_transitions}")
            print()

            if trans_result.mixins:
                print("Mixins (track entries):")
                for i, mixin in enumerate(trans_result.mixins[:5], 1):
                    filter_info = f" [FILTER: {mixin.filter_direction}]" if mixin.filter_detected else ""
                    print(f"  {i}. {format_time(mixin.time_sec)} "
                          f"(conf: {mixin.confidence:.2f}, "
                          f"bass+: {mixin.bass_introduction:.2f}){filter_info}")
                if len(trans_result.mixins) > 5:
                    print(f"  ... and {len(trans_result.mixins) - 5} more")
                print()

            if trans_result.mixouts:
                print("Mixouts (track exits):")
                for i, mixout in enumerate(trans_result.mixouts[:5], 1):
                    filter_info = f" [FILTER: {mixout.filter_direction}]" if mixout.filter_detected else ""
                    print(f"  {i}. {format_time(mixout.time_sec)} "
                          f"(conf: {mixout.confidence:.2f}, "
                          f"bass-: {mixout.bass_removal:.2f}){filter_info}")
                if len(trans_result.mixouts) > 5:
                    print(f"  ... and {len(trans_result.mixouts) - 5} more")
                print()

            if trans_result.transitions:
                print("Paired transitions:")
                for i, t in enumerate(trans_result.transitions, 1):
                    print(f"  {i}. {format_time(t.mixin.time_sec)} -> "
                          f"{format_time(t.mixout.time_sec)} "
                          f"({t.overlap_sec:.0f}s, {t.transition_type.name})")
                print()

                print("Type distribution:")
                for type_name, count in trans_result.transition_type_distribution.items():
                    print(f"  {type_name}: {count}")

            results['transitions'] = trans_result.to_dict()
        else:
            print(f"Error: {trans_result.error}")

        print()

    # Genre Analysis
    if analyze_genre:
        print("=" * 50)
        print("GENRE ANALYSIS (Essentia)")
        print("=" * 50)

        try:
            genre_task = GenreAnalysisTask(top_n=5)
            genre_result = genre_task.execute(ctx)

            if genre_result.success:
                print(f"Genre: {genre_result.genre}")
                print(f"Subgenre: {genre_result.subgenre}")
                print(f"DJ Category: {genre_result.dj_category}")
                print(f"Confidence: {genre_result.confidence:.0%}")
                print()

                if genre_result.all_styles:
                    print("Top styles:")
                    for style, score in genre_result.all_styles:
                        print(f"  {style}: {score:.2%}")

                if genre_result.mood_tags:
                    # Handle both string and tuple formats
                    tags = [t[0] if isinstance(t, tuple) else t for t in genre_result.mood_tags]
                    print(f"\nMood tags: {', '.join(tags)}")

                results['genre'] = genre_result.to_dict()
            else:
                print(f"Error: {genre_result.error}")
                print("(Genre analysis requires Essentia library)")
        except ImportError:
            print("Genre analysis not available (Essentia not installed)")

        print()

    # JSON output
    if output_json:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        results = convert_numpy(results)

        json_path = path.with_suffix('.analysis.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {json_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze track for drops, transitions, and genre"
    )
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--json", action="store_true", help="Save results as JSON")
    parser.add_argument("--drops-only", action="store_true", help="Only analyze drops")
    parser.add_argument("--transitions-only", action="store_true", help="Only analyze transitions")
    parser.add_argument("--genre-only", action="store_true", help="Only analyze genre")

    args = parser.parse_args()

    # Determine what to analyze
    if args.drops_only:
        analyze_drops, analyze_transitions, analyze_genre = True, False, False
    elif args.transitions_only:
        analyze_drops, analyze_transitions, analyze_genre = False, True, False
    elif args.genre_only:
        analyze_drops, analyze_transitions, analyze_genre = False, False, True
    else:
        analyze_drops = analyze_transitions = analyze_genre = True

    analyze_track(
        args.file,
        analyze_drops=analyze_drops,
        analyze_transitions=analyze_transitions,
        analyze_genre=analyze_genre,
        output_json=args.json
    )


if __name__ == "__main__":
    main()