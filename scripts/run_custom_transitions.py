#!/usr/bin/env python3
"""
Run transition detection with custom calibrated parameters.

Usage:
    python scripts/run_custom_transitions.py <audio_file> [output_dir]
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import TransitionConfig, SetAnalysisConfig, SegmentationConfig, DropDetectionConfig
from src.core.pipelines.set_analysis import SetAnalysisPipeline


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def run_with_custom_params(
    audio_path: str,
    bass_weight: float = 0.80,
    timbral_weight: float = 0.19,
    peak_percentile: float = 90.4,
    min_gap_sec: float = 54.2,
    energy_threshold: float = 0.092,
    smooth_sigma: float = 5.06,
    output_dir: str = None,
):
    """
    Run transition detection with custom parameters.
    """
    audio_path = Path(audio_path)

    print("=" * 60)
    print("TRANSITION DETECTION - Custom Calibrated Parameters")
    print("=" * 60)
    print(f"\nInput: {audio_path.name}")
    print(f"\nParameters:")
    print(f"  bass_weight:      {bass_weight}")
    print(f"  timbral_weight:   {timbral_weight}")
    print(f"  peak_percentile:  {peak_percentile}")
    print(f"  min_gap_sec:      {min_gap_sec}")
    print(f"  energy_threshold: {energy_threshold}")
    print(f"  smooth_sigma:     {smooth_sigma}")
    print()

    # Create custom TransitionConfig
    transition_cfg = TransitionConfig(
        bass_weight=bass_weight,
        timbral_weight=timbral_weight,
        peak_percentile=peak_percentile,
        min_transition_gap_sec=min_gap_sec,
        energy_threshold=energy_threshold,
        smooth_sigma=smooth_sigma,
        detect_filters=True,
        filter_velocity_threshold=500.0,
        transition_merge_window_sec=90.0,
    )

    # Create full config
    config = SetAnalysisConfig(
        transition=transition_cfg,
        segmentation=SegmentationConfig(
            min_track_duration=60.0,
            min_transition_duration=30.0,
        ),
        drop_detection=DropDetectionConfig(
            min_drop_magnitude=0.12,
            min_confidence=0.20,
            use_multiband=False,
        ),
        analyze_genres=False,
    )

    # Create and run pipeline
    print("Running analysis...")
    pipeline = SetAnalysisPipeline(
        config=config,
        analyze_genres=False,
        verbose=True,
    )

    result = pipeline.analyze(str(audio_path))

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nDuration: {result.duration_sec/60:.1f} min")
    print(f"Transitions detected: {result.n_transitions}")
    print(f"Transition density: {result.transition_density:.2f}/min")
    print(f"Segments: {result.n_segments}")
    print(f"Drops: {result.total_drops}")

    # Print transition details
    if result.transition_times:
        print(f"\n--- Transitions ({result.n_transitions}) ---")
        for i, (mixin, mixout) in enumerate(result.transition_times, 1):
            duration = mixout - mixin
            print(f"  {i:2d}. Mixin: {format_time(mixin)} -> Mixout: {format_time(mixout)} ({duration:.0f}s)")

    # Print segments
    if result.segments:
        print(f"\n--- Timeline ({result.n_segments} segments) ---")
        for seg in result.segments:
            seg_type = "TRANSITION" if seg.is_transition_zone else "TRACK"
            start = format_time(seg.start_time)
            end = format_time(seg.end_time)
            dur = seg.duration / 60
            print(f"  [{seg_type:10}] {start} - {end} ({dur:.1f} min)")

    # Save results if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{audio_path.stem}_transitions.json"
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nSaved: {output_file}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: Josh Baker Boiler Room London
        audio_path = "data/dj_sets/josh-baker/boiler-room---josh-baker-boiler-room-london.m4a"
    else:
        audio_path = sys.argv[1]

    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/transitions"

    # Run with calibrated parameters
    run_with_custom_params(
        audio_path,
        bass_weight=0.80,
        timbral_weight=0.19,
        peak_percentile=90.4,
        min_gap_sec=54.2,
        energy_threshold=0.092,
        smooth_sigma=5.06,
        output_dir=output_dir,
    )