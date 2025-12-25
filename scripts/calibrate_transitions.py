#!/usr/bin/env python3
"""
CLI for TransitionDetection calibration using CalibrationPipeline.

Uses mir-aidj dataset for ground truth.

Usage:
    # Test current config
    python scripts/calibrate_transitions.py --test

    # Run calibration
    python scripts/calibrate_transitions.py --max-iter 100

    # Use specific mixes
    python scripts/calibrate_transitions.py --max-mixes 10

    # Use parallel loading (4 workers)
    python scripts/calibrate_transitions.py --max-iter 100 --workers 4

    # Use parallel loading (8 workers) for faster processing
    python scripts/calibrate_transitions.py --max-mixes 20 --workers 8
"""

import sys
import re
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.training.pipelines import CalibrationPipeline
from src.core.config import MixingStyle


# Exclude techno/hard style mixes - not suitable for smooth mixing calibration
EXCLUDE_MIXES = {
    'mix2050',  # Felix Kröcher - Hardliner (hard techno)
    'mix2085',  # the CZAP - CLR Podcast (techno)
    'mix2087',  # Fernanda Martins - Cubbo Podcast (hard techno)
    'mix5005',  # Karotte @ Harry Klein (techno)
}


def load_mir_aidj_dataset(
    audio_dir: Path,
    dataset_path: Path,
    max_mixes: int = None,
    exclude: set = None
):
    """
    Load mir-aidj dataset.

    Returns:
        audio_paths: List of audio file paths
        ground_truth: Dict {mix_id: [transition_times]}
    """
    exclude = exclude or set()

    with open(dataset_path) as f:
        dataset = json.load(f)

    # Find audio files
    audio_files = sorted(
        list(audio_dir.glob("mix*.m4a")) +
        list(audio_dir.glob("mix*.mp3")) +
        list(audio_dir.glob("mix*.opus"))
    )

    audio_paths = []
    ground_truth = {}

    for audio_path in audio_files:
        mix_id = audio_path.stem

        if mix_id in exclude:
            print(f"  {mix_id}: excluded")
            continue

        # Find ground truth
        gt_times = []
        for mix in dataset:
            if mix['id'] == mix_id:
                tracklist = mix.get('tracklist', [])
                for track in tracklist:
                    title = track.get('title', '')
                    if '??' in title[:10]:
                        continue
                    match = re.match(r'\[(\d{1,3})(?::(\d{2}))?\]', title)
                    if match:
                        minutes = int(match.group(1))
                        seconds = int(match.group(2)) if match.group(2) else 0
                        ts = minutes * 60 + seconds
                        if ts > 0:
                            gt_times.append(ts)
                break

        gt_times = list(np.unique(gt_times))

        if len(gt_times) < 3:
            print(f"  {mix_id}: only {len(gt_times)} GT, skipping")
            continue

        audio_paths.append(str(audio_path))
        ground_truth[mix_id] = gt_times
        print(f"  {mix_id}: {len(gt_times)} transitions")

        if max_mixes and len(audio_paths) >= max_mixes:
            break

    return audio_paths, ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate TransitionDetection parameters using CalibrationPipeline"
    )
    parser.add_argument("--test", action="store_true", help="Test current config only")
    parser.add_argument("--max-iter", type=int, default=100, help="Max optimization iterations")
    parser.add_argument("--max-mixes", type=int, default=None, help="Max mixes to use")
    parser.add_argument("--style", choices=["smooth", "standard", "hard"], default="smooth",
                       help="Mixing style to calibrate")
    parser.add_argument("--audio-dir", type=str,
                       default="experiments/transition_calibration/audio_cache",
                       help="Directory with audio files")
    parser.add_argument("--dataset", type=str,
                       default="experiments/transition_calibration/djmix-dataset.json",
                       help="Path to mir-aidj dataset JSON")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers for loading (default: 4)")

    args = parser.parse_args()

    # Convert style string to enum
    style_map = {
        'smooth': MixingStyle.SMOOTH,
        'standard': MixingStyle.STANDARD,
        'hard': MixingStyle.HARD,
    }
    mixing_style = style_map[args.style]

    print("=" * 60)
    print("TransitionDetection Calibration (CalibrationPipeline)")
    print("=" * 60)
    print()

    # Load dataset
    audio_dir = Path(args.audio_dir)
    dataset_path = Path(args.dataset)

    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}")
        return 1

    if not dataset_path.exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        return 1

    print(f"Loading mir-aidj dataset from {audio_dir}...")
    audio_paths, ground_truth = load_mir_aidj_dataset(
        audio_dir, dataset_path,
        max_mixes=args.max_mixes,
        exclude=EXCLUDE_MIXES
    )

    if not audio_paths:
        print("ERROR: No valid mixes found!")
        return 1

    print(f"\nLoaded {len(audio_paths)} mixes")
    total_gt = sum(len(gt) for gt in ground_truth.values())
    print(f"Total ground truth transitions: {total_gt}")
    print()

    # Create pipeline
    pipeline = CalibrationPipeline(
        mixing_style=mixing_style,
        max_iter=args.max_iter,
        verbose=True,
        num_workers=args.workers
    )

    if args.test:
        # Test current config
        f1, precision, recall = pipeline.test_current_config(audio_paths, ground_truth)
        print(f"\nCurrent config F1: {f1:.3f}")
        return 0

    # Run calibration
    result = pipeline.calibrate(
        audio_paths=audio_paths,
        ground_truth=ground_truth,
        max_iter=args.max_iter
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"calibrated_{args.style}_weights.json")

    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nSaved results to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())