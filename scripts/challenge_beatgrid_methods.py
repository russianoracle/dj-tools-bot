#!/usr/bin/env python3
"""
Challenge all beat grid detection methods against Rekordbox ground truth.

ARCHITECTURE: Uses Tasks layer (BeatGridTask) - NOT primitives directly.
This follows the three-layer architecture: Scripts → Tasks → Primitives.

Compares:
1. STATIC - librosa beat_track (single global tempo)
2. PLP - Predominant Local Pulse (adaptive tempo)
3. ML_ENHANCED - XGBoost-enhanced detection
4. HYBRID - Consensus of all methods

Metrics:
- BPM accuracy (within 1 BPM of GT)
- Downbeat accuracy (within 50ms of GT)
- First downbeat accuracy
- Phase accuracy (correct 4-beat grouping)

Usage:
    python scripts/challenge_beatgrid_methods.py --sample 100
    python scripts/challenge_beatgrid_methods.py --full  # All tracks
"""

import sys
import argparse
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.tasks.beat_grid import BeatGridTask, BeatGridMode, BeatGridAnalysisResult
from src.core.tasks.base import create_audio_context

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MethodResult:
    """Result for a single method on a single track."""
    method: str
    bpm_detected: float
    bpm_gt: float
    bpm_error: float
    bpm_correct: bool  # Within tolerance
    first_downbeat_detected: float
    first_downbeat_gt: float
    first_downbeat_error: float
    first_downbeat_correct: bool
    downbeat_precision: float
    downbeat_recall: float
    downbeat_f1: float
    confidence: float


@dataclass
class ChallengeResult:
    """Aggregated results for a method across all tracks."""
    method: str
    n_tracks: int = 0
    bpm_accuracy: float = 0.0
    bpm_mean_error: float = 0.0
    bpm_median_error: float = 0.0
    first_downbeat_accuracy: float = 0.0
    first_downbeat_mean_error: float = 0.0
    downbeat_mean_f1: float = 0.0
    mean_confidence: float = 0.0

    # Per-track results
    track_results: List[MethodResult] = field(default_factory=list)


def load_ground_truth(gt_path: str, max_tracks: Optional[int] = None) -> List[Dict]:
    """Load Rekordbox ground truth."""
    with open(gt_path) as f:
        data = json.load(f)

    # Filter to existing files
    valid = []
    for track in data:
        if Path(track.get('file', '')).exists():
            valid.append(track)
            if max_tracks and len(valid) >= max_tracks:
                break

    return valid


def compute_downbeat_metrics(
    detected_downbeats: np.ndarray,
    gt_downbeats: np.ndarray,
    tolerance_sec: float = 0.05
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 for downbeat detection.

    Args:
        detected_downbeats: Detected downbeat times
        gt_downbeats: Ground truth downbeat times
        tolerance_sec: Tolerance for matching (default 50ms)

    Returns:
        (precision, recall, f1)
    """
    if len(detected_downbeats) == 0 or len(gt_downbeats) == 0:
        return 0.0, 0.0, 0.0

    # For each detected, find nearest GT
    true_positives = 0
    for det in detected_downbeats:
        min_dist = np.min(np.abs(gt_downbeats - det))
        if min_dist <= tolerance_sec:
            true_positives += 1

    precision = true_positives / len(detected_downbeats) if len(detected_downbeats) > 0 else 0
    recall = true_positives / len(gt_downbeats) if len(gt_downbeats) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def run_method(audio_context, mode: BeatGridMode) -> Dict:
    """
    Run beat grid method using BeatGridTask.

    ARCHITECTURE: Uses Tasks layer - does NOT call primitives directly.
    """
    task = BeatGridTask(
        beats_per_bar=4,
        bars_per_phrase=4,
        mode=mode,
    )
    result: BeatGridAnalysisResult = task.execute(audio_context)

    # Extract bar times (downbeats)
    downbeat_times = np.array([b.start_time for b in result.beat_grid.bars])

    return {
        'tempo': result.tempo,
        'confidence': result.tempo_confidence,
        'first_downbeat': downbeat_times[0] if len(downbeat_times) > 0 else 0.0,
        'downbeat_times': downbeat_times,
    }


def evaluate_method(
    method_name: str,
    result: Dict,
    gt: Dict,
    bpm_tolerance: float = 1.0,
    downbeat_tolerance: float = 0.05
) -> MethodResult:
    """Evaluate a method result against ground truth."""
    gt_bpm = gt['bpm']
    gt_first_db = gt['first_downbeat_sec']
    gt_downbeats = np.array(gt.get('downbeat_times', []))

    # BPM
    bpm_error = abs(result['tempo'] - gt_bpm)
    bpm_correct = bpm_error <= bpm_tolerance

    # Also check half/double (common octave errors)
    if not bpm_correct:
        if abs(result['tempo'] * 2 - gt_bpm) <= bpm_tolerance:
            bpm_error = abs(result['tempo'] * 2 - gt_bpm)
            bpm_correct = True
        elif abs(result['tempo'] / 2 - gt_bpm) <= bpm_tolerance:
            bpm_error = abs(result['tempo'] / 2 - gt_bpm)
            bpm_correct = True

    # First downbeat
    first_db_error = abs(result['first_downbeat'] - gt_first_db)
    first_db_correct = first_db_error <= downbeat_tolerance

    # Downbeat precision/recall
    precision, recall, f1 = compute_downbeat_metrics(
        result['downbeat_times'],
        gt_downbeats,
        tolerance_sec=downbeat_tolerance
    )

    return MethodResult(
        method=method_name,
        bpm_detected=result['tempo'],
        bpm_gt=gt_bpm,
        bpm_error=bpm_error,
        bpm_correct=bpm_correct,
        first_downbeat_detected=result['first_downbeat'],
        first_downbeat_gt=gt_first_db,
        first_downbeat_error=first_db_error,
        first_downbeat_correct=first_db_correct,
        downbeat_precision=precision,
        downbeat_recall=recall,
        downbeat_f1=f1,
        confidence=result['confidence'],
    )


def challenge_track(
    track: Dict,
    modes: Dict[str, BeatGridMode],
) -> Dict[str, MethodResult]:
    """
    Challenge all methods on a single track.

    ARCHITECTURE: Uses Tasks layer via create_audio_context() and BeatGridTask.
    Does NOT call primitives (compute_stft) directly.
    """
    from src.audio.loader import AudioLoader

    file_path = track['file']

    # Load audio
    loader = AudioLoader(sample_rate=22050)
    y, sr = loader.load(file_path)

    # Create audio context (computes STFT internally)
    audio_context = create_audio_context(
        y=y,
        sr=sr,
        file_path=file_path,
        n_fft=2048,
        hop_length=512,
    )

    results = {}
    for name, mode in modes.items():
        try:
            method_result = run_method(audio_context, mode)
            results[name] = evaluate_method(name, method_result, track)
        except Exception as e:
            logger.warning(f"{name} failed on {Path(file_path).name}: {e}")
            continue

    return results


def aggregate_results(all_results: List[Dict[str, MethodResult]]) -> Dict[str, ChallengeResult]:
    """Aggregate results across all tracks."""
    aggregated = {}

    # Collect per-method
    method_results = {}
    for track_results in all_results:
        for method_name, result in track_results.items():
            if method_name not in method_results:
                method_results[method_name] = []
            method_results[method_name].append(result)

    # Compute aggregates
    for method_name, results in method_results.items():
        n = len(results)

        bpm_correct = sum(1 for r in results if r.bpm_correct)
        bpm_errors = [r.bpm_error for r in results]
        first_db_correct = sum(1 for r in results if r.first_downbeat_correct)
        first_db_errors = [r.first_downbeat_error for r in results]
        f1_scores = [r.downbeat_f1 for r in results]
        confidences = [r.confidence for r in results]

        aggregated[method_name] = ChallengeResult(
            method=method_name,
            n_tracks=n,
            bpm_accuracy=bpm_correct / n if n > 0 else 0,
            bpm_mean_error=float(np.mean(bpm_errors)),
            bpm_median_error=float(np.median(bpm_errors)),
            first_downbeat_accuracy=first_db_correct / n if n > 0 else 0,
            first_downbeat_mean_error=float(np.mean(first_db_errors)),
            downbeat_mean_f1=float(np.mean(f1_scores)),
            mean_confidence=float(np.mean(confidences)),
            track_results=results,
        )

    return aggregated


def print_results(results: Dict[str, ChallengeResult]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("BEAT GRID METHOD CHALLENGE RESULTS")
    print("=" * 80)

    # Sort by BPM accuracy (primary metric)
    sorted_methods = sorted(results.values(), key=lambda r: r.bpm_accuracy, reverse=True)

    print(f"\n{'Method':<15} {'BPM Acc':<10} {'BPM Err':<10} {'1st DB Acc':<12} {'DB F1':<10} {'Conf':<10}")
    print("-" * 80)

    for r in sorted_methods:
        print(f"{r.method:<15} {r.bpm_accuracy*100:>6.1f}%    {r.bpm_mean_error:>6.2f}    "
              f"{r.first_downbeat_accuracy*100:>8.1f}%    {r.downbeat_mean_f1:>6.3f}    {r.mean_confidence:>6.3f}")

    # Winner
    winner = sorted_methods[0]
    print("\n" + "=" * 80)
    print(f"WINNER: {winner.method}")
    print(f"  - BPM accuracy: {winner.bpm_accuracy*100:.1f}%")
    print(f"  - Downbeat F1: {winner.downbeat_mean_f1:.3f}")
    print(f"  - Mean confidence: {winner.mean_confidence:.3f}")
    print("=" * 80)

    return winner.method


def main():
    parser = argparse.ArgumentParser(description='Challenge beat grid methods')
    parser.add_argument('--gt-path', default='data/beatgrid_training.json',
                        help='Ground truth JSON path')
    parser.add_argument('--sample', '-n', type=int, default=50,
                        help='Number of tracks to test')
    parser.add_argument('--full', action='store_true',
                        help='Test all tracks (overrides --sample)')
    parser.add_argument('--output', '-o', help='Save results to JSON')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    # Load ground truth
    max_tracks = None if args.full else args.sample
    gt_data = load_ground_truth(args.gt_path, max_tracks)

    print(f"Loaded {len(gt_data)} tracks for challenge")

    # Define methods using BeatGridMode enum
    # ARCHITECTURE: Uses Tasks layer via BeatGridTask(mode=...)
    modes = {
        'STATIC': BeatGridMode.STATIC,
        'PLP': BeatGridMode.PLP,
        'ML_ENHANCED': BeatGridMode.ML_ENHANCED,
        'HYBRID': BeatGridMode.HYBRID,
    }

    # Run challenge
    all_results = []
    for i, track in enumerate(gt_data):
        name = Path(track['file']).stem[:40]

        if args.verbose or (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(gt_data)}] {name}...")

        track_results = challenge_track(track, modes)
        all_results.append(track_results)

        if args.verbose:
            for method, result in track_results.items():
                status = "✓" if result.bpm_correct else "✗"
                print(f"  {method}: {status} BPM={result.bpm_detected:.1f} (GT={result.bpm_gt:.1f})")

    # Aggregate and print
    aggregated = aggregate_results(all_results)
    winner = print_results(aggregated)

    # Save results
    if args.output:
        output_data = {
            method: {
                'bpm_accuracy': r.bpm_accuracy,
                'bpm_mean_error': r.bpm_mean_error,
                'bpm_median_error': r.bpm_median_error,
                'first_downbeat_accuracy': r.first_downbeat_accuracy,
                'first_downbeat_mean_error': r.first_downbeat_mean_error,
                'downbeat_mean_f1': r.downbeat_mean_f1,
                'mean_confidence': r.mean_confidence,
                'n_tracks': r.n_tracks,
            }
            for method, r in aggregated.items()
        }
        output_data['winner'] = winner

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
