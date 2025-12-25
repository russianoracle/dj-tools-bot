#!/usr/bin/env python3
"""
Beat Grid Validation Script - Compare our detection with Rekordbox ground truth.

Uses existing project entities (AudioLoader, create_audio_context, BeatGridTask).

Usage:
    python scripts/validate_beat_grid.py --xml data/17-57.xml --sample 50
    python scripts/validate_beat_grid.py --xml data/17-57.xml --bpm-range 125-135
    python scripts/validate_beat_grid.py --xml data/17-57.xml --genre "Techno" --verbose
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use existing project entities
from src.audio.loader import AudioLoader
from src.core.tasks import (
    create_audio_context,
    BeatGridTask,
    BeatGridMode,
)


def parse_rekordbox_xml_for_bpm(
    xml_path: str,
    bpm_range: Optional[Tuple[float, float]] = None,
    genre_filter: Optional[str] = None,
    max_tracks: Optional[int] = None
) -> List[Dict]:
    """
    Parse Rekordbox XML and extract BPM ground truth.

    Args:
        xml_path: Path to Rekordbox XML
        bpm_range: Optional (min_bpm, max_bpm) filter
        genre_filter: Optional genre substring filter
        max_tracks: Maximum number of tracks to return

    Returns:
        List of track dicts with 'file', 'bpm', 'name', 'genre'
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracks = []

    for track in root.findall('.//TRACK'):
        bpm = float(track.get('AverageBpm', 0))
        if bpm <= 0:
            continue

        # BPM range filter
        if bpm_range:
            if bpm < bpm_range[0] or bpm > bpm_range[1]:
                continue

        genre = track.get('Genre', '')

        # Genre filter
        if genre_filter:
            if genre_filter.lower() not in genre.lower():
                continue

        location = track.get('Location', '')
        if not location:
            continue

        # Decode file path
        file_path = unquote(location).replace('file://localhost', '')

        # Check file exists
        if not Path(file_path).exists():
            continue

        tracks.append({
            'file': file_path,
            'bpm': bpm,
            'name': track.get('Name', ''),
            'artist': track.get('Artist', ''),
            'genre': genre,
            'duration_sec': int(track.get('TotalTime', 0)),
            'key': track.get('Tonality', '')
        })

        if max_tracks and len(tracks) >= max_tracks:
            break

    return tracks


def validate_beat_grid(
    tracks: List[Dict],
    mode: BeatGridMode = BeatGridMode.STATIC,
    verbose: bool = False
) -> Dict:
    """
    Run beat grid detection on tracks and compare with Rekordbox.

    Uses existing project entities: AudioLoader, create_audio_context, BeatGridTask.

    Args:
        tracks: List of track dicts with 'file' and 'bpm' (ground truth)
        mode: BeatGridMode.STATIC or BeatGridMode.PLP
        verbose: Print per-track results

    Returns:
        Validation results dict
    """
    loader = AudioLoader()
    task = BeatGridTask(mode=mode)

    results = {
        'total_tracks': len(tracks),
        'processed': 0,
        'failed': 0,
        'bpm_errors': [],
        'bpm_errors_abs': [],
        'exact_matches': 0,  # Within 0.5 BPM
        'close_matches': 0,  # Within 2 BPM
        'half_double_matches': 0,  # BPM is half or double
        'failed_tracks': [],
        'details': []
    }

    for i, track in enumerate(tracks):
        file_path = track['file']
        gt_bpm = track['bpm']

        if verbose:
            print(f"[{i+1}/{len(tracks)}] {track['name'][:40]}... ", end='', flush=True)

        try:
            # Load audio using project's AudioLoader (sr=22050 set in constructor)
            y, sr = loader.load(file_path)

            # Create AudioContext using project's factory function
            context = create_audio_context(
                y=y,
                sr=sr,
                file_path=file_path,
                hop_length=512,
                n_fft=2048
            )

            # Run beat grid detection
            start_time = time.time()
            result = task.execute(context)
            proc_time = time.time() - start_time

            if not result.beat_grid:
                results['failed'] += 1
                results['failed_tracks'].append({
                    'file': file_path,
                    'name': track['name'],
                    'error': 'No beat grid result'
                })
                if verbose:
                    print("FAILED: No beat grid")
                continue

            detected_bpm = result.tempo

            # Calculate BPM error
            bpm_error = detected_bpm - gt_bpm
            bpm_error_abs = abs(bpm_error)

            # Check for half/double BPM detection
            is_half = abs(detected_bpm * 2 - gt_bpm) < 2.0
            is_double = abs(detected_bpm / 2 - gt_bpm) < 2.0

            if is_half:
                corrected_bpm = detected_bpm * 2
                bpm_error = corrected_bpm - gt_bpm
                bpm_error_abs = abs(bpm_error)
                results['half_double_matches'] += 1
            elif is_double:
                corrected_bpm = detected_bpm / 2
                bpm_error = corrected_bpm - gt_bpm
                bpm_error_abs = abs(bpm_error)
                results['half_double_matches'] += 1

            results['bpm_errors'].append(bpm_error)
            results['bpm_errors_abs'].append(bpm_error_abs)
            results['processed'] += 1

            # Categorize match quality
            if bpm_error_abs <= 0.5:
                results['exact_matches'] += 1
            if bpm_error_abs <= 2.0:
                results['close_matches'] += 1

            detail = {
                'file': file_path,
                'name': track['name'],
                'gt_bpm': gt_bpm,
                'detected_bpm': detected_bpm,
                'bpm_error': bpm_error,
                'n_phrases': result.n_phrases,
                'tempo_confidence': result.tempo_confidence,
                'processing_time': proc_time,
                'is_half_double': is_half or is_double
            }
            results['details'].append(detail)

            if verbose:
                status = "OK" if bpm_error_abs <= 2.0 else "MISMATCH"
                half_note = " (half/double)" if is_half or is_double else ""
                print(f"{status} GT={gt_bpm:.1f} Det={detected_bpm:.1f} Err={bpm_error:+.2f}{half_note}")

        except Exception as e:
            results['failed'] += 1
            results['failed_tracks'].append({
                'file': file_path,
                'name': track['name'],
                'error': str(e)
            })
            if verbose:
                print(f"ERROR: {e}")

    # Calculate statistics
    if results['bpm_errors_abs']:
        errors = np.array(results['bpm_errors_abs'])
        results['stats'] = {
            'mean_abs_error': float(np.mean(errors)),
            'median_abs_error': float(np.median(errors)),
            'std_abs_error': float(np.std(errors)),
            'max_abs_error': float(np.max(errors)),
            'p95_abs_error': float(np.percentile(errors, 95)),
            'exact_match_rate': results['exact_matches'] / results['processed'],
            'close_match_rate': results['close_matches'] / results['processed'],
            'half_double_rate': results['half_double_matches'] / results['processed']
        }

    return results


def print_report(results: Dict):
    """Print validation report."""
    print("\n" + "=" * 70)
    print("BEAT GRID VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTracks processed: {results['processed']} / {results['total_tracks']}")
    print(f"Failed: {results['failed']}")

    if 'stats' in results:
        stats = results['stats']
        print(f"\n--- BPM Accuracy ---")
        print(f"Mean Absolute Error:   {stats['mean_abs_error']:.2f} BPM")
        print(f"Median Absolute Error: {stats['median_abs_error']:.2f} BPM")
        print(f"Std Dev:               {stats['std_abs_error']:.2f} BPM")
        print(f"95th Percentile:       {stats['p95_abs_error']:.2f} BPM")
        print(f"Max Error:             {stats['max_abs_error']:.2f} BPM")

        print(f"\n--- Match Quality ---")
        print(f"Exact matches (±0.5 BPM): {results['exact_matches']:>3} ({stats['exact_match_rate']*100:.1f}%)")
        print(f"Close matches (±2.0 BPM): {results['close_matches']:>3} ({stats['close_match_rate']*100:.1f}%)")
        print(f"Half/Double corrections:  {results['half_double_matches']:>3} ({stats['half_double_rate']*100:.1f}%)")

    # Show worst mismatches
    if results['details']:
        worst = sorted(results['details'], key=lambda x: abs(x['bpm_error']), reverse=True)[:5]
        if worst and abs(worst[0]['bpm_error']) > 2.0:
            print(f"\n--- Worst Mismatches ---")
            for d in worst:
                if abs(d['bpm_error']) > 2.0:
                    print(f"  {d['name'][:40]}: GT={d['gt_bpm']:.1f} Det={d['detected_bpm']:.1f} Err={d['bpm_error']:+.1f}")

    # Show failures
    if results['failed_tracks']:
        print(f"\n--- Failed Tracks ---")
        for ft in results['failed_tracks'][:5]:
            print(f"  {ft['name'][:40]}: {ft['error'][:50]}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Validate beat grid against Rekordbox')
    parser.add_argument('--xml', '-x', default='data/17-57.xml', help='Rekordbox XML path')
    parser.add_argument('--sample', '-n', type=int, default=50, help='Number of tracks to sample')
    parser.add_argument('--bpm-range', '-b', help='BPM range filter (e.g., "125-135")')
    parser.add_argument('--genre', '-g', help='Genre filter substring')
    parser.add_argument('--output', '-o', help='Output JSON path')
    parser.add_argument('--plp', action='store_true', help='Use PLP mode (for DJ sets with tempo changes)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--random', '-r', action='store_true', help='Random sample instead of first N')

    args = parser.parse_args()

    # Parse BPM range
    bpm_range = None
    if args.bpm_range:
        parts = args.bpm_range.split('-')
        bpm_range = (float(parts[0]), float(parts[1]))

    print(f"Loading tracks from {args.xml}...")
    tracks = parse_rekordbox_xml_for_bpm(
        args.xml,
        bpm_range=bpm_range,
        genre_filter=args.genre,
        max_tracks=None if args.random else args.sample
    )

    print(f"Found {len(tracks)} tracks matching criteria")

    # Random sampling
    if args.random and len(tracks) > args.sample:
        import random
        tracks = random.sample(tracks, args.sample)
        print(f"Randomly sampled {len(tracks)} tracks")
    elif len(tracks) > args.sample:
        tracks = tracks[:args.sample]

    if not tracks:
        print("No tracks found!")
        return

    # Select mode
    mode = BeatGridMode.PLP if args.plp else BeatGridMode.STATIC

    print(f"\nValidating {len(tracks)} tracks (mode={mode.name})...")
    print("-" * 70)

    results = validate_beat_grid(
        tracks,
        mode=mode,
        verbose=args.verbose
    )

    print_report(results)

    # Save results
    if args.output:
        # Remove numpy arrays for JSON serialization
        save_results = {k: v for k, v in results.items() if k not in ['bpm_errors', 'bpm_errors_abs']}
        with open(args.output, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
