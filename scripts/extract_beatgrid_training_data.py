#!/usr/bin/env python3
"""
Extract beat grid training data from Rekordbox database.

Combines:
- Track metadata from master.db (BPM, path, key)
- Beat grid from ANLZ files (beat positions, downbeats)

Output: JSON with tracks having:
- file: path to audio file
- bpm: Rekordbox BPM (ground truth)
- first_downbeat: position of first downbeat in seconds
- beat_times: list of beat times
- downbeat_times: list of downbeat times (beat 1 of each bar)
- phrase_times: list of phrase boundaries (every 16 beats)

Usage:
    python scripts/extract_beatgrid_training_data.py --output data/beatgrid_training.json
    python scripts/extract_beatgrid_training_data.py --bpm-range 120-140 --sample 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrekordbox import Rekordbox6Database
from pyrekordbox.anlz import AnlzFile


# Database encryption key (publicly known for Rekordbox 6)
RB6_DB_KEY = "402fd482c38817c35ffa8ffb8c7d93143b749e7d315df7a81732a1ff43608497"


def parse_anlz_beatgrid(anlz_path: str) -> Optional[Dict]:
    """
    Parse beat grid from ANLZ file.

    Returns dict with:
    - beat_times: all beat times in seconds
    - downbeat_times: times of beat 1 (bar start)
    - phrase_times: times of phrase boundaries (every 4 bars)
    - tempo_values: BPM at each beat
    """
    try:
        anlz = AnlzFile.parse_file(anlz_path)

        # Find PQTZ tag (quantization/beat grid)
        pqtz_tag = None
        for tag in anlz.tags:
            if tag.type == 'PQTZ':
                pqtz_tag = tag
                break

        if pqtz_tag is None or not hasattr(pqtz_tag.content, 'entries'):
            return None

        entries = pqtz_tag.content.entries
        if len(entries) < 4:
            return None

        beat_times = []
        downbeat_times = []
        tempo_values = []

        for entry in entries:
            beat_num = entry.beat  # 1-4
            tempo = entry.tempo / 100.0  # BPM
            time_sec = entry.time / 1000.0  # seconds

            beat_times.append(time_sec)
            tempo_values.append(tempo)

            if beat_num == 1:
                downbeat_times.append(time_sec)

        # Phrase boundaries: every 4 bars (every 4 downbeats)
        phrase_times = [downbeat_times[i] for i in range(0, len(downbeat_times), 4)]

        return {
            'beat_times': beat_times,
            'downbeat_times': downbeat_times,
            'phrase_times': phrase_times,
            'tempo_values': tempo_values,
            'tempo_mean': float(np.mean(tempo_values)) if tempo_values else 0,
            'tempo_std': float(np.std(tempo_values)) if len(tempo_values) > 1 else 0,
            'n_beats': len(beat_times),
            'n_bars': len(downbeat_times),
            'n_phrases': len(phrase_times),
        }

    except Exception as e:
        return None


def extract_training_data(
    db_dir: Path,
    bpm_range: Optional[Tuple[float, float]] = None,
    max_tracks: Optional[int] = None,
    require_local_file: bool = True,
    verbose: bool = False
) -> List[Dict]:
    """
    Extract training data from Rekordbox database.

    Args:
        db_dir: Path to Rekordbox backup directory with master.db
        bpm_range: Optional (min_bpm, max_bpm) filter
        max_tracks: Maximum number of tracks to extract
        require_local_file: Only include tracks with existing local files
        verbose: Print progress

    Returns:
        List of training examples with audio path, BPM, beat grid
    """
    # Open database
    db = Rekordbox6Database(db_dir=db_dir, key=RB6_DB_KEY)

    # ANLZ search paths (try multiple locations)
    anlz_search_paths = [
        db_dir / 'share',
        Path.home() / 'Library/Pioneer/rekordbox/share',
    ]

    training_data = []
    skipped_no_file = 0
    skipped_no_anlz = 0
    skipped_bpm_range = 0

    for content in db.get_content():
        # Get file path
        path = content.OrgFolderPath
        if not path:
            skipped_no_file += 1
            continue

        # Check file exists
        if require_local_file and not Path(path).exists():
            skipped_no_file += 1
            continue

        # Get BPM (stored as BPM × 100)
        bpm = content.BPM / 100.0 if content.BPM else 0

        # BPM filter
        if bpm_range:
            if bpm < bpm_range[0] or bpm > bpm_range[1]:
                skipped_bpm_range += 1
                continue

        # Get ANLZ path and parse beat grid
        anlz_path_rel = content.AnalysisDataPath
        if not anlz_path_rel:
            skipped_no_anlz += 1
            continue

        # Try to find ANLZ file in multiple locations
        anlz_path = None
        for search_base in anlz_search_paths:
            # AnalysisDataPath is like "/PIONEER/USBANLZ/xxx/uuid/ANLZ0000.DAT"
            candidate = search_base / anlz_path_rel.lstrip('/')
            if candidate.exists():
                anlz_path = candidate
                break

        if anlz_path is None:
            skipped_no_anlz += 1
            continue

        # Parse beat grid
        beatgrid = parse_anlz_beatgrid(str(anlz_path))
        if beatgrid is None:
            skipped_no_anlz += 1
            continue

        # Create training example
        example = {
            'file': path,
            'name': content.Title or '',
            'artist': str(content.ArtistName) if content.ArtistName else '',
            'bpm': bpm,
            'key': str(content.KeyName) if content.KeyName else '',
            'duration_sec': content.Length or 0,
            'first_downbeat_sec': beatgrid['downbeat_times'][0] if beatgrid['downbeat_times'] else 0,
            **beatgrid
        }

        training_data.append(example)

        if verbose:
            print(f"[{len(training_data)}] {content.Title[:40]}: "
                  f"BPM={bpm:.1f}, beats={beatgrid['n_beats']}, "
                  f"1st downbeat={example['first_downbeat_sec']:.3f}s")

        if max_tracks and len(training_data) >= max_tracks:
            break

    print(f"\nExtracted {len(training_data)} tracks")
    print(f"Skipped: {skipped_no_file} no file, {skipped_no_anlz} no ANLZ, {skipped_bpm_range} BPM range")

    return training_data


def print_statistics(data: List[Dict]):
    """Print statistics about extracted data."""
    if not data:
        return

    bpms = [d['bpm'] for d in data]
    first_downbeats = [d['first_downbeat_sec'] for d in data]
    n_beats = [d['n_beats'] for d in data]

    print(f"\n{'='*60}")
    print("TRAINING DATA STATISTICS")
    print(f"{'='*60}")
    print(f"Total tracks: {len(data)}")
    print(f"\nBPM:")
    print(f"  Mean: {np.mean(bpms):.1f}")
    print(f"  Std: {np.std(bpms):.1f}")
    print(f"  Range: {min(bpms):.1f} - {max(bpms):.1f}")

    print(f"\nFirst downbeat position:")
    print(f"  Mean: {np.mean(first_downbeats):.3f}s")
    print(f"  Std: {np.std(first_downbeats):.3f}s")
    print(f"  Range: {min(first_downbeats):.3f}s - {max(first_downbeats):.3f}s")

    print(f"\nBeats per track:")
    print(f"  Mean: {np.mean(n_beats):.0f}")
    print(f"  Range: {min(n_beats)} - {max(n_beats)}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Extract beat grid training data from Rekordbox')
    parser.add_argument('--db-dir', '-d', help='Rekordbox backup directory (default: auto-detect)')
    parser.add_argument('--output', '-o', default='data/beatgrid_training.json', help='Output JSON path')
    parser.add_argument('--bpm-range', '-b', help='BPM range filter (e.g., "120-140")')
    parser.add_argument('--sample', '-n', type=int, help='Limit number of tracks')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-require-file', action='store_true',
                        help='Include tracks without local files')

    args = parser.parse_args()

    # Find database directory
    if args.db_dir:
        db_dir = Path(args.db_dir)
    else:
        # Auto-detect from project
        project_root = Path(__file__).parent.parent
        backup_dirs = list(project_root.glob('data/rekordbox_bak_*'))
        if not backup_dirs:
            print("ERROR: No Rekordbox backup found in data/rekordbox_bak_*/")
            print("Copy your Rekordbox database there first.")
            return
        db_dir = sorted(backup_dirs)[-1]  # Latest backup

    print(f"Using database: {db_dir}")

    # Parse BPM range
    bpm_range = None
    if args.bpm_range:
        parts = args.bpm_range.split('-')
        bpm_range = (float(parts[0]), float(parts[1]))
        print(f"BPM filter: {bpm_range[0]:.0f} - {bpm_range[1]:.0f}")

    # Extract data
    data = extract_training_data(
        db_dir=db_dir,
        bpm_range=bpm_range,
        max_tracks=args.sample,
        require_local_file=not args.no_require_file,
        verbose=args.verbose
    )

    print_statistics(data)

    # Save
    print(f"\nSaving to {args.output}...")

    # Convert numpy arrays to lists for JSON
    def to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_serializable(i) for i in obj]
        return obj

    with open(args.output, 'w') as f:
        json.dump(to_json_serializable(data), f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
