#!/usr/bin/env python3
"""
Extract beat grid training data from Rekordbox - SIMPLE VERSION.

Uses pyrekordbox to directly link tracks to ANLZ files via AnalysisDataPath.
No BPM/duration matching needed - database provides direct mapping!

Usage:
    python scripts/extract_beatgrid_simple.py --output data/beatgrid_training.json
    python scripts/extract_beatgrid_simple.py --sample 100 --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrekordbox import Rekordbox6Database
from pyrekordbox.anlz import AnlzFile


# Rekordbox 6 database encryption key (publicly known)
RB6_DB_KEY = "402fd482c38817c35ffa8ffb8c7d93143b749e7d315df7a81732a1ff43608497"


def parse_anlz_beatgrid(anlz_path: Path) -> Optional[Dict]:
    """Parse beat grid from ANLZ file."""
    try:
        anlz = AnlzFile.parse_file(str(anlz_path))

        # Find PQTZ tag (beat grid)
        for tag in anlz.tags:
            if tag.type == 'PQTZ':
                entries = tag.content.entries
                if len(entries) < 4:
                    return None

                beat_times = []
                downbeat_times = []
                tempo_values = []

                for entry in entries:
                    time_sec = entry.time / 1000.0
                    tempo = entry.tempo / 100.0

                    beat_times.append(time_sec)
                    tempo_values.append(tempo)

                    if entry.beat == 1:
                        downbeat_times.append(time_sec)

                # Phrase = every 4 bars (16 beats)
                phrase_times = [downbeat_times[i] for i in range(0, len(downbeat_times), 4)]

                return {
                    'beat_times': beat_times,
                    'downbeat_times': downbeat_times,
                    'phrase_times': phrase_times,
                    'tempo_mean': float(np.mean(tempo_values)),
                    'tempo_std': float(np.std(tempo_values)) if len(tempo_values) > 1 else 0,
                    'first_downbeat_sec': downbeat_times[0] if downbeat_times else 0,
                    'n_beats': len(beat_times),
                    'n_bars': len(downbeat_times),
                    'n_phrases': len(phrase_times),
                }
        return None
    except Exception:
        return None


def extract_training_data(
    db_dir: Path,
    max_tracks: Optional[int] = None,
    require_local_file: bool = True,
    verbose: bool = False
) -> List[Dict]:
    """
    Extract training data from Rekordbox database.

    Direct approach: use AnalysisDataPath from database to find ANLZ files.
    """
    # Open database
    db = Rekordbox6Database(db_dir=db_dir, key=RB6_DB_KEY)

    # ANLZ files - try backup first, then system library
    anlz_search_paths = [
        db_dir / 'share',  # Backup location
        Path.home() / 'Library/Pioneer/rekordbox/share',  # System library
    ]

    training_data = []
    skipped_no_file = 0
    skipped_no_anlz = 0

    for content in db.get_content():
        # Get audio file path
        audio_path = content.OrgFolderPath or ''

        # Check file exists (optional)
        if require_local_file:
            if not audio_path or not Path(audio_path).exists():
                skipped_no_file += 1
                continue

        # Get ANLZ path - try multiple locations
        anlz_rel = content.AnalysisDataPath
        if not anlz_rel:
            skipped_no_anlz += 1
            continue

        anlz_path = None
        for search_base in anlz_search_paths:
            candidate = search_base / anlz_rel.lstrip('/')
            if candidate.exists():
                anlz_path = candidate
                break

        if anlz_path is None:
            skipped_no_anlz += 1
            continue

        # Parse beat grid
        beatgrid = parse_anlz_beatgrid(anlz_path)
        if beatgrid is None:
            skipped_no_anlz += 1
            continue

        # BPM from database (×100)
        bpm = content.BPM / 100.0 if content.BPM else 0

        example = {
            'file': audio_path,
            'name': content.Title or '',
            'artist': str(content.ArtistName) if content.ArtistName else '',
            'bpm': bpm,
            'key': str(content.KeyName) if content.KeyName else '',
            'duration_sec': content.Length or 0,
            **beatgrid
        }

        training_data.append(example)

        if verbose:
            print(f"[{len(training_data)}] {content.Title[:40] if content.Title else 'Unknown'}: "
                  f"BPM={bpm:.1f}, beats={beatgrid['n_beats']}")

        if max_tracks and len(training_data) >= max_tracks:
            break

    print(f"\nExtracted {len(training_data)} tracks")
    print(f"Skipped: {skipped_no_file} no file, {skipped_no_anlz} no ANLZ")

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
    parser = argparse.ArgumentParser(description='Extract beat grid from Rekordbox (simple)')
    parser.add_argument('--db-dir', '-d', default='data/rekordbox_bak_20251211',
                        help='Rekordbox database directory')
    parser.add_argument('--output', '-o', default='data/beatgrid_training.json',
                        help='Output JSON path')
    parser.add_argument('--sample', '-n', type=int, help='Limit number of tracks')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-require-file', action='store_true',
                        help='Include tracks without local audio files')

    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    if not db_dir.exists():
        print(f"ERROR: Database directory not found: {db_dir}")
        return

    print(f"Using database: {db_dir}")
    print(f"ANLZ files from: ~/Library/Pioneer/rekordbox/share/")

    # Extract data
    data = extract_training_data(
        db_dir=db_dir,
        max_tracks=args.sample,
        require_local_file=not args.no_require_file,
        verbose=args.verbose
    )

    print_statistics(data)

    # Save
    print(f"\nSaving to {args.output}...")

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
