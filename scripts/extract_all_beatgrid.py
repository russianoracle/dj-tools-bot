#!/usr/bin/env python3
"""
Extract ALL beat grid data from Rekordbox ANLZ files.

Scans all ANLZ*.DAT files in backup directory and extracts:
- Beat positions (every beat)
- Downbeat positions (beat 1 of each bar)
- Phrase boundaries (every 4 bars)
- Tempo values at each beat

Then matches to XML tracks by BPM + duration for training data.

Usage:
    python scripts/extract_all_beatgrid.py --output data/beatgrid_all.json
    python scripts/extract_all_beatgrid.py --match-xml data/17-57.xml --output data/beatgrid_training.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrekordbox.anlz import AnlzFile


def parse_anlz_beatgrid(anlz_path: Path) -> Optional[Dict]:
    """
    Parse beat grid from ANLZ file.

    Returns dict with:
    - beat_times: all beat times in seconds
    - downbeat_times: times of beat 1 (bar start)
    - phrase_times: times of phrase boundaries (every 4 bars)
    - tempo_values: BPM at each beat
    """
    try:
        anlz = AnlzFile.parse_file(str(anlz_path))

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

        # Calculate duration from last beat
        duration_sec = beat_times[-1] if beat_times else 0

        return {
            'beat_times': beat_times,
            'downbeat_times': downbeat_times,
            'phrase_times': phrase_times,
            'tempo_mean': float(np.mean(tempo_values)) if tempo_values else 0,
            'tempo_std': float(np.std(tempo_values)) if len(tempo_values) > 1 else 0,
            'first_downbeat_sec': downbeat_times[0] if downbeat_times else 0,
            'n_beats': len(beat_times),
            'n_bars': len(downbeat_times),
            'n_phrases': len(phrase_times),
            'duration_sec': duration_sec,
        }

    except Exception as e:
        return None


def scan_all_anlz_files(backup_dir: Path, verbose: bool = False) -> Dict[str, Dict]:
    """
    Scan all ANLZ files in backup directory.

    Args:
        backup_dir: Path to Rekordbox backup (containing share/PIONEER/USBANLZ)
        verbose: Print progress

    Returns:
        Dict mapping UUID to beat grid data
    """
    anlz_base = backup_dir / 'share' / 'PIONEER' / 'USBANLZ'
    if not anlz_base.exists():
        print(f"ERROR: ANLZ directory not found: {anlz_base}")
        return {}

    anlz_files = list(anlz_base.rglob('ANLZ*.DAT'))
    print(f"Found {len(anlz_files)} ANLZ files")

    results = {}
    parsed = 0
    failed = 0

    for i, anlz_path in enumerate(anlz_files):
        if verbose and i % 100 == 0:
            print(f"Processing {i}/{len(anlz_files)}...")

        # Extract UUID from path: .../xxx/uuid-part/ANLZ0000.DAT
        # UUID is the folder name containing the ANLZ file
        uuid_folder = anlz_path.parent.name
        prefix_folder = anlz_path.parent.parent.name
        uuid = f"{prefix_folder}{uuid_folder}"  # e.g., "79750016-28d5-44dc-bb86-8c27ef4f0c3a"

        beatgrid = parse_anlz_beatgrid(anlz_path)
        if beatgrid:
            results[uuid] = {
                'anlz_path': str(anlz_path),
                **beatgrid
            }
            parsed += 1
        else:
            failed += 1

    print(f"Parsed {parsed} beat grids, {failed} failed")
    return results


def load_xml_tracks(xml_path: str) -> List[Dict]:
    """Load tracks from Rekordbox XML export."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracks = []
    for track in root.findall('.//TRACK'):
        bpm = float(track.get('AverageBpm', 0))
        if bpm <= 0:
            continue

        location = track.get('Location', '')
        if not location:
            continue

        file_path = unquote(location).replace('file://localhost', '')
        duration = int(track.get('TotalTime', 0))

        tracks.append({
            'file': file_path,
            'name': track.get('Name', ''),
            'artist': track.get('Artist', ''),
            'bpm': bpm,
            'duration_sec': duration,
            'key': track.get('Tonality', ''),
            'genre': track.get('Genre', ''),
        })

    return tracks


def match_beatgrid_to_tracks(
    beatgrid_data: Dict[str, Dict],
    xml_tracks: List[Dict],
    bpm_tolerance: float = 1.0,
    duration_tolerance: float = 5.0,
    verbose: bool = False
) -> List[Dict]:
    """
    Match beat grid data to XML tracks by BPM and duration.

    Args:
        beatgrid_data: Dict of UUID -> beat grid data
        xml_tracks: List of track dicts from XML
        bpm_tolerance: Max BPM difference for match
        duration_tolerance: Max duration difference in seconds
        verbose: Print matches

    Returns:
        List of matched training examples
    """
    training_data = []
    matched = 0
    unmatched = 0

    # Build lookup by BPM (bucketed)
    bpm_buckets = {}
    for uuid, bg in beatgrid_data.items():
        bpm_bucket = round(bg['tempo_mean'])
        if bpm_bucket not in bpm_buckets:
            bpm_buckets[bpm_bucket] = []
        bpm_buckets[bpm_bucket].append((uuid, bg))

    for track in xml_tracks:
        track_bpm = track['bpm']
        track_dur = track['duration_sec']

        # Check nearby BPM buckets
        candidates = []
        for bpm_offset in range(-2, 3):
            bucket = round(track_bpm) + bpm_offset
            if bucket in bpm_buckets:
                candidates.extend(bpm_buckets[bucket])

        # Find best match by BPM + duration
        best_match = None
        best_score = float('inf')

        for uuid, bg in candidates:
            bpm_diff = abs(bg['tempo_mean'] - track_bpm)
            dur_diff = abs(bg['duration_sec'] - track_dur)

            if bpm_diff <= bpm_tolerance and dur_diff <= duration_tolerance:
                score = bpm_diff + dur_diff * 0.1  # Weight BPM more
                if score < best_score:
                    best_score = score
                    best_match = (uuid, bg)

        if best_match:
            uuid, bg = best_match
            example = {
                **track,
                'uuid': uuid,
                'anlz_bpm': bg['tempo_mean'],
                'first_downbeat_sec': bg['first_downbeat_sec'],
                'beat_times': bg['beat_times'],
                'downbeat_times': bg['downbeat_times'],
                'phrase_times': bg['phrase_times'],
                'n_beats': bg['n_beats'],
                'n_bars': bg['n_bars'],
                'n_phrases': bg['n_phrases'],
            }
            training_data.append(example)
            matched += 1

            if verbose:
                print(f"Matched: {track['name'][:40]} BPM={track_bpm:.1f} -> {bg['tempo_mean']:.1f}")
        else:
            unmatched += 1

    print(f"\nMatched {matched} tracks, {unmatched} unmatched")
    return training_data


def print_statistics(data: List[Dict]):
    """Print statistics about extracted data."""
    if not data:
        print("No data to analyze")
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
    parser = argparse.ArgumentParser(description='Extract all beat grid data from Rekordbox')
    parser.add_argument('--backup-dir', '-d',
                        default='data/rekordbox_bak_20251211',
                        help='Rekordbox backup directory')
    parser.add_argument('--output', '-o', default='data/beatgrid_all.json',
                        help='Output JSON path')
    parser.add_argument('--match-xml', '-x', help='XML file to match tracks')
    parser.add_argument('--bpm-tolerance', type=float, default=1.0,
                        help='BPM tolerance for matching')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    backup_dir = Path(args.backup_dir)
    if not backup_dir.exists():
        print(f"ERROR: Backup directory not found: {backup_dir}")
        return

    print(f"Scanning ANLZ files in {backup_dir}...")
    beatgrid_data = scan_all_anlz_files(backup_dir, args.verbose)

    if not beatgrid_data:
        print("No beat grid data extracted!")
        return

    # If XML provided, match and create training data
    if args.match_xml:
        print(f"\nLoading XML tracks from {args.match_xml}...")
        xml_tracks = load_xml_tracks(args.match_xml)
        print(f"Loaded {len(xml_tracks)} tracks from XML")

        # Filter to tracks with local files
        local_tracks = [t for t in xml_tracks if Path(t['file']).exists()]
        print(f"{len(local_tracks)} tracks have local files")

        print("\nMatching beat grid to tracks...")
        training_data = match_beatgrid_to_tracks(
            beatgrid_data,
            local_tracks,
            bpm_tolerance=args.bpm_tolerance,
            verbose=args.verbose
        )

        print_statistics(training_data)
        output_data = training_data
    else:
        # Just output raw beat grid data
        output_data = beatgrid_data

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
        json.dump(to_json_serializable(output_data), f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
