#!/usr/bin/env python3
"""
Extract beat grid ground truth from Rekordbox ANLZ files.

Rekordbox stores beat grid in PQTZ tags within ANLZ*.DAT files.
Each entry contains:
- beat: Position in bar (1-4 for 4/4 time)
- tempo: BPM × 100 (e.g., 12675 = 126.75 BPM)
- time: Time in milliseconds

Usage:
    python scripts/extract_rekordbox_beatgrid.py --output data/rekordbox_beatgrid.json
    python scripts/extract_rekordbox_beatgrid.py --sample 100 --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrekordbox.anlz import AnlzFile


def find_anlz_files(rekordbox_path: str = None) -> List[Path]:
    """
    Find all ANLZ*.DAT files in Rekordbox share folder.

    Args:
        rekordbox_path: Path to Rekordbox folder (default: ~/Library/Pioneer/rekordbox)

    Returns:
        List of paths to ANLZ*.DAT files
    """
    if rekordbox_path is None:
        rekordbox_path = os.path.expanduser('~/Library/Pioneer/rekordbox')

    share_path = Path(rekordbox_path) / 'share' / 'PIONEER' / 'USBANLZ'

    if not share_path.exists():
        print(f"Warning: USBANLZ folder not found at {share_path}")
        return []

    anlz_files = list(share_path.rglob('ANLZ*.DAT'))
    return anlz_files


def parse_anlz_beatgrid(anlz_path: Path) -> Optional[Dict]:
    """
    Parse beat grid from ANLZ file.

    Returns:
        Dict with beat_entries, bpm_mean, downbeat_times, or None if failed
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

        # Extract beat data
        beats = []
        tempos = []
        downbeat_times = []  # Times of beat 1 (downbeats)

        for entry in entries:
            beat_num = entry.beat  # 1-4
            tempo = entry.tempo / 100.0  # Convert to BPM
            time_ms = entry.time
            time_sec = time_ms / 1000.0

            beats.append({
                'beat': beat_num,
                'time_sec': time_sec,
                'tempo': tempo
            })
            tempos.append(tempo)

            if beat_num == 1:
                downbeat_times.append(time_sec)

        # Calculate phrase boundaries (every 4 bars = 16 beats)
        phrase_times = []
        for i, db_time in enumerate(downbeat_times):
            if i % 4 == 0:  # Every 4 bars
                phrase_times.append(db_time)

        return {
            'n_beats': len(beats),
            'bpm_mean': sum(tempos) / len(tempos) if tempos else 0,
            'bpm_min': min(tempos) if tempos else 0,
            'bpm_max': max(tempos) if tempos else 0,
            'bpm_std': (sum((t - sum(tempos)/len(tempos))**2 for t in tempos) / len(tempos))**0.5 if len(tempos) > 1 else 0,
            'first_downbeat_sec': downbeat_times[0] if downbeat_times else 0,
            'n_downbeats': len(downbeat_times),
            'n_phrases': len(phrase_times),
            'downbeat_times': downbeat_times,
            'phrase_times': phrase_times,
            'beats': beats  # Full beat data
        }

    except Exception as e:
        return None


def match_anlz_to_tracks(
    anlz_files: List[Path],
    xml_path: str,
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Match ANLZ files to tracks from Rekordbox XML using path patterns.

    Rekordbox ANLZ files are organized by UUID, we match via XML track info.
    """
    # Parse XML for track metadata
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracks_by_path = {}
    for track in root.findall('.//TRACK'):
        location = track.get('Location', '')
        if not location:
            continue

        file_path = unquote(location).replace('file://localhost', '')
        tracks_by_path[file_path] = {
            'name': track.get('Name', ''),
            'artist': track.get('Artist', ''),
            'bpm_xml': float(track.get('AverageBpm', 0)),
            'key': track.get('Tonality', ''),
            'duration_sec': int(track.get('TotalTime', 0))
        }

    # For now, just process ANLZ files and store with UUID
    results = {}

    for anlz_path in anlz_files:
        # Extract UUID from path
        parts = anlz_path.parts
        uuid_idx = parts.index('USBANLZ') + 2 if 'USBANLZ' in parts else -1
        if uuid_idx > 0 and uuid_idx < len(parts):
            uuid = parts[uuid_idx - 1] + parts[uuid_idx]
        else:
            uuid = anlz_path.stem

        beatgrid = parse_anlz_beatgrid(anlz_path)
        if beatgrid:
            results[uuid] = {
                'anlz_path': str(anlz_path),
                **beatgrid
            }

            if verbose:
                print(f"Parsed {uuid}: {beatgrid['n_beats']} beats, "
                      f"BPM={beatgrid['bpm_mean']:.2f}")

    return results


def extract_training_data(
    beatgrid_data: Dict[str, Dict],
    xml_path: str,
    max_tracks: int = None
) -> List[Dict]:
    """
    Extract training data combining XML metadata with beat grid.

    Returns list of tracks with:
    - file path
    - BPM from Rekordbox (ground truth)
    - First downbeat position
    - Beat positions
    """
    # Parse XML for file paths
    tree = ET.parse(xml_path)
    root = tree.getroot()

    training_data = []

    for track in root.findall('.//TRACK'):
        location = track.get('Location', '')
        if not location:
            continue

        file_path = unquote(location).replace('file://localhost', '')

        # Check file exists
        if not Path(file_path).exists():
            continue

        bpm_xml = float(track.get('AverageBpm', 0))
        if bpm_xml <= 0:
            continue

        training_data.append({
            'file': file_path,
            'name': track.get('Name', ''),
            'artist': track.get('Artist', ''),
            'bpm': bpm_xml,
            'key': track.get('Tonality', ''),
            'duration_sec': int(track.get('TotalTime', 0))
        })

        if max_tracks and len(training_data) >= max_tracks:
            break

    return training_data


def main():
    parser = argparse.ArgumentParser(description='Extract Rekordbox beat grid')
    parser.add_argument('--xml', '-x', default='data/17-57.xml', help='Rekordbox XML path')
    parser.add_argument('--output', '-o', help='Output JSON path')
    parser.add_argument('--sample', '-n', type=int, help='Limit number of files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--training-data', '-t', action='store_true',
                        help='Output training data format (file + BPM)')

    args = parser.parse_args()

    print("Finding ANLZ files...")
    anlz_files = find_anlz_files()
    print(f"Found {len(anlz_files)} ANLZ files")

    if args.sample:
        anlz_files = anlz_files[:args.sample]

    print("Parsing beat grids...")
    beatgrid_data = match_anlz_to_tracks(anlz_files, args.xml, args.verbose)
    print(f"Parsed {len(beatgrid_data)} beat grids")

    # Statistics
    bpms = [d['bpm_mean'] for d in beatgrid_data.values()]
    if bpms:
        print(f"\nBPM Statistics:")
        print(f"  Mean: {sum(bpms)/len(bpms):.1f}")
        print(f"  Min: {min(bpms):.1f}")
        print(f"  Max: {max(bpms):.1f}")

    # Output
    if args.training_data:
        print("\nExtracting training data...")
        training_data = extract_training_data(beatgrid_data, args.xml, args.sample)
        output_data = training_data
        print(f"Extracted {len(training_data)} tracks for training")
    else:
        output_data = beatgrid_data

    output_path = args.output or 'data/rekordbox_beatgrid.json'
    print(f"\nSaving to {output_path}...")

    # Convert for JSON (remove non-serializable)
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(i) for i in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(clean_for_json(output_data), f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
