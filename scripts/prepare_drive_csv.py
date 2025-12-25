#!/usr/bin/env python3
"""
Prepare CSV files from Serato database for feature extraction.

Parses Serato DJ Pro database V2 and exports labeled tracks to CSV format
compatible with extract_unified_features.py.

Usage:
    python scripts/prepare_drive_csv.py
    python scripts/prepare_drive_csv.py --output results/my_tracks.csv
    python scripts/prepare_drive_csv.py --filter _drive --remap-path
"""

import argparse
import csv
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Serato color to zone mapping
COLOR_TO_ZONE = {
    '00ffff99': 'YELLOW',      # Yellow
    '0099ff99': 'GREEN',       # Green
    '00ddff99': 'GREEN',       # Light green
    '00bbff99': 'GREEN',       # Green variant
    '00ff99ff': 'PURPLE',      # Pink/Purple
}


def parse_serato_database(filepath: str) -> List[Dict]:
    """
    Parse Serato database V2 binary file.

    Args:
        filepath: Path to the database V2 file

    Returns:
        List of track dictionaries with path, zone, color_hex, bpm
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    tracks = []
    pos = 0

    while True:
        pos = data.find(b'otrk', pos)
        if pos == -1:
            break

        track_len = struct.unpack('>I', data[pos+4:pos+8])[0]
        track_data = data[pos+8:pos+8+track_len]

        track_info = parse_track_block(track_data)

        if track_info.get('path'):
            # Add zone based on color
            color = track_info.get('color_hex', '')
            if color in COLOR_TO_ZONE:
                track_info['zone'] = COLOR_TO_ZONE[color]
                tracks.append(track_info)

        pos += 1

    return tracks


def parse_track_block(track_data: bytes) -> Dict:
    """Parse a single track data block."""
    track_info = {}
    inner_pos = 0

    while inner_pos < len(track_data) - 8:
        tag = track_data[inner_pos:inner_pos+4]

        if not tag.isalpha():
            inner_pos += 1
            continue

        try:
            field_len = struct.unpack('>I', track_data[inner_pos+4:inner_pos+8])[0]
        except struct.error:
            inner_pos += 1
            continue

        if field_len > len(track_data) - inner_pos - 8:
            inner_pos += 1
            continue

        field_data = track_data[inner_pos+8:inner_pos+8+field_len]
        tag_str = tag.decode('ascii', errors='ignore')

        if tag_str == 'pfil':  # File path
            track_info['path'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag_str == 'ulbl':  # User label (color)
            track_info['color_hex'] = field_data.hex()
        elif tag_str == 'tbpm':  # BPM
            track_info['bpm'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag_str == 'tsng':  # Song title
            track_info['title'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag_str == 'tart':  # Artist
            track_info['artist'] = field_data.decode('utf-16-be', errors='ignore')

        inner_pos += 8 + field_len

    return track_info


def remap_path(path: str, old_prefix: str, new_prefix: str) -> str:
    """
    Remap path from Serato DB to local filesystem.

    Args:
        path: Original path from Serato
        old_prefix: Prefix to replace (e.g., 'Users/r/Music')
        new_prefix: New prefix (e.g., '/Users/artemgusarov/Yandex.Disk.localized/Music (1)')

    Returns:
        Remapped path
    """
    # Remove leading slash if present
    path = path.lstrip('/')

    if path.startswith(old_prefix.lstrip('/')):
        remaining = path[len(old_prefix.lstrip('/')):]
        return new_prefix + remaining

    return '/' + path


def scan_local_files(directory: str) -> Dict[str, str]:
    """
    Scan local directory and build filename -> full_path mapping.

    Args:
        directory: Directory to scan

    Returns:
        Dictionary mapping lowercase filename to full path
    """
    file_map = {}
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Warning: Directory not found: {directory}")
        return file_map

    extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.mp4', '*.aif', '*.aiff']
    for ext in extensions:
        for file_path in dir_path.glob(ext):
            # Use lowercase filename as key for case-insensitive matching
            file_map[file_path.name.lower()] = str(file_path)

    print(f"Scanned {len(file_map)} audio files in {directory}")
    return file_map


def match_tracks_to_local(
    tracks: List[Dict],
    local_files: Dict[str, str]
) -> Tuple[List[Dict], int, int]:
    """
    Match Serato tracks to local files by filename.

    Args:
        tracks: List of track dictionaries from Serato
        local_files: Dictionary mapping filename to full path

    Returns:
        Tuple of (matched_tracks, matched_count, unmatched_count)
    """
    matched = []
    unmatched = 0

    for track in tracks:
        serato_path = track.get('path', '')
        filename = Path(serato_path).name.lower()

        if filename in local_files:
            track_copy = track.copy()
            track_copy['local_path'] = local_files[filename]
            matched.append(track_copy)
        else:
            unmatched += 1

    return matched, len(matched), unmatched


def export_to_csv(
    tracks: List[Dict],
    output_path: str,
    use_local_path: bool = False,
    remap: bool = False,
    old_prefix: str = 'Users/r/Music',
    new_prefix: str = '/Users/artemgusarov/Yandex.Disk.localized/Music (1)'
):
    """
    Export tracks to CSV file.

    Args:
        tracks: List of track dictionaries
        output_path: Output CSV path
        use_local_path: Use matched local_path instead of Serato path
        remap: Whether to remap paths (ignored if use_local_path=True)
        old_prefix: Old path prefix to replace
        new_prefix: New path prefix
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'zone'])

        for track in tracks:
            if use_local_path and 'local_path' in track:
                path = track['local_path']
            else:
                path = track['path']
                if remap:
                    path = remap_path(path, old_prefix, new_prefix)

            writer.writerow([path, track['zone']])

    print(f"Exported {len(tracks)} tracks to {output_path}")


def print_statistics(tracks: List[Dict], folder_filter: Optional[str] = None):
    """Print statistics about parsed tracks."""
    print("\n" + "="*60)
    print("SERATO DATABASE STATISTICS")
    print("="*60)

    # Filter if specified
    if folder_filter:
        tracks = [t for t in tracks if folder_filter in t.get('path', '')]
        print(f"\nFiltered to tracks containing '{folder_filter}'")

    # Zone counts
    zone_counts = {}
    for track in tracks:
        zone = track.get('zone', 'UNKNOWN')
        zone_counts[zone] = zone_counts.get(zone, 0) + 1

    print(f"\nTotal labeled tracks: {len(tracks)}")
    print("\nBy zone:")
    for zone, count in sorted(zone_counts.items()):
        emoji = {'YELLOW': 'Y', 'GREEN': 'G', 'PURPLE': 'P'}.get(zone, '?')
        pct = 100 * count / len(tracks) if tracks else 0
        print(f"  [{emoji}] {zone}: {count} ({pct:.1f}%)")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare CSV from Serato database for feature extraction'
    )
    parser.add_argument(
        '--db', type=str,
        default='serato/_Serato_/database V2',
        help='Path to Serato database V2 file'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default='results/serato_drive_tracks.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--filter', '-f', type=str,
        default='_drive',
        help='Filter tracks by path substring (default: _drive)'
    )
    parser.add_argument(
        '--local-dir', '-l', type=str,
        help='Local directory to match files by filename'
    )
    parser.add_argument(
        '--remap-path', action='store_true',
        help='Remap paths from Serato to local filesystem'
    )
    parser.add_argument(
        '--old-prefix', type=str,
        default='Users/r/Music',
        help='Old path prefix to replace'
    )
    parser.add_argument(
        '--new-prefix', type=str,
        default='/Users/artemgusarov/Yandex.Disk.localized/Music (1)',
        help='New path prefix'
    )
    parser.add_argument(
        '--stats-only', action='store_true',
        help='Only print statistics, do not export'
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent
    db_path = project_root / args.db

    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        return 1

    print(f"Parsing Serato database: {db_path}")

    # Parse database
    all_tracks = parse_serato_database(str(db_path))
    print(f"Found {len(all_tracks)} labeled tracks total")

    # Filter by path
    if args.filter:
        filtered = [t for t in all_tracks if args.filter in t.get('path', '')]
        print(f"Filtered to {len(filtered)} tracks containing '{args.filter}'")
        all_tracks = filtered

    # Match to local files if directory specified
    use_local_path = False
    if args.local_dir:
        local_files = scan_local_files(args.local_dir)
        all_tracks, matched, unmatched = match_tracks_to_local(all_tracks, local_files)
        print(f"Matched {matched} tracks, {unmatched} not found locally")
        use_local_path = True

    # Print statistics
    print_statistics(all_tracks)

    # Export to CSV
    if not args.stats_only:
        output_path = project_root / args.output
        export_to_csv(
            all_tracks,
            str(output_path),
            use_local_path=use_local_path,
            remap=args.remap_path,
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix
        )

    return 0


if __name__ == '__main__':
    exit(main())
