#!/usr/bin/env python3
"""
Serato Database Parser

Parses Serato DJ Pro database V2 binary file and extracts track metadata
including color labels. Exports to CSV with zone classification.

Usage:
    python scripts/parse_serato_db.py
    python scripts/parse_serato_db.py --db "path/to/database V2"
    python scripts/parse_serato_db.py --filter-folder 1901-1
"""

import argparse
import csv
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional


# Serato color to zone mapping
COLOR_TO_ZONE = {
    '00ffff99': 'YELLOW',      # Yellow
    '0099ff99': 'GREEN',       # Green
    '00ddff99': 'GREEN',       # Light green
    '00bbff99': 'GREEN',       # Green variant
    '00ff99ff': 'PURPLE',      # Pink/Purple
}

# Human-readable color names
COLOR_NAMES = {
    '00ffff99': 'Yellow',
    '0099ff99': 'Green',
    '00ddff99': 'Light Green',
    '00bbff99': 'Green 2',
    '00ff99ff': 'Pink/Purple',
    '00ffffff': 'White (no label)',
    '0099ffff': 'Cyan',
    '00ffdd99': 'Orange',
}


def parse_serato_database(filepath: str) -> List[Dict]:
    """
    Parse Serato database V2 binary file.

    The format uses 4-char tags followed by 4-byte big-endian length and data.
    Strings are UTF-16-BE encoded.

    Args:
        filepath: Path to the database V2 file

    Returns:
        List of track dictionaries
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    tracks = []
    pos = 0

    while True:
        # Find next track entry (otrk tag)
        pos = data.find(b'otrk', pos)
        if pos == -1:
            break

        # Read track block length
        track_len = struct.unpack('>I', data[pos+4:pos+8])[0]
        track_data = data[pos+8:pos+8+track_len]

        track_info = parse_track_block(track_data)

        if track_info.get('path'):
            tracks.append(track_info)

        pos += 1

    return tracks


def parse_track_block(track_data: bytes) -> Dict:
    """Parse a single track data block."""
    track_info = {}
    inner_pos = 0

    while inner_pos < len(track_data) - 8:
        # Read 4-char tag
        tag = track_data[inner_pos:inner_pos+4].decode('ascii', errors='ignore')

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

        # Parse known tags
        if tag == 'pfil':  # File path
            track_info['path'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag == 'tsng':  # Song title
            track_info['title'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag == 'tart':  # Artist
            track_info['artist'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag == 'talb':  # Album
            track_info['album'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag == 'tgen':  # Genre
            track_info['genre'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag == 'ulbl':  # User label (color!)
            track_info['color_hex'] = field_data.hex()
        elif tag == 'tbpm':  # BPM
            track_info['bpm'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag == 'tkey':  # Musical key
            track_info['key'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag == 'tlen':  # Track length
            track_info['length'] = field_data.decode('utf-16-be', errors='ignore')
        elif tag == 'tbit':  # Bitrate
            track_info['bitrate'] = field_data.decode('utf-16-be', errors='ignore')

        inner_pos += 8 + field_len

    return track_info


def classify_tracks(tracks: List[Dict]) -> List[Dict]:
    """Add zone classification based on color labels."""
    classified = []

    for track in tracks:
        color = track.get('color_hex', '')

        if color in COLOR_TO_ZONE:
            track['zone'] = COLOR_TO_ZONE[color]
            track['color_name'] = COLOR_NAMES.get(color, 'Unknown')
            classified.append(track)

    return classified


def extract_filename(path: str) -> str:
    """Extract filename from full path."""
    return Path(path).name if path else ''


def extract_folder(path: str, depth: int = 4) -> str:
    """Extract folder path up to specified depth."""
    parts = path.split('/')
    return '/'.join(parts[:depth]) if len(parts) >= depth else '/'.join(parts[:-1])


def export_to_csv(tracks: List[Dict], output_path: str):
    """Export tracks to CSV file."""
    fieldnames = [
        'file_path', 'filename', 'title', 'artist', 'album',
        'bpm', 'key', 'length', 'zone', 'serato_color', 'color_name'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for track in tracks:
            row = {
                'file_path': track.get('path', ''),
                'filename': extract_filename(track.get('path', '')),
                'title': track.get('title', ''),
                'artist': track.get('artist', ''),
                'album': track.get('album', ''),
                'bpm': track.get('bpm', ''),
                'key': track.get('key', ''),
                'length': track.get('length', ''),
                'zone': track.get('zone', ''),
                'serato_color': track.get('color_hex', ''),
                'color_name': track.get('color_name', ''),
            }
            writer.writerow(row)

    print(f"Exported {len(tracks)} tracks to {output_path}")


def print_statistics(tracks: List[Dict]):
    """Print statistics about the parsed tracks."""
    print("\n" + "="*60)
    print("SERATO DATABASE STATISTICS")
    print("="*60)

    # Zone counts
    zone_counts = {}
    for track in tracks:
        zone = track.get('zone', 'UNKNOWN')
        zone_counts[zone] = zone_counts.get(zone, 0) + 1

    print(f"\nTotal labeled tracks: {len(tracks)}")
    print("\nBy zone:")
    for zone, count in sorted(zone_counts.items()):
        emoji = {'YELLOW': '🟨', 'GREEN': '🟩', 'PURPLE': '🟪'}.get(zone, '❓')
        print(f"  {emoji} {zone}: {count}")

    # Folder breakdown
    folder_counts = {}
    for track in tracks:
        folder = extract_folder(track.get('path', ''))
        if folder not in folder_counts:
            folder_counts[folder] = {'total': 0, 'YELLOW': 0, 'GREEN': 0, 'PURPLE': 0}
        folder_counts[folder]['total'] += 1
        folder_counts[folder][track.get('zone', 'UNKNOWN')] += 1

    print("\nTop folders:")
    for folder, stats in sorted(folder_counts.items(), key=lambda x: -x[1]['total'])[:5]:
        print(f"  📁 {folder}: {stats['total']} tracks")
        print(f"     🟨 {stats['YELLOW']} | 🟩 {stats['GREEN']} | 🟪 {stats['PURPLE']}")


def main():
    parser = argparse.ArgumentParser(description='Parse Serato database and export to CSV')
    parser.add_argument('--db', type=str,
                       default='serato/_Serato_/database V2',
                       help='Path to Serato database V2 file')
    parser.add_argument('--output', type=str,
                       default='results/serato_labeled_tracks.csv',
                       help='Output CSV file path')
    parser.add_argument('--filter-folder', type=str,
                       help='Filter tracks by folder name (e.g., "1901-1")')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only print statistics, do not export')

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
    print(f"Found {len(all_tracks)} total tracks")

    # Classify tracks (filter by color)
    labeled_tracks = classify_tracks(all_tracks)
    print(f"Found {len(labeled_tracks)} labeled tracks (with target colors)")

    # Filter by folder if specified
    if args.filter_folder:
        filtered = [t for t in labeled_tracks if args.filter_folder in t.get('path', '')]
        print(f"Filtered to {len(filtered)} tracks in folder containing '{args.filter_folder}'")
        labeled_tracks = filtered

    # Print statistics
    print_statistics(labeled_tracks)

    # Export to CSV
    if not args.stats_only:
        output_path = project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_csv(labeled_tracks, str(output_path))

    return 0


if __name__ == '__main__':
    exit(main())
