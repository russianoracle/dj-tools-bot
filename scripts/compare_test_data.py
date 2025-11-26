#!/usr/bin/env python3
"""Compare test_data.txt files and create delta."""

import re
from pathlib import Path

def parse_test_data(filepath):
    """Parse test_data.txt TSV and return dict of path -> zone.

    Supports two formats:
    - Old: BPM | Key | Zone | Artist | Track Title | Location
    - New: # | My Tag | Artwork | BPM | Key | Artist | Track Title | Time | Bitrate | Date Added | Location | Album
    """
    tracks = {}

    with open(filepath, 'r', encoding='utf-16') as f:
        lines = f.readlines()

    if not lines:
        return tracks

    # Detect format from header
    header = lines[0].strip().split('\t')

    # Find Zone and Location column indices
    zone_idx = None
    location_idx = None

    for i, col in enumerate(header):
        col_lower = col.strip().lower()
        if col_lower in ['zone', 'my tag']:
            zone_idx = i
        elif col_lower == 'location':
            location_idx = i

    if zone_idx is None or location_idx is None:
        print(f"Warning: Could not detect columns in {filepath}")
        print(f"  Header: {header}")
        return tracks

    # Parse tracks
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) <= max(zone_idx, location_idx):
            continue

        zone = parts[zone_idx].strip().upper()
        path = parts[location_idx].strip().strip('"')  # Remove quotes if present

        if zone in ['YELLOW', 'GREEN', 'PURPLE'] and path:
            tracks[path] = zone

    return tracks


def main():
    old_file = 'tests/test_data.txt'
    new_file = 'tests/test_data_2.txt'
    delta_file = 'tests/test_data_delta.txt'

    old_tracks = parse_test_data(old_file)
    new_tracks = parse_test_data(new_file)

    print(f'Old file: {len(old_tracks)} labeled tracks')
    print(f'New file: {len(new_tracks)} labeled tracks')

    # Find differences
    new_entries = {}
    changed_entries = {}

    for path, zone in new_tracks.items():
        if path not in old_tracks:
            new_entries[path] = zone
        elif old_tracks[path] != zone:
            changed_entries[path] = {'old': old_tracks[path], 'new': zone}

    print(f'\nDelta:')
    print(f'  New tracks: {len(new_entries)}')
    print(f'  Changed zones: {len(changed_entries)}')

    # Create delta file
    with open(delta_file, 'w', encoding='utf-8') as f:
        f.write('# Delta: New and Changed Tracks\n')
        f.write(f'# Compared: {old_file} vs {new_file}\n')
        f.write(f'# New tracks: {len(new_entries)}\n')
        f.write(f'# Changed zones: {len(changed_entries)}\n\n')

        # New tracks by zone
        if new_entries:
            f.write('## NEW TRACKS\n\n')

            for zone in ['YELLOW', 'GREEN', 'PURPLE']:
                zone_tracks = [p for p, z in new_entries.items() if z == zone]
                if zone_tracks:
                    f.write(f'### {zone} ({len(zone_tracks)} tracks)\n')
                    for path in sorted(zone_tracks):
                        f.write(f'{path}\n')
                    f.write('\n')

        # Changed tracks
        if changed_entries:
            f.write('## CHANGED ZONES\n\n')
            for path, change in sorted(changed_entries.items()):
                f.write(f'# {change["old"]} -> {change["new"]}\n')
                f.write(f'{path}\n')

    print(f'\nDelta saved to: {delta_file}')

    # Summary by zone
    if new_entries:
        print(f'\nNew tracks by zone:')
        for zone in ['YELLOW', 'GREEN', 'PURPLE']:
            count = sum(1 for z in new_entries.values() if z == zone)
            if count:
                print(f'  {zone}: {count}')

    if changed_entries:
        print(f'\nChanged zones:')
        for path, change in list(changed_entries.items())[:10]:
            fname = path.split('/')[-1][:40]
            print(f'  {change["old"]} -> {change["new"]}: {fname}')
        if len(changed_entries) > 10:
            print(f'  ... and {len(changed_entries) - 10} more')


if __name__ == '__main__':
    main()
