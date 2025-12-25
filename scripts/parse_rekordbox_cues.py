#!/usr/bin/env python3
"""
Parse Rekordbox XML to extract cue points as ground truth drops.

Usage:
    python scripts/parse_rekordbox_cues.py data/17-57.xml --output data/ground_truth_from_rekordbox.json
    python scripts/parse_rekordbox_cues.py data/17-57.xml --filter josh-baker --merge data/ground_truth_drops.json
"""

import xml.etree.ElementTree as ET
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote


def parse_rekordbox_xml(xml_path: str, filter_pattern: Optional[str] = None) -> Dict:
    """
    Parse Rekordbox XML and extract cue points.

    Args:
        xml_path: Path to Rekordbox XML file
        filter_pattern: Optional filter for track names (case-insensitive)

    Returns:
        Dict with track data and cue points
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = {}

    for track in root.findall('.//TRACK'):
        name = track.get('Name', '')
        location = track.get('Location', '')

        # Apply filter if specified
        if filter_pattern:
            if filter_pattern.lower() not in name.lower() and \
               filter_pattern.lower() not in location.lower():
                continue

        # Get cue points
        position_marks = track.findall('POSITION_MARK')
        if not position_marks:
            continue

        # Parse cue points
        cues = []
        green_cues = []
        memory_cues = []

        for pm in position_marks:
            start = float(pm.get('Start', 0))
            num = int(pm.get('Num', -1))
            cue_name = pm.get('Name', '')
            cue_type = pm.get('Type', '0')

            # Determine cue type based on Num and color
            is_memory = num == -1

            cue_data = {
                'time': start,
                'description': cue_name if cue_name else f"{'Memory' if is_memory else 'Hot'} cue {len(cues)}",
                'type': 'drop' if is_memory else 'cue',
                'num': num
            }
            cues.append(cue_data)

            if is_memory:
                memory_cues.append(start)
            else:
                green_cues.append(start)

        # Sort by time
        cues.sort(key=lambda x: x['time'])
        green_cues.sort()
        memory_cues.sort()

        # Create safe key from name
        safe_name = name.lower().replace(' ', '_').replace('-', '_').replace('&', 'and')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')

        # Decode URL path
        file_path = unquote(location).replace('file://localhost', '')

        results[safe_name] = {
            'file': file_path,
            'original_name': name,
            'duration_sec': int(track.get('TotalTime', 0)),
            'bpm': float(track.get('AverageBpm', 0)),
            'key': track.get('Tonality', ''),
            'drops': cues,
            'green_cues': green_cues,
            'memory_cues': memory_cues,
            'source': f'Rekordbox XML ({Path(xml_path).name})',
            'notes': f"{len(cues)} cue points: {len(green_cues)} hot cues + {len(memory_cues)} memory cues"
        }

    return results


def merge_ground_truth(existing_path: str, new_data: Dict, overwrite: bool = False) -> Dict:
    """
    Merge new ground truth data with existing file.

    Args:
        existing_path: Path to existing ground truth JSON
        new_data: New data to merge
        overwrite: If True, overwrite existing entries; if False, skip

    Returns:
        Merged data dict
    """
    with open(existing_path) as f:
        existing = json.load(f)

    for key, value in new_data.items():
        if key in existing and not overwrite:
            print(f"  Skipping {key} (already exists)")
            continue
        existing[key] = value
        print(f"  Added {key} ({len(value['drops'])} drops)")

    return existing


def main():
    parser = argparse.ArgumentParser(description='Parse Rekordbox XML for ground truth drops')
    parser.add_argument('xml_path', help='Path to Rekordbox XML file')
    parser.add_argument('--output', '-o', help='Output JSON path')
    parser.add_argument('--filter', '-f', help='Filter track names (e.g., "josh-baker")')
    parser.add_argument('--merge', '-m', help='Merge with existing ground truth file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing entries when merging')
    parser.add_argument('--list-only', action='store_true', help='Only list tracks with cues, do not save')

    args = parser.parse_args()

    print(f"Parsing {args.xml_path}...")
    data = parse_rekordbox_xml(args.xml_path, args.filter)

    print(f"\nFound {len(data)} tracks with cue points:")
    print("=" * 70)

    total_drops = 0
    for key, value in data.items():
        n_drops = len(value['drops'])
        total_drops += n_drops
        duration = value['duration_sec']
        duration_str = f"{duration//60}:{duration%60:02d}" if duration else "N/A"
        print(f"  {value['original_name'][:50]:<50} | {n_drops:>3} drops | {duration_str}")

    print("=" * 70)
    print(f"Total: {total_drops} drops across {len(data)} sets")

    if args.list_only:
        return

    # Save or merge
    if args.merge:
        print(f"\nMerging with {args.merge}...")
        data = merge_ground_truth(args.merge, data, args.overwrite)
        output_path = args.merge
    else:
        output_path = args.output or 'data/ground_truth_from_rekordbox.json'

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
