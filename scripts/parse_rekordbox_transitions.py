#!/usr/bin/env python3
"""
Parse Rekordbox XML to extract transition ground truth.

Cue point color convention:
- GREEN (RGB ~40,226,20) = mixin point (new track starts entering)
- RED (RGB ~232,20,20) = mixout point (old track finishes exiting)

For DJ sets, user marks mixin/mixout in Rekordbox, this script extracts them.

Usage:
    python scripts/parse_rekordbox_transitions.py data/17-57.xml --output data/ground_truth_transitions.json
    python scripts/parse_rekordbox_transitions.py data/17-57.xml --filter josh-baker --verbose
    python scripts/parse_rekordbox_transitions.py data/17-57.xml --list-only
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Color thresholds for cue point classification
# Rekordbox uses specific colors, we match approximately
CUE_COLORS = {
    'green': {'r': (20, 80), 'g': (180, 255), 'b': (0, 80)},   # Mixin
    'red': {'r': (180, 255), 'g': (0, 80), 'b': (0, 80)},       # Mixout
    'yellow': {'r': (200, 255), 'g': (180, 255), 'b': (0, 80)}, # Other
    'blue': {'r': (0, 80), 'g': (80, 180), 'b': (180, 255)},    # Other
    'pink': {'r': (200, 255), 'g': (0, 120), 'b': (180, 255)},  # Other
}


def classify_cue_color(r: int, g: int, b: int) -> str:
    """
    Classify cue point color.

    Args:
        r, g, b: RGB values (0-255)

    Returns:
        Color name: 'green', 'red', 'yellow', 'blue', 'pink', or 'unknown'
    """
    for color_name, ranges in CUE_COLORS.items():
        r_match = ranges['r'][0] <= r <= ranges['r'][1]
        g_match = ranges['g'][0] <= g <= ranges['g'][1]
        b_match = ranges['b'][0] <= b <= ranges['b'][1]
        if r_match and g_match and b_match:
            return color_name
    return 'unknown'


def parse_rekordbox_transitions(
    xml_path: str,
    filter_pattern: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """
    Parse Rekordbox XML and extract transition cue points.

    Green cues → mixin points
    Red cues → mixout points

    Args:
        xml_path: Path to Rekordbox XML
        filter_pattern: Optional filter for track names (case-insensitive)
        verbose: Print debug info

    Returns:
        Dict with transition ground truth per track
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

        # Parse cue points by color
        mixin_times = []
        mixout_times = []
        other_cues = []

        for pm in position_marks:
            start = float(pm.get('Start', 0))
            r = int(pm.get('Red', 0))
            g = int(pm.get('Green', 0))
            b = int(pm.get('Blue', 0))
            cue_name = pm.get('Name', '')
            num = int(pm.get('Num', -1))

            color = classify_cue_color(r, g, b)

            cue_data = {
                'time': start,
                'color': color,
                'rgb': (r, g, b),
                'name': cue_name,
                'num': num
            }

            if color == 'green':
                mixin_times.append(start)
            elif color == 'red':
                mixout_times.append(start)
            else:
                other_cues.append(cue_data)

            if verbose:
                print(f"  Cue at {start:.1f}s: {color} (R={r}, G={g}, B={b})")

        # Sort times
        mixin_times.sort()
        mixout_times.sort()

        # Only include tracks with transition markers
        if not mixin_times and not mixout_times:
            continue

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
            'mixin_times': mixin_times,
            'mixout_times': mixout_times,
            'n_transitions': max(len(mixin_times), len(mixout_times)),
            'source': f'Rekordbox XML ({Path(xml_path).name})',
            'notes': f"{len(mixin_times)} mixin (green) + {len(mixout_times)} mixout (red) cues"
        }

        if verbose:
            print(f"\n{name}:")
            print(f"  Mixin times: {mixin_times}")
            print(f"  Mixout times: {mixout_times}")

    return results


def convert_to_training_format(results: Dict) -> Dict:
    """
    Convert parsed results to TrainingPipeline-compatible format.

    Format expected by TransitionGroundTruthLoader:
    {
        'set_name': {
            'file': str,
            'labels': List[float],  # mixin times
            'metadata': {'bpm': float, 'mixout_times': List[float], ...}
        }
    }
    """
    training_format = {}

    for key, data in results.items():
        training_format[key] = {
            'file': data['file'],
            'labels': data['mixin_times'],  # Primary labels = mixin times
            'metadata': {
                'bpm': data['bpm'],
                'key': data['key'],
                'duration_sec': data['duration_sec'],
                'mixout_times': data['mixout_times'],
                'original_name': data['original_name']
            }
        }

    return training_format


def pair_transitions(mixin_times: List[float], mixout_times: List[float]) -> List[Tuple[float, float]]:
    """
    Pair mixin and mixout times into transition zones.

    Simple heuristic: each mixin pairs with the next mixout after it.

    Args:
        mixin_times: Sorted list of mixin times
        mixout_times: Sorted list of mixout times

    Returns:
        List of (mixin_time, mixout_time) tuples
    """
    pairs = []
    mixout_idx = 0

    for mixin_t in mixin_times:
        # Find next mixout after this mixin
        while mixout_idx < len(mixout_times) and mixout_times[mixout_idx] <= mixin_t:
            mixout_idx += 1

        if mixout_idx < len(mixout_times):
            mixout_t = mixout_times[mixout_idx]
            # Sanity check: transition should be < 3 minutes
            if 10.0 < (mixout_t - mixin_t) < 180.0:
                pairs.append((mixin_t, mixout_t))
                mixout_idx += 1

    return pairs


def print_summary(results: Dict):
    """Print summary of parsed transitions."""
    print("\n" + "=" * 70)
    print("REKORDBOX TRANSITION PARSING SUMMARY")
    print("=" * 70)

    total_mixin = 0
    total_mixout = 0

    for key, data in results.items():
        n_mixin = len(data['mixin_times'])
        n_mixout = len(data['mixout_times'])
        total_mixin += n_mixin
        total_mixout += n_mixout

        duration = data['duration_sec']
        duration_str = f"{duration//60}:{duration%60:02d}" if duration else "N/A"

        pairs = pair_transitions(data['mixin_times'], data['mixout_times'])

        print(f"\n{data['original_name'][:50]}")
        print(f"  Duration: {duration_str} | BPM: {data['bpm']:.1f}")
        print(f"  Green (mixin): {n_mixin} | Red (mixout): {n_mixout} | Paired: {len(pairs)}")

        if pairs:
            for i, (mi, mo) in enumerate(pairs[:3]):
                dur = mo - mi
                print(f"    Transition {i+1}: {mi:.1f}s → {mo:.1f}s ({dur:.1f}s)")
            if len(pairs) > 3:
                print(f"    ... and {len(pairs) - 3} more")

    print("\n" + "-" * 70)
    print(f"TOTAL: {total_mixin} mixin + {total_mixout} mixout across {len(results)} tracks")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Parse Rekordbox XML for transition ground truth')
    parser.add_argument('xml_path', help='Path to Rekordbox XML file')
    parser.add_argument('--output', '-o', help='Output JSON path')
    parser.add_argument('--filter', '-f', help='Filter track names (e.g., "josh-baker")')
    parser.add_argument('--list-only', action='store_true', help='Only list tracks, do not save')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--training-format', '-t', action='store_true',
                        help='Output in TrainingPipeline format')

    args = parser.parse_args()

    print(f"Parsing {args.xml_path}...")
    results = parse_rekordbox_transitions(args.xml_path, args.filter, args.verbose)

    print_summary(results)

    if args.list_only:
        return

    # Convert to training format if requested
    if args.training_format:
        output_data = convert_to_training_format(results)
    else:
        output_data = results

    # Save
    output_path = args.output or 'data/ground_truth_transitions.json'
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
