#!/usr/bin/env python3
"""
Convert labels.json to CSV format for feature extraction.

Usage:
    python scripts/convert_labels_to_csv.py
"""

import json
import csv
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent

    # Load labels.json
    labels_path = project_root / 'data' / 'labels.json'
    output_path = project_root / 'results' / 'user_tracks.csv'

    print(f"Loading labels from: {labels_path}")

    with open(labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels = data.get('labels', {})
    print(f"Found {len(labels)} labeled tracks")

    # Filter tracks that exist on filesystem
    existing = []
    missing = 0

    for path, info in labels.items():
        if Path(path).exists():
            existing.append({
                'path': path,
                'zone': info['zone']
            })
        else:
            missing += 1

    print(f"Existing files: {len(existing)}")
    print(f"Missing files: {missing}")

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'zone'])

        for track in existing:
            writer.writerow([track['path'], track['zone']])

    print(f"\nExported {len(existing)} tracks to: {output_path}")

    # Print zone distribution
    zones = {}
    for track in existing:
        zone = track['zone']
        zones[zone] = zones.get(zone, 0) + 1

    print("\nZone distribution:")
    for zone in ['YELLOW', 'GREEN', 'PURPLE']:
        count = zones.get(zone, 0)
        pct = 100 * count / len(existing) if existing else 0
        print(f"  {zone}: {count} ({pct:.1f}%)")

    return 0


if __name__ == '__main__':
    exit(main())
