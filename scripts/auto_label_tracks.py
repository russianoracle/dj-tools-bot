#!/usr/bin/env python3
"""
Automatically label tracks in test_data.txt using BPM heuristics.
Then you can manually review and correct the labels.
"""

import csv
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.audio import AudioLoader, FeatureExtractor
from src.classification import EnergyZoneClassifier
from src.utils import get_config


def auto_label_from_analysis(input_file: str, output_file: str):
    """
    Analyze tracks and add Zone column based on full audio analysis.

    Args:
        input_file: Path to input TSV file (UTF-16)
        output_file: Path to output TSV file with Zone column
    """
    print(f"Reading {input_file}...")

    # Read existing data
    with open(input_file, 'r', encoding='utf-16') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Found {len(rows)} tracks")

    # Add Zone column if not present
    if 'Zone' not in fieldnames:
        fieldnames = list(fieldnames) + ['Zone']

    # Initialize classifier
    config = get_config()
    audio_loader = AudioLoader(config.get('audio.sample_rate', 22050))
    feature_extractor = FeatureExtractor(config)
    classifier = EnergyZoneClassifier(config)

    # Analyze each track
    for i, row in enumerate(rows, 1):
        file_path = row.get('Location', '').strip()

        if not file_path:
            print(f"  [{i}/{len(rows)}] Skipping - no path")
            continue

        if not Path(file_path).exists():
            print(f"  [{i}/{len(rows)}] Skipping - file not found: {Path(file_path).name}")
            continue

        # Skip if already labeled
        if row.get('Zone', '').strip():
            print(f"  [{i}/{len(rows)}] Already labeled: {Path(file_path).name}")
            continue

        try:
            print(f"  [{i}/{len(rows)}] Analyzing: {Path(file_path).name}...")

            # Load and analyze
            y, sr = audio_loader.load(file_path)
            features = feature_extractor.extract(y, sr)
            result = classifier.classify(features)

            # Add zone
            row['Zone'] = result.zone.value

            print(f"      → {result.zone.emoji} {result.zone.value} (confidence: {result.confidence:.1%}, BPM: {features.tempo:.0f})")

        except Exception as e:
            print(f"      ✗ Error: {e}")
            continue

    # Write output
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-16', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    zones_count = {}
    for row in rows:
        zone = row.get('Zone', '').strip()
        if zone:
            zones_count[zone] = zones_count.get(zone, 0) + 1

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for zone, count in sorted(zones_count.items()):
        print(f"  {zone}: {count} tracks")
    print(f"\nTotal labeled: {sum(zones_count.values())}/{len(rows)}")
    print(f"\nOutput saved to: {output_file}")
    print("\n⚠️  IMPORTANT: Manually review and correct labels before training!")


def auto_label_from_bpm(input_file: str, output_file: str):
    """
    Add Zone column based on BPM heuristics (fast, no audio analysis).

    Simple rules:
    - BPM < 110 → yellow
    - BPM 110-128 → green
    - BPM > 128 → purple

    Args:
        input_file: Path to input TSV file (UTF-16)
        output_file: Path to output TSV file with Zone column
    """
    print(f"Reading {input_file}...")

    # Read existing data
    with open(input_file, 'r', encoding='utf-16') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Found {len(rows)} tracks")

    # Add Zone column if not present
    if 'Zone' not in fieldnames:
        fieldnames = list(fieldnames) + ['Zone']

    # Label based on BPM
    for i, row in enumerate(rows, 1):
        bpm_str = row.get('BPM', '').strip()

        if not bpm_str:
            print(f"  [{i}/{len(rows)}] Skipping - no BPM")
            continue

        try:
            bpm = float(bpm_str)

            # Simple heuristics
            if bpm < 110:
                zone = 'yellow'
            elif bpm <= 128:
                zone = 'green'
            else:
                zone = 'purple'

            row['Zone'] = zone

            track = row.get('Track Title', 'Unknown')
            print(f"  [{i}/{len(rows)}] {track[:40]:40} → {zone:8} (BPM: {bpm:.0f})")

        except ValueError:
            print(f"  [{i}/{len(rows)}] Skipping - invalid BPM: {bpm_str}")
            continue

    # Write output
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-16', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    zones_count = {}
    for row in rows:
        zone = row.get('Zone', '').strip()
        if zone:
            zones_count[zone] = zones_count.get(zone, 0) + 1

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for zone, count in sorted(zones_count.items()):
        emoji = {'yellow': '🟨', 'green': '🟩', 'purple': '🟪'}.get(zone, '')
        print(f"  {emoji} {zone}: {count} tracks")
    print(f"\nTotal labeled: {sum(zones_count.values())}/{len(rows)}")
    print(f"\nOutput saved to: {output_file}")
    print("\n⚠️  IMPORTANT: These are BPM-based heuristics!")
    print("   Manually review and correct labels before training.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Auto-label tracks for zone classification training')
    parser.add_argument('input', help='Input TSV file (UTF-16)')
    parser.add_argument('output', help='Output TSV file with Zone column')
    parser.add_argument('--method', choices=['bpm', 'analysis'], default='bpm',
                      help='Labeling method: bpm (fast, heuristic) or analysis (slow, accurate)')

    args = parser.parse_args()

    if args.method == 'bpm':
        auto_label_from_bpm(args.input, args.output)
    else:
        auto_label_from_analysis(args.input, args.output)
