#!/usr/bin/env python3
"""Quick script to add Zone column based on BPM."""

import csv

input_file = 'tests/test_data.txt'
output_file = 'tests/test_data_labeled.txt'

print(f"Reading {input_file}...")

# Read data
with open(input_file, 'r', encoding='utf-16') as f:
    reader = csv.DictReader(f, delimiter='\t')
    fieldnames = list(reader.fieldnames) + ['Zone']
    rows = list(reader)

print(f"Found {len(rows)} tracks")

# Add zones based on BPM
zones_count = {'yellow': 0, 'green': 0, 'purple': 0}

for i, row in enumerate(rows, 1):
    bpm_str = row.get('BPM', '').strip()

    if not bpm_str:
        continue

    try:
        bpm = float(bpm_str)

        # Zone rules
        if bpm < 110:
            zone = 'yellow'
        elif bpm <= 128:
            zone = 'green'
        else:
            zone = 'purple'

        row['Zone'] = zone
        zones_count[zone] += 1

        track = row.get('Track Title', 'Unknown')[:40]
        print(f"  [{i}/{len(rows)}] {track:40} → {zone:8} (BPM: {bpm:.0f})")

    except ValueError:
        print(f"  [{i}/{len(rows)}] Skipping - invalid BPM: {bpm_str}")

# Write output
print(f"\nWriting to {output_file}...")
with open(output_file, 'w', encoding='utf-16', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  🟨 yellow: {zones_count['yellow']} tracks")
print(f"  🟩 green:  {zones_count['green']} tracks")
print(f"  🟪 purple: {zones_count['purple']} tracks")
print(f"\nTotal: {sum(zones_count.values())}/{len(rows)}")
print(f"\nSaved to: {output_file}")
print("\n⚠️  Now use this file for training:")
print(f"   In Training Window, select: {output_file}")
