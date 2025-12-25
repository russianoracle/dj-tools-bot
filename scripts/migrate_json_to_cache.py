#!/usr/bin/env python3
"""
Migrate existing JSON analysis results to the new CacheManager architecture.

Converts results/nk_set_analysis.json format to SetAnalysisResult format
and saves to predictions.db via CacheManager.

Usage:
    python scripts/migrate_json_to_cache.py results/nk_set_analysis.json
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pipelines import CacheManager


def convert_json_to_set_analysis_format(json_data: dict) -> list[tuple[str, dict]]:
    """
    Convert JSON analysis format to SetAnalysisResult.to_dict() format.

    Returns:
        List of (file_path, result_dict) tuples
    """
    input_dir = json_data.get('input_dir', '')
    results = []

    for set_data in json_data.get('sets', []):
        file_name = set_data['file_name']
        file_path = os.path.abspath(os.path.join(input_dir, file_name))

        # Convert minutes to seconds
        duration_sec = set_data['duration_min'] * 60

        # Convert segments
        segments = []
        for i, seg in enumerate(set_data.get('segments', [])):
            segment_dict = {
                'start_time': seg['start_min'] * 60,
                'end_time': seg['end_min'] * 60,
                'duration': seg['duration_min'] * 60,
                'segment_index': i,
                'zone': None,
                'features': None,
                'genre': None,  # No genre analysis was done
            }
            segments.append(segment_dict)

        # Build SetAnalysisResult-compatible dict
        result_dict = {
            'file_path': file_path,
            'file_name': file_name,
            'duration_sec': duration_sec,
            'n_transitions': set_data.get('n_transitions', 0),
            'transition_times': [],  # Not available in old format
            'transition_density': set_data.get('transition_density', 0.0),
            'n_segments': set_data.get('n_segments', len(segments)),
            'segments': segments,
            'total_drops': set_data.get('total_drops', 0),
            'drop_density': set_data.get('drop_density', 0.0),
            'genre_distribution': None,
            'processing_time_sec': set_data.get('processing_time_sec', 0.0),
            'success': True,
            'error': None,
        }

        results.append((file_path, result_dict))

    return results


def migrate_json_to_cache(json_path: str, cache_dir: str = "~/.mood-classifier/cache"):
    """
    Migrate JSON results to CacheManager.

    Args:
        json_path: Path to JSON file with analysis results
        cache_dir: Cache directory for CacheManager
    """
    # Read JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    print(f"Loaded: {json_path}")
    print(f"  Analysis date: {json_data.get('analysis_date', 'unknown')}")
    print(f"  Input dir: {json_data.get('input_dir', 'unknown')}")
    print(f"  Sets: {len(json_data.get('sets', []))}")

    # Convert format
    results = convert_json_to_set_analysis_format(json_data)

    # Initialize CacheManager
    cache_manager = CacheManager(cache_dir)

    # Save each result
    migrated = 0
    skipped = 0
    errors = 0

    for file_path, result_dict in results:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"  SKIP (file not found): {Path(file_path).name}")
                skipped += 1
                continue

            # Check if already cached
            existing = cache_manager.get_set_analysis(file_path)
            if existing:
                print(f"  SKIP (already cached): {Path(file_path).name}")
                skipped += 1
                continue

            # Save to cache
            cache_manager.save_set_analysis(file_path, result_dict)

            duration_min = result_dict['duration_sec'] / 60
            n_segments = result_dict['n_segments']
            print(f"  OK: {Path(file_path).name} ({duration_min:.1f}min, {n_segments} segments)")
            migrated += 1

        except Exception as e:
            print(f"  ERROR: {Path(file_path).name} - {e}")
            errors += 1

    print(f"\nMigration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

    # Show cache stats
    stats = cache_manager.get_set_analysis_stats()
    print(f"\nCache stats:")
    print(f"  Total cached sets: {stats['count']}")

    return migrated, skipped, errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_json_to_cache.py <json_file> [cache_dir]")
        print("\nExample:")
        print("  python scripts/migrate_json_to_cache.py results/nk_set_analysis.json")
        sys.exit(1)

    json_path = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "~/.mood-classifier/cache"

    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    migrate_json_to_cache(json_path, cache_dir)


if __name__ == '__main__':
    main()
