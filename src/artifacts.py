#!/usr/bin/env python3
"""
Artifact Management System

Provides safe loading of datasets with validation against registry.
Prevents accidental use of wrong datasets.

Usage:
    from src.artifacts import load_dataset, get_active_dataset, list_datasets

    # Load active dataset
    df = load_dataset()

    # Load specific dataset
    df = load_dataset("user_ultimate_v1")

    # Get info about active dataset
    info = get_active_dataset()
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any
from collections import Counter
import sys

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = PROJECT_ROOT / "config" / "artifacts.yaml"


class ArtifactError(Exception):
    """Raised when artifact validation fails."""
    pass


def load_registry() -> Dict:
    """Load artifact registry from YAML."""
    if not REGISTRY_PATH.exists():
        raise ArtifactError(f"Registry not found: {REGISTRY_PATH}")

    with open(REGISTRY_PATH, 'r') as f:
        return yaml.safe_load(f)


def get_active_dataset() -> Dict:
    """Get info about currently active dataset."""
    registry = load_registry()
    active_name = registry.get('active_dataset')

    if not active_name:
        raise ArtifactError("No active_dataset defined in registry")

    datasets = registry.get('datasets', {})
    if active_name not in datasets:
        raise ArtifactError(f"Active dataset '{active_name}' not found in registry")

    info = datasets[active_name].copy()
    info['name'] = active_name
    return info


def list_datasets() -> Dict[str, Dict]:
    """List all registered datasets with their status."""
    registry = load_registry()
    active = registry.get('active_dataset', '')

    result = {}
    for name, info in registry.get('datasets', {}).items():
        result[name] = {
            'status': info.get('status', 'unknown'),
            'description': info.get('description', ''),
            'tracks': info.get('expected_tracks', '?'),
            'is_active': name == active,
        }
    return result


def print_dataset_summary(name: str, info: Dict, df: pd.DataFrame = None):
    """Print clear summary of dataset being loaded."""
    print("\n" + "=" * 60)
    print(f"LOADING DATASET: {name}")
    print("=" * 60)
    print(f"Description: {info.get('description', 'N/A')}")
    print(f"Status: {info.get('status', 'unknown').upper()}")
    print(f"Source CSV: {info.get('source_csv', 'N/A')}")
    print(f"Extractor: {info.get('extractor_script', 'N/A')}")
    print(f"Extraction Date: {info.get('extraction_date', 'N/A')}")

    expected_zones = info.get('zones', {})
    print(f"\nExpected: {info.get('expected_tracks', '?')} tracks, {info.get('expected_features', '?')} features")
    print(f"Zones: Y:{expected_zones.get('YELLOW', '?')} G:{expected_zones.get('GREEN', '?')} P:{expected_zones.get('PURPLE', '?')}")

    if df is not None:
        actual_tracks = df['track_id'].nunique() if 'track_id' in df.columns else len(df)
        actual_zones = Counter(df.groupby('track_id')['zone'].first()) if 'track_id' in df.columns else Counter(df['zone'])
        print(f"\nActual: {actual_tracks} tracks, {len(df.columns)} columns")
        print(f"Zones: Y:{actual_zones.get('YELLOW', 0)} G:{actual_zones.get('GREEN', 0)} P:{actual_zones.get('PURPLE', 0)}")

    if info.get('notes'):
        print(f"\nNotes: {info['notes']}")

    print("=" * 60 + "\n")


def validate_dataset(df: pd.DataFrame, info: Dict) -> bool:
    """Validate loaded dataset against registry expectations."""
    errors = []

    # Check track count
    expected_tracks = info.get('expected_tracks')
    if expected_tracks:
        actual_tracks = df['track_id'].nunique() if 'track_id' in df.columns else len(df)
        if abs(actual_tracks - expected_tracks) > expected_tracks * 0.1:  # 10% tolerance
            errors.append(f"Track count mismatch: expected ~{expected_tracks}, got {actual_tracks}")

    # Check zone distribution
    expected_zones = info.get('zones', {})
    if expected_zones and 'zone' in df.columns:
        if 'track_id' in df.columns:
            actual_zones = Counter(df.groupby('track_id')['zone'].first())
        else:
            actual_zones = Counter(df['zone'])

        for zone, expected in expected_zones.items():
            actual = actual_zones.get(zone, 0)
            if abs(actual - expected) > expected * 0.15:  # 15% tolerance
                errors.append(f"Zone {zone} mismatch: expected ~{expected}, got {actual}")

    if errors:
        print("\nVALIDATION WARNINGS:")
        for e in errors:
            print(f"  - {e}")
        return False

    return True


def load_dataset(name: str = None, artifact_type: str = 'frames',
                 validate: bool = True, quiet: bool = False) -> pd.DataFrame:
    """
    Load dataset from registry with validation.

    Args:
        name: Dataset name from registry. If None, uses active_dataset.
        artifact_type: 'frames', 'spectrograms', or 'track_features'
        validate: Whether to validate against expected values
        quiet: Suppress summary output

    Returns:
        Loaded DataFrame

    Raises:
        ArtifactError: If dataset not found or validation fails
    """
    registry = load_registry()

    # Resolve dataset name
    if name is None:
        name = registry.get('active_dataset')
        if not name:
            raise ArtifactError("No dataset specified and no active_dataset in registry")

    datasets = registry.get('datasets', {})
    if name not in datasets:
        available = list(datasets.keys())
        raise ArtifactError(f"Dataset '{name}' not found. Available: {available}")

    info = datasets[name]

    # Check status
    status = info.get('status', 'unknown')
    if status == 'archived':
        print(f"\nWARNING: Dataset '{name}' is ARCHIVED. Consider using active dataset.")
    elif status == 'experimental':
        print(f"\nWARNING: Dataset '{name}' is EXPERIMENTAL. Results may be unreliable.")

    # Get path
    path_key = f"{artifact_type}_path"
    if path_key not in info:
        raise ArtifactError(f"No {path_key} defined for dataset '{name}'")

    file_path = PROJECT_ROOT / info[path_key]
    if not file_path.exists():
        raise ArtifactError(f"File not found: {file_path}")

    # Load
    df = pd.read_pickle(file_path)

    # Print summary
    if not quiet:
        print_dataset_summary(name, info, df)

    # Validate
    if validate:
        validate_dataset(df, info)

    return df


def load_active_frames(validate: bool = True, quiet: bool = False) -> pd.DataFrame:
    """Convenience function to load active dataset frames."""
    return load_dataset(artifact_type='frames', validate=validate, quiet=quiet)


# CLI interface
def main():
    """Command-line interface for artifact management."""
    import argparse

    parser = argparse.ArgumentParser(description="Artifact Management")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List command
    subparsers.add_parser('list', help='List all datasets')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show dataset info')
    info_parser.add_argument('name', nargs='?', help='Dataset name (default: active)')

    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate dataset files')
    val_parser.add_argument('name', nargs='?', help='Dataset name (default: all)')

    args = parser.parse_args()

    if args.command == 'list':
        datasets = list_datasets()
        print("\nRegistered Datasets:")
        print("-" * 60)
        for name, info in datasets.items():
            marker = " [ACTIVE]" if info['is_active'] else ""
            status = info['status'].upper()
            print(f"  {name}{marker}")
            print(f"    Status: {status} | Tracks: {info['tracks']}")
            print(f"    {info['description']}")
            print()

    elif args.command == 'info':
        if args.name:
            registry = load_registry()
            info = registry['datasets'].get(args.name)
            if info:
                info['name'] = args.name
                print_dataset_summary(args.name, info)
            else:
                print(f"Dataset '{args.name}' not found")
        else:
            info = get_active_dataset()
            print_dataset_summary(info['name'], info)

    elif args.command == 'validate':
        registry = load_registry()
        datasets_to_check = [args.name] if args.name else list(registry['datasets'].keys())

        for name in datasets_to_check:
            try:
                print(f"\nValidating {name}...")
                load_dataset(name, validate=True, quiet=False)
                print(f"  OK")
            except Exception as e:
                print(f"  FAILED: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
