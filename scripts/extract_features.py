#!/usr/bin/env python3
"""
Extract audio features from music collection for ML training.

This script processes tracks from a CSV dataset, extracts all 16 audio features,
and saves them for ML model training.

Usage:
    python scripts/extract_features.py dataset.csv --output features.csv
    python scripts/extract_features.py dataset.csv --output features.csv --workers 8
    python scripts/extract_features.py dataset.csv --output features.pkl --format pickle
"""

import argparse
import csv
import logging
import pickle
import sys
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.loader import AudioLoader
from src.audio.extractors import FeatureExtractor, AudioFeatures
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractionPipeline:
    """Pipeline for extracting features from music collection."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.loader = AudioLoader(self.config)
        self.extractor = FeatureExtractor(self.config)

    def extract_single_track(
        self,
        track_path: str,
        ground_truth_bpm: Optional[float] = None,
        zone: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Extract features from a single track.

        Args:
            track_path: Path to audio file
            ground_truth_bpm: Ground truth BPM from metadata
            zone: Energy zone label if known

        Returns:
            Dictionary with features and metadata, or None if extraction failed
        """
        try:
            # Load audio
            audio_data = self.loader.load(track_path)
            if audio_data is None:
                logger.warning(f"Failed to load: {track_path}")
                return None

            # Extract features
            features = self.extractor.extract(
                audio_data.audio,
                audio_data.sample_rate,
                track_path
            )

            if features is None:
                logger.warning(f"Failed to extract features: {track_path}")
                return None

            # Prepare result
            result = {
                'path': track_path,
                'ground_truth_bpm': ground_truth_bpm,
                'zone': zone or '',
                **asdict(features)
            }

            return result

        except Exception as e:
            logger.error(f"Error processing {track_path}: {e}")
            return None


def process_track_wrapper(args: Tuple[str, Optional[float], Optional[str]]) -> Optional[Dict]:
    """
    Wrapper function for multiprocessing.

    Args:
        args: Tuple of (track_path, ground_truth_bpm, zone)

    Returns:
        Feature dictionary or None
    """
    track_path, ground_truth_bpm, zone = args
    pipeline = FeatureExtractionPipeline()
    return pipeline.extract_single_track(track_path, ground_truth_bpm, zone)


def load_dataset(csv_path: Path) -> List[Tuple[str, Optional[float], Optional[str]]]:
    """
    Load dataset from CSV file.

    Expected columns: path, bpm, zone (optional)

    Args:
        csv_path: Path to dataset CSV

    Returns:
        List of (path, bpm, zone) tuples
    """
    tracks = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            path = row['path']

            # Parse BPM
            bpm = None
            if 'bpm' in row and row['bpm']:
                try:
                    bpm = float(row['bpm'])
                except ValueError:
                    logger.warning(f"Invalid BPM value: {row['bpm']}")

            # Get zone if present
            zone = row.get('zone', '').strip()
            if not zone:
                zone = None

            tracks.append((path, bpm, zone))

    return tracks


def extract_features_batch(
    dataset_path: Path,
    output_path: Path,
    workers: int = None,
    output_format: str = 'csv'
) -> None:
    """
    Extract features from all tracks in dataset.

    Args:
        dataset_path: Path to dataset CSV
        output_path: Output file path
        workers: Number of parallel workers (default: CPU count - 1)
        output_format: Output format ('csv' or 'pickle')
    """
    # Determine number of workers
    if workers is None:
        workers = max(1, cpu_count() - 1)

    logger.info(f"Loading dataset from: {dataset_path}")
    tracks = load_dataset(dataset_path)
    logger.info(f"Found {len(tracks)} tracks in dataset")

    logger.info(f"Extracting features using {workers} workers...")

    # Process tracks in parallel with progress bar
    features_list = []

    if workers > 1:
        with Pool(processes=workers) as pool:
            results = list(tqdm(
                pool.imap(process_track_wrapper, tracks),
                total=len(tracks),
                desc="Extracting features"
            ))
            features_list = [r for r in results if r is not None]
    else:
        # Single-threaded processing (useful for debugging)
        for track_args in tqdm(tracks, desc="Extracting features"):
            result = process_track_wrapper(track_args)
            if result is not None:
                features_list.append(result)

    logger.info(f"Successfully extracted features from {len(features_list)}/{len(tracks)} tracks")

    # Save results
    if output_format == 'csv':
        save_features_csv(features_list, output_path)
    elif output_format == 'pickle':
        save_features_pickle(features_list, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    logger.info(f"Features saved to: {output_path}")

    # Print summary statistics
    print_summary_statistics(features_list)


def save_features_csv(features_list: List[Dict], output_path: Path) -> None:
    """Save features to CSV file."""
    if not features_list:
        logger.warning("No features to save")
        return

    # Get all field names from first item
    fieldnames = list(features_list[0].keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features_list)


def save_features_pickle(features_list: List[Dict], output_path: Path) -> None:
    """Save features to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(features_list, f, protocol=pickle.HIGHEST_PROTOCOL)


def print_summary_statistics(features_list: List[Dict]) -> None:
    """Print summary statistics of extracted features."""
    if not features_list:
        return

    import numpy as np

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)

    # BPM statistics
    detected_bpms = [f['tempo'] for f in features_list if f.get('tempo')]
    ground_truth_bpms = [f['ground_truth_bpm'] for f in features_list if f.get('ground_truth_bpm')]

    if detected_bpms:
        print(f"\nDetected BPM:")
        print(f"  Mean: {np.mean(detected_bpms):.1f}")
        print(f"  Std:  {np.std(detected_bpms):.1f}")
        print(f"  Min:  {np.min(detected_bpms):.1f}")
        print(f"  Max:  {np.max(detected_bpms):.1f}")

    if ground_truth_bpms:
        print(f"\nGround Truth BPM:")
        print(f"  Mean: {np.mean(ground_truth_bpms):.1f}")
        print(f"  Std:  {np.std(ground_truth_bpms):.1f}")
        print(f"  Min:  {np.min(ground_truth_bpms):.1f}")
        print(f"  Max:  {np.max(ground_truth_bpms):.1f}")

    # BPM accuracy if both available
    if detected_bpms and ground_truth_bpms and len(detected_bpms) == len(ground_truth_bpms):
        errors = np.abs(np.array(detected_bpms) - np.array(ground_truth_bpms))
        print(f"\nBPM Detection Accuracy:")
        print(f"  MAE (Mean Absolute Error): {np.mean(errors):.2f} BPM")
        print(f"  Within 2 BPM: {100 * np.mean(errors <= 2):.1f}%")
        print(f"  Within 5 BPM: {100 * np.mean(errors <= 5):.1f}%")

    # Zone distribution
    zones = [f['zone'] for f in features_list if f.get('zone')]
    if zones:
        from collections import Counter
        zone_counts = Counter(zones)
        print(f"\nZone Distribution:")
        for zone, count in sorted(zone_counts.items()):
            print(f"  {zone}: {count} ({100*count/len(zones):.1f}%)")

    # Feature ranges
    print(f"\nFeature Ranges:")
    numeric_features = ['energy_variance', 'brightness', 'drop_intensity',
                       'spectral_centroid', 'spectral_rolloff', 'zcr']

    for feature in numeric_features:
        values = [f[feature] for f in features_list if f.get(feature) is not None]
        if values:
            print(f"  {feature:20s}: [{np.min(values):.4f}, {np.max(values):.4f}]")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract audio features from music collection for ML training'
    )
    parser.add_argument(
        'dataset',
        type=Path,
        help='Dataset CSV file (output from create_dataset.py)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output file path (features.csv or features.pkl)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['csv', 'pickle'],
        default='csv',
        help='Output format (default: csv)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input
    if not args.dataset.exists():
        logger.error(f"Dataset file not found: {args.dataset}")
        sys.exit(1)

    # Extract features
    extract_features_batch(
        dataset_path=args.dataset,
        output_path=args.output,
        workers=args.workers,
        output_format=args.format
    )


if __name__ == '__main__':
    main()
