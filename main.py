#!/usr/bin/env python3
"""
Mood Classifier - Main Application Entry Point

Desktop application for classifying DJ tracks into energy zones.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils import get_config, setup_logger
from src.audio import AudioLoader, FeatureExtractor
from src.classification import EnergyZoneClassifier
from src.metadata import MetadataWriter, MetadataReader


def setup_application():
    """Initialize application components."""
    config = get_config()

    # Setup logging
    log_config = config.logging
    logger = setup_logger(
        level=log_config.get('level', 'INFO'),
        log_file=log_config.get('log_file'),
        max_bytes=log_config.get('max_log_size_mb', 10) * 1024 * 1024,
        backup_count=log_config.get('backup_count', 3)
    )

    return config, logger


def run_cli(args):
    """Run command-line interface mode."""
    config, logger = setup_application()

    logger.info("Starting Mood Classifier (CLI mode)")

    # Initialize components
    audio_loader = AudioLoader(sample_rate=config.get('audio.sample_rate', 22050))
    feature_extractor = FeatureExtractor(config)
    classifier = EnergyZoneClassifier(config)
    metadata_writer = MetadataWriter(config)
    metadata_reader = MetadataReader(config)

    # Process single file or batch
    if args.file:
        process_single_file(
            args.file, audio_loader, feature_extractor,
            classifier, metadata_writer, metadata_reader, args
        )
    elif args.batch:
        process_batch(
            args.batch, audio_loader, feature_extractor,
            classifier, metadata_writer, metadata_reader, args
        )
    else:
        logger.error("No input specified. Use --file or --batch")
        return 1

    logger.info("Processing complete")
    return 0


def process_single_file(
    file_path, audio_loader, feature_extractor,
    classifier, metadata_writer, metadata_reader, args
):
    """Process a single audio file."""
    from src.utils import get_logger
    logger = get_logger()

    logger.info(f"Processing: {file_path}")

    # Check if already classified
    if not args.force and metadata_reader.has_classification(file_path):
        existing_zone = metadata_reader.read_zone(file_path)
        logger.info(f"File already classified as: {existing_zone.display_name}")
        if not args.overwrite:
            return

    try:
        # Load audio
        y, sr = audio_loader.load(file_path)

        # Extract features
        features = feature_extractor.extract(y, sr)

        # Classify
        result = classifier.classify(features)

        # Display result
        print(f"\n{result}")
        print(f"\nFeatures:")
        print(f"  Tempo: {features.tempo:.1f} BPM")
        print(f"  Energy Variance: {features.energy_variance:.4f}")
        print(f"  Drop Intensity: {features.drop_intensity:.2f}")
        print(f"  Brightness: {features.brightness:.2%}")
        print(f"  Spectral Centroid: {features.spectral_centroid:.1f} Hz")

        # Write metadata
        if args.write_metadata:
            if metadata_writer.write(file_path, result):
                logger.info("Metadata written successfully")
            else:
                logger.error("Failed to write metadata")

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}", exc_info=True)


def process_batch(
    directory, audio_loader, feature_extractor,
    classifier, metadata_writer, metadata_reader, args
):
    """Process all audio files in a directory."""
    from src.utils import get_logger
    from tqdm import tqdm
    import multiprocessing as mp

    logger = get_logger()

    directory = Path(directory)
    if not directory.is_dir():
        logger.error(f"Not a directory: {directory}")
        return

    # Find all audio files
    audio_files = []
    for ext in AudioLoader.SUPPORTED_FORMATS:
        audio_files.extend(directory.rglob(f'*{ext}'))

    logger.info(f"Found {len(audio_files)} audio files")

    if not audio_files:
        logger.warning("No audio files found")
        return

    # Filter already classified files
    if not args.force:
        files_to_process = [
            f for f in audio_files
            if not metadata_reader.has_classification(str(f))
        ]
        logger.info(f"Skipping {len(audio_files) - len(files_to_process)} already classified files")
        audio_files = files_to_process

    # Process files
    results = []
    for file_path in tqdm(audio_files, desc="Processing tracks"):
        try:
            # Load audio
            y, sr = audio_loader.load(str(file_path))

            # Extract features
            features = feature_extractor.extract(y, sr)

            # Classify
            result = classifier.classify(features)

            results.append({
                'file': file_path.name,
                'path': str(file_path),
                'zone': result.zone,
                'confidence': result.confidence,
                'tempo': features.tempo,
                'result': result
            })

            # Write metadata
            if args.write_metadata:
                metadata_writer.write(str(file_path), result)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    # Print summary
    print("\n" + "="*80)
    print("CLASSIFICATION SUMMARY")
    print("="*80)

    from collections import Counter
    zone_counts = Counter(r['zone'] for r in results)

    for zone, count in zone_counts.items():
        print(f"{zone.emoji} {zone.display_name}: {count} tracks ({count/len(results)*100:.1f}%)")

    # Export CSV if requested
    if args.export_csv:
        export_to_csv(results, args.export_csv)
        logger.info(f"Results exported to {args.export_csv}")


def export_to_csv(results, output_path):
    """Export results to CSV file."""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['File', 'Path', 'Zone', 'Confidence', 'Tempo (BPM)'])

        for r in results:
            writer.writerow([
                r['file'],
                r['path'],
                r['zone'].display_name,
                f"{r['confidence']:.2%}",
                f"{r['tempo']:.1f}"
            ])


def run_gui():
    """Run graphical user interface mode."""
    config, logger = setup_application()

    logger.info("Starting Mood Classifier (GUI mode)")

    try:
        from PyQt5.QtWidgets import QApplication
        from src.gui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("Mood Classifier")
        app.setOrganizationName("DJ Tools")

        window = MainWindow(config)
        window.show()

        return app.exec_()

    except ImportError as e:
        logger.error(f"GUI dependencies not available: {e}")
        logger.error("Please install PyQt5: pip install PyQt5")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Mood Classifier - DJ Track Energy Zone Classification'
    )

    parser.add_argument(
        '--file', '-f',
        help='Single audio file to classify'
    )

    parser.add_argument(
        '--batch', '-b',
        help='Directory of audio files to process'
    )

    parser.add_argument(
        '--write-metadata', '-w',
        action='store_true',
        help='Write classification to file metadata'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Process files even if already classified'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing classification'
    )

    parser.add_argument(
        '--export-csv',
        help='Export batch results to CSV file'
    )

    parser.add_argument(
        '--gui', '-g',
        action='store_true',
        help='Launch graphical interface'
    )

    parser.add_argument(
        '--config', '-c',
        help='Path to custom configuration file'
    )

    args = parser.parse_args()

    # Load custom config if specified
    if args.config:
        get_config(args.config)

    # Run in appropriate mode
    if args.gui or (not args.file and not args.batch):
        return run_gui()
    else:
        return run_cli(args)


if __name__ == '__main__':
    sys.exit(main())
