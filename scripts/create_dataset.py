#!/usr/bin/env python3
"""
Script to create dataset CSV from music collection with BPM metadata.

This script scans a directory for audio files and extracts BPM from their metadata,
creating a CSV file that can be used for training the ML models.

Usage:
    python scripts/create_dataset.py /path/to/music/folder --output dataset.csv
    python scripts/create_dataset.py /path/to/music/folder --output dataset.csv --recursive
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Optional, Dict
import sys

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.wave import WAVE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BPMExtractor:
    """Extract BPM from audio file metadata."""

    SUPPORTED_EXTENSIONS = {'.mp3', '.flac', '.m4a', '.mp4', '.wav'}

    # Common BPM tag names across different formats
    BPM_TAG_NAMES = [
        'BPM',           # Standard tag
        'TBPM',          # ID3v2.3
        'bpm',           # Vorbis comment (FLAC)
        'tmpo',          # MP4/M4A
        '----:com.apple.iTunes:BPM',  # iTunes
        'TEMPO',         # Some DJ software
    ]

    @staticmethod
    def extract_bpm(file_path: Path) -> Optional[float]:
        """
        Extract BPM from audio file metadata.

        Args:
            file_path: Path to audio file

        Returns:
            BPM value if found, None otherwise
        """
        try:
            audio = MutagenFile(file_path)
            if audio is None:
                return None

            # Try different BPM tag names
            for tag_name in BPMExtractor.BPM_TAG_NAMES:
                if tag_name in audio:
                    value = audio[tag_name]

                    # Handle different value types
                    if isinstance(value, list):
                        value = value[0]

                    if isinstance(value, bytes):
                        value = value.decode('utf-8')

                    # Try to convert to float
                    try:
                        bpm = float(str(value))
                        if 40 <= bpm <= 250:  # Reasonable BPM range
                            return bpm
                    except (ValueError, TypeError):
                        continue

            # Special handling for MP3 ID3 tags
            if isinstance(audio, MP3):
                # Try TBPM frame
                if 'TBPM' in audio:
                    try:
                        bpm = float(audio['TBPM'].text[0])
                        if 40 <= bpm <= 250:
                            return bpm
                    except:
                        pass

            # Special handling for MP4/M4A
            elif isinstance(audio, MP4):
                if 'tmpo' in audio:
                    try:
                        bpm = float(audio['tmpo'][0])
                        if 40 <= bpm <= 250:
                            return bpm
                    except:
                        pass

            return None

        except Exception as e:
            logger.debug(f"Error extracting BPM from {file_path}: {e}")
            return None

    @staticmethod
    def extract_genre(file_path: Path) -> Optional[str]:
        """Extract genre from metadata if available."""
        try:
            audio = MutagenFile(file_path)
            if audio is None:
                return None

            # Try common genre tags
            for tag in ['genre', 'TCON', '©gen', 'GENRE']:
                if tag in audio:
                    value = audio[tag]
                    if isinstance(value, list):
                        value = value[0]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    return str(value)

            return None
        except:
            return None


def scan_directory(directory: Path, recursive: bool = False) -> List[Path]:
    """
    Scan directory for audio files.

    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories

    Returns:
        List of audio file paths
    """
    audio_files = []

    if recursive:
        for ext in BPMExtractor.SUPPORTED_EXTENSIONS:
            audio_files.extend(directory.rglob(f'*{ext}'))
    else:
        for ext in BPMExtractor.SUPPORTED_EXTENSIONS:
            audio_files.extend(directory.glob(f'*{ext}'))

    return sorted(audio_files)


def create_dataset(
    music_dir: Path,
    output_file: Path,
    recursive: bool = False,
    min_bpm: float = 40,
    max_bpm: float = 250
) -> None:
    """
    Create dataset CSV from music directory.

    Args:
        music_dir: Directory containing music files
        output_file: Output CSV file path
        recursive: Whether to scan subdirectories
        min_bpm: Minimum valid BPM
        max_bpm: Maximum valid BPM
    """
    logger.info(f"Scanning directory: {music_dir}")
    logger.info(f"Recursive: {recursive}")

    # Scan for audio files
    audio_files = scan_directory(music_dir, recursive)
    logger.info(f"Found {len(audio_files)} audio files")

    # Extract BPM from each file
    dataset = []
    files_with_bpm = 0
    files_without_bpm = 0

    for i, file_path in enumerate(audio_files, 1):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(audio_files)} files...")

        bpm = BPMExtractor.extract_bpm(file_path)
        genre = BPMExtractor.extract_genre(file_path)

        if bpm is not None and min_bpm <= bpm <= max_bpm:
            dataset.append({
                'path': str(file_path),
                'bpm': bpm,
                'genre': genre or '',
                'zone': ''  # To be filled manually or by classifier
            })
            files_with_bpm += 1
        else:
            files_without_bpm += 1
            logger.debug(f"No valid BPM found for: {file_path.name}")

    # Write to CSV
    logger.info(f"Writing dataset to: {output_file}")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if dataset:
            writer = csv.DictWriter(f, fieldnames=['path', 'bpm', 'genre', 'zone'])
            writer.writeheader()
            writer.writerows(dataset)

    # Summary
    logger.info("=" * 60)
    logger.info(f"Dataset created successfully!")
    logger.info(f"Total files scanned: {len(audio_files)}")
    logger.info(f"Files with BPM: {files_with_bpm}")
    logger.info(f"Files without BPM: {files_without_bpm}")
    logger.info(f"Dataset saved to: {output_file}")
    logger.info("=" * 60)

    if files_with_bpm == 0:
        logger.warning("⚠️  No files with BPM metadata found!")
        logger.warning("Make sure your files have BPM tags in their metadata.")
        logger.warning("If BPM is stored externally (e.g., in DJ software database),")
        logger.warning("you'll need to export it first or provide a different input format.")


def main():
    parser = argparse.ArgumentParser(
        description='Create training dataset from music collection with BPM metadata'
    )
    parser.add_argument(
        'music_dir',
        type=Path,
        help='Directory containing music files'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('dataset.csv'),
        help='Output CSV file (default: dataset.csv)'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Scan subdirectories recursively'
    )
    parser.add_argument(
        '--min-bpm',
        type=float,
        default=40,
        help='Minimum valid BPM (default: 40)'
    )
    parser.add_argument(
        '--max-bpm',
        type=float,
        default=250,
        help='Maximum valid BPM (default: 250)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (debug level)'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input directory
    if not args.music_dir.exists():
        logger.error(f"Directory not found: {args.music_dir}")
        sys.exit(1)

    if not args.music_dir.is_dir():
        logger.error(f"Not a directory: {args.music_dir}")
        sys.exit(1)

    # Create dataset
    create_dataset(
        music_dir=args.music_dir,
        output_file=args.output,
        recursive=args.recursive,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm
    )


if __name__ == '__main__':
    main()
