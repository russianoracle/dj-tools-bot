#!/usr/bin/env python3
"""
Extract Unified Features for DEAM-Compatible Training

Extracts 21 common features from audio files or DEAM CSV files.
Supports merging datasets for unified training.

Usage:
    # From User audio (test_data.txt format)
    python scripts/extract_unified_features.py \
        --input tests/test_data.txt \
        --output dataset/user_unified.pkl \
        --source audio

    # From DEAM CSV features
    python scripts/extract_unified_features.py \
        --input dataset/features \
        --output dataset/deam_unified.pkl \
        --source deam \
        --audio-dir dataset/MEMD_audio

    # From directory of audio files
    python scripts/extract_unified_features.py \
        --input /path/to/music \
        --output dataset/extracted.pkl \
        --source audio-dir

    # Merge two datasets
    python scripts/extract_unified_features.py \
        --merge dataset/user_unified.pkl dataset/deam_unified.pkl \
        --output dataset/combined.pkl \
        --user-weight 3.0 \
        --deam-weight 1.0
"""

import argparse
import sys
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.unified_features import (
    UnifiedFeatureExtractor,
    COMMON_FEATURES,
    FRAME_FEATURES,
    merge_datasets
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_tracks_from_txt(txt_path: Path) -> List[Tuple[str, Optional[str]]]:
    """
    Load track paths and zones from test_data.txt format.

    Supports two formats:
    1. Simple: path<TAB>zone (one per line)
    2. Full: BPM<TAB>Key<TAB>Zone<TAB>Artist<TAB>Title<TAB>Location (with header)

    Returns:
        List of (path, zone) tuples
    """
    tracks = []

    # Try to detect encoding (UTF-16 or UTF-8)
    try:
        with open(txt_path, 'rb') as f:
            raw = f.read(2)
            if raw == b'\xff\xfe' or raw == b'\xfe\xff':
                encoding = 'utf-16'
            else:
                encoding = 'utf-8'
    except Exception:
        encoding = 'utf-8'

    with open(txt_path, 'r', encoding=encoding) as f:
        lines = f.readlines()

    if not lines:
        return tracks

    # Check if first line is header
    first_line = lines[0].strip()
    has_header = 'Location' in first_line or 'BPM' in first_line

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Skip header
        if i == 0 and has_header:
            continue

        parts = line.split('\t')

        # Format 2: BPM, Key, Zone, Artist, Title, Location
        if len(parts) >= 6 and 'Location' not in parts[0]:
            zone = parts[2] if parts[2] else None
            path = parts[5]
            tracks.append((path, zone))

        # Format 1: path<TAB>zone
        elif len(parts) >= 2:
            path, zone = parts[0], parts[1]
            tracks.append((path, zone))

        # Just path
        elif len(parts) == 1:
            tracks.append((parts[0], None))

    return tracks


def extract_from_audio_list(
    txt_path: Path,
    output_path: Path,
    workers: int = 1
) -> pd.DataFrame:
    """
    Extract features from audio files listed in txt file.

    Args:
        txt_path: Path to txt file with track paths and zones
        output_path: Output file path
        workers: Number of parallel workers (not used yet)

    Returns:
        DataFrame with extracted features
    """
    logger.info(f"Loading track list from: {txt_path}")
    tracks = load_tracks_from_txt(txt_path)
    logger.info(f"Found {len(tracks)} tracks")

    extractor = UnifiedFeatureExtractor()

    records = []
    failed = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(tracks, desc="Extracting features")
    except ImportError:
        iterator = tracks

    for path, zone in iterator:
        if not Path(path).exists():
            logger.warning(f"File not found: {path}")
            failed += 1
            continue

        features = extractor.extract_from_audio(path)
        if features:
            features['path'] = path
            features['zone'] = zone
            features['source'] = 'user'
            records.append(features)
        else:
            failed += 1

    logger.info(f"Successfully extracted: {len(records)}, Failed: {failed}")

    df = pd.DataFrame(records)

    # Save
    save_dataframe(df, output_path)

    return df


def extract_from_audio_dir(
    input_dir: Path,
    output_path: Path,
    workers: int = 1
) -> pd.DataFrame:
    """
    Extract features from all audio files in directory.

    Args:
        input_dir: Directory with audio files
        output_path: Output file path
        workers: Number of parallel workers (not used yet)

    Returns:
        DataFrame with extracted features
    """
    # Find audio files
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.mp4']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"**/{ext}"))

    audio_files = sorted(audio_files)
    logger.info(f"Found {len(audio_files)} audio files in {input_dir}")

    extractor = UnifiedFeatureExtractor()
    df = extractor.extract_batch_from_audio(
        [str(f) for f in audio_files],
        show_progress=True
    )

    # Save
    save_dataframe(df, output_path)

    return df


def extract_from_deam(
    features_dir: Path,
    output_path: Path,
    audio_dir: Optional[Path] = None,
    annotations_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Extract features from DEAM CSV files.

    Args:
        features_dir: Directory with DEAM feature CSV files
        output_path: Output file path
        audio_dir: Optional directory with audio files (for tempo)
        annotations_path: Optional path to annotations CSV

    Returns:
        DataFrame with extracted features
    """
    logger.info(f"Extracting from DEAM features: {features_dir}")

    extractor = UnifiedFeatureExtractor()
    df = extractor.extract_batch_from_deam(
        str(features_dir),
        audio_dir=str(audio_dir) if audio_dir else None,
        show_progress=True
    )

    # Load annotations if provided
    if annotations_path and annotations_path.exists():
        logger.info(f"Loading annotations from: {annotations_path}")
        annotations = pd.read_csv(annotations_path)

        # Merge arousal/valence
        if 'song_id' in annotations.columns:
            annotations = annotations.rename(columns={'song_id': 'track_id'})

        if 'track_id' in df.columns and 'track_id' in annotations.columns:
            df = df.merge(
                annotations[['track_id', 'arousal_mean', 'valence_mean']].rename(
                    columns={'arousal_mean': 'arousal', 'valence_mean': 'valence'}
                ),
                on='track_id',
                how='left'
            )

            # Convert arousal to zone
            df['zone'] = df['arousal'].apply(arousal_to_zone)

    # Save
    save_dataframe(df, output_path)

    return df


def extract_frames_from_audio_list(
    txt_path: Path,
    output_path: Path,
    frame_size: float = 0.5,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract frame-level features from audio files listed in txt file.

    Args:
        txt_path: Path to txt file with track paths and zones
        output_path: Output file path
        frame_size: Frame size in seconds (default 0.5)
        limit: Optional limit on number of tracks

    Returns:
        DataFrame with frame features
    """
    logger.info(f"Loading track list from: {txt_path}")
    tracks = load_tracks_from_txt(txt_path)

    if limit:
        tracks = tracks[:limit]

    logger.info(f"Processing {len(tracks)} tracks with frame extraction")

    extractor = UnifiedFeatureExtractor()

    audio_paths = [path for path, _ in tracks]
    zones = [zone for _, zone in tracks]

    df = extractor.extract_batch_frames_from_audio(
        audio_paths,
        zones=zones,
        frame_size=frame_size,
        show_progress=True
    )

    save_dataframe(df, output_path)
    return df


def extract_frames_from_deam(
    features_dir: Path,
    output_path: Path,
    annotations_path: Optional[Path] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract frame-level features from DEAM CSV files.

    Args:
        features_dir: Directory with DEAM CSV files
        output_path: Output file path
        annotations_path: Optional path to annotations CSV
        limit: Optional limit on number of tracks

    Returns:
        DataFrame with frame features
    """
    logger.info(f"Extracting frames from DEAM: {features_dir}")

    # Load zones from annotations if available
    zones = {}
    if annotations_path and annotations_path.exists():
        logger.info(f"Loading annotations from: {annotations_path}")
        annotations = pd.read_csv(annotations_path)
        annotations.columns = [c.strip() for c in annotations.columns]

        if 'song_id' in annotations.columns:
            for _, row in annotations.iterrows():
                track_id = int(row['song_id'])
                arousal = row.get('arousal_mean', 5.0)
                zones[track_id] = arousal_to_zone(arousal)

    extractor = UnifiedFeatureExtractor()

    # Limit CSV files if needed
    csv_dir = Path(features_dir)
    csv_files = sorted(csv_dir.glob("*.csv"))
    if limit:
        csv_files = csv_files[:limit]
        # Create temp dir with limited files
        import tempfile
        import shutil
        temp_dir = Path(tempfile.mkdtemp())
        for csv_file in csv_files:
            shutil.copy(csv_file, temp_dir / csv_file.name)
        features_dir = temp_dir

    df = extractor.extract_batch_frames_from_deam(
        str(features_dir),
        zones=zones if zones else None,
        show_progress=True
    )

    save_dataframe(df, output_path)
    return df


def arousal_to_zone(arousal: float, yellow_thresh: float = 4.0, purple_thresh: float = 6.0) -> str:
    """Convert arousal value to zone."""
    if pd.isna(arousal):
        return 'GREEN'
    if arousal < yellow_thresh:
        return 'YELLOW'
    elif arousal > purple_thresh:
        return 'PURPLE'
    else:
        return 'GREEN'


def merge_datasets_cmd(
    input_paths: List[Path],
    output_path: Path,
    user_weight: float = 1.0,
    deam_weight: float = 1.0
) -> pd.DataFrame:
    """
    Merge multiple datasets.

    Args:
        input_paths: List of input pickle/csv files
        output_path: Output file path
        user_weight: Weight for user source
        deam_weight: Weight for deam source

    Returns:
        Merged DataFrame
    """
    dfs = []

    for path in input_paths:
        logger.info(f"Loading: {path}")
        if path.suffix == '.pkl':
            df = pd.read_pickle(path)
        else:
            df = pd.read_csv(path)
        dfs.append(df)

    if len(dfs) == 2:
        # Assume first is user, second is deam
        merged = merge_datasets(dfs[0], dfs[1], user_weight, deam_weight)
    else:
        # Simple concat
        merged = pd.concat(dfs, ignore_index=True)

    # Save
    save_dataframe(merged, output_path)

    return merged


def save_dataframe(df: pd.DataFrame, output_path: Path):
    """Save DataFrame to file (pickle or CSV)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == '.pkl':
        df.to_pickle(output_path)
    else:
        df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(df)} records to: {output_path}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    print(f"\nTotal tracks: {len(df)}")

    if 'source' in df.columns:
        print(f"\nBy source:")
        for source, count in df['source'].value_counts().items():
            print(f"  {source}: {count}")

    if 'zone' in df.columns:
        print(f"\nBy zone:")
        for zone, count in df['zone'].value_counts().items():
            pct = 100 * count / len(df)
            print(f"  {zone}: {count} ({pct:.1f}%)")

    print(f"\nFeatures extracted: {len(COMMON_FEATURES)}")
    print("  " + ", ".join(COMMON_FEATURES[:7]))
    print("  " + ", ".join(COMMON_FEATURES[7:14]))
    print("  " + ", ".join(COMMON_FEATURES[14:]))

    # Feature statistics
    print("\nFeature statistics:")
    for feature in ['tempo', 'rms_energy', 'drop_count', 'brightness']:
        if feature in df.columns:
            values = df[feature].dropna()
            print(f"  {feature:20s}: mean={values.mean():.3f}, std={values.std():.3f}")

    print("=" * 60 + "\n")


def print_frame_summary(df: pd.DataFrame):
    """Print summary statistics for frame-level extraction."""
    print("\n" + "=" * 60)
    print("FRAME EXTRACTION SUMMARY")
    print("=" * 60)

    print(f"\nTotal frames: {len(df)}")

    if 'track_id' in df.columns:
        n_tracks = df['track_id'].nunique()
        avg_frames = len(df) / n_tracks if n_tracks > 0 else 0
        print(f"Total tracks: {n_tracks}")
        print(f"Average frames/track: {avg_frames:.1f}")

    if 'source' in df.columns:
        print(f"\nBy source:")
        for source, count in df['source'].value_counts().items():
            n_tracks = df[df['source'] == source]['track_id'].nunique()
            print(f"  {source}: {count} frames ({n_tracks} tracks)")

    if 'zone' in df.columns:
        print(f"\nBy zone:")
        for zone in ['GREEN', 'PURPLE', 'YELLOW']:
            zone_df = df[df['zone'] == zone]
            if len(zone_df) > 0:
                n_tracks = zone_df['track_id'].nunique()
                pct = 100 * len(zone_df) / len(df)
                print(f"  {zone}: {len(zone_df)} frames ({n_tracks} tracks, {pct:.1f}%)")

    print(f"\nFrame features: {len(FRAME_FEATURES)}")
    print("  " + ", ".join(FRAME_FEATURES[:8]))
    print("  " + ", ".join(FRAME_FEATURES[8:16]))
    print("  " + ", ".join(FRAME_FEATURES[16:24]))
    print("  " + ", ".join(FRAME_FEATURES[24:]))

    # Key frame statistics
    print("\nKey frame statistics:")
    for feature in ['rms_energy', 'spectral_centroid', 'onset_strength', 'drop_candidate']:
        if feature in df.columns:
            values = df[feature].dropna()
            print(f"  {feature:25s}: mean={values.mean():.4f}, std={values.std():.4f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract unified features for DEAM-compatible training'
    )

    # Input/output
    parser.add_argument(
        '--input', '-i',
        type=Path,
        help='Input path (txt file, directory, or DEAM features dir)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output file path (.pkl or .csv)'
    )

    # Source type
    parser.add_argument(
        '--source', '-s',
        choices=['audio', 'audio-dir', 'deam'],
        default='audio',
        help='Source type: audio (txt file), audio-dir (directory), deam (DEAM CSV)'
    )

    # Extraction method
    parser.add_argument(
        '--method', '-m',
        choices=['track', 'frames'],
        default='track',
        help='Extraction method: track (aggregated) or frames (per 0.5s frame)'
    )
    parser.add_argument(
        '--frame-size',
        type=float,
        default=0.5,
        help='Frame size in seconds for frame extraction (default: 0.5)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of tracks to process'
    )

    # DEAM options
    parser.add_argument(
        '--audio-dir',
        type=Path,
        help='Audio directory for DEAM tempo extraction'
    )
    parser.add_argument(
        '--annotations',
        type=Path,
        help='DEAM annotations CSV for arousal/valence'
    )

    # Merge mode
    parser.add_argument(
        '--merge',
        nargs='+',
        type=Path,
        help='Merge multiple datasets'
    )
    parser.add_argument(
        '--user-weight',
        type=float,
        default=1.0,
        help='Sample weight for user source (default: 1.0)'
    )
    parser.add_argument(
        '--deam-weight',
        type=float,
        default=1.0,
        help='Sample weight for DEAM source (default: 1.0)'
    )

    # Other options
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Merge mode
    if args.merge:
        df = merge_datasets_cmd(
            args.merge,
            args.output,
            args.user_weight,
            args.deam_weight
        )
        print_summary(df)
        return 0

    # Extraction mode
    if not args.input:
        parser.error("--input is required for extraction mode")

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        return 1

    # Frame extraction mode
    if args.method == 'frames':
        if args.source == 'audio':
            df = extract_frames_from_audio_list(
                args.input,
                args.output,
                frame_size=args.frame_size,
                limit=args.limit
            )
        elif args.source == 'deam':
            df = extract_frames_from_deam(
                args.input,
                args.output,
                annotations_path=args.annotations,
                limit=args.limit
            )
        else:
            logger.error(f"Frame extraction not supported for source: {args.source}")
            return 1

        print_frame_summary(df)
        return 0

    # Track-level extraction mode
    if args.source == 'audio':
        # Extract from txt file with track list
        df = extract_from_audio_list(args.input, args.output, args.workers)

    elif args.source == 'audio-dir':
        # Extract from directory of audio files
        df = extract_from_audio_dir(args.input, args.output, args.workers)

    elif args.source == 'deam':
        # Extract from DEAM CSV files
        df = extract_from_deam(
            args.input,
            args.output,
            args.audio_dir,
            args.annotations
        )

    print_summary(df)
    return 0


if __name__ == '__main__':
    sys.exit(main())
