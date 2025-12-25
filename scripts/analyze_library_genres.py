#!/usr/bin/env python3
"""
Analyze music library genres and find gaps.

Usage:
    python scripts/analyze_library_genres.py [--input FOLDER] [--sample N] [--full]

Examples:
    # Quick analysis (sample 100 tracks from default library)
    python scripts/analyze_library_genres.py

    # Full library analysis
    python scripts/analyze_library_genres.py --full

    # Custom folder
    python scripts/analyze_library_genres.py --input "/path/to/music folder"
"""

# Suppress ALL logs/warnings BEFORE any imports
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ESSENTIA_LOG_LEVEL"] = "silent"
os.environ["GLOG_minloglevel"] = "3"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)

# Redirect stderr to /dev/null at OS level (catches C++ output too)
import io
_stderr_fd = os.dup(2)  # Save original stderr file descriptor
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)  # Redirect stderr to /dev/null

import argparse
from pathlib import Path
from collections import Counter, defaultdict
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
import pandas as pd

# Restore stderr after imports
os.dup2(_stderr_fd, 2)
os.close(_devnull)

# Audio extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".mp4", ".ogg", ".opus", ".aiff", ".aif"}


def find_audio_files(folder: Path) -> list:
    """Recursively find all audio files in folder."""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(folder.rglob(f"*{ext}"))
        audio_files.extend(folder.rglob(f"*{ext.upper()}"))
    return [str(f) for f in audio_files if f.is_file()]


def get_folder_genre(path: str, library_root: Path) -> str:
    """Extract genre from folder structure.

    Assumes structure like: library_root/Genre/Subgenre/track.mp3
    Returns the first folder after library_root.
    """
    p = Path(path)
    try:
        rel_path = p.relative_to(library_root)
        parts = rel_path.parts
        if len(parts) > 1:
            return parts[0]  # First folder = genre
        return "Root"
    except ValueError:
        return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="Analyze library genres")
    parser.add_argument("--input", "-i", type=str,
                       default="/Users/artemgusarov/Music/dj library",
                       help="Input folder with music files")
    parser.add_argument("--sample", "-s", type=int, default=100,
                       help="Number of tracks to sample (default: 100)")
    parser.add_argument("--full", "-f", action="store_true",
                       help="Analyze full library (slow)")
    parser.add_argument("--output", "-o", type=str,
                       help="Output CSV with results")
    args = parser.parse_args()

    # Find input folder
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Folder not found: {input_path}")
        return 1

    if not input_path.is_dir():
        print(f"Error: Not a directory: {input_path}")
        return 1

    # Find audio files
    print(f"Scanning folder: {input_path}")
    valid_paths = find_audio_files(input_path)
    print(f"Found {len(valid_paths)} audio files")

    if len(valid_paths) == 0:
        print("No valid audio files found!")
        return 1

    # Sample if not full
    if not args.full and len(valid_paths) > args.sample:
        print(f"Sampling {args.sample} tracks (use --full for complete analysis)")
        valid_paths = random.sample(valid_paths, args.sample)

    # Initialize genre classifier (suppress all output during init)
    print("\nInitializing genre classifier...")

    # Suppress stderr at OS level during classifier init (catches C++ output)
    _devnull2 = os.open(os.devnull, os.O_WRONLY)
    _stderr_fd2 = os.dup(2)
    os.dup2(_devnull2, 2)

    from src.genre import GenreClassifier
    classifier = GenreClassifier()

    # Restore stderr
    os.dup2(_stderr_fd2, 2)
    os.close(_devnull2)

    # Analyze tracks with M2-optimized parallel processing
    num_workers = 4  # M2 has 4 performance cores
    print(f"\nAnalyzing {len(valid_paths)} tracks using {num_workers} parallel workers...")

    # Suppress stderr during predictions (catches C++ TensorFlow noise)
    _pred_devnull = os.open(os.devnull, os.O_WRONLY)
    _pred_stderr = os.dup(2)
    os.dup2(_pred_devnull, 2)

    # Use batch prediction with parallel processing
    genre_results = classifier.predict_batch(
        valid_paths,
        show_progress=True,
        parallel=True,
        workers=num_workers
    )

    # Restore stderr
    os.dup2(_pred_stderr, 2)
    os.close(_pred_devnull)

    # Process results
    results = []
    genre_counter = Counter()
    dj_category_counter = Counter()
    mood_counter = Counter()
    errors = []
    folder_genre_counter = Counter()
    mismatches = []

    print("Processing results...")
    for path, result in zip(valid_paths, genre_results):
        if result.genre == "Error":
            errors.append((path, "Classification error"))
            continue

        folder_genre = get_folder_genre(path, input_path)

        results.append({
            "path": path,
            "folder_genre": folder_genre,
            "detected_genre": result.genre,
            "detected_category": result.dj_category,
            "confidence": result.confidence,
            "subgenre": result.subgenre,
            "moods": ", ".join([m[0] for m in result.mood_tags[:3]])
        })

        genre_counter[result.genre] += 1
        dj_category_counter[result.dj_category] += 1
        folder_genre_counter[folder_genre] += 1

        for mood, _ in result.mood_tags[:3]:
            mood_counter[mood] += 1

        # Check for mismatch (detected genre doesn't match folder)
        folder_lower = folder_genre.lower()
        detected_lower = result.dj_category.lower()
        genre_lower = result.genre.lower()

        # Simple mismatch detection
        if (folder_lower not in detected_lower and
            folder_lower not in genre_lower and
            detected_lower not in folder_lower):
            mismatches.append({
                "path": path,
                "folder": folder_genre,
                "detected": result.dj_category,
                "genre": result.genre,
                "confidence": result.confidence
            })

    # Print results
    print("\n" + "=" * 60)
    print("GENRE ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nAnalyzed: {len(results)} tracks")
    if errors:
        print(f"Errors: {len(errors)} tracks")

    # DJ Categories distribution
    print("\n📊 DJ CATEGORIES:")
    print("-" * 40)
    total = sum(dj_category_counter.values())
    for cat, count in dj_category_counter.most_common():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:15} {count:4} ({pct:5.1f}%) {bar}")

    # Top genres
    print("\n🎵 TOP GENRES:")
    print("-" * 40)
    for genre, count in genre_counter.most_common(15):
        pct = count / total * 100
        print(f"  {genre:35} {count:4} ({pct:5.1f}%)")

    # Mood distribution
    print("\n🎭 TOP MOODS:")
    print("-" * 40)
    for mood, count in mood_counter.most_common(10):
        print(f"  {mood:20} {count:4}")

    # Gap analysis
    print("\n⚠️  GENRE GAPS (potentially missing):")
    print("-" * 40)

    # Define expected DJ genres
    expected_categories = {
        "Techno": 15,
        "House": 15,
        "Trance": 5,
        "Bass": 10,
        "Hip-Hop": 10,
        "Disco/Funk": 10,
        "Ambient": 5,
        "Electronic": 10,
    }

    gaps = []
    for cat, expected_pct in expected_categories.items():
        actual_pct = dj_category_counter.get(cat, 0) / total * 100 if total > 0 else 0
        if actual_pct < expected_pct / 2:  # Less than half expected
            gaps.append((cat, actual_pct, expected_pct))

    if gaps:
        for cat, actual, expected in gaps:
            print(f"  🔴 {cat}: {actual:.1f}% (expected ~{expected}%)")
    else:
        print("  ✅ No major gaps detected!")

    # Categories with low diversity
    print("\n📉 LOW DIVERSITY (only 1-2 subgenres):")
    print("-" * 40)
    genre_by_category = defaultdict(set)
    for r in results:
        genre_by_category[r["detected_category"]].add(r["detected_genre"])

    for cat, genres in genre_by_category.items():
        if len(genres) <= 2 and dj_category_counter[cat] > 5:
            print(f"  {cat}: only {len(genres)} subgenres ({', '.join(list(genres)[:3])})")

    # Folder structure analysis
    print("\n📁 FOLDER STRUCTURE:")
    print("-" * 40)
    for folder, count in folder_genre_counter.most_common():
        print(f"  {folder:20} {count:4} tracks")

    # MISMATCHES - tracks in wrong folders
    print("\n" + "=" * 60)
    print("🔀 MISMATCHED TRACKS (possibly in wrong folder)")
    print("=" * 60)

    if mismatches:
        print(f"\nFound {len(mismatches)} potential mismatches:\n")

        # Group by folder
        by_folder = defaultdict(list)
        for m in mismatches:
            by_folder[m["folder"]].append(m)

        for folder, tracks in sorted(by_folder.items()):
            print(f"\n📁 {folder}/ ({len(tracks)} mismatches):")
            for t in tracks[:10]:  # Show max 10 per folder
                filename = Path(t["path"]).name
                print(f"  ❌ {filename[:50]:50}")
                print(f"     → Detected: {t['detected']} / {t['genre']} ({t['confidence']:.0%})")

            if len(tracks) > 10:
                print(f"  ... and {len(tracks) - 10} more")
    else:
        print("\n✅ No mismatches found! All tracks seem to be in correct folders.")

    # Save results
    output_path = Path(args.output) if args.output else project_root / "results" / "library_genre_analysis.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n💾 Full results saved to: {output_path}")

    # Save mismatches separately
    if mismatches:
        mismatches_path = output_path.parent / "genre_mismatches.csv"
        pd.DataFrame(mismatches).to_csv(mismatches_path, index=False)
        print(f"💾 Mismatches saved to: {mismatches_path}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)

    if gaps:
        print("\nConsider adding more tracks in these categories:")
        for cat, actual, expected in gaps:
            print(f"  • {cat} - currently underrepresented")

    # Check for balance
    top_cat = dj_category_counter.most_common(1)[0] if dj_category_counter else None
    if top_cat and top_cat[1] / total > 0.5:
        print(f"\n⚠️  Library is heavily skewed towards {top_cat[0]} ({top_cat[1]/total*100:.0f}%)")
        print("   Consider diversifying for more versatile sets")

    return 0


if __name__ == "__main__":
    sys.exit(main())