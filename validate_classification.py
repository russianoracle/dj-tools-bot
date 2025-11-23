#!/usr/bin/env python3
"""
Validation script for energy zone classification.
Analyzes tracks and shows detailed feature breakdown.
"""

import sys
from pathlib import Path
from src.audio import AudioLoader, FeatureExtractor
from src.classification import EnergyZoneClassifier
from src.utils import setup_logger

logger = setup_logger()


def analyze_track(file_path: str, show_details: bool = True):
    """
    Analyze a single track and show detailed classification breakdown.

    Args:
        file_path: Path to audio file
        show_details: Whether to show detailed feature breakdown
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {Path(file_path).name}")
    print(f"{'='*70}")

    try:
        # Load audio
        loader = AudioLoader()
        y, sr = loader.load(file_path)
        print(f"✓ Loaded audio: {len(y)/sr:.1f} seconds at {sr} Hz")

        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract(y, sr)

        # Classify
        classifier = EnergyZoneClassifier()
        result = classifier.classify(features)

        # Display results
        print(f"\n{result.zone.emoji} CLASSIFICATION: {result.zone.display_name}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Method: {result.method}")

        if show_details:
            print(f"\n--- TEMPORAL FEATURES ---")
            conf_indicator = "✓" if features.tempo_confidence >= 0.8 else "⚠" if features.tempo_confidence >= 0.6 else "✗"
            print(f"Tempo:              {features.tempo:>7.1f} BPM  {conf_indicator} (confidence: {features.tempo_confidence:.1%})")
            if features.tempo_confidence < 0.8:
                print(f"                    WARNING: Low tempo confidence! Results may be unreliable.")
            print(f"Zero Crossing Rate: {features.zero_crossing_rate:>7.4f}")
            print(f"Low Energy:         {features.low_energy:>7.1%}")
            print(f"RMS Energy:         {features.rms_energy:>7.4f}")

            print(f"\n--- SPECTRAL FEATURES ---")
            print(f"Spectral Rolloff:   {features.spectral_rolloff:>7.1f} Hz")
            print(f"Brightness:         {features.brightness:>7.1%}")
            print(f"Spectral Centroid:  {features.spectral_centroid:>7.1f} Hz")

            print(f"\n--- DYNAMICS ---")
            print(f"Energy Variance:    {features.energy_variance:>7.4f}")
            print(f"Drop Intensity:     {features.drop_intensity:>7.4f}")

            print(f"\n--- CLASSIFICATION LOGIC ---")
            _show_classification_reasoning(features, result)

        return result

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _show_classification_reasoning(features, result):
    """Show why a track was classified into a specific zone."""
    print("\nWhy this classification?")

    # Check tempo
    if features.tempo < 110:
        print(f"  • Low tempo ({features.tempo:.1f} BPM) → favors YELLOW")
    elif features.tempo > 128:
        print(f"  • High tempo ({features.tempo:.1f} BPM) → favors PURPLE")
    else:
        print(f"  • Medium tempo ({features.tempo:.1f} BPM) → favors GREEN")

    # Check energy variance
    if features.energy_variance < 0.15:
        print(f"  • Low energy variance ({features.energy_variance:.3f}) → favors YELLOW")
    elif features.energy_variance > 0.3:
        print(f"  • High energy variance ({features.energy_variance:.3f}) → favors PURPLE")
    else:
        print(f"  • Medium energy variance ({features.energy_variance:.3f}) → favors GREEN")

    # Check drops
    if features.drop_intensity > 0.5:
        print(f"  • Strong drops ({features.drop_intensity:.3f}) → favors PURPLE")
    elif features.drop_intensity < 0.2:
        print(f"  • Minimal drops ({features.drop_intensity:.3f}) → favors YELLOW")

    # Check brightness
    if features.brightness > 0.5:
        print(f"  • High brightness ({features.brightness:.1%}) → favors PURPLE")
    elif features.brightness < 0.3:
        print(f"  • Low brightness ({features.brightness:.1%}) → favors YELLOW")


def validate_folder(folder_path: str, expected_zone: str = None):
    """
    Validate all tracks in a folder.

    Args:
        folder_path: Path to folder with audio files
        expected_zone: Expected zone (yellow/green/purple) for validation
    """
    from src.audio.loader import AudioLoader

    loader = AudioLoader()
    audio_files = []

    folder = Path(folder_path)
    for ext in AudioLoader.SUPPORTED_FORMATS:
        audio_files.extend(folder.glob(f"**/*{ext}"))

    if not audio_files:
        print(f"No audio files found in {folder_path}")
        return

    print(f"\nFound {len(audio_files)} audio file(s)")
    print(f"Expected zone: {expected_zone.upper() if expected_zone else 'ANY'}\n")

    results = []
    for file_path in audio_files:
        result = analyze_track(str(file_path), show_details=False)
        if result:
            results.append((file_path.name, result))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    zone_counts = {'yellow': 0, 'green': 0, 'purple': 0, 'uncertain': 0}
    correct = 0

    for filename, result in results:
        zone_counts[result.zone.value] += 1
        if expected_zone and result.zone.value == expected_zone.lower():
            correct += 1
            status = "✓"
        elif expected_zone:
            status = "✗"
        else:
            status = " "

        print(f"{status} {result.zone.emoji} {filename:<40} {result.confidence:>6.1%}")

    print(f"\nDistribution:")
    print(f"  🟨 Yellow:  {zone_counts['yellow']:>3} ({zone_counts['yellow']/len(results)*100:.1f}%)")
    print(f"  🟩 Green:   {zone_counts['green']:>3} ({zone_counts['green']/len(results)*100:.1f}%)")
    print(f"  🟪 Purple:  {zone_counts['purple']:>3} ({zone_counts['purple']/len(results)*100:.1f}%)")
    print(f"  ❓ Uncertain: {zone_counts['uncertain']:>3} ({zone_counts['uncertain']/len(results)*100:.1f}%)")

    if expected_zone:
        accuracy = correct / len(results) * 100
        print(f"\nAccuracy: {correct}/{len(results)} ({accuracy:.1f}%)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:  python validate_classification.py <file.mp3>")
        print("  Folder:       python validate_classification.py <folder> [expected_zone]")
        print("")
        print("Examples:")
        print("  python validate_classification.py track.mp3")
        print("  python validate_classification.py ./yellow_tracks yellow")
        print("  python validate_classification.py ./purple_bangers purple")
        sys.exit(1)

    path = sys.argv[1]

    if Path(path).is_file():
        analyze_track(path, show_details=True)
    elif Path(path).is_dir():
        expected = sys.argv[2] if len(sys.argv) > 2 else None
        validate_folder(path, expected)
    else:
        print(f"Error: {path} is not a file or directory")
        sys.exit(1)
