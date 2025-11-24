#!/usr/bin/env python3
"""Test BPM accuracy against known values."""

import sys
from src.audio import AudioLoader, FeatureExtractor
from src.utils import setup_logger

def test_bpm(file_path: str, expected_bpm: float):
    """Test BPM detection against expected value."""
    print(f"\nTesting: {file_path}")
    print(f"Expected BPM: {expected_bpm}")
    
    loader = AudioLoader()
    extractor = FeatureExtractor()
    
    y, sr = loader.load(file_path)
    features = extractor.extract(y, sr)
    
    detected = features.tempo
    confidence = features.tempo_confidence
    
    # Check if detected BPM is close to expected (within 2 BPM or octave)
    error = abs(detected - expected_bpm)
    error_half = abs(detected/2 - expected_bpm)
    error_double = abs(detected*2 - expected_bpm)
    
    min_error = min(error, error_half, error_double)
    
    if min_error <= 2:
        result = "✓ CORRECT"
    elif min_error <= 5:
        result = "⚠ CLOSE"
    else:
        result = "✗ WRONG"
    
    print(f"Detected BPM: {detected:.1f} (confidence: {confidence:.1%})")
    print(f"Error: {min_error:.1f} BPM - {result}")
    
    return min_error <= 5

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python test_bpm_accuracy.py <file.mp3> <expected_bpm>")
        print("Example: python test_bpm_accuracy.py track.mp3 128")
        sys.exit(1)
    
    file_path = sys.argv[1]
    expected_bpm = float(sys.argv[2])
    
    test_bpm(file_path, expected_bpm)
