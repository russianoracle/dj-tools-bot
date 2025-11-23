#!/usr/bin/env python3
"""Batch test BPM accuracy."""

import sys
from pathlib import Path
from src.audio import AudioLoader, FeatureExtractor

def test_tracks(tracks_with_bpm):
    """Test multiple tracks."""
    loader = AudioLoader()
    extractor = FeatureExtractor()
    
    results = []
    
    print("="*70)
    print("BPM ACCURACY TEST")
    print("="*70)
    
    for file_path, expected_bpm in tracks_with_bpm:
        print(f"\nFile: {Path(file_path).name}")
        print(f"Expected: {expected_bpm} BPM")
        
        try:
            y, sr = loader.load(file_path)
            features = extractor.extract(y, sr)
            
            detected = features.tempo
            confidence = features.tempo_confidence
            
            # Check octave errors
            error = abs(detected - expected_bpm)
            error_half = abs(detected/2 - expected_bpm)
            error_double = abs(detected*2 - expected_bpm)
            
            min_error = min(error, error_half, error_double)
            
            # Determine which octave was used
            if error == min_error:
                octave = "correct"
            elif error_half == min_error:
                octave = "2x too high"
            else:
                octave = "2x too low"
            
            if min_error <= 2:
                status = "✓"
            elif min_error <= 5:
                status = "⚠"
            else:
                status = "✗"
            
            print(f"Detected: {detected:.1f} BPM (conf: {confidence:.0%}) - {octave}")
            print(f"Error: {min_error:.1f} BPM {status}")
            
            results.append({
                'file': Path(file_path).name,
                'expected': expected_bpm,
                'detected': detected,
                'error': min_error,
                'octave': octave,
                'correct': min_error <= 2
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'file': Path(file_path).name,
                'expected': expected_bpm,
                'detected': 0,
                'error': 999,
                'octave': 'failed',
                'correct': False
            })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    correct = sum(1 for r in results if r['correct'])
    octave_errors = sum(1 for r in results if r['octave'] in ['2x too high', '2x too low'])
    
    print(f"Total tracks: {len(results)}")
    print(f"Correct BPM (±2): {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    print(f"Octave errors: {octave_errors}/{len(results)} ({octave_errors/len(results)*100:.1f}%)")
    print(f"Average error: {sum(r['error'] for r in results)/len(results):.1f} BPM")

if __name__ == '__main__':
    # You can add tracks here: (file_path, expected_bpm)
    tracks = [
        ('/Users/artemgusarov/Music/DJ Library/Tech House/3-IDEMI-Ain-t-Nothin--Ordinary-PQ8NV2.flac', 132),
    ]
    
    test_tracks(tracks)
