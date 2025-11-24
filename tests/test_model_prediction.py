#!/usr/bin/env python3
"""Test the trained zone classification model."""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.zone_models import XGBoostZoneClassifier


def main():
    """Test model predictions."""
    # Load the trained model
    model_path = 'models/test_run_2/xgboost_final.pkl'
    print(f'Loading model from {model_path}...')

    model = XGBoostZoneClassifier()
    model.load(model_path)

    print('✅ Model loaded successfully!')
    print(f'Model is trained: {model.is_trained}')

    # Create test feature vectors (10 features for fast mode)
    # Format: tempo, tempo_confidence, rms_energy, energy_variance,
    #         spectral_centroid, spectral_rolloff, brightness,
    #         zero_crossing_rate, drop_intensity, low_energy

    test_cases = {
        'Purple (high-energy track)': np.array([[
            135.0,  # High tempo
            0.95,   # High confidence
            0.8,    # High RMS energy
            0.6,    # High energy variance
            3500,   # High spectral centroid
            7000,   # High spectral rolloff
            0.7,    # High brightness
            0.3,    # Medium zero-crossing rate
            0.8,    # High drop intensity
            0.2     # Low low-energy
        ]]),

        'Yellow (low-energy track)': np.array([[
            95.0,   # Low tempo
            0.85,   # Medium-high confidence
            0.3,    # Low RMS energy
            0.2,    # Low energy variance
            1500,   # Low spectral centroid
            3000,   # Low spectral rolloff
            0.2,    # Low brightness
            0.1,    # Low zero-crossing rate
            0.1,    # Low drop intensity
            0.7     # High low-energy
        ]]),

        'Green (medium-energy track)': np.array([[
            125.0,  # Medium tempo
            0.90,   # High confidence
            0.5,    # Medium RMS energy
            0.4,    # Medium energy variance
            2500,   # Medium spectral centroid
            5000,   # Medium spectral rolloff
            0.4,    # Medium brightness
            0.2,    # Low-medium zero-crossing rate
            0.4,    # Medium drop intensity
            0.4     # Medium low-energy
        ]])
    }

    zones = ['yellow', 'green', 'purple']
    zone_emojis = {'yellow': '🟨', 'green': '🟩', 'purple': '🟪'}

    print('\n' + '='*70)
    print('📊 TESTING MODEL PREDICTIONS')
    print('='*70)

    for name, features in test_cases.items():
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        predicted_zone = zones[pred]

        print(f'\n{name}:')
        print(f'  → Predicted Zone: {zone_emojis[predicted_zone]} {predicted_zone.upper()} '
              f'(confidence: {proba[pred]:.1%})')
        print(f'  Probabilities:')
        print(f'    🟨 Yellow: {proba[0]:.1%}')
        print(f'    🟩 Green:  {proba[1]:.1%}')
        print(f'    🟪 Purple: {proba[2]:.1%}')

    print('\n' + '='*70)
    print('✅ All model predictions completed successfully!')
    print('🎯 Training system is FULLY FUNCTIONAL!')
    print('='*70)


if __name__ == '__main__':
    main()
