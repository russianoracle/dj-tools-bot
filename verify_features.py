"""Quick verification of feature count."""

from src.training.zone_features import ZoneFeatureExtractor
import numpy as np

# Initialize extractor exactly as in training
extractor = ZoneFeatureExtractor(
    sample_rate=22050,
    use_gpu=True,
    use_embeddings=False,  # Same as --no-embeddings flag
    use_music_emotion=False
)

# Extract from one track
test_file = "/Users/artemgusarov/Music/DJ Library/Tech House/1-not-without-friends-without-u-Z869Y3.flac"
print(f"Extracting features from test track...")

features = extractor.extract(test_file)
vec = features.to_vector(include_embeddings=False)

print(f"\n{'='*60}")
print(f"FEATURE VERIFICATION RESULTS")
print(f"{'='*60}")
print(f"Feature vector shape: {vec.shape}")
print(f"Total features: {len(vec)}")
print(f"Expected: 51 (with DJ-specific features)")
print(f"\nFirst 15 values: {vec[:15]}")
print(f"\nDJ-specific features:")
print(f"  energy_slope: {features.energy_slope:.4f}")
print(f"  drop_frequency: {features.drop_frequency:.2f}")
print(f"  rhythmic_regularity: {features.rhythmic_regularity:.4f}")
print(f"  has_climax: {features.has_climax:.1f}")
print(f"{'='*60}")

if len(vec) == 51:
    print("\n✓ CORRECT! Using full 51-feature extractor")
elif len(vec) == 10:
    print("\n✗ ERROR! Still using fast 10-feature extractor")
else:
    print(f"\n? UNEXPECTED! Got {len(vec)} features")
