"""Test DJ-specific features extraction."""

from src.training.zone_features import ZoneFeatureExtractor
from pathlib import Path

# Initialize extractor (fast mode - no embeddings, no music_emotion)
print("Initializing feature extractor...")
extractor = ZoneFeatureExtractor(
    sample_rate=22050,
    use_gpu=False,
    use_embeddings=False,
    use_music_emotion=False
)

# Test on first track from dataset
test_file = "/Users/artemgusarov/Music/DJ Library/Tech House/1-not-without-friends-without-u-Z869Y3.flac"

print(f"\nExtracting features from: {Path(test_file).name}")
print("=" * 60)

features = extractor.extract(test_file)

# Print all features
print("\n### BASIC LIBROSA FEATURES (19) ###")
print(f"tempo: {features.tempo:.2f} BPM")
print(f"zero_crossing_rate: {features.zero_crossing_rate:.4f}")
print(f"rms_energy: {features.rms_energy:.4f}")
print(f"energy_variance: {features.energy_variance:.4f}")
print(f"spectral_rolloff: {features.spectral_rolloff:.2f}")
print(f"spectral_centroid: {features.spectral_centroid:.2f}")
print(f"brightness: {features.brightness:.4f}")
print(f"drop_strength: {features.drop_strength:.4f}")
# ... (skipping individual MFCCs for brevity)

print("\n### MUSIC EMOTION (2) ###")
print(f"arousal: {features.arousal:.4f}")
print(f"valence: {features.valence:.4f}")

print("\n### BUILD-UP DETECTION (3) - NEW ###")
print(f"energy_slope: {features.energy_slope:.4f} (GREEN zone indicator)")
print(f"energy_buildup_ratio: {features.energy_buildup_ratio:.4f}")
print(f"onset_acceleration: {features.onset_acceleration:.4f}")

print("\n### DRIVE ENHANCEMENT (2) - NEW ###")
print(f"drop_frequency: {features.drop_frequency:.2f} drops/min (PURPLE zone indicator)")
print(f"peak_energy_ratio: {features.peak_energy_ratio:.2f}")

print("\n### EUPHORIA INDICATORS (2) - NEW ###")
print(f"rhythmic_regularity: {features.rhythmic_regularity:.4f} (PURPLE zone indicator)")
print(f"harmonic_complexity: {features.harmonic_complexity:.4f}")

print("\n### CLIMAX STRUCTURE (3) - NEW ###")
print(f"has_climax: {features.has_climax:.1f} (binary: {bool(features.has_climax)})")
print(f"climax_position: {features.climax_position:.2f} (0=start, 1=end)")
print(f"dynamic_range: {features.dynamic_range:.2f} dB")

# Convert to vector
print("\n### FEATURE VECTOR ###")
vec = features.to_vector(include_embeddings=False)
print(f"Total features: {len(vec)}")
print(f"Expected: 51 (19 basic + 2 emotion + 4 temporal + 14 spectral + 2 hp + 3 buildup + 2 drive + 2 euphoria + 3 climax)")
print(f"Vector shape: {vec.shape}")
print(f"First 10 values: {vec[:10]}")

print("\n✓ Feature extraction test completed!")
