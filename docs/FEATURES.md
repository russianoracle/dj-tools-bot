# Audio Features Documentation

## Overview

Two extraction modes are available:
1. **Full extraction** - computes all features including DJ-specific and drop detection (slow, ~3-5s per track)
2. **Frame-based extraction** - outputs frame-level features compatible with DEAM format (fast, ~0.5s per track)

---

## Full Extraction Features (ZoneFeatures)

Total: **55 features** (without embeddings)

### Basic Librosa Features (19)

| # | Feature | Calculation | Description |
|---|---------|-------------|-------------|
| 1 | `tempo` | `librosa.beat.beat_track()` | BPM (beats per minute) |
| 2 | `zero_crossing_rate` | `librosa.feature.zero_crossing_rate().mean()` | Signal noisiness indicator |
| 3 | `rms_energy` | `librosa.feature.rms().mean()` | Root mean square energy |
| 4 | `spectral_centroid` | `librosa.feature.spectral_centroid().mean()` | Spectral center of mass (Hz) |
| 5 | `spectral_rolloff` | `librosa.feature.spectral_rolloff().mean()` | 85% energy frequency (Hz) |
| 6 | `energy_variance` | `np.std(rms)` | Energy variation over time |
| 7-16 | `mfcc_1_mean` ... `mfcc_5_std` | `librosa.feature.mfcc(n_mfcc=5)` | First 5 MFCCs (mean + std) |
| 17 | `low_energy` | `sum(rms < mean_rms) / len(rms)` | Fraction of low-energy frames |
| 18 | `brightness` | `high_freq_energy / total_energy` | Energy above 3kHz |
| 19 | `drop_strength` | `np.percentile(abs(diff(rms)), 90)` | 90th percentile of energy derivative |

### Music Emotion Features (2)

| # | Feature | Calculation | Description |
|---|---------|-------------|-------------|
| 20 | `arousal` | Pre-trained model or 0.0 | Energy level (-1 to +1) |
| 21 | `valence` | Pre-trained model or 0.0 | Mood (-1 to +1) |

### Temporal Features (4)

| # | Feature | Calculation | Description |
|---|---------|-------------|-------------|
| 22 | `onset_strength_mean` | `librosa.onset.onset_strength().mean()` | Average onset strength |
| 23 | `onset_strength_std` | `librosa.onset.onset_strength().std()` | Onset strength variation |
| 24 | `beat_strength` | Energy at beat frames | Average energy on beats |
| 25 | `tempo_stability` | `1 / (std(tempogram_peaks) + 1)` | How stable the tempo is |

### Spectral Contrast Features (14)

| # | Feature | Calculation | Description |
|---|---------|-------------|-------------|
| 26-32 | `spectral_contrast_mean[0-6]` | `librosa.feature.spectral_contrast()` | 7 frequency bands, mean |
| 33-39 | `spectral_contrast_std[0-6]` | Same, std | 7 frequency bands, std |

### Harmonic-Percussive Features (2)

| # | Feature | Calculation | Description |
|---|---------|-------------|-------------|
| 40 | `harmonic_ratio` | `sum(y_harmonic^2) / total` | Harmonic content ratio |
| 41 | `percussive_ratio` | `sum(y_percussive^2) / total` | Percussive content ratio |

### Build-up Detection Features (3)

| # | Feature | Calculation | Description |
|---|---------|-------------|-------------|
| 42 | `energy_slope` | Linear regression of RMS | Energy trend (-1 to +1) |
| 43 | `energy_buildup_ratio` | `sum(diff(rms) > 0) / len` | Fraction of rising energy |
| 44 | `onset_acceleration` | `mean(abs(diff(diff(onset))))` | Onset second derivative |

### Drive Enhancement Features (2)

| # | Feature | Calculation | Description |
|---|---------|-------------|-------------|
| 45 | `drop_frequency` | Count sharp drops / duration | Drops per minute |
| 46 | `peak_energy_ratio` | `max(rms) / mean(rms)` | Peak-to-average energy |

### Improved Drop Detection Features (4) - NEW

| # | Feature | Calculation | Description |
|---|---------|-------------|-------------|
| 47 | `drop_contrast_mean` | `mean((peak - valley) / mean_energy)` | Avg breakdown→drop contrast |
| 48 | `drop_contrast_max` | `max((peak - valley) / mean_energy)` | Max breakdown→drop contrast |
| 49 | `drop_count` | Number of breakdown→drop patterns | True drop count |
| 50 | `drop_intensity` | `mean(drop_peak_energy) / mean_energy` | Relative drop peak energy |

**Drop Detection Algorithm:**
1. Smooth RMS with median filter (0.5s window)
2. Find valleys (local minima < mean - 0.3*std) = breakdowns
3. Find peaks (local maxima > mean + 0.3*std) = potential drops
4. Match each valley to following peak within 8 seconds
5. Calculate contrast = (peak_energy - valley_energy) / mean_energy
6. Only count if contrast > 0.5 (50% of mean energy)

### Euphoria Indicators (2)

| # | Feature | Calculation | Description |
|---|---------|-------------|-------------|
| 51 | `rhythmic_regularity` | `1 / (1 + cv(beat_intervals))` | Beat consistency (0-1) |
| 52 | `harmonic_complexity` | Mean entropy of chromagram | Harmonic richness |

### Climax Structure Features (3)

| # | Feature | Calculation | Description |
|---|---------|-------------|-------------|
| 53 | `has_climax` | 1 if max > mean + 2*std | Clear peak exists |
| 54 | `climax_position` | `argmax(rms) / len(rms)` | Position of peak (0-1) |
| 55 | `dynamic_range` | `max(rms_db) - min(rms_db)` | Range in dB |

---

## Frame-Based Extraction (DEAM-compatible)

**Frame size:** 0.5 seconds (matches DEAM openSMILE config)
**Output:** DataFrame with columns per frame

### Frame Features (15 per frame)

| # | Feature | librosa equivalent | DEAM column |
|---|---------|-------------------|-------------|
| 1 | `rms_energy` | `rms[frame]` | `pcm_RMSenergy_sma_amean` |
| 2 | `rms_energy_delta` | `diff(rms)[frame]` | `pcm_RMSenergy_sma_de_amean` |
| 3 | `zcr` | `zcr[frame]` | `pcm_zcr_sma_amean` |
| 4 | `spectral_centroid` | `centroid[frame]` | `pcm_fftMag_spectralCentroid_sma_amean` |
| 5 | `spectral_rolloff` | `rolloff[frame]` | `pcm_fftMag_spectralRollOff90.0_sma_amean` |
| 6-10 | `mfcc_1` ... `mfcc_5` | `mfcc[0-4, frame]` | `pcm_fftMag_mfcc_sma[1-5]_amean` |
| 11-15 | `mfcc_1_delta` ... `mfcc_5_delta` | `diff(mfcc)[0-4, frame]` | Computed delta |

### Aggregation to Track-Level

```python
# From frames to single features
track_features = {
    'rms_energy': frames['rms_energy'].mean(),
    'energy_variance': frames['rms_energy'].std(),
    'drop_intensity': np.percentile(abs(frames['rms_energy_delta']), 90),
    'low_energy': (frames['rms_energy'] < frames['rms_energy'].mean()).mean(),
    # ... etc
}
```

---

## Usage Examples

### Full Extraction (slow, all features)
```python
from src.training.zone_features import ZoneFeatureExtractor

extractor = ZoneFeatureExtractor(use_gpu=True)
features = extractor.extract('/path/to/track.mp3')

print(features.drop_contrast_mean)  # New drop detection
print(features.tempo)
```

### Frame-Based Extraction (fast, DEAM-compatible)
```python
from src.training.zone_features import ZoneFeatureExtractor

extractor = ZoneFeatureExtractor(use_gpu=True)
frames_df = extractor.extract_frames('/path/to/track.mp3', frame_size=0.5)

# Each row = 0.5 second frame
print(frames_df.shape)  # (num_frames, 15)
print(frames_df.columns)
```

---

## Zone Classification Criteria (User-defined)

Based on user feedback, zone classification is based on:

| Zone | Criteria |
|------|----------|
| **YELLOW** | Low energy, calm, no prominent drops |
| **GREEN** | Transitional, moderate energy, gradual build-ups |
| **PURPLE** | High drop energy, high contrast breakdown→drop, frequent drops, OR no drops but constant high energy |

Key features for zone separation:
- `drop_contrast_mean` / `drop_contrast_max` - higher for GREEN/PURPLE with drops
- `drop_count` - more drops = more likely PURPLE
- `drop_intensity` - stronger drops = PURPLE
- `rms_energy` + low `drop_count` = "driving" PURPLE (no drops but high energy)
