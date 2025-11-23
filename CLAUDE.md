# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a desktop application for DJs that analyzes music tracks and automatically classifies them into three energy zones based on audio features. The classification is saved to the track's metadata for use in DJ software.

### Three Energy Zones

- **🟨 YELLOW** (rest zone): Low-energy, calm tracks for audience rest periods
  - Low tempo (<110 BPM), minimal dynamics, low brightness

- **🟩 GREEN** (transitional): Medium-energy tracks bridging yellow and purple zones
  - Medium tempo, gradual energy build-up

- **🟪 PURPLE** (hits/energy): High-energy tracks with clear build-up/drop structure
  - High tempo (>128 BPM), energetic drops, high energy variance

## Tech Stack

- **Python 3.8+**
- **librosa** - audio feature extraction
- **PyQt5** - GUI framework
- **mutagen** - audio metadata read/write
- **scikit-learn** - ML classification

Supported formats: MP3, WAV, FLAC, M4A, MP4

## Implemented Architecture

```
mood-classifier/
├── main.py                     # Application entry point (CLI + GUI launcher)
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── loader.py           # AudioLoader - file loading & validation
│   │   └── extractors.py       # FeatureExtractor - all audio features
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── classifier.py       # EnergyZoneClassifier - main classifier
│   │   └── rules.py            # RuleBasedClassifier - scoring logic
│   ├── metadata/
│   │   ├── __init__.py
│   │   ├── reader.py           # MetadataReader - read zone from files
│   │   └── writer.py           # MetadataWriter - write to MP3/M4A/FLAC/WAV
│   ├── gui/
│   │   ├── __init__.py
│   │   └── main_window.py      # MainWindow - PyQt5 GUI with drag-drop
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Config - YAML config management
│       └── logger.py           # Logging setup
├── config/
│   └── default_config.yaml     # Default configuration
├── models/                     # ML models (optional)
├── tests/                      # Unit tests
└── requirements.txt
```

## Audio Feature Extraction

The classifier uses these audio features (from research paper):

### Temporal Features
- **Tempo (BPM)** - beats per minute
- **Zero-Crossing Rate** - signal noisiness indicator
- **Low-Energy** - percentage of low-energy segments
- **RMS Energy** - overall signal energy

### Spectral Features
- **Spectral Rolloff** - frequency containing 85% of energy
- **Brightness** - high-frequency energy (>3kHz)
- **Spectral Centroid** - spectral center of mass

### MFCC & Dynamics
- **MFCC 1-5** - first 5 mel-frequency cepstral coefficients (mean/std)
- **Energy Variance** - energy changes over time
- **Drop Detection** - sudden energy changes (for purple zone identification)

## Classification Logic

Uses hybrid rule-based + ML approach:

**Purple tracks criteria:**
- High tempo (>128 BPM)
- High energy variance
- Pronounced drops (sharp energy changes)
- High spectral centroid

**Yellow tracks criteria:**
- Low tempo (<110 BPM)
- Low energy variance
- Minimal sharp transitions
- Low brightness

**Green tracks:** intermediate values between yellow and purple

## Metadata Storage

Classification results are saved to audio file metadata:
- Zone stored in **Comment** or **Grouping** field
- Color label in **Genre** field
- Custom **EnergyZone** field if format supports it

## Performance Targets

- Analysis time: <30 seconds for 3-5 minute track
- Classification accuracy: >80% vs expert assessment
- Batch processing without memory leaks
- Directory selection for batch operations

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run GUI application
python main.py
# or explicitly:
python main.py --gui

# Analyze single file (CLI)
python main.py --file path/to/track.mp3

# Analyze and write metadata
python main.py --file track.mp3 --write-metadata

# Batch processing
python main.py --batch /path/to/music/folder

# Batch with metadata writing and CSV export
python main.py --batch /path/to/music --write-metadata --export-csv results.csv

# Force re-analysis of already classified files
python main.py --batch /path/to/music --force

# Use custom configuration
python main.py --config custom_config.yaml

# Run tests
pytest tests/
pytest tests/ --cov=src  # with coverage
```

## GUI Features

- Drag & drop track loading
- Progress bar for analysis
- Color-coded classification visualization
- Feature inspection view
- Batch file processing with directory picker

## Key Implementation Details

### Component Interactions

1. **Audio Processing Pipeline:**
   - `AudioLoader.load()` → returns (audio_array, sample_rate)
   - `FeatureExtractor.extract()` → returns `AudioFeatures` dataclass
   - Features include 16 values: tempo, temporal features (3), spectral features (3), MFCC stats (10)

2. **Classification Flow:**
   - `RuleBasedClassifier.classify()` scores each zone (yellow/green/purple)
   - Each zone score is 0.0-1.0 based on feature thresholds
   - Highest score wins; if confidence < 0.6, flagged as "uncertain"
   - Optional ML fallback if rule-based confidence is low

3. **Metadata Handling:**
   - Format-specific writers for MP3 (ID3), M4A (MP4), FLAC (Vorbis), WAV
   - Always creates backup with `.backup` suffix before modification
   - Stores zone in Comment field, Grouping field, and custom ENERGYZONE field

### Configuration System

- `get_config()` returns singleton Config instance
- Access nested values with dot notation: `config.get('audio.sample_rate')`
- All thresholds tunable in `config/default_config.yaml`
- Paths automatically expand `~` to home directory

### Feature Extraction Notes

- Default sample rate: 22050 Hz (good balance of speed/quality)
- Tempo extraction uses librosa's beat tracking (can fail on ambient tracks)
- Drop detection analyzes energy derivative (90th percentile threshold)
- MFCC uses first 5 coefficients as per research spec
- All features normalized/scaled for classification

### GUI Architecture

- `ProcessingThread` runs analysis in background (QThread)
- Emits signals: `progress`, `file_processed`, `error`, `finished`
- Main window connects to signals to update UI
- Drag-drop accepts files and directories recursively
- Table rows color-coded by zone (yellow/green/purple backgrounds)

### Testing Strategy

When writing tests:

- Mock librosa calls (slow, require audio files)
- Test feature extraction with known synthetic signals
- Validate classification thresholds with edge cases
- Test metadata writing on temporary file copies
- Fixtures in `tests/fixtures/` for sample audio

## Key Development Notes

- The specification document (Задание.md) contains the full Russian-language requirements
- Classification is based on research paper methodology using 16 audio features
- Focus on DJ workflow: quick analysis, reliable categorization, seamless metadata integration
- Validation should use test set of 50 manually-labeled tracks with confusion matrix
- Sample rate of 22050 Hz chosen for 2x speed improvement vs 44100 Hz with minimal accuracy loss

## Common Modification Patterns

### Adding a New Feature

1. Add extraction method in `src/audio/extractors.py`
2. Add field to `AudioFeatures` dataclass
3. Update `to_vector()` method to include new feature
4. Update classification rules in `src/classification/rules.py` to use it
5. Add to config thresholds if needed

### Adjusting Classification Thresholds

Edit `config/default_config.yaml`:

```yaml
classification:
  yellow_max_bpm: 110      # Adjust BPM threshold for yellow
  purple_min_bpm: 128      # Adjust BPM threshold for purple
  # etc.
```

### Custom ML Model Integration

```python
# Train model
classifier = EnergyZoneClassifier()
classifier.train_model(X_features, y_labels)
classifier.save_model('models/my_model.pkl')

# Use in application
classifier = EnergyZoneClassifier(model_path='models/my_model.pkl')
```
