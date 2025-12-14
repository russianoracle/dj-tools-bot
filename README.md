# Mood Classifier - DJ Track Energy Zone Analyzer

A desktop application for DJs that automatically analyzes music tracks and classifies them into three energy zones based on audio features. Classification results are saved to track metadata for seamless integration with DJ software.

## 🎵 Energy Zones

### 🟨 Yellow Zone (Rest)
Low-energy, calm tracks for audience rest periods
- Low tempo (<110 BPM)
- Minimal dynamics
- Low brightness and energy variance

### 🟩 Green Zone (Transition)
Medium-energy tracks that bridge yellow and purple zones
- Medium tempo (110-128 BPM)
- Gradual energy build-up
- Balanced characteristics

### 🟪 Purple Zone (Energy/Hits)
High-energy tracks with pronounced build-ups and drops
- High tempo (>128 BPM)
- Energetic drops
- High energy variance and spectral characteristics

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Platform-Specific Notes

**macOS/Linux:**
```bash
# Install ffmpeg for additional audio format support
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg
```

**Windows:**
Download and install ffmpeg from https://ffmpeg.org/

## 📖 Usage

### Graphical Interface (GUI)

Launch the GUI application:

```bash
python main.py --gui
```

Or simply:

```bash
python main.py
```

#### GUI Features:
- **Drag & Drop**: Drop audio files or folders directly into the window
- **Batch Processing**: Analyze multiple tracks simultaneously
- **Visual Results**: Color-coded classification with confidence scores
- **Metadata Writing**: Save results directly to file metadata
- **CSV Export**: Export analysis results for external use
- **Detailed View**: Right-click tracks for detailed feature analysis

### Command-Line Interface (CLI)

#### Analyze Single File

```bash
python main.py --file /path/to/track.mp3
```

Display analysis without writing metadata:

```bash
python main.py -f track.mp3
```

Analyze and write results to metadata:

```bash
python main.py -f track.mp3 --write-metadata
```

#### Batch Processing

Analyze entire directory:

```bash
python main.py --batch /path/to/music/folder
```

Process folder and write metadata:

```bash
python main.py -b /path/to/music --write-metadata
```

Export results to CSV:

```bash
python main.py -b /path/to/music --export-csv results.csv
```

Force re-analysis of already classified tracks:

```bash
python main.py -b /path/to/music --force --write-metadata
```

#### CLI Options

```
--file, -f          Single audio file to classify
--batch, -b         Directory of audio files to process
--write-metadata, -w   Write classification to file metadata
--force             Process files even if already classified
--overwrite         Overwrite existing classification
--export-csv        Export batch results to CSV file
--gui, -g           Launch graphical interface
--config, -c        Path to custom configuration file
```

## 🎛️ Configuration

Configuration is stored in [config/default_config.yaml](config/default_config.yaml). You can create a custom config file and use it:

```bash
python main.py --config my_config.yaml
```

### Key Configuration Options

```yaml
# Audio Processing
audio:
  sample_rate: 22050  # Balance between quality and speed

# Classification Thresholds
classification:
  yellow_max_bpm: 110
  purple_min_bpm: 128
  yellow_max_energy_variance: 0.15
  purple_min_energy_variance: 0.40
  min_confidence: 0.6  # Flag uncertain classifications

# Metadata
metadata:
  use_comment_field: true
  use_grouping_field: true
  create_backup: true  # Backup files before modifying

# Performance
performance:
  use_multiprocessing: true
  enable_cache: true
  cache_dir: "~/.mood-classifier/cache"
```

## 🔬 How It Works

### Audio Feature Extraction

The classifier analyzes the following audio features:

**Temporal Features:**
- **Tempo (BPM)** - Beats per minute
- **Zero-Crossing Rate** - Signal noisiness
- **Low-Energy Percentage** - Proportion of quiet segments
- **RMS Energy** - Overall signal energy

**Spectral Features:**
- **Spectral Rolloff** - Frequency containing 85% of energy
- **Brightness** - High-frequency energy (>3kHz)
- **Spectral Centroid** - Spectral center of mass

**MFCC & Dynamics:**
- **MFCC 1-5** - Mel-frequency cepstral coefficients (mean/std)
- **Energy Variance** - Energy changes over time
- **Drop Intensity** - Sudden energy spikes (build-up/drop detection)

### Classification Method

The system uses a hybrid approach:

1. **Rule-Based Classification**: Evaluates each track against thresholds for each zone
2. **Confidence Scoring**: Each zone receives a score (0-1)
3. **Zone Selection**: Highest-scoring zone is selected
4. **ML Enhancement** (Optional): If confidence is low, use trained ML model

### Metadata Storage

Results are saved to multiple metadata fields for maximum compatibility:

- **Comment Field**: Full classification with confidence
- **Grouping Field**: Zone name (Yellow/Green/Purple)
- **Custom Field** (ENERGYZONE): Zone value for programmatic access

## 📊 Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- FLAC (.flac)
- M4A/AAC (.m4a, .mp4)
- OGG (.ogg)

## 🔧 Development

### Project Structure

```
mood-classifier/
├── src/
│   ├── audio/              # Audio loading and feature extraction
│   │   ├── loader.py
│   │   └── extractors.py
│   ├── classification/     # Classification logic
│   │   ├── classifier.py
│   │   └── rules.py
│   ├── metadata/           # Metadata read/write
│   │   ├── reader.py
│   │   └── writer.py
│   ├── gui/                # PyQt5 GUI
│   │   └── main_window.py
│   └── utils/              # Configuration and logging
│       ├── config.py
│       └── logger.py
├── config/                 # Configuration files
├── models/                 # Trained ML models
├── tests/                  # Unit tests
├── main.py                 # Application entry point
└── requirements.txt
```

### Running Tests

```bash
pytest tests/
```

### Training ML Model

To train a custom ML model on your labeled dataset:

```python
from src import EnergyZoneClassifier
import numpy as np

# Load your labeled data (features and labels)
X = np.load('features.npy')  # Shape: (N, 16)
y = np.load('labels.npy')    # Shape: (N,) - 0=yellow, 1=green, 2=purple

# Train classifier
classifier = EnergyZoneClassifier()
accuracy = classifier.train_model(X, y)

# Save model
classifier.save_model('models/classifier_custom.pkl')
```

## 🎯 Performance Targets

- **Analysis Speed**: <30 seconds per 3-5 minute track
- **Accuracy**: >80% agreement with expert classification
- **Memory**: <500MB for batch of 100 tracks
- **CPU**: Utilizes 75-85% of available cores

## 📝 Tips for Best Results

1. **Consistent Audio Quality**: Higher quality files (FLAC, high-bitrate MP3) yield better feature extraction
2. **Full Tracks**: Analyze complete tracks rather than snippets for accurate tempo/energy analysis
3. **Genre Considerations**: The classifier is optimized for electronic dance music but works across genres
4. **Threshold Tuning**: Adjust thresholds in config file to match your music library's characteristics
5. **Backup First**: When writing metadata, use the backup feature to preserve original files

## 🤝 Contributing

This project is based on research in music information retrieval and DJ workflow optimization. Contributions are welcome!

## 📄 License

This project uses the following open-source libraries:
- librosa - Audio analysis
- PyQt5 - GUI framework
- scikit-learn - Machine learning
- mutagen - Metadata handling

## 🐛 Troubleshooting

### "Failed to load audio file"
- Ensure ffmpeg is installed for format support
- Check file is not corrupted
- Verify file format is supported

### "Low confidence classification"
- Track may have ambiguous characteristics
- Review detailed features to understand why
- Consider manual review for uncertain tracks

### GUI not launching
- Ensure PyQt5 is installed: `pip install PyQt5`
- Check Python version is 3.8+

### Metadata not writing
- Verify you have write permissions to files
- Check backup directory has space
- Some formats may not support all metadata fields

## 📚 References

This implementation is based on research in music information retrieval, specifically audio feature extraction for music classification and DJ workflow optimization.
