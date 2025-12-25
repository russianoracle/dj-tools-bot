# Scripts Directory

## Recommended Workflow

### 1. Feature Extraction
```bash
# M2-optimized extraction (79 features per frame)
/Applications/miniforge3/bin/python3 scripts/extract_features_m2.py \
    --input results/user_tracks.csv \
    --output results/user_frames.pkl \
    --workers 4

# Ultimate GPU extraction (327 features, more comprehensive)
/Applications/miniforge3/bin/python3 scripts/extract_ultimate_gpu.py \
    --input results/user_tracks.csv \
    --output results/user_frames_full.pkl
```

### 2. Training
```bash
# RECOMMENDED: XGBoost on user data only
/Applications/miniforge3/bin/python3 scripts/train_user_only.py \
    --input results/user_frames.pkl \
    --output models/production/zone_classifier.pkl

# Alternative: Production pipeline with DEAM (if available)
/Applications/miniforge3/bin/python3 scripts/train_production.py \
    --user results/user_frames.pkl \
    --deam results/deam_frames.pkl \
    --annotations data/deam/annotations/annotations_averaged.csv
```

## Active Scripts

| Script | Purpose |
|--------|---------|
| `train_user_only.py` | Train XGBoost on labeled user data |
| `train_production.py` | Full pipeline with DEAM integration |
| `extract_features_m2.py` | M2-optimized feature extraction (79 features) |
| `extract_ultimate_gpu.py` | Comprehensive extraction (327 features) |
| `parse_serato_db.py` | Parse Serato database for labels |
| `convert_labels_to_csv.py` | Convert labels.json to CSV |
| `prepare_drive_csv.py` | Prepare Google Drive tracks for extraction |

## Archived Scripts

Experimental scripts moved to `_archive/`:
- `train_cascade.py` - Cascade classification experiment
- `train_ensemble.py` - Ensemble model experiment
- `train_spectrogram_model.py` - CNN on spectrograms
- `train_track_level.py` - Track-level aggregation
- `train_track_balanced.py` - Balanced sampling experiment
- Others...

These are kept for reference but not recommended for production use.
