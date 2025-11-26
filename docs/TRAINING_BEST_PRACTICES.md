# Training Best Practices

This document consolidates all optimization findings from experiments.

## Summary of Best Configuration

| Parameter | Value | Impact |
|-----------|-------|--------|
| Algorithm | XGBoost | +3% accuracy vs RandomForest |
| Features | Top-50 | Optimal signal/noise balance |
| Drop features | Normalized | Fixes YELLOW paradox |
| Source norm | Yes | DEAM -> User distribution |

**Best CV Accuracy: 65.48%** (vs 62% with RandomForest defaults)

## Quick Start

```bash
# 1. Extract features (once)
python scripts/extract_unified_features.py \
    --input tests/test_sample_50.txt \
    --output results/user_50_frames.pkl \
    --source audio --method frames

python scripts/extract_unified_features.py \
    --input data/deam/annotations/annotations_averaged.csv \
    --output results/deam_50_frames.pkl \
    --source deam --method frames

# 2. Train production model
python scripts/train_production.py \
    --user results/user_50_frames.pkl \
    --deam results/deam_50_frames.pkl \
    --annotations data/deam/annotations/annotations_averaged.csv

# 3. Use in production
python main.py --model models/production/zone_classifier_latest.pkl --file track.mp3
```

## Key Insights

### 1. XGBoost > RandomForest > GradientBoosting

Tested with 5-fold CV:
- XGBoost: 65.48%
- RandomForest: 63.21%
- GradientBoosting: 62.15%

XGBoost parameters (tuned):
```python
{
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

### 2. Top-50 Feature Selection

Why 50?
- < 30 features: loses important signals
- > 70 features: adds noise
- 50: optimal balance

Feature selection is done by training a preliminary model and taking
top N features by importance.

### 3. Normalized Drop Features

**The Problem (YELLOW Paradox):**
Quiet tracks (YELLOW) showed HIGHER `energy_buildup_score_max` than
energetic tracks (PURPLE). This is counterintuitive.

**The Cause:**
In quiet music, any energy change appears dramatic relative to the
low baseline energy.

**The Solution:**
Normalize drop features by energy level:
```python
normalized_buildup_max = buildup_score / rms_energy_mean
```

**6 Normalized Features Added:**
1. `normalized_buildup_max` - buildup relative to energy level
2. `normalized_buildup_mean` - average buildup relative to energy
3. `relative_drop_intensity` - drop strength vs energy variance
4. `drop_prominence` - buildup range vs energy range
5. `energy_dynamics_score` - combined PURPLE indicator
6. `drop_frequency` - drops per frame

### 4. Source Normalization

DEAM and User audio have different distributions. Without normalization,
the model may overfit to DEAM statistics.

Solution: Z-score normalize DEAM features to match User distribution:
```python
deam_normalized = (deam - deam_mean) / deam_std * user_std + user_mean
```

## File Structure

```
src/training/
├── production_pipeline.py  # Main training pipeline (USE THIS)
├── zone_trainer.py         # Legacy trainer
└── zone_models.py          # Model definitions

scripts/
├── train_production.py     # CLI for production training
├── optimize_frame_model.py # Experimentation script
└── analyze_drop_features.py # Feature analysis

models/
├── production/             # Production models
│   ├── zone_classifier_latest.pkl
│   └── zone_classifier_YYYYMMDD.pkl
└── checkpoints/            # Training checkpoints
```

## When to Retrain

1. After adding new labeled tracks to training set
2. After changing feature extraction
3. After discovering better hyperparameters

## Experimental Scripts

For experimentation (not production):

```bash
# Test different configurations
python scripts/optimize_frame_model.py \
    --user results/user_50_frames.pkl \
    --deam results/deam_50_frames.pkl \
    --output-dir models/experiments

# Analyze feature importance
python scripts/analyze_drop_features.py \
    --user results/user_50_frames.pkl \
    --deam results/deam_50_frames.pkl
```

## Updating Best Practices

If you find a better configuration:

1. Document the experiment and results
2. Update `PipelineConfig` defaults in `production_pipeline.py`
3. Update this document
4. Retrain production model
