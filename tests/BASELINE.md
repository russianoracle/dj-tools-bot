# Baseline Test Results

**Date:** 2025-12-12
**Python:** 3.12.10
**Pytest:** 8.4.1

## Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 624 |
| **Passed** | 613 |
| **Skipped** | 11 |
| **Failed** | 0 |
| **Warnings** | 11 |
| **Duration** | 46.07s |

## Test Categories

| Category | Tests | Status |
|----------|-------|--------|
| Architecture | ~130 | All passed |
| Audio Loader | ~14 | All passed |
| Beat Grid | ~10 | All passed |
| BPM Accuracy | ~20 | All passed |
| Cache Repository | ~15 | All passed |
| DJ Features | ~10 | All passed |
| Feature Factory | ~25 | All passed |
| Integration | ~30 | All passed |
| Invariants | ~20 | All passed |
| Librosa Consumers | ~50 | All passed |
| Model Prediction | ~15 | All passed |
| No Duplication | ~10 | All passed |
| Regression | ~20 | All passed |
| STFT Cache | ~30 | All passed |
| Tasks Functional | ~50 | All passed |
| Vectorized Primitives | ~40 | All passed |
| Vectorized Tasks | ~30 | All passed |

## Deprecation Warnings (expected)

- `compute_rms()` → use `STFTCache.get_rms()`
- `compute_centroid()` → use `STFTCache.get_spectral_centroid()`
- `librosa.core.audio.__audioread_load` deprecated in 0.10.0

## Command to Reproduce

```bash
/Applications/miniforge3/bin/python3 -m pytest tests/ -v --tb=short
```

## Notes

- All critical architecture tests pass (layer isolation, no librosa in primitives)
- STFTCache lazy methods working correctly
- Tasks functional tests verify core analysis pipeline
- Ready for cloud migration refactoring
