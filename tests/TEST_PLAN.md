# Test System Plan - Mood Classifier

## Overview

This document describes the comprehensive test system for the mood-classifier project.
The system covers three main areas:

1. **Critical Class Coverage (100%)** - Core classes that must have complete test coverage
2. **Architectural Invariant Tests** - Rules that must never be violated
3. **Regression Tests** - Golden baselines with real audio data

---

## 1. Critical Classes Requiring 100% Coverage

### Priority 1: CRITICAL (Data Integrity)

| Class | File | Methods | Test File |
|-------|------|---------|-----------|
| **STFTCache** | `src/core/primitives/stft.py` | 25+ lazy methods | `test_stft_cache_full.py` |
| **CacheRepository** | `src/core/cache/repository.py` | 40+ methods | `test_cache_repository.py` |
| **BeatGridResult** | `src/core/primitives/beat_grid.py` | 15+ methods | `test_beat_grid_result.py` |

### Priority 2: HIGH (Core Functionality)

| Class | File | Test File |
|-------|------|-----------|
| **BeatGridTask** | `src/core/tasks/beat_grid.py` | `test_beat_grid_task.py` |
| **DropDetectionTask** | `src/core/tasks/drop_detection.py` | `test_drop_detection_task.py` |
| **TransitionDetectionTask** | `src/core/tasks/transition_detection.py` | `test_transition_detection_task.py` |

---

## 2. Architectural Invariant Tests

### Invariant 1: STFTCache is Single Source for Librosa Features

**Rule:** No direct librosa calls outside `stft.py`

```python
# test_invariant_stft_single_source.py
def test_no_librosa_in_primitives_except_stft():
    """Primitives (except stft.py) must not import librosa."""

def test_no_librosa_feature_calls_in_tasks():
    """Tasks must use STFTCache.get_*() instead of librosa.feature.*"""

def test_no_librosa_in_scripts():
    """Scripts must use FeatureFactory or STFTCache."""
```

### Invariant 2: Float32 Contiguity (M2 Optimization)

**Rule:** All array returns must be `np.ascontiguousarray(arr, dtype=np.float32)`

```python
# test_invariant_float32_contiguous.py
def test_stft_cache_returns_float32():
    """All STFTCache methods must return float32 arrays."""

def test_stft_cache_returns_contiguous():
    """All STFTCache arrays must be C-contiguous."""

def test_beat_grid_boundaries_contiguous():
    """BeatGridResult boundary arrays must be contiguous float32."""
```

### Invariant 3: Three-Layer Architecture

**Rule:** Primitives -> Tasks -> Pipelines (no reverse imports)

```python
# test_invariant_layer_dependencies.py
def test_primitives_no_task_imports():
    """Primitives cannot import from tasks."""

def test_primitives_no_pipeline_imports():
    """Primitives cannot import from pipelines."""

def test_tasks_no_pipeline_imports():
    """Tasks cannot import from pipelines (except types)."""
```

### Invariant 4: CacheRepository is Only Cache API

**Rule:** Never import CacheManager directly

```python
# test_invariant_cache_api.py
def test_no_direct_cache_manager_import():
    """No module should import CacheManager directly."""

def test_cache_repository_is_singleton():
    """CacheRepository.get_instance() returns same object."""
```

### Invariant 5: Hierarchical BeatGrid Structure

**Rule:** beats ⊂ bars ⊂ phrases (proper nesting)

```python
# test_invariant_beat_grid_hierarchy.py
def test_beats_per_bar_consistent():
    """Each bar must have exactly 4 beats."""

def test_bars_per_phrase_consistent():
    """Each phrase must have exactly 4 bars."""

def test_downbeat_has_bar_position_1():
    """Downbeats must have bar_position == 1."""
```

---

## 3. Regression Test Framework

### Design Principles

1. **Golden Baselines:** Pre-computed expected outputs for real audio files
2. **Hash-Based Verification:** MD5 of output arrays with tolerance
3. **Automatic Failure Detection:** CI fails if output changes unexpectedly
4. **Easy Updates:** Script to regenerate baselines when intentional changes made

### Baseline Storage

```
tests/
├── fixtures/
│   ├── baselines/
│   │   ├── stft_cache_baseline.json       # STFTCache feature hashes
│   │   ├── beat_grid_baseline.json        # BeatGrid structure
│   │   ├── drop_detection_baseline.json   # Drop detection results
│   │   └── transition_baseline.json       # Transition detection results
│   └── audio/
│       └── README.md  # Instructions for test audio
```

### Baseline Format

```json
{
  "version": "1.0",
  "created_at": "2025-12-12T00:00:00Z",
  "track": {
    "name": "Demo Track 1",
    "path": "data/dj_sets/josh-baker/boiler-room.mp3",
    "duration_sec": 3600.0,
    "file_hash": "abc123..."
  },
  "baselines": {
    "rms_mean": 0.0234,
    "rms_std": 0.0089,
    "tempo": 128.0,
    "n_drops": 15,
    "drop_times_hash": "def456...",
    "beat_grid_hash": "ghi789..."
  },
  "tolerances": {
    "rms_mean": 0.001,
    "tempo": 0.5,
    "n_drops": 1
  }
}
```

### Regression Test Structure

```python
# tests/test_regression.py

class TestRegressionSTFTCache:
    """Regression tests for STFTCache outputs."""

    @pytest.fixture
    def baseline(self):
        return load_baseline("stft_cache_baseline.json")

    def test_rms_consistency(self, real_audio, baseline):
        """RMS output must match baseline within tolerance."""
        cache = compute_stft(real_audio.y, real_audio.sr)
        rms = cache.get_rms()

        assert_within_tolerance(
            np.mean(rms),
            baseline["rms_mean"],
            baseline["tolerances"]["rms_mean"]
        )

    def test_mfcc_shape_consistency(self, real_audio, baseline):
        """MFCC shape must match baseline."""
        cache = compute_stft(real_audio.y, real_audio.sr)
        mfcc = cache.get_mfcc()

        assert mfcc.shape[0] == baseline["mfcc_n_coeffs"]


class TestRegressionBeatGrid:
    """Regression tests for BeatGrid."""

    def test_tempo_consistency(self, real_audio, baseline):
        """Detected tempo must match baseline."""
        grid = compute_beat_grid(...)

        assert abs(grid.tempo - baseline["tempo"]) < baseline["tolerances"]["tempo"]

    def test_phrase_count_consistency(self, real_audio, baseline):
        """Number of phrases must match baseline."""
        grid = compute_beat_grid(...)

        assert len(grid.phrases) == baseline["n_phrases"]


class TestRegressionDropDetection:
    """Regression tests for drop detection."""

    def test_drop_count_consistency(self, real_audio, baseline):
        """Number of detected drops must match baseline."""
        result = DropDetectionTask().execute(context)

        assert abs(len(result.drops) - baseline["n_drops"]) <= baseline["tolerances"]["n_drops"]

    def test_drop_times_consistency(self, real_audio, baseline):
        """Drop times must be within tolerance of baseline."""
        result = DropDetectionTask().execute(context)
        drop_times = np.array([d.time for d in result.drops])

        # Check that drops match baseline within 1 second
        for expected_time in baseline["drop_times"]:
            assert np.any(np.abs(drop_times - expected_time) < 1.0)
```

### Baseline Management Scripts

```python
# scripts/update_baselines.py

def update_baseline(baseline_name: str, force: bool = False):
    """Update a baseline file with current outputs."""

    # Compute current outputs
    results = compute_results_for_baseline(baseline_name)

    # Load existing baseline
    existing = load_baseline(baseline_name)

    # Compare
    changes = compare_baselines(existing, results)

    if changes and not force:
        print("Changes detected:")
        for change in changes:
            print(f"  - {change}")
        print("Use --force to update.")
        return

    # Save new baseline
    save_baseline(baseline_name, results)
    print(f"Updated {baseline_name}")
```

---

## 4. Test Organization

### Directory Structure

```
tests/
├── conftest.py                         # Global fixtures
├── fixtures/
│   ├── baselines/                      # Golden baseline files
│   ├── audio/                          # Test audio (gitignored)
│   └── synthetic.py                    # Synthetic audio generators
│
├── # Architecture & Invariant Tests
├── test_architecture.py                # Existing - layer rules
├── test_no_duplication.py              # Existing - no reimplementation
├── test_invariant_stft_single_source.py   # NEW
├── test_invariant_float32_contiguous.py   # NEW
├── test_invariant_cache_api.py            # NEW
├── test_invariant_beat_grid_hierarchy.py  # NEW
│
├── # Critical Class Tests (100% coverage)
├── test_stft_cache_full.py             # NEW - comprehensive STFTCache
├── test_cache_repository.py            # NEW - CacheRepository
├── test_beat_grid_result.py            # NEW - BeatGridResult
│
├── # Task Tests
├── test_beat_grid_task.py              # NEW
├── test_drop_detection_task.py         # NEW
├── test_transition_detection_task.py   # NEW
│
├── # Regression Tests
├── test_regression_stft.py             # NEW
├── test_regression_beat_grid.py        # NEW
├── test_regression_drop_detection.py   # NEW
├── test_regression_transition.py       # NEW
│
├── # Existing tests (keep)
├── test_audio_loader.py
├── test_baseline.py
├── test_baseline_real_tracks.py
├── test_feature_factory.py
├── test_integration.py
├── test_librosa_consumers.py
├── test_stft_cache.py
├── test_tasks_functional.py
├── test_vectorized_primitives.py
└── test_vectorized_tasks.py
```

### Test Categories & Markers

```python
# conftest.py

# Markers for test categories
pytest.register_marker("critical", description="Critical class tests (100% coverage)")
pytest.register_marker("invariant", description="Architectural invariant tests")
pytest.register_marker("regression", description="Regression tests with golden baselines")
pytest.register_marker("slow", description="Slow tests (real audio)")
pytest.register_marker("requires_audio", description="Requires real audio files")
```

### Running Tests by Category

```bash
# Run all tests
pytest tests/

# Run only critical class tests
pytest tests/ -m critical

# Run only architectural invariants
pytest tests/ -m invariant

# Run regression tests
pytest tests/ -m regression

# Run fast tests only (no real audio)
pytest tests/ -m "not slow"

# Run with coverage for critical classes
pytest tests/ -m critical --cov=src/core/primitives/stft --cov=src/core/cache --cov-report=html
```

---

## 5. Implementation Priority

### Phase 1: Critical Class Tests (Week 1)
1. `test_stft_cache_full.py` - 50+ tests for STFTCache
2. `test_cache_repository.py` - 40+ tests for CacheRepository
3. `test_beat_grid_result.py` - 30+ tests for BeatGridResult

### Phase 2: Architectural Invariants (Week 1-2)
1. `test_invariant_stft_single_source.py`
2. `test_invariant_float32_contiguous.py`
3. `test_invariant_cache_api.py`
4. `test_invariant_beat_grid_hierarchy.py`

### Phase 3: Regression Framework (Week 2)
1. Create baseline format and storage
2. Implement baseline loading/comparison utilities
3. Create initial baselines for:
   - STFTCache feature outputs
   - BeatGrid structure
   - Drop detection results
   - Transition detection results

### Phase 4: Task Tests (Week 2-3)
1. `test_beat_grid_task.py`
2. `test_drop_detection_task.py`
3. `test_transition_detection_task.py`

---

## 6. CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run fast tests
        run: pytest tests/ -m "not slow" -v

  full-tests:
    runs-on: ubuntu-latest
    needs: fast-tests
    steps:
      - uses: actions/checkout@v3
      - name: Run all tests
        run: pytest tests/ -v --cov=src --cov-report=xml

  critical-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check critical class coverage
        run: |
          pytest tests/ -m critical --cov=src/core/primitives/stft --cov=src/core/cache --cov-fail-under=100
```

---

## 7. Fixtures & Utilities

### Synthetic Audio Fixture

```python
# tests/fixtures/synthetic.py

def create_synthetic_track(
    duration_sec: float = 30.0,
    sr: int = 22050,
    tempo: float = 128.0,
    has_drops: bool = True,
    has_buildups: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Create synthetic audio with musical structure.

    Structure:
    - 0-25%: Intro (quiet, building)
    - 25-50%: Buildup (increasing energy)
    - 50%: DROP (sudden energy increase)
    - 50-75%: Main section (high energy)
    - 75-100%: Outro (decreasing energy)
    """
    ...
```

### Baseline Utilities

```python
# tests/fixtures/baseline_utils.py

def load_baseline(name: str) -> Dict:
    """Load baseline from JSON file."""

def save_baseline(name: str, data: Dict):
    """Save baseline to JSON file."""

def compute_array_hash(arr: np.ndarray, precision: int = 6) -> str:
    """Compute deterministic hash of numpy array."""

def assert_within_tolerance(actual, expected, tolerance, msg=""):
    """Assert value is within tolerance of expected."""

def compare_baselines(old: Dict, new: Dict) -> List[str]:
    """Compare two baselines and return list of changes."""
```

---

## 8. Coverage Requirements

### Critical Classes (MUST be 100%)

| Class | Target Coverage | Current |
|-------|-----------------|---------|
| STFTCache | 100% | ~60% |
| CacheRepository | 100% | ~20% |
| BeatGridResult | 100% | ~40% |

### High Priority (Target: 90%+)

| Class | Target Coverage | Current |
|-------|-----------------|---------|
| BeatGridTask | 90% | ~30% |
| DropDetectionTask | 90% | ~30% |
| TransitionDetectionTask | 90% | ~30% |

### Overall Project Target

- **Statements:** 80%+
- **Branches:** 75%+
- **Critical paths:** 100%

---

## 9. Acceptance Criteria

### For PR Merge

1. All existing tests pass
2. No decrease in overall coverage
3. Critical class coverage >= 100%
4. All architectural invariant tests pass
5. All regression tests pass (or baselines explicitly updated)

### For Release

1. All tests pass
2. Coverage targets met
3. Regression baselines verified against real audio
4. No flaky tests
