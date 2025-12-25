"""
BeatGridResult Full Coverage Tests - 100% coverage target.

Tests BeatGridResult for:
1. Hierarchical structure (beats ⊂ bars ⊂ phrases)
2. Boundary caching
3. Snap operations
4. Position calculations
5. Edge cases
"""

import numpy as np
import pytest
from typing import List

from app.common.primitives.beat_grid import (
    BeatInfo, BarInfo, PhraseInfo, BeatGridResult,
    compute_beat_grid, detect_downbeat, snap_events_to_grid,
    compute_event_offsets, compute_alignment_score, calibrate_grid_phase,
    apply_phase_correction
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_beat_grid() -> BeatGridResult:
    """Create sample beat grid (128 BPM, 32 beats = 2 phrases)."""
    tempo = 128.0
    beat_duration = 60.0 / tempo  # ~0.469 sec

    # Create 32 beats (2 phrases)
    beats = []
    for i in range(32):
        beats.append(BeatInfo(
            time_sec=i * beat_duration,
            frame_idx=int(i * beat_duration * 22050 / 512),
            bar_position=(i % 4) + 1,
            phrase_position=(i % 16) + 1,
            strength=1.0 if i % 4 == 0 else 0.5
        ))

    # Create 8 bars (2 phrases)
    bars = []
    for i in range(8):
        bars.append(BarInfo(
            index=i,
            start_time=i * 4 * beat_duration,
            end_time=(i + 1) * 4 * beat_duration,
            beat_indices=list(range(i * 4, (i + 1) * 4)),
            phrase_idx=i // 4,
            bar_in_phrase=(i % 4) + 1
        ))

    # Create 2 phrases
    phrases = []
    for i in range(2):
        phrases.append(PhraseInfo(
            index=i,
            start_time=i * 16 * beat_duration,
            end_time=(i + 1) * 16 * beat_duration,
            bar_indices=list(range(i * 4, (i + 1) * 4)),
            duration_sec=16 * beat_duration
        ))

    return BeatGridResult(
        beats=beats,
        bars=bars,
        phrases=phrases,
        tempo=tempo,
        tempo_confidence=0.9,
        downbeat_idx=0,
        beat_duration_sec=beat_duration,
        bar_duration_sec=4 * beat_duration,
        phrase_duration_sec=16 * beat_duration,
        sr=22050,
        hop_length=512
    )


@pytest.fixture
def empty_beat_grid() -> BeatGridResult:
    """Create empty beat grid."""
    return BeatGridResult(
        beats=[],
        bars=[],
        phrases=[],
        tempo=128.0,
        tempo_confidence=0.0,
        downbeat_idx=0,
        beat_duration_sec=0.469,
        bar_duration_sec=1.875,
        phrase_duration_sec=7.5,
        sr=22050,
        hop_length=512
    )


# =============================================================================
# Test: BeatGridResult Structure
# =============================================================================

class TestBeatGridResultStructure:
    """Tests for BeatGridResult structure."""

    def test_has_beats_bars_phrases(self, sample_beat_grid):
        """Grid has beats, bars, and phrases."""
        assert len(sample_beat_grid.beats) > 0
        assert len(sample_beat_grid.bars) > 0
        assert len(sample_beat_grid.phrases) > 0

    def test_hierarchical_counts(self, sample_beat_grid):
        """4 beats per bar, 4 bars per phrase."""
        # 32 beats = 8 bars = 2 phrases
        assert len(sample_beat_grid.beats) == 32
        assert len(sample_beat_grid.bars) == 8
        assert len(sample_beat_grid.phrases) == 2

    def test_bar_has_4_beats(self, sample_beat_grid):
        """Each bar references exactly 4 beats."""
        for bar in sample_beat_grid.bars:
            assert len(bar.beat_indices) == 4

    def test_phrase_has_4_bars(self, sample_beat_grid):
        """Each phrase references exactly 4 bars."""
        for phrase in sample_beat_grid.phrases:
            assert len(phrase.bar_indices) == 4


# =============================================================================
# Test: Boundary Caching
# =============================================================================

class TestBoundaryCaching:
    """Tests for boundary array caching."""

    def test_get_phrase_boundaries_returns_array(self, sample_beat_grid):
        """get_phrase_boundaries returns numpy array."""
        boundaries = sample_beat_grid.get_phrase_boundaries()
        assert isinstance(boundaries, np.ndarray)

    def test_phrase_boundaries_cached(self, sample_beat_grid):
        """Phrase boundaries are cached (same object)."""
        b1 = sample_beat_grid.get_phrase_boundaries()
        b2 = sample_beat_grid.get_phrase_boundaries()
        assert b1 is b2

    def test_phrase_boundaries_count(self, sample_beat_grid):
        """n_phrases + 1 boundaries (start + end of each)."""
        boundaries = sample_beat_grid.get_phrase_boundaries()
        assert len(boundaries) == len(sample_beat_grid.phrases) + 1

    def test_bar_boundaries_cached(self, sample_beat_grid):
        """Bar boundaries are cached (same object)."""
        b1 = sample_beat_grid.get_bar_boundaries()
        b2 = sample_beat_grid.get_bar_boundaries()
        assert b1 is b2

    def test_beat_times_cached(self, sample_beat_grid):
        """Beat times are cached (same object)."""
        t1 = sample_beat_grid.get_beat_times()
        t2 = sample_beat_grid.get_beat_times()
        assert t1 is t2

    def test_boundaries_float32(self, sample_beat_grid):
        """Boundaries are float32 (M2 optimization)."""
        assert sample_beat_grid.get_phrase_boundaries().dtype == np.float32
        assert sample_beat_grid.get_bar_boundaries().dtype == np.float32
        assert sample_beat_grid.get_beat_times().dtype == np.float32

    def test_boundaries_contiguous(self, sample_beat_grid):
        """Boundaries are C-contiguous (M2 optimization)."""
        assert sample_beat_grid.get_phrase_boundaries().flags['C_CONTIGUOUS']
        assert sample_beat_grid.get_bar_boundaries().flags['C_CONTIGUOUS']
        assert sample_beat_grid.get_beat_times().flags['C_CONTIGUOUS']


# =============================================================================
# Test: Snap Operations
# =============================================================================

class TestSnapOperations:
    """Tests for snap-to-grid operations."""

    def test_snap_to_phrase_on_boundary(self, sample_beat_grid):
        """Snap returns same time when on boundary."""
        boundaries = sample_beat_grid.get_phrase_boundaries()
        for boundary in boundaries:
            snapped = sample_beat_grid.snap_to_phrase(float(boundary))
            assert abs(snapped - boundary) < 0.001

    def test_snap_to_phrase_near_boundary(self, sample_beat_grid):
        """Snap moves time to nearest boundary."""
        boundaries = sample_beat_grid.get_phrase_boundaries()
        # Time slightly after first phrase boundary
        test_time = float(boundaries[1]) + 0.1
        snapped = sample_beat_grid.snap_to_phrase(test_time)
        assert abs(snapped - boundaries[1]) < 0.001

    def test_snap_to_bar(self, sample_beat_grid):
        """snap_to_bar works correctly."""
        # Bar at 1.875 sec (4 beats at 128 BPM)
        bar_boundaries = sample_beat_grid.get_bar_boundaries()
        test_time = float(bar_boundaries[1]) + 0.1
        snapped = sample_beat_grid.snap_to_bar(test_time)
        assert abs(snapped - bar_boundaries[1]) < 0.001

    def test_snap_to_beat(self, sample_beat_grid):
        """snap_to_beat works correctly."""
        beat_times = sample_beat_grid.get_beat_times()
        test_time = float(beat_times[5]) + 0.1
        snapped = sample_beat_grid.snap_to_beat(test_time)
        assert abs(snapped - beat_times[5]) < 0.15  # Within 1 beat


# =============================================================================
# Test: Boundary Checks
# =============================================================================

class TestBoundaryChecks:
    """Tests for is_on_*_boundary methods."""

    def test_is_on_phrase_boundary_true(self, sample_beat_grid):
        """is_on_phrase_boundary returns True on boundary."""
        boundaries = sample_beat_grid.get_phrase_boundaries()
        result = sample_beat_grid.is_on_phrase_boundary(float(boundaries[0]))
        assert result == True

    def test_is_on_phrase_boundary_false(self, sample_beat_grid):
        """is_on_phrase_boundary returns False away from boundary."""
        boundaries = sample_beat_grid.get_phrase_boundaries()
        # Midway between boundaries
        mid = (boundaries[0] + boundaries[1]) / 2
        result = sample_beat_grid.is_on_phrase_boundary(float(mid), tolerance_beats=1)
        assert result == False

    def test_is_on_phrase_boundary_with_tolerance(self, sample_beat_grid):
        """is_on_phrase_boundary respects tolerance."""
        boundaries = sample_beat_grid.get_phrase_boundaries()
        # Slightly off boundary
        offset_time = float(boundaries[1]) + sample_beat_grid.beat_duration_sec * 0.5
        # Should be True with 2-beat tolerance
        result = sample_beat_grid.is_on_phrase_boundary(offset_time, tolerance_beats=2)
        assert result == True

    def test_is_on_bar_boundary(self, sample_beat_grid):
        """is_on_bar_boundary works correctly."""
        bar_boundaries = sample_beat_grid.get_bar_boundaries()
        result = sample_beat_grid.is_on_bar_boundary(float(bar_boundaries[2]))
        assert result == True


# =============================================================================
# Test: Position Calculations
# =============================================================================

class TestPositionCalculations:
    """Tests for position calculation methods."""

    def test_get_phrase_at_time(self, sample_beat_grid):
        """get_phrase_at_time returns correct phrase."""
        # First phrase covers 0 to ~7.5 seconds
        phrase = sample_beat_grid.get_phrase_at_time(1.0)
        assert phrase is not None
        assert phrase.index == 0

    def test_get_phrase_at_time_second_phrase(self, sample_beat_grid):
        """get_phrase_at_time returns second phrase."""
        phrase = sample_beat_grid.get_phrase_at_time(10.0)  # In second phrase
        assert phrase is not None
        assert phrase.index == 1

    def test_get_phrase_at_time_out_of_range(self, sample_beat_grid):
        """get_phrase_at_time returns None for out-of-range."""
        phrase = sample_beat_grid.get_phrase_at_time(1000.0)
        assert phrase is None

    def test_get_bar_at_time(self, sample_beat_grid):
        """get_bar_at_time returns correct bar."""
        bar = sample_beat_grid.get_bar_at_time(2.0)  # In first bar
        assert bar is not None
        assert bar.index == 1  # Second bar (0-indexed)

    def test_time_to_phrase_position(self, sample_beat_grid):
        """time_to_phrase_position returns (phrase, bar, beat)."""
        position = sample_beat_grid.time_to_phrase_position(0.1)

        assert isinstance(position, tuple)
        assert len(position) == 3
        # First beat of first bar of first phrase
        phrase_num, bar_in_phrase, beat_in_bar = position
        assert phrase_num == 1  # 1-indexed
        assert 1 <= bar_in_phrase <= 4
        assert 1 <= beat_in_bar <= 4


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_grid_snap_to_phrase(self, empty_beat_grid):
        """Empty grid snap returns original time."""
        result = empty_beat_grid.snap_to_phrase(5.0)
        assert result == 5.0

    def test_empty_grid_snap_to_bar(self, empty_beat_grid):
        """Empty grid bar snap returns original time."""
        result = empty_beat_grid.snap_to_bar(5.0)
        assert result == 5.0

    def test_empty_grid_snap_to_beat(self, empty_beat_grid):
        """Empty grid beat snap returns original time."""
        result = empty_beat_grid.snap_to_beat(5.0)
        assert result == 5.0

    def test_empty_grid_is_on_boundary(self, empty_beat_grid):
        """Empty grid boundary check returns False."""
        assert empty_beat_grid.is_on_phrase_boundary(5.0) is False
        assert empty_beat_grid.is_on_bar_boundary(5.0) is False

    def test_empty_grid_boundaries_empty(self, empty_beat_grid):
        """Empty grid returns empty boundary arrays."""
        assert len(empty_beat_grid.get_phrase_boundaries()) == 0
        assert len(empty_beat_grid.get_bar_boundaries()) == 0
        assert len(empty_beat_grid.get_beat_times()) == 0

    def test_empty_grid_get_phrase_at_time(self, empty_beat_grid):
        """Empty grid get_phrase returns None."""
        assert empty_beat_grid.get_phrase_at_time(5.0) is None

    def test_empty_grid_position(self, empty_beat_grid):
        """Empty grid position returns (0, 0, 0)."""
        position = empty_beat_grid.time_to_phrase_position(5.0)
        assert position == (0, 0, 0)


# =============================================================================
# Test: Serialization
# =============================================================================

class TestSerialization:
    """Tests for to_dict serialization."""

    def test_to_dict_returns_dict(self, sample_beat_grid):
        """to_dict returns dictionary."""
        result = sample_beat_grid.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_required_fields(self, sample_beat_grid):
        """to_dict has all required fields."""
        result = sample_beat_grid.to_dict()

        assert 'tempo' in result
        assert 'tempo_confidence' in result
        assert 'n_beats' in result
        assert 'n_bars' in result
        assert 'n_phrases' in result
        assert 'phrase_boundaries' in result

    def test_to_dict_values_correct(self, sample_beat_grid):
        """to_dict values are correct."""
        result = sample_beat_grid.to_dict()

        assert result['tempo'] == 128.0
        assert result['n_beats'] == 32
        assert result['n_bars'] == 8
        assert result['n_phrases'] == 2


# =============================================================================
# Test: snap_events_to_grid function
# =============================================================================

class TestSnapEventsToGrid:
    """Tests for snap_events_to_grid function."""

    def test_snaps_events_to_phrases(self, sample_beat_grid):
        """Events are snapped to phrase boundaries."""
        boundaries = sample_beat_grid.get_phrase_boundaries()
        events = np.array([float(boundaries[1]) + 0.2], dtype=np.float32)

        snapped = snap_events_to_grid(
            events,
            sample_beat_grid,
            snap_level='phrase',
            max_shift_beats=4.0
        )

        assert abs(snapped[0] - boundaries[1]) < 0.001

    def test_max_shift_respected(self, sample_beat_grid):
        """Events beyond max_shift are not snapped."""
        boundaries = sample_beat_grid.get_phrase_boundaries()
        # Event very far from any boundary
        far_event = (float(boundaries[0]) + float(boundaries[1])) / 2
        events = np.array([far_event], dtype=np.float32)

        snapped = snap_events_to_grid(
            events,
            sample_beat_grid,
            snap_level='phrase',
            max_shift_beats=0.5  # Very tight tolerance
        )

        # Should not have moved
        assert abs(snapped[0] - far_event) < 0.001

    def test_empty_events(self, sample_beat_grid):
        """Empty events array returns empty."""
        events = np.array([], dtype=np.float32)
        snapped = snap_events_to_grid(events, sample_beat_grid)
        assert len(snapped) == 0


# =============================================================================
# Test: calibrate_grid_phase function
# =============================================================================

class TestCalibrateGridPhase:
    """Tests for grid phase calibration."""

    def test_calibration_returns_result(self, sample_beat_grid):
        """calibrate_grid_phase returns GridCalibrationResult."""
        from app.core.primitives.beat_grid import GridCalibrationResult

        events = np.array([7.5, 15.0], dtype=np.float32)
        result = calibrate_grid_phase(sample_beat_grid, events)

        assert isinstance(result, GridCalibrationResult)

    def test_insufficient_events(self, sample_beat_grid):
        """Calibration with few events has low confidence."""
        events = np.array([7.5], dtype=np.float32)
        result = calibrate_grid_phase(sample_beat_grid, events, min_events=2)

        assert result.calibration_confidence == 0.0

    def test_aligned_events_low_correction(self, sample_beat_grid):
        """Already aligned events need minimal correction."""
        boundaries = sample_beat_grid.get_phrase_boundaries()
        # Events exactly on boundaries
        events = np.array([float(boundaries[1]), float(boundaries[2])], dtype=np.float32)

        result = calibrate_grid_phase(sample_beat_grid, events)

        assert abs(result.phase_offset_sec) < 0.1


# =============================================================================
# Test: apply_phase_correction function
# =============================================================================

class TestApplyPhaseCorrection:
    """Tests for phase correction application."""

    def test_zero_correction_returns_same(self, sample_beat_grid):
        """Zero correction returns same grid."""
        result = apply_phase_correction(sample_beat_grid, 0.0)
        assert result is sample_beat_grid

    def test_correction_shifts_times(self, sample_beat_grid):
        """Correction shifts all times."""
        offset = 0.5
        result = apply_phase_correction(sample_beat_grid, offset)

        original_times = sample_beat_grid.get_beat_times()
        new_times = result.get_beat_times()

        for orig, new in zip(original_times[:5], new_times[:5]):
            assert abs(new - orig - offset) < 0.001


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
