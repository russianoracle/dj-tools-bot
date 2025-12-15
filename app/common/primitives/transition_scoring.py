"""
Transition Scoring Primitives - Score track-to-track compatibility.

Pure functions for scoring transitions between two tracks:
- Harmonic compatibility (Camelot wheel distance)
- Energy flow (outro → intro smoothness)
- Drop conflict detection
- Spectral similarity
- Genre matching

All functions are stateless and operate on simple inputs.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


# ============== Camelot Wheel Logic ==============

# Camelot wheel as numeric positions (1-12, A=minor, B=major)
CAMELOT_TO_NUMERIC = {
    '1A': (1, 'A'), '2A': (2, 'A'), '3A': (3, 'A'), '4A': (4, 'A'),
    '5A': (5, 'A'), '6A': (6, 'A'), '7A': (7, 'A'), '8A': (8, 'A'),
    '9A': (9, 'A'), '10A': (10, 'A'), '11A': (11, 'A'), '12A': (12, 'A'),
    '1B': (1, 'B'), '2B': (2, 'B'), '3B': (3, 'B'), '4B': (4, 'B'),
    '5B': (5, 'B'), '6B': (6, 'B'), '7B': (7, 'B'), '8B': (8, 'B'),
    '9B': (9, 'B'), '10B': (10, 'B'), '11B': (11, 'B'), '12B': (12, 'B'),
}


def camelot_distance(key_a: str, key_b: str) -> int:
    """
    Calculate Camelot wheel distance between two keys.

    Compatible moves:
    - Same key: distance 0
    - +1/-1 (adjacent on wheel): distance 1
    - Same number, A↔B (relative major/minor): distance 1

    Args:
        key_a: First Camelot key (e.g., '8A', '11B')
        key_b: Second Camelot key

    Returns:
        Distance on Camelot wheel (0 = same, 1 = compatible, 2+ = risky)
    """
    if key_a not in CAMELOT_TO_NUMERIC or key_b not in CAMELOT_TO_NUMERIC:
        return 6  # Unknown keys - max distance

    num_a, mode_a = CAMELOT_TO_NUMERIC[key_a]
    num_b, mode_b = CAMELOT_TO_NUMERIC[key_b]

    # Same key
    if key_a == key_b:
        return 0

    # Same number (relative major/minor)
    if num_a == num_b:
        return 1

    # Calculate circular distance
    diff = abs(num_a - num_b)
    circular_dist = min(diff, 12 - diff)

    # Same mode (inner or outer ring)
    if mode_a == mode_b:
        return circular_dist

    # Different mode - add 1 for mode switch
    return circular_dist + 1


def score_harmonic_compatibility(camelot_a: str, camelot_b: str) -> float:
    """
    Score harmonic compatibility between two tracks.

    Args:
        camelot_a: First track's Camelot key
        camelot_b: Second track's Camelot key

    Returns:
        Score 0-1 (1 = perfect match, 0 = incompatible)
    """
    dist = camelot_distance(camelot_a, camelot_b)

    # Scoring: 0 → 1.0, 1 → 0.9, 2 → 0.6, 3 → 0.3, 4+ → 0.1
    scores = {0: 1.0, 1: 0.9, 2: 0.6, 3: 0.3}
    return scores.get(dist, 0.1)


def score_harmonic_progression(camelot_a: str, camelot_b: str) -> float:
    """
    Score harmonic PROGRESSION - prefer movement over staying same key.

    For set building, we want to move around the Camelot wheel,
    not stay in one key the entire set. This function rewards:
    - +1/-1 moves (adjacent keys): highest score
    - A↔B moves (relative major/minor): high score
    - Same key: good but not optimal
    - Distance 2: acceptable
    - Distance 3+: avoid

    Args:
        camelot_a: Current track's Camelot key
        camelot_b: Next track's Camelot key

    Returns:
        Score 0-1 (1 = ideal progression, 0 = bad)
    """
    if camelot_a not in CAMELOT_TO_NUMERIC or camelot_b not in CAMELOT_TO_NUMERIC:
        return 0.3  # Unknown keys

    num_a, mode_a = CAMELOT_TO_NUMERIC[camelot_a]
    num_b, mode_b = CAMELOT_TO_NUMERIC[camelot_b]

    # Same key - good but not optimal for progression
    if camelot_a == camelot_b:
        return 0.7

    # Calculate circular distance on the wheel
    diff = abs(num_a - num_b)
    circular_dist = min(diff, 12 - diff)

    # Adjacent keys on same ring (+1 or -1) - BEST for energy building
    if mode_a == mode_b and circular_dist == 1:
        return 1.0

    # Relative major/minor (same number, A↔B) - very smooth
    if num_a == num_b and mode_a != mode_b:
        return 0.95

    # Adjacent + mode switch (e.g., 5A → 6B) - interesting
    if circular_dist == 1 and mode_a != mode_b:
        return 0.85

    # Distance 2 on same ring - acceptable
    if mode_a == mode_b and circular_dist == 2:
        return 0.5

    # Distance 2 with mode switch
    if circular_dist == 2:
        return 0.4

    # Distance 3+ - risky
    if circular_dist >= 3:
        return 0.2

    return 0.3


# ============== Energy Flow Scoring ==============

def score_energy_flow(
    outro_energy: float,
    intro_energy: float,
    tolerance: float = 0.2
) -> float:
    """
    Score energy flow from track A outro to track B intro.

    Good transitions have similar energy levels at the mix point,
    or a gentle rise (building energy).

    Args:
        outro_energy: Track A's energy at mix-out point (0-1)
        intro_energy: Track B's energy at mix-in point (0-1)
        tolerance: Acceptable energy difference

    Returns:
        Score 0-1 (1 = smooth flow, 0 = jarring)
    """
    diff = intro_energy - outro_energy

    # Slight energy rise is good (building energy)
    if 0 <= diff <= 0.15:
        return 1.0

    # Same energy is also good
    if abs(diff) <= tolerance:
        return 0.9

    # Larger differences are progressively worse
    abs_diff = abs(diff)
    if abs_diff <= 0.3:
        return 0.7
    elif abs_diff <= 0.5:
        return 0.4
    else:
        return 0.2


def score_energy_curve_compatibility(
    energy_a: np.ndarray,
    energy_b: np.ndarray,
    overlap_ratio: float = 0.15
) -> float:
    """
    Score energy curve compatibility in overlap zone.

    Analyzes how well the energy curves blend in the overlap region.

    Args:
        energy_a: Track A energy curve (normalized 0-1)
        energy_b: Track B energy curve (normalized 0-1)
        overlap_ratio: What fraction of tracks overlap (default 15%)

    Returns:
        Score 0-1 (1 = smooth blend, 0 = clash)
    """
    # Get overlap regions
    overlap_len_a = int(len(energy_a) * overlap_ratio)
    overlap_len_b = int(len(energy_b) * overlap_ratio)

    if overlap_len_a < 2 or overlap_len_b < 2:
        return 0.5  # Not enough data

    # Track A outro (last portion)
    outro_a = energy_a[-overlap_len_a:]
    # Track B intro (first portion)
    intro_b = energy_b[:overlap_len_b]

    # Resample to same length for comparison
    target_len = min(len(outro_a), len(intro_b))
    outro_a = np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, len(outro_a)),
        outro_a
    )
    intro_b = np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, len(intro_b)),
        intro_b
    )

    # Score based on correlation and level match
    if np.std(outro_a) < 0.01 or np.std(intro_b) < 0.01:
        # Flat energy - check level match only
        level_diff = abs(np.mean(outro_a) - np.mean(intro_b))
        return max(0.2, 1.0 - level_diff * 2)

    correlation = np.corrcoef(outro_a, intro_b)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0

    # Combine correlation with level matching
    level_diff = abs(np.mean(outro_a) - np.mean(intro_b))
    level_score = max(0.0, 1.0 - level_diff * 2)

    # Positive correlation is good (both rising/falling together)
    corr_score = max(0.0, (correlation + 1) / 2)

    return 0.6 * level_score + 0.4 * corr_score


# ============== Drop Conflict Detection ==============

def score_drop_conflict(
    drop_times_a: List[float],
    drop_times_b: List[float],
    duration_a: float,
    duration_b: float,
    mix_zone_sec: float = 32.0
) -> float:
    """
    Score potential drop conflicts in mix zone (VECTORIZED).

    Drops should not occur in the overlap zone (last N seconds of A,
    first N seconds of B), as they may clash.

    M2 Optimized: Uses numpy comparison instead of Python loops.

    Args:
        drop_times_a: Drop timestamps in track A (seconds)
        drop_times_b: Drop timestamps in track B (seconds)
        duration_a: Total duration of track A (seconds)
        duration_b: Total duration of track B (seconds)
        mix_zone_sec: Mix zone duration (default 32 sec = ~16 bars @ 120 BPM)

    Returns:
        Score 0-1 (1 = no conflict, 0 = drops in mix zone)
    """
    # Convert to arrays for vectorized operations
    drops_a = np.asarray(drop_times_a, dtype=np.float32)
    drops_b = np.asarray(drop_times_b, dtype=np.float32)

    # Vectorized conflict counting
    # Track A: drops in outro (time_from_end < mix_zone_sec)
    conflicts_a = np.sum((duration_a - drops_a) < mix_zone_sec) if len(drops_a) > 0 else 0

    # Track B: drops in intro (drop_time < mix_zone_sec)
    conflicts_b = np.sum(drops_b < mix_zone_sec) if len(drops_b) > 0 else 0

    conflicts = conflicts_a + conflicts_b

    # Score: 0 conflicts → 1.0, 1 conflict → 0.7, 2+ → 0.4
    if conflicts == 0:
        return 1.0
    elif conflicts == 1:
        return 0.7
    elif conflicts == 2:
        return 0.4
    else:
        return 0.2


# ============== Spectral Compatibility ==============

def score_spectral_compatibility(
    centroid_a: float,
    centroid_b: float,
    max_diff_hz: float = 2000.0
) -> float:
    """
    Score spectral compatibility based on spectral centroid.

    Similar spectral centroids indicate similar timbral characteristics.

    Args:
        centroid_a: Track A mean spectral centroid (Hz)
        centroid_b: Track B mean spectral centroid (Hz)
        max_diff_hz: Maximum acceptable difference

    Returns:
        Score 0-1 (1 = similar timbre, 0 = very different)
    """
    diff = abs(centroid_a - centroid_b)
    score = max(0.0, 1.0 - (diff / max_diff_hz))
    return score


# ============== Genre Matching ==============

# Genre compatibility matrix (which genres mix well)
GENRE_COMPATIBILITY = {
    ('Techno', 'Techno'): 1.0,
    ('Techno', 'House'): 0.8,
    ('Techno', 'Trance'): 0.7,
    ('House', 'House'): 1.0,
    ('House', 'Techno'): 0.8,
    ('House', 'Disco/Funk'): 0.7,
    ('Trance', 'Trance'): 1.0,
    ('Trance', 'Techno'): 0.7,
    ('Bass', 'Bass'): 1.0,
    ('Bass', 'Hip-Hop'): 0.6,
    ('Hip-Hop', 'Hip-Hop'): 1.0,
    ('Ambient', 'Ambient'): 1.0,
    ('Ambient', 'House'): 0.5,
}


def score_genre_compatibility(
    genre_a: str,
    genre_b: str,
    confidence_a: float = 1.0,
    confidence_b: float = 1.0
) -> float:
    """
    Score genre compatibility between two tracks.

    Args:
        genre_a: Track A genre/category
        genre_b: Track B genre/category
        confidence_a: Confidence of genre_a classification
        confidence_b: Confidence of genre_b classification

    Returns:
        Score 0-1 (1 = same genre, 0 = incompatible)
    """
    # Same genre
    if genre_a == genre_b:
        return 1.0

    # Check compatibility matrix (both directions)
    key1 = (genre_a, genre_b)
    key2 = (genre_b, genre_a)

    if key1 in GENRE_COMPATIBILITY:
        base_score = GENRE_COMPATIBILITY[key1]
    elif key2 in GENRE_COMPATIBILITY:
        base_score = GENRE_COMPATIBILITY[key2]
    else:
        base_score = 0.4  # Unknown combination - moderate penalty

    # Weight by classification confidence
    confidence_weight = (confidence_a + confidence_b) / 2

    # If confidence is low, be more lenient
    if confidence_weight < 0.5:
        base_score = 0.5 + base_score * 0.5  # Pull towards neutral

    return base_score


# ============== BPM Compatibility ==============

def score_bpm_compatibility(
    bpm_a: float,
    bpm_b: float,
    max_diff_percent: float = 6.0
) -> float:
    """
    Score BPM compatibility between two tracks.

    DJ mixers can typically adjust pitch ±6-8%, so tracks
    within this range are mixable.

    Args:
        bpm_a: Track A tempo in BPM
        bpm_b: Track B tempo in BPM
        max_diff_percent: Maximum acceptable BPM difference (%)

    Returns:
        Score 0-1 (1 = same BPM, 0 = unmixable)
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.5  # Unknown BPM

    # Calculate percentage difference
    diff_percent = abs(bpm_a - bpm_b) / min(bpm_a, bpm_b) * 100

    # Also check half/double tempo (for house↔techno transitions)
    half_double_diff = min(
        abs(bpm_a - bpm_b * 2) / bpm_a * 100,
        abs(bpm_a * 2 - bpm_b) / (bpm_a * 2) * 100
    )

    # Use the better match (direct or half/double)
    effective_diff = min(diff_percent, half_double_diff)

    if effective_diff <= 2.0:
        return 1.0
    elif effective_diff <= max_diff_percent:
        return 0.8
    elif effective_diff <= max_diff_percent * 1.5:
        return 0.5
    else:
        return 0.2


# ============== Combined Transition Score ==============

@dataclass
class TransitionScore:
    """
    Combined transition compatibility score.

    Weights:
    - harmonic: 30% - Key compatibility is crucial
    - energy: 25% - Energy flow affects dancefloor
    - drop_conflict: 20% - Avoid clashing drops
    - spectral: 15% - Timbral similarity
    - genre: 10% - Genre matching
    """
    harmonic: float        # 0-1, Camelot wheel compatibility
    energy: float          # 0-1, outro→intro energy flow
    drop_conflict: float   # 0-1, no drops in mix zone = good
    spectral: float        # 0-1, spectral centroid similarity
    genre: float           # 0-1, genre match
    bpm: float = 1.0       # 0-1, BPM compatibility (bonus factor)

    @property
    def total(self) -> float:
        """
        Calculate weighted total score.

        Returns:
            Combined score 0-1
        """
        base_score = (
            0.30 * self.harmonic +
            0.25 * self.energy +
            0.20 * self.drop_conflict +
            0.15 * self.spectral +
            0.10 * self.genre
        )

        # BPM as multiplier (if BPM is bad, reduce total)
        return base_score * (0.5 + 0.5 * self.bpm)

    @property
    def quality(self) -> str:
        """
        Human-readable quality rating.

        Returns:
            Quality string: "Excellent", "Good", "Fair", "Poor"
        """
        total = self.total
        if total >= 0.8:
            return "Excellent"
        elif total >= 0.6:
            return "Good"
        elif total >= 0.4:
            return "Fair"
        else:
            return "Poor"

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'harmonic': float(self.harmonic),
            'energy': float(self.energy),
            'drop_conflict': float(self.drop_conflict),
            'spectral': float(self.spectral),
            'genre': float(self.genre),
            'bpm': float(self.bpm),
            'total': float(self.total),
            'quality': self.quality,
        }


def compute_transition_score(
    # Track A (outgoing)
    camelot_a: str,
    outro_energy_a: float,
    drop_times_a: List[float],
    duration_a: float,
    spectral_centroid_a: float,
    genre_a: str,
    bpm_a: float,
    # Track B (incoming)
    camelot_b: str,
    intro_energy_b: float,
    drop_times_b: List[float],
    duration_b: float,
    spectral_centroid_b: float,
    genre_b: str,
    bpm_b: float,
    # Optional
    genre_confidence_a: float = 1.0,
    genre_confidence_b: float = 1.0,
    mix_zone_sec: float = 32.0,
) -> TransitionScore:
    """
    Compute complete transition score between two tracks.

    Args:
        camelot_a: Track A Camelot key (e.g., '8A')
        outro_energy_a: Track A energy at outro (0-1)
        drop_times_a: Track A drop timestamps
        duration_a: Track A duration (seconds)
        spectral_centroid_a: Track A mean spectral centroid
        genre_a: Track A genre
        bpm_a: Track A tempo
        camelot_b: Track B Camelot key
        intro_energy_b: Track B energy at intro (0-1)
        drop_times_b: Track B drop timestamps
        duration_b: Track B duration
        spectral_centroid_b: Track B mean spectral centroid
        genre_b: Track B genre
        bpm_b: Track B tempo
        genre_confidence_a: Classification confidence for A
        genre_confidence_b: Classification confidence for B
        mix_zone_sec: Mix zone duration for drop conflict check

    Returns:
        TransitionScore with all component scores
    """
    return TransitionScore(
        harmonic=score_harmonic_compatibility(camelot_a, camelot_b),
        energy=score_energy_flow(outro_energy_a, intro_energy_b),
        drop_conflict=score_drop_conflict(
            drop_times_a, drop_times_b, duration_a, duration_b, mix_zone_sec
        ),
        spectral=score_spectral_compatibility(
            spectral_centroid_a, spectral_centroid_b
        ),
        genre=score_genre_compatibility(
            genre_a, genre_b, genre_confidence_a, genre_confidence_b
        ),
        bpm=score_bpm_compatibility(bpm_a, bpm_b),
    )
