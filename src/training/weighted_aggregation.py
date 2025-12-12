"""
📊 Weighted Frame Aggregation for Track-Level Prediction

Instead of simple majority voting, this module provides smarter
aggregation strategies that:
1. Weight frames by model confidence
2. Give more weight to later frames (drop zone in tracks)
3. Consider local agreement between neighboring frames

This is particularly important for PURPLE classification where
the drop section (usually at the end) should dominate prediction.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class AggregationResult:
    """Result of frame aggregation."""
    prediction: np.ndarray       # (3,) class probabilities
    predicted_class: int         # argmax of prediction
    confidence: float            # max probability
    attention_weights: np.ndarray  # weights assigned to each frame


def simple_weighted_aggregation(
    frame_probs: np.ndarray,
    temporal_boost_start: float = 0.6,
    temporal_boost_factor: float = 3.0,
    confidence_weight: float = 0.3,
    temporal_weight: float = 0.5,
    agreement_weight: float = 0.2
) -> AggregationResult:
    """
    Simple but effective weighted aggregation without neural networks.

    Args:
        frame_probs: (n_frames, 3) - predicted probabilities for each frame
        temporal_boost_start: Where to start boosting (0.7 = last 30%)
        temporal_boost_factor: How much to boost late frames
        confidence_weight: Weight for confidence-based weighting
        temporal_weight: Weight for temporal position
        agreement_weight: Weight for local agreement

    Returns:
        AggregationResult with final prediction
    """
    n_frames = frame_probs.shape[0]

    if n_frames == 0:
        return AggregationResult(
            prediction=np.array([0.33, 0.34, 0.33]),
            predicted_class=1,  # GREEN default
            confidence=0.34,
            attention_weights=np.array([])
        )

    # === Weight 1: Model confidence ===
    # Frames where model is more confident get higher weight
    conf_weights = np.max(frame_probs, axis=1)  # (n_frames,)

    # === Weight 2: Temporal position ===
    # Later frames (drop zone) get boosted
    position = np.arange(n_frames) / (n_frames - 1) if n_frames > 1 else np.array([0.5])
    temp_weights = np.ones(n_frames)

    # Boost frames in the last portion of track
    late_mask = position >= temporal_boost_start
    temp_weights[late_mask] = np.linspace(1.0, temporal_boost_factor,
                                           np.sum(late_mask))

    # Extra boost for final section (potential big drop)
    final_mask = position >= 0.9
    temp_weights[final_mask] = temporal_boost_factor * 1.2

    # === Weight 3: Local agreement ===
    # If neighboring frames agree, boost confidence
    agreement_weights = np.ones(n_frames)

    if n_frames >= 3:
        for i in range(1, n_frames - 1):
            top_class = np.argmax(frame_probs[i])
            prev_class = np.argmax(frame_probs[i - 1])
            next_class = np.argmax(frame_probs[i + 1])

            # Boost if 3 consecutive frames agree
            if top_class == prev_class == next_class:
                agreement_weights[i] = 1.5
            # Slight boost if 2 agree
            elif top_class == prev_class or top_class == next_class:
                agreement_weights[i] = 1.2

    # === Combine weights ===
    combined_weights = (
        confidence_weight * _normalize_weights(conf_weights) +
        temporal_weight * _normalize_weights(temp_weights) +
        agreement_weight * _normalize_weights(agreement_weights)
    )

    # Normalize to sum to 1
    combined_weights = combined_weights / combined_weights.sum()

    # === Weighted aggregation ===
    # Weighted sum of probabilities
    weighted_probs = (frame_probs.T @ combined_weights)  # (3,)

    return AggregationResult(
        prediction=weighted_probs,
        predicted_class=int(np.argmax(weighted_probs)),
        confidence=float(np.max(weighted_probs)),
        attention_weights=combined_weights
    )


def confidence_threshold_aggregation(
    frame_probs: np.ndarray,
    frame_positions: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.7,
    fallback_strategy: str = 'weighted'
) -> Tuple[int, float, str]:
    """
    Aggregation with confidence threshold for production use.

    Returns prediction only if confidence is above threshold,
    otherwise returns 'uncertain' status.

    Args:
        frame_probs: (n_frames, 3) - predicted probabilities
        frame_positions: Optional (n_frames,) - position in track [0, 1]
        confidence_threshold: Minimum confidence for prediction
        fallback_strategy: 'weighted' or 'majority'

    Returns:
        Tuple of (predicted_class, confidence, status)
        status is 'confident' or 'uncertain'
    """
    if fallback_strategy == 'weighted':
        result = simple_weighted_aggregation(frame_probs)
        pred_class = result.predicted_class
        confidence = result.confidence
    else:
        # Simple majority voting
        frame_preds = np.argmax(frame_probs, axis=1)
        pred_class = int(np.bincount(frame_preds, minlength=3).argmax())
        # Confidence = proportion of agreeing frames
        confidence = float((frame_preds == pred_class).mean())

    if confidence >= confidence_threshold:
        return pred_class, confidence, 'confident'
    else:
        return pred_class, confidence, 'uncertain'


def purple_boosted_aggregation(
    frame_probs: np.ndarray,
    purple_boost: float = 1.3,
    late_purple_boost: float = 1.5
) -> AggregationResult:
    """
    Aggregation strategy that boosts PURPLE predictions.

    This compensates for the fact that PURPLE is underrepresented
    and critical for DJ workflow.

    Args:
        frame_probs: (n_frames, 3) - probabilities [YELLOW, GREEN, PURPLE]
        purple_boost: Base boost for PURPLE probabilities
        late_purple_boost: Extra boost for PURPLE in late frames

    Returns:
        AggregationResult
    """
    n_frames = frame_probs.shape[0]
    if n_frames == 0:
        return AggregationResult(
            prediction=np.array([0.33, 0.34, 0.33]),
            predicted_class=1,
            confidence=0.34,
            attention_weights=np.array([])
        )

    # Apply PURPLE boost
    boosted_probs = frame_probs.copy()
    boosted_probs[:, 2] *= purple_boost  # PURPLE is index 2

    # Extra boost for late frames
    late_start = int(n_frames * 0.7)
    boosted_probs[late_start:, 2] *= late_purple_boost / purple_boost

    # Renormalize probabilities
    row_sums = boosted_probs.sum(axis=1, keepdims=True)
    boosted_probs = boosted_probs / row_sums

    # Use weighted aggregation on boosted probs
    return simple_weighted_aggregation(boosted_probs)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize weights to [0, 1] range."""
    if len(weights) == 0:
        return weights
    w_min = weights.min()
    w_max = weights.max()
    if w_max - w_min < 1e-6:
        return np.ones_like(weights) / len(weights)
    return (weights - w_min) / (w_max - w_min)


# ============================================================
# Comparison utilities
# ============================================================

def compare_aggregation_strategies(
    frame_probs: np.ndarray,
    true_label: int
) -> dict:
    """
    Compare different aggregation strategies on a single track.

    Args:
        frame_probs: (n_frames, 3) - predicted probabilities
        true_label: Ground truth label (0, 1, or 2)

    Returns:
        Dict with results from each strategy
    """
    results = {}

    # Majority voting
    preds = np.argmax(frame_probs, axis=1)
    majority_pred = int(np.bincount(preds, minlength=3).argmax())
    results['majority'] = {
        'prediction': majority_pred,
        'correct': majority_pred == true_label,
        'confidence': (preds == majority_pred).mean()
    }

    # Weighted aggregation
    weighted_result = simple_weighted_aggregation(frame_probs)
    results['weighted'] = {
        'prediction': weighted_result.predicted_class,
        'correct': weighted_result.predicted_class == true_label,
        'confidence': weighted_result.confidence
    }

    # Purple boosted
    purple_result = purple_boosted_aggregation(frame_probs)
    results['purple_boosted'] = {
        'prediction': purple_result.predicted_class,
        'correct': purple_result.predicted_class == true_label,
        'confidence': purple_result.confidence
    }

    return results


# ============================================================
# Test
# ============================================================

if __name__ == '__main__':
    # Create synthetic test data
    np.random.seed(42)

    # Simulate a PURPLE track with drop at the end
    n_frames = 100
    frame_probs = np.zeros((n_frames, 3))

    # First 70% looks like GREEN
    frame_probs[:70] = [0.2, 0.6, 0.2]

    # Last 30% is the drop - PURPLE
    frame_probs[70:] = [0.1, 0.3, 0.6]

    # Add some noise
    frame_probs += np.random.normal(0, 0.05, frame_probs.shape)
    frame_probs = np.clip(frame_probs, 0.01, 0.99)
    frame_probs = frame_probs / frame_probs.sum(axis=1, keepdims=True)

    print("Test: PURPLE track with drop in last 30%")
    print("=" * 50)

    results = compare_aggregation_strategies(frame_probs, true_label=2)  # PURPLE

    zones = ['YELLOW', 'GREEN', 'PURPLE']
    for strategy, res in results.items():
        status = "✅" if res['correct'] else "❌"
        print(f"{strategy:20s}: {zones[res['prediction']]:8s} "
              f"conf={res['confidence']:.2f} {status}")
