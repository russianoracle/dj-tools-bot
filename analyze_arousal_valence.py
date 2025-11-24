#!/usr/bin/env python3
"""
Quick test script to analyze arousal/valence features on a subset of tracks.

This script:
1. Creates a stratified subset of tracks (balanced by zone)
2. Extracts arousal/valence using music emotion model
3. Analyzes variance and correlation with zones
4. Generates visualizations (boxplots, scatter plots)
5. Provides statistical analysis (ANOVA, means, std)

Usage:
    python analyze_arousal_valence.py tests/test_data_prepared.tsv --subset-size 60
    python analyze_arousal_valence.py tests/test_data_prepared.tsv --subset-size 60 --output results/arousal_valence_analysis.csv
"""

import sys
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.zone_features import ZoneFeatureExtractor
from src.utils import get_logger

logger = get_logger(__name__)

# Zone colors for visualization
ZONE_COLORS = {
    'yellow': '#FFD700',
    'green': '#90EE90',
    'purple': '#DDA0DD'
}


def load_data_file(data_path: Path) -> List[Dict]:
    """Load TSV data file with zone labels."""
    logger.info(f"Loading data from {data_path}")

    with open(data_path, 'r', encoding='utf-16') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)

    # Process rows to extract path and zone
    processed = []
    for row in rows:
        file_path = (
            row.get('Location', '').strip() or
            row.get('Path', '').strip() or
            row.get('путь', '').strip()
        )

        zone = (
            row.get('Zone', '').strip() or
            row.get('zone', '').strip() or
            row.get('зона', '').strip()
        ).lower()

        if file_path and zone in ['yellow', 'green', 'purple']:
            # Check if file exists
            if Path(file_path).exists():
                processed.append({
                    'path': file_path,
                    'zone': zone
                })
            else:
                logger.warning(f"File not found: {file_path}")

    logger.info(f"Loaded {len(processed)} valid tracks")
    return processed


def create_stratified_subset(data: List[Dict], subset_size: int) -> List[Dict]:
    """Create stratified subset with balanced zones."""
    logger.info(f"Creating stratified subset of {subset_size} tracks")

    # Group by zone
    zones = {}
    for item in data:
        zone = item['zone']
        if zone not in zones:
            zones[zone] = []
        zones[zone].append(item)

    # Calculate samples per zone
    samples_per_zone = subset_size // 3

    # Sample from each zone
    subset = []
    for zone, items in zones.items():
        available = len(items)
        sample_size = min(samples_per_zone, available)

        # Random sampling
        np.random.seed(42)  # For reproducibility
        sampled = np.random.choice(items, size=sample_size, replace=False)
        subset.extend(sampled)

        logger.info(f"  {zone}: sampled {sample_size} from {available} tracks")

    logger.info(f"Total subset size: {len(subset)} tracks")
    return subset


def extract_arousal_valence(subset: List[Dict], use_gpu: bool = True) -> pd.DataFrame:
    """Extract arousal/valence for subset of tracks."""
    logger.info("Initializing music emotion feature extractor")
    logger.info("⚠️  This will take ~15 seconds per track!")

    extractor = ZoneFeatureExtractor(
        use_gpu=use_gpu,
        use_embeddings=False,
        use_music_emotion=True  # ENABLE emotion extraction!
    )

    results = []
    total = len(subset)

    for i, item in enumerate(subset, 1):
        path = item['path']
        zone = item['zone']

        logger.info(f"[{i}/{total}] Extracting features: {Path(path).name}")

        try:
            # Extract features (this will include arousal/valence)
            features = extractor.extract(path)

            # Get arousal and valence
            # These are in the features object
            feature_dict = {
                'path': path,
                'filename': Path(path).name,
                'zone': zone,
                'arousal': features.arousal,
                'valence': features.valence
            }

            results.append(feature_dict)
            logger.info(f"  ✓ arousal={features.arousal:.3f}, valence={features.valence:.3f}")

        except Exception as e:
            logger.error(f"  ✗ Failed to extract features: {e}")
            # Add with NaN values
            results.append({
                'path': path,
                'filename': Path(path).name,
                'zone': zone,
                'arousal': np.nan,
                'valence': np.nan
            })

    df = pd.DataFrame(results)
    logger.info(f"✓ Extraction complete: {len(df)} tracks processed")

    return df


def analyze_variance(df: pd.DataFrame) -> Dict:
    """Analyze variance of arousal/valence features."""
    logger.info("\n" + "="*70)
    logger.info("VARIANCE ANALYSIS")
    logger.info("="*70)

    results = {}

    for feature in ['arousal', 'valence']:
        values = df[feature].dropna()

        logger.info(f"\n{feature.upper()}:")
        logger.info(f"  Min: {values.min():.3f}")
        logger.info(f"  Max: {values.max():.3f}")
        logger.info(f"  Mean: {values.mean():.3f}")
        logger.info(f"  Std: {values.std():.3f}")
        logger.info(f"  Range: {values.max() - values.min():.3f}")

        results[feature] = {
            'min': values.min(),
            'max': values.max(),
            'mean': values.mean(),
            'std': values.std(),
            'range': values.max() - values.min()
        }

        # Check if variance is meaningful
        if values.std() < 0.01:
            logger.warning(f"  ⚠️  VERY LOW VARIANCE! Feature may not be useful.")
        elif values.std() < 0.1:
            logger.warning(f"  ⚠️  Low variance detected.")
        else:
            logger.info(f"  ✓ Good variance detected!")

    return results


def analyze_by_zone(df: pd.DataFrame) -> Dict:
    """Analyze arousal/valence statistics by zone."""
    logger.info("\n" + "="*70)
    logger.info("ZONE-WISE ANALYSIS")
    logger.info("="*70)

    results = {}

    for zone in ['yellow', 'green', 'purple']:
        zone_data = df[df['zone'] == zone]

        if len(zone_data) == 0:
            continue

        logger.info(f"\n{zone.upper()} ZONE (n={len(zone_data)}):")

        zone_results = {}
        for feature in ['arousal', 'valence']:
            values = zone_data[feature].dropna()

            if len(values) > 0:
                logger.info(f"  {feature}: {values.mean():.3f} ± {values.std():.3f}")
                zone_results[feature] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': values.median()
                }
            else:
                logger.warning(f"  {feature}: No valid values")
                zone_results[feature] = {'mean': np.nan, 'std': np.nan, 'median': np.nan}

        results[zone] = zone_results

    return results


def perform_anova(df: pd.DataFrame) -> Dict:
    """Perform ANOVA test to check if zones differ significantly."""
    logger.info("\n" + "="*70)
    logger.info("STATISTICAL SIGNIFICANCE TESTING (ANOVA)")
    logger.info("="*70)

    results = {}

    for feature in ['arousal', 'valence']:
        # Get values for each zone
        yellow = df[df['zone'] == 'yellow'][feature].dropna()
        green = df[df['zone'] == 'green'][feature].dropna()
        purple = df[df['zone'] == 'purple'][feature].dropna()

        # Perform one-way ANOVA
        if len(yellow) > 0 and len(green) > 0 and len(purple) > 0:
            f_stat, p_value = stats.f_oneway(yellow, green, purple)

            logger.info(f"\n{feature.upper()}:")
            logger.info(f"  F-statistic: {f_stat:.3f}")
            logger.info(f"  p-value: {p_value:.4f}")

            if p_value < 0.001:
                logger.info(f"  ✓✓✓ HIGHLY SIGNIFICANT! Zones differ strongly (p < 0.001)")
                significance = "highly_significant"
            elif p_value < 0.01:
                logger.info(f"  ✓✓ VERY SIGNIFICANT! Zones differ (p < 0.01)")
                significance = "very_significant"
            elif p_value < 0.05:
                logger.info(f"  ✓ SIGNIFICANT! Zones differ (p < 0.05)")
                significance = "significant"
            else:
                logger.info(f"  ✗ NOT SIGNIFICANT. Zones do not differ meaningfully (p >= 0.05)")
                significance = "not_significant"

            results[feature] = {
                'f_stat': f_stat,
                'p_value': p_value,
                'significance': significance
            }
        else:
            logger.warning(f"{feature}: Insufficient data for ANOVA")
            results[feature] = None

    return results


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create boxplots and scatter plots."""
    logger.info("\n" + "="*70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 5)

    # 1. Boxplots for arousal and valence by zone
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Arousal boxplot
    sns.boxplot(data=df, x='zone', y='arousal', ax=axes[0],
                order=['yellow', 'green', 'purple'],
                palette=ZONE_COLORS)
    axes[0].set_title('Arousal by Energy Zone', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Energy Zone', fontsize=12)
    axes[0].set_ylabel('Arousal', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Valence boxplot
    sns.boxplot(data=df, x='zone', y='valence', ax=axes[1],
                order=['yellow', 'green', 'purple'],
                palette=ZONE_COLORS)
    axes[1].set_title('Valence by Energy Zone', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Energy Zone', fontsize=12)
    axes[1].set_ylabel('Valence', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    boxplot_path = output_dir / 'arousal_valence_boxplots.png'
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Saved boxplots: {boxplot_path}")
    plt.close()

    # 2. Scatter plot: arousal vs valence colored by zone
    plt.figure(figsize=(10, 8))

    for zone in ['yellow', 'green', 'purple']:
        zone_data = df[df['zone'] == zone]
        plt.scatter(zone_data['arousal'], zone_data['valence'],
                   c=ZONE_COLORS[zone], label=zone.capitalize(),
                   alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    plt.xlabel('Arousal', fontsize=12)
    plt.ylabel('Valence', fontsize=12)
    plt.title('Arousal-Valence Space by Energy Zone', fontsize=14, fontweight='bold')
    plt.legend(title='Energy Zone', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    scatter_path = output_dir / 'arousal_valence_scatter.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Saved scatter plot: {scatter_path}")
    plt.close()

    logger.info(f"✓ Visualizations saved to: {output_dir}")


def make_recommendation(variance_results: Dict, anova_results: Dict, zone_results: Dict) -> str:
    """Make recommendation based on analysis."""
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATION")
    logger.info("="*70)

    # Check arousal variance
    arousal_std = variance_results['arousal']['std']
    arousal_range = variance_results['arousal']['range']

    # Check arousal significance
    arousal_sig = anova_results.get('arousal', {})
    arousal_p = arousal_sig.get('p_value', 1.0) if arousal_sig else 1.0

    # Check zone separation for arousal
    yellow_arousal = zone_results.get('yellow', {}).get('arousal', {}).get('mean', 0)
    purple_arousal = zone_results.get('purple', {}).get('arousal', {}).get('mean', 0)
    arousal_separation = abs(purple_arousal - yellow_arousal)

    logger.info(f"\nArousal Analysis:")
    logger.info(f"  Standard deviation: {arousal_std:.3f}")
    logger.info(f"  Range: {arousal_range:.3f}")
    logger.info(f"  ANOVA p-value: {arousal_p:.4f}")
    logger.info(f"  Purple-Yellow separation: {arousal_separation:.3f}")

    # Decision criteria
    has_variance = arousal_std > 0.1
    is_significant = arousal_p < 0.05
    good_separation = arousal_separation > 0.2

    logger.info(f"\nDecision Criteria:")
    logger.info(f"  ✓ Has meaningful variance (std > 0.1): {has_variance}")
    logger.info(f"  ✓ Statistically significant (p < 0.05): {is_significant}")
    logger.info(f"  ✓ Good zone separation (diff > 0.2): {good_separation}")

    # Make recommendation
    if has_variance and is_significant and good_separation:
        recommendation = "ENABLE_FULL_DATASET"
        logger.info(f"\n{'='*70}")
        logger.info(f"✓✓✓ RECOMMENDATION: ENABLE FOR FULL DATASET")
        logger.info(f"{'='*70}")
        logger.info("Arousal/valence features show:")
        logger.info("  • Meaningful variance across tracks")
        logger.info("  • Statistically significant differences between zones")
        logger.info("  • Good separation between yellow and purple zones")
        logger.info("\nNext steps:")
        logger.info("  1. Extract arousal/valence for all 415 tracks (~2 hours)")
        logger.info("  2. Retrain model with these features included")
        logger.info("  3. Expected improvement: +5-10% accuracy, especially for purple zone")

    elif has_variance and is_significant:
        recommendation = "ENABLE_WITH_CAUTION"
        logger.info(f"\n{'='*70}")
        logger.info(f"⚠️  RECOMMENDATION: ENABLE WITH CAUTION")
        logger.info(f"{'='*70}")
        logger.info("Arousal/valence features show some promise but need validation:")
        logger.info(f"  • Variance: {'Good' if has_variance else 'Low'}")
        logger.info(f"  • Significance: {'Yes' if is_significant else 'No'}")
        logger.info(f"  • Separation: {'Good' if good_separation else 'Marginal'}")
        logger.info("\nNext steps:")
        logger.info("  1. Try extracting for larger subset (100-150 tracks)")
        logger.info("  2. Validate results before committing to full extraction")

    else:
        recommendation = "DO_NOT_ENABLE"
        logger.info(f"\n{'='*70}")
        logger.info(f"✗ RECOMMENDATION: DO NOT ENABLE")
        logger.info(f"{'='*70}")
        logger.info("Arousal/valence features do NOT show sufficient value:")
        logger.info(f"  • Variance: {'Good' if has_variance else 'TOO LOW'}")
        logger.info(f"  • Significance: {'Yes' if is_significant else 'NO'}")
        logger.info(f"  • Separation: {'Good' if good_separation else 'INSUFFICIENT'}")
        logger.info("\nNext steps:")
        logger.info("  1. Remove arousal/valence from feature set")
        logger.info("  2. Proceed with feature selection (TOP-20 strategy)")
        logger.info("  3. Focus on proven features (peak_energy_ratio, energy_slope, etc.)")

    return recommendation


def main():
    parser = argparse.ArgumentParser(
        description='Analyze arousal/valence features on subset of tracks'
    )

    parser.add_argument(
        'data_file',
        type=str,
        help='Path to TSV file with zone labels'
    )

    parser.add_argument(
        '--subset-size',
        type=int,
        default=60,
        help='Size of subset to analyze (default: 60, ~20 per zone)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file to save results'
    )

    parser.add_argument(
        '--viz-dir',
        type=str,
        default='results/arousal_valence_viz',
        help='Directory to save visualizations (default: results/arousal_valence_viz)'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU (slower)'
    )

    args = parser.parse_args()

    print("="*70)
    print("AROUSAL/VALENCE FEATURE ANALYSIS")
    print("="*70)
    print(f"Data file: {args.data_file}")
    print(f"Subset size: {args.subset_size}")
    print(f"Use GPU: {not args.no_gpu}")
    print("="*70)
    print()

    try:
        # 1. Load data
        data = load_data_file(Path(args.data_file))

        if len(data) < args.subset_size:
            logger.warning(f"Only {len(data)} tracks available, requested {args.subset_size}")
            args.subset_size = len(data)

        # 2. Create subset
        subset = create_stratified_subset(data, args.subset_size)

        # 3. Extract arousal/valence
        df = extract_arousal_valence(subset, use_gpu=not args.no_gpu)

        # 4. Analyze variance
        variance_results = analyze_variance(df)

        # 5. Analyze by zone
        zone_results = analyze_by_zone(df)

        # 6. Perform ANOVA
        anova_results = perform_anova(df)

        # 7. Create visualizations
        viz_dir = Path(args.viz_dir)
        create_visualizations(df, viz_dir)

        # 8. Make recommendation
        recommendation = make_recommendation(variance_results, anova_results, zone_results)

        # 9. Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"\n✓ Results saved to: {output_path}")

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"Final recommendation: {recommendation}")
        print(f"Visualizations: {viz_dir}")
        if args.output:
            print(f"Data: {args.output}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Analysis failed")
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
