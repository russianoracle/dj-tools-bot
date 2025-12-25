#!/usr/bin/env python3
"""
Batch DJ Set Profiling

Processes all audio files in a directory and saves them to DJ profile database.

Usage:
    python scripts/batch_profile_dj.py --dj "Nina Kraviz" --folder data/dj_sets/nina-kraviz/ --save-to-db
    python scripts/batch_profile_dj.py --folder data/dj_sets/ --auto-detect-dj
"""

import sys
import argparse
from pathlib import Path
import time
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress librosa warnings
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
warnings.filterwarnings("ignore", message="librosa.beat.tempo")

from src.core.pipelines import (
    Pipeline,
    PipelineContext,
    LoadAudioStage,
    ComputeSTFTStage,
)
from src.core.pipelines.dj_profiling import DJProfilingStage
from src.core.cache import CacheRepository


# Supported audio formats
AUDIO_EXTENSIONS = {'.mp3', '.m4a', '.flac', '.wav', '.opus', '.ogg'}


# ============== Progress Display ==============

class ProgressReporter:
    """
    Displays pipeline progress with stages and tasks.

    Output format:
    ┌─ LoadAudio ─────────────────────────────────────────┐
    │  Loading audio...                                    │
    │  ✓ 3842.5s @ 22050 Hz                              │  1.2s
    └──────────────────────────────────────────────────────┘

    ┌─ ComputeSTFT ───────────────────────────────────────┐
    │  Computing STFT...                                   │
    │  ✓ shape=(1025, 165076), hop_length=2048           │  2.3s
    └──────────────────────────────────────────────────────┘
    """

    STAGE_NAMES = {
        'LoadAudioStage': '🎵 Load Audio',
        'ComputeSTFTStage': '📊 Compute STFT',
        'DJProfilingStage': '🎛️ DJ Profiling',
        'EnergyArcProfilingStage': '📈 Energy Arc',
        'DropPatternProfilingStage': '💥 Drop Detection',
        'TempoDistributionProfilingStage': '🥁 Tempo Analysis',
        'KeyAnalysisProfilingStage': '🎹 Key Analysis',
        'GenreProfilingStage': '🎸 Genre Analysis',
    }

    SUBSTAGE_NAMES = {
        'beat_grid': '🎼 Beat Grid',
        'energy_arc': '📈 Energy Arc',
        'drop_pattern': '💥 Drop Detection',
        'tempo_distribution': '🥁 Tempo Analysis',
        'key_analysis': '🎹 Key Analysis',
        'genre': '🎸 Genre Analysis',
    }

    def __init__(self, verbose: bool = True, show_substages: bool = True):
        self.verbose = verbose
        self.show_substages = show_substages
        self.current_stage = None
        self.stage_start_time = None
        self.substage_times = {}

    def on_stage_start(self, stage_name: str):
        """Called when a pipeline stage starts."""
        self.current_stage = stage_name
        self.stage_start_time = time.time()

        display_name = self.STAGE_NAMES.get(stage_name, stage_name)

        if self.verbose:
            print(f"  ├─ {display_name}...", end='', flush=True)

    def on_stage_complete(self, stage_name: str, context):
        """Called when a pipeline stage completes."""
        elapsed = time.time() - self.stage_start_time if self.stage_start_time else 0

        if self.verbose:
            print(f" ✓ ({elapsed:.1f}s)")

        # Show substage details for DJProfilingStage
        if stage_name == 'DJProfilingStage' and self.show_substages:
            self._print_profiling_results(context)

    def _print_profiling_results(self, context):
        """Print detailed results from DJ profiling."""
        # Energy Arc
        energy_arc = context.get_result('energy_arc_metrics')
        if energy_arc:
            print(f"  │   └─ Energy Arc: {energy_arc.arc_shape} "
                  f"(open={energy_arc.opening_energy:.2f}, "
                  f"peak={energy_arc.peak_energy:.2f}, "
                  f"close={energy_arc.closing_energy:.2f})")

        # Drop Pattern
        drop_pattern = context.get_result('drop_pattern_metrics')
        if drop_pattern:
            print(f"  │   └─ Drop Pattern: {drop_pattern.drop_clustering} "
                  f"({drop_pattern.drops_per_hour:.1f} drops/h, "
                  f"avg={drop_pattern.avg_drop_magnitude:.2f})")

        # Tempo
        tempo = context.get_result('tempo_distribution_metrics')
        if tempo:
            print(f"  │   └─ Tempo: {tempo.tempo_mean:.1f} BPM "
                  f"(range {tempo.tempo_min:.0f}-{tempo.tempo_max:.0f})")

        # Key
        key_analysis = context.get_result('key_analysis_metrics')
        if key_analysis:
            print(f"  │   └─ Key: {key_analysis.dominant_camelot} "
                  f"({key_analysis.dominant_key})")


def create_progress_pipeline(fast_mode: bool, skip_energy_arc: bool, skip_drops: bool,
                              include_genre: bool, progress_reporter: ProgressReporter):
    """Create pipeline with progress callbacks."""

    hop_length = 2048 if fast_mode else 512

    # Patch DJProfilingStage to report substage progress
    original_process = DJProfilingStage.process

    def patched_process(self, context):
        """Enhanced process method with progress reporting."""
        reporter = getattr(self, '_progress_reporter', None)

        # Run each stage with timing (beat_grid FIRST for alignment)
        stages = [
            ('beat_grid', self.beat_grid_stage),
            ('energy_arc', self.energy_arc_stage),
            ('drop_pattern', self.drop_pattern_stage),
            ('tempo_distribution', self.tempo_distribution_stage),
            ('key_analysis', self.key_analysis_stage),
            ('genre', self.genre_stage),
        ]

        for stage_key, stage in stages:
            if stage is None:
                continue

            stage_name = ProgressReporter.SUBSTAGE_NAMES.get(stage_key, stage_key)

            if reporter and reporter.verbose:
                print(f"\n  │   ├─ {stage_name}...", end='', flush=True)

            start = time.time()
            context = stage.process(context)
            elapsed = time.time() - start

            if reporter and reporter.verbose:
                print(f" ✓ ({elapsed:.1f}s)", end='')

        if reporter and reporter.verbose:
            print()  # Newline after all substages

        # Aggregate results
        from src.core.pipelines.dj_profiling import DJProfileMetrics
        profile = DJProfileMetrics(
            energy_arc=context.get_result('energy_arc_metrics'),
            drop_pattern=context.get_result('drop_pattern_metrics'),
            tempo_distribution=context.get_result('tempo_distribution_metrics'),
            key_analysis=context.get_result('key_analysis_metrics'),
            genre=context.get_result('genre_metrics'),
        )
        context.set_result('dj_profile', profile)

        return context

    # Create pipeline stages
    load_stage = LoadAudioStage(sr=22050, mono=True)
    stft_stage = ComputeSTFTStage(hop_length=hop_length)
    profiling_stage = DJProfilingStage(
        include_energy_arc=not skip_energy_arc,
        include_drop_pattern=not skip_drops,
        include_genre=include_genre,
    )

    # Attach progress reporter to profiling stage
    profiling_stage._progress_reporter = progress_reporter
    profiling_stage.process = lambda ctx: patched_process(profiling_stage, ctx)

    def on_stage_complete(stage_name: str, context):
        progress_reporter.on_stage_complete(stage_name, context)

    pipeline = Pipeline(
        stages=[load_stage, stft_stage, profiling_stage],
        name="DJProfilingPipeline",
        on_stage_complete=on_stage_complete
    )

    return pipeline


def extract_dj_name_from_filename(filename: str) -> str:
    """
    Extract DJ name from filename.

    Examples:
        "nina-kraviz---@-trip-2023.mp3" -> "Nina Kraviz"
        "amelie_lens_fabric_2024.mp3" -> "Amelie Lens"
        "boiler-room---josh-baker-london.mp3" -> "Josh Baker"
    """
    # Remove extension
    name = Path(filename).stem

    # Split on common delimiters
    if '---' in name:
        # Format: "dj-name---venue-date"
        dj_part = name.split('---')[0]
    elif '_' in name:
        # Format: "dj_name_venue_date"
        parts = name.split('_')
        dj_part = parts[0] if len(parts) > 0 else name
    elif '-' in name:
        # Format: "dj-name-venue-date"
        parts = name.split('-')
        # Take first 2-3 parts as DJ name
        dj_part = '-'.join(parts[:2])
    else:
        dj_part = name

    # Convert to title case and replace delimiters with spaces
    dj_name = dj_part.replace('-', ' ').replace('_', ' ').strip()
    dj_name = ' '.join(word.capitalize() for word in dj_name.split())

    return dj_name


def find_audio_files(folder: Path) -> list:
    """Find all audio files in folder (recursive)."""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(folder.rglob(f'*{ext}'))
    return sorted(audio_files)


def process_set(
    file_path: Path,
    dj_name: str,
    cache_repo: CacheRepository,
    save_to_db: bool = False,
    fast_mode: bool = True,
    skip_energy_arc: bool = False,
    skip_drops: bool = False,
    include_genre: bool = False,
    verbose: bool = True
) -> dict:
    """Process single DJ set with detailed progress output."""

    # Header
    print(f"\n{'─'*70}")
    print(f"  📁 {file_path.name}")
    print(f"  👤 {dj_name}")
    print(f"{'─'*70}")

    # Check cache first (via CacheRepository)
    cached_set = cache_repo.get_set(str(file_path.absolute()))
    cached_result = cached_set.to_dict() if cached_set else None
    if cached_result:
        # Check if it's NEW format (DJ profiling) or OLD format (segmentation)
        has_dj_profile = any(k in cached_result for k in ['energy_arc', 'tempo_distribution', 'drop_pattern'])

        if has_dj_profile:
            print(f"  ✅ Loaded from cache (skipped analysis)")

            # Print cached results summary
            if verbose:
                _print_cached_summary(cached_result)

            # Return cached profile data (will be aggregated later)
            return {
                'success': True,
                'file': file_path.name,
                'from_cache': True,
                'profile_data': cached_result,
                'file_path': str(file_path.absolute()),
            }
        else:
            # Old format - need to re-analyze
            print(f"  ⚠️  Cache found but old format - re-analyzing...")

    # Create progress reporter
    progress_reporter = ProgressReporter(verbose=verbose, show_substages=True)

    # Create pipeline with progress callbacks
    pipeline = create_progress_pipeline(
        fast_mode=fast_mode,
        skip_energy_arc=skip_energy_arc,
        skip_drops=skip_drops,
        include_genre=include_genre,
        progress_reporter=progress_reporter
    )

    # Run pipeline
    start_time = time.time()

    try:
        context = PipelineContext(
            input_path=str(file_path),
            cache_dir=str(cache_repo.cache_dir)
        )

        # Run stages with progress reporting
        print(f"  ┌─ Pipeline")

        # Manually run stages with progress
        for i, stage in enumerate(pipeline.stages):
            progress_reporter.on_stage_start(stage.name)
            stage_start = time.time()
            context = stage.process(context)
            # on_stage_complete is called via callback

        elapsed = time.time() - start_time

        # Get results
        profile = context.get_result('dj_profile')

        if profile is None:
            print(f"  ❌ Failed to analyze")
            return {'success': False, 'error': 'No profile returned'}

        # Prepare set metadata
        set_data = profile.to_dict()
        set_data.update({
            'file_path': str(file_path.absolute()),
            'file_name': file_path.name,
            'duration_sec': context.get_result('_duration'),
            'analyzed_at': time.time(),
        })

        # Save to analysis cache (for fast reload)
        cache_repo.save_set_dict(str(file_path.absolute()), set_data)

        # Print final summary
        duration_min = context.get_result('_duration') / 60
        print(f"  └─ ✅ Complete ({elapsed:.1f}s total, {duration_min:.0f} min audio)")

        summary = {
            'success': True,
            'file': file_path.name,
            'duration_min': duration_min,
            'elapsed_sec': elapsed,
            'profile_data': set_data,
            'file_path': str(file_path.absolute()),
        }

        if profile.tempo_distribution:
            summary['tempo_mean'] = profile.tempo_distribution.tempo_mean
        if profile.drop_pattern:
            summary['drops_per_hour'] = profile.drop_pattern.drops_per_hour
        if profile.energy_arc:
            summary['arc_shape'] = profile.energy_arc.arc_shape

        return summary

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {'success': False, 'error': str(e)}


def _print_cached_summary(cached_result: dict):
    """Print summary from cached results."""
    # Energy Arc
    if 'energy_arc' in cached_result and cached_result['energy_arc']:
        arc = cached_result['energy_arc']
        print(f"      └─ Energy: {arc.get('arc_shape', '?')} "
              f"(peak={arc.get('peak_energy', 0):.2f})")

    # Drop Pattern (support both old and new format)
    drops_per_hour = _extract_drops_per_hour(cached_result)
    if drops_per_hour > 0:
        drop_clustering = ''
        if 'drop_pattern' in cached_result and cached_result['drop_pattern']:
            drop_clustering = f" ({cached_result['drop_pattern'].get('drop_clustering', '')})"
        print(f"      └─ Drops: {drops_per_hour:.1f}/h{drop_clustering}")

    # Tempo
    if 'tempo_distribution' in cached_result and cached_result['tempo_distribution']:
        tempo = cached_result['tempo_distribution']
        print(f"      └─ Tempo: {tempo.get('tempo_mean', 0):.0f} BPM")

    # Key
    if 'key_analysis' in cached_result and cached_result['key_analysis']:
        key = cached_result['key_analysis']
        print(f"      └─ Key: {key.get('dominant_camelot', '?')}")


def _extract_drops_per_hour(profile_data: dict) -> float:
    """
    Extract drops_per_hour from profile data.

    Supports both formats:
    - New format: drop_pattern.drops_per_hour
    - Old format: drops.density_per_min * 60
    """
    # New format (DJProfilingStage)
    if 'drop_pattern' in profile_data and profile_data['drop_pattern']:
        return profile_data['drop_pattern'].get('drops_per_hour', 0.0)

    # Old format (analyze_dj_set.py)
    if 'drops' in profile_data and profile_data['drops']:
        drops = profile_data['drops']
        if isinstance(drops, dict):
            density_per_min = drops.get('density_per_min', 0.0)
            return density_per_min * 60  # Convert to per hour

    return 0.0


def _aggregate_dj_profile(sets: list) -> dict:
    """
    Aggregate profile data from multiple DJ sets.

    Instead of just using first set, properly aggregates:
    - drops_per_hour: average across all sets
    - duration: total sum
    - energy metrics: average across sets
    - tempo: weighted average by duration
    """
    if not sets:
        return {}

    # Collect metrics from all sets
    durations = []
    drops_per_hour_list = []
    tempos = []
    energy_avgs = []

    for s in sets:
        profile = s['profile_data']

        # Duration
        duration_sec = profile.get('duration_sec', 0)
        durations.append(duration_sec)

        # Drops - support both formats
        drops_per_hour = _extract_drops_per_hour(profile)
        if drops_per_hour > 0:
            drops_per_hour_list.append(drops_per_hour)

        # Tempo
        if 'tempo_distribution' in profile and profile['tempo_distribution']:
            tempo = profile['tempo_distribution'].get('tempo_mean', 0)
            if tempo > 0:
                tempos.append((tempo, duration_sec))  # Weighted by duration

        # Energy arc
        if 'energy_arc' in profile and profile['energy_arc']:
            avg_energy = profile['energy_arc'].get('energy_mean', 0)
            if avg_energy > 0:
                energy_avgs.append(avg_energy)

    # Start with first set as base (for structure)
    aggregated = sets[0]['profile_data'].copy()

    # Aggregate metadata
    aggregated['n_sets_analyzed'] = len(sets)

    total_duration = sum(durations)
    aggregated['total_duration_hours'] = total_duration / 3600

    # Aggregate drops
    if drops_per_hour_list:
        aggregated['drops_per_hour'] = sum(drops_per_hour_list) / len(drops_per_hour_list)
        # Also update drop_pattern if exists
        if 'drop_pattern' not in aggregated:
            aggregated['drop_pattern'] = {}
        if isinstance(aggregated.get('drop_pattern'), dict):
            aggregated['drop_pattern']['drops_per_hour'] = aggregated['drops_per_hour']
    else:
        aggregated['drops_per_hour'] = 0.0

    # Aggregate tempo (weighted by duration)
    if tempos:
        total_weight = sum(w for _, w in tempos)
        if total_weight > 0:
            weighted_tempo = sum(t * w for t, w in tempos) / total_weight
            aggregated['tempo_mean'] = weighted_tempo
            if 'tempo_distribution' in aggregated and isinstance(aggregated['tempo_distribution'], dict):
                aggregated['tempo_distribution']['tempo_mean'] = weighted_tempo

    # Aggregate energy
    if energy_avgs:
        aggregated['energy_avg'] = sum(energy_avgs) / len(energy_avgs)

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Batch process DJ sets and save to profile database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sets in folder with explicit DJ name
  python scripts/batch_profile_dj.py \\
      --dj "Nina Kraviz" \\
      --folder data/dj_sets/nina-kraviz/ \\
      --save-to-db

  # Auto-detect DJ names from filenames
  python scripts/batch_profile_dj.py \\
      --folder data/dj_sets/ \\
      --auto-detect-dj \\
      --save-to-db

  # Process without saving (preview mode)
  python scripts/batch_profile_dj.py \\
      --folder data/dj_sets/nina-kraviz/
        """
    )
    parser.add_argument('--folder', required=True,
                        help='Folder containing DJ set audio files')
    parser.add_argument('--dj', type=str, default=None,
                        help='DJ name (applies to all files in folder)')
    parser.add_argument('--auto-detect-dj', action='store_true',
                        help='Auto-detect DJ name from filename')
    parser.add_argument('--save-to-db', action='store_true', default=True,
                        help='Save results to DJ profile database (default: True)')
    parser.add_argument('--no-save-to-db', dest='save_to_db', action='store_false',
                        help='Do not save results to database')
    parser.add_argument('--cache-dir', default='~/.mood-classifier/cache',
                        help='Cache directory')
    parser.add_argument('--fast', action='store_true', default=True,
                        help='Fast mode: use hop_length=2048 (default: True)')
    parser.add_argument('--no-fast', dest='fast', action='store_false',
                        help='Disable fast mode (use hop_length=512)')
    parser.add_argument('--no-energy-arc', action='store_true',
                        help='Skip energy arc analysis')
    parser.add_argument('--no-drops', action='store_true',
                        help='Skip drop pattern analysis')
    parser.add_argument('--genre', action='store_true',
                        help='Enable ML-based genre classification (requires essentia-tensorflow)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to process (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Show detailed progress for each stage (default: True)')
    parser.add_argument('--quiet', '-q', dest='verbose', action='store_false',
                        help='Minimal output (suppress stage progress)')

    args = parser.parse_args()

    # Validate arguments
    if not args.dj and not args.auto_detect_dj:
        print("Error: Must specify either --dj or --auto-detect-dj")
        parser.print_help()
        sys.exit(1)

    folder = Path(args.folder).expanduser()
    if not folder.exists():
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)

    # Find audio files
    audio_files = find_audio_files(folder)

    if not audio_files:
        print(f"No audio files found in {folder}")
        print(f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        sys.exit(1)

    # Apply limit if specified
    if args.limit:
        audio_files = audio_files[:args.limit]

    print(f"╔═══════════════════════════════════════════════════════════╗")
    print(f"║           BATCH DJ SET PROFILING                          ║")
    print(f"╠═══════════════════════════════════════════════════════════╣")
    print(f"║  Folder: {str(folder):<50} ║")
    print(f"║  Files found: {len(audio_files):<44} ║")
    print(f"║  Save to DB: {str(args.save_to_db):<45} ║")
    print(f"╚═══════════════════════════════════════════════════════════╝")
    print()

    # Create cache repository
    cache_repo = CacheRepository(cache_dir=args.cache_dir)

    # Process each file
    results = []
    total_start = time.time()

    for i, file_path in enumerate(audio_files, 1):
        # Determine DJ name
        if args.auto_detect_dj:
            dj_name = extract_dj_name_from_filename(file_path.name)
        else:
            dj_name = args.dj

        print(f"\n[{i}/{len(audio_files)}]")

        result = process_set(
            file_path=file_path,
            dj_name=dj_name,
            cache_repo=cache_repo,
            save_to_db=args.save_to_db,
            fast_mode=args.fast,
            skip_energy_arc=args.no_energy_arc,
            skip_drops=args.no_drops,
            include_genre=args.genre,
            verbose=args.verbose
        )

        results.append(result)

    total_elapsed = time.time() - total_start

    # Print summary
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total files: {len(audio_files)}")
    print(f"Successful: {sum(1 for r in results if r.get('success'))}")
    print(f"Failed: {sum(1 for r in results if not r.get('success'))}")
    print(f"Total time: {total_elapsed/60:.1f} min")
    print(f"Avg time per set: {total_elapsed/len(audio_files):.1f}s")
    print()

    # Show failures
    failures = [r for r in results if not r.get('success')]
    if failures:
        print("Failed files:")
        for r in failures:
            print(f"  ✗ {r.get('file', 'unknown')}: {r.get('error', 'unknown error')}")
        print()

    # Aggregate and save to database
    if args.save_to_db:
        print("Saving aggregated profiles to database...")

        # Group successful results by DJ name
        dj_sets = {}  # dj_name -> list of (file_path, profile_data)
        for i, result in enumerate(results):
            if not result.get('success'):
                continue

            # Get DJ name for this file
            file_path = audio_files[i]
            if args.auto_detect_dj:
                dj_name = extract_dj_name_from_filename(file_path.name)
            else:
                dj_name = args.dj

            if dj_name not in dj_sets:
                dj_sets[dj_name] = []

            dj_sets[dj_name].append({
                'file_path': result.get('file_path', str(file_path.absolute())),
                'profile_data': result.get('profile_data', {}),
            })

        # Save aggregated profile for each DJ
        for dj_name, sets in dj_sets.items():
            # Collect all file paths
            all_paths = [s['file_path'] for s in sets]

            # Aggregate profile data across ALL sets (not just first one!)
            if sets:
                aggregated_profile = _aggregate_dj_profile(sets)

                # Save to database via CacheRepository
                cache_repo.save_dj_profile_dict(
                    dj_name=dj_name,
                    profile_dict=aggregated_profile,
                    set_paths=all_paths
                )
                print(f"  ✓ {dj_name}: {len(sets)} sets saved")

        print()
        print("DJ Profile Database:")
        all_profiles = cache_repo.get_all_dj_profiles_info()
        for profile in all_profiles:
            print(f"  → {profile['dj_name']}: {profile['n_sets']} sets, {profile['total_hours']:.1f}h")
        print()


if __name__ == '__main__':
    main()
