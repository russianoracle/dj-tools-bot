#!/usr/bin/env python3
"""
Generate DJ Set - CLI tool for creating set plans.

Usage:
    # Phase 1: Quick plan (from Rekordbox metadata)
    python scripts/generate_set.py --dj "Josh Baker" --duration 60

    # Phase 2: Verify with ML analysis
    python scripts/generate_set.py --dj "Josh Baker" --duration 60 --verify

    # Phase 3: Interactive optimization of weak transitions
    python scripts/generate_set.py --dj "Josh Baker" --duration 60 --optimize
    python scripts/generate_set.py --dj "Josh Baker" --duration 60 --optimize --threshold 0.7

    # Auto-optimize (no interaction)
    python scripts/generate_set.py --dj "Josh Baker" --duration 60 --auto-optimize

    # Full workflow: verify + optimize + export
    python scripts/generate_set.py --dj "Josh Baker" --duration 60 --verify --optimize --save "My Set"

    # Export to Rekordbox/M3U
    python scripts/generate_set.py --dj "Josh Baker" --duration 60 --verify --save "My Set"

    # List available DJ profiles
    python scripts/generate_set.py --list-profiles

    # Show plan details
    python scripts/generate_set.py --dj "Josh Baker" --duration 60 --verbose
"""

import argparse
import json
import sys
import time
import os
import warnings
from pathlib import Path

# Suppress noisy warnings BEFORE any imports
warnings.filterwarnings("ignore", message=".*aifc.*deprecated.*")
warnings.filterwarnings("ignore", message=".*audioop.*deprecated.*")
warnings.filterwarnings("ignore", message=".*sunau.*deprecated.*")
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress TensorFlow/MLIR logs BEFORE any TF imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=""'

# Suppress absl logging (MLIR warnings)
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Redirect stderr temporarily to suppress C++ level warnings
import contextlib
import io

@contextlib.contextmanager
def suppress_stderr():
    """Suppress stderr output (for C++ level TF warnings)."""
    stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = stderr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import with stderr suppression to avoid MLIR/TF C++ warnings
_stderr_backup = sys.stderr
sys.stderr = io.StringIO()
try:
    from src.core.cache import CacheRepository
    from src.core.pipelines.set_generator import (
        SetGeneratorPipeline,
        SetPlan,
        SetPhase,
        RekordboxTrack,
    )
finally:
    sys.stderr = _stderr_backup


def format_duration(seconds: float) -> str:
    """Format duration as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def format_transition_score(score) -> str:
    """Format transition score with quality indicator."""
    if score is None:
        return "---"

    total = score.total
    quality = score.quality

    # Color-code quality (ANSI colors)
    colors = {
        "Excellent": "\033[92m",  # Green
        "Good": "\033[93m",       # Yellow
        "Fair": "\033[33m",       # Orange
        "Poor": "\033[91m",       # Red
    }
    reset = "\033[0m"

    color = colors.get(quality, "")
    return f"{color}{total:.2f} ({quality}){reset}"


def print_plan_summary(plan: SetPlan, verbose: bool = False):
    """Print set plan summary."""
    print("\n" + "=" * 60)
    print(f"DJ SET PLAN: {plan.dj_name}")
    print("=" * 60)
    print(f"Status: {plan.status.upper()}")
    print(f"Target Duration: {plan.duration_min} min")
    print(f"Actual Duration: {plan.actual_duration_min:.1f} min")
    print(f"Tracks: {plan.n_tracks}")
    print(f"Avg Transition Score: {plan.avg_transition_score:.2f}")
    print()

    # Phase breakdown
    phase_counts = {}
    for phase in SetPhase:
        tracks = plan.get_tracks_by_phase(phase)
        if tracks:
            phase_counts[phase.value] = len(tracks)
            duration = sum(t.duration_sec for t in tracks)
            print(f"  {phase.value.upper():10s}: {len(tracks)} tracks ({format_duration(duration)})")

    print()

    # Track list
    print("-" * 125)
    print(f"{'#':<3} {'Title':<24} {'Artist':<15} {'Genre':<16} {'Year':>4} {'BPM':>5} {'Key':>4} {'Score':>15}")
    print("-" * 125)

    current_phase = None
    for track in plan.tracks:
        # Phase separator
        if track.phase != current_phase:
            current_phase = track.phase
            print(f"\n{current_phase.value.upper()}")
            print("-" * 50)

        title = track.title[:22] + ".." if len(track.title) > 24 else track.title
        artist = track.artist[:13] + ".." if len(track.artist) > 15 else track.artist

        # Get genre from Rekordbox (more specific) or ML analysis (fallback)
        genre = ""
        if track.genre:
            genre = track.genre[:14] + ".." if len(track.genre) > 16 else track.genre
        elif track.analysis and track.analysis.dj_category != "Unknown":
            genre = track.analysis.dj_category[:14]

        year_str = str(track.year) if track.year > 0 else "----"
        score_str = format_transition_score(track.transition_score)

        print(f"{track.position:<3} {title:<24} {artist:<15} {genre:<16} {year_str:>4} {track.bpm:>5.1f} {track.camelot:>4} {score_str:>15}")

        if verbose and track.analysis:
            print(f"    └─ Energy: {track.analysis.intro_energy:.2f}→{track.analysis.outro_energy:.2f} | "
                  f"Drops: {track.analysis.drop_count} | "
                  f"Centroid: {track.analysis.spectral_centroid_mean:.0f}Hz")

    print("-" * 125)
    print()


def list_profiles(cache: CacheRepository):
    """List all available DJ profiles."""
    profiles = cache.get_all_dj_profiles_info()

    if not profiles:
        print("No DJ profiles found in cache.")
        print("Run 'python scripts/batch_profile_dj.py' to create profiles.")
        return

    print("\n" + "=" * 50)
    print("AVAILABLE DJ PROFILES")
    print("=" * 50)

    for p in profiles:
        updated = time.strftime("%Y-%m-%d", time.localtime(p['updated_at']))
        print(f"  {p['dj_name']:<20} | {p['n_sets']:>2} sets | {p['total_hours']:.1f}h | Updated: {updated}")

    print()


def interactive_optimize_callback(
    weak_idx: int,
    weak_total: int,
    position: int,
    old_track,
    old_score,
    prev_track,
    candidates,
):
    """
    Interactive callback for optimize mode.
    Returns: candidate index (0-based), -1 to skip, None to stop
    """
    print("\n" + "=" * 70)
    print(f"🔧 СЛАБЫЙ ПЕРЕХОД {weak_idx}/{weak_total}")
    print("=" * 70)

    # Show context
    print(f"\n📍 Позиция #{position}")
    print(f"   ОТ: {prev_track.title[:35]} ({prev_track.artist[:20]})")
    print(f"       BPM: {prev_track.bpm:.1f} | Key: {prev_track.camelot}")
    print()
    print(f"   К:  \033[91m{old_track.title[:35]}\033[0m ({old_track.artist[:20]})")
    print(f"       BPM: {old_track.bpm:.1f} | Key: {old_track.camelot}")
    print(f"       Score: \033[91m{old_score.total:.2f} ({old_score.quality})\033[0m")
    print()

    # Show candidates
    print("ВАРИАНТЫ ЗАМЕНЫ (из библиотеки Rekordbox):")
    print("-" * 70)
    for i, (track, score_in, score_out) in enumerate(candidates):
        avg_score = ((score_in.total if score_in else 0.5) + (score_out.total if score_out else 0.5)) / 2

        # Color code by quality
        if avg_score >= 0.8:
            color = "\033[92m"  # Green
        elif avg_score >= 0.6:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[33m"  # Orange

        title = track.title[:28] + ".." if len(track.title) > 30 else track.title
        artist = track.artist[:14] + ".." if len(track.artist) > 16 else track.artist
        genre = track.genre[:12] + ".." if len(track.genre) > 14 else track.genre

        # Rating stars
        rating_str = "★" * track.rating + "☆" * (5 - track.rating) if track.rating > 0 else "-----"

        print(f"  [{i+1}] {color}{title:<30}\033[0m {artist:<16}")
        print(f"      BPM: {track.bpm:>5.1f} | Key: {track.camelot:>3} | {genre:<14} | {rating_str}")
        print(f"      Score: {color}{avg_score:.2f}\033[0m "
              f"(← {score_in.total:.2f if score_in else 0:.2f} | "
              f"{score_out.total:.2f if score_out else 0:.2f} →)")

    print("-" * 70)
    print("  [S] Пропустить этот переход")
    print("  [Q] Завершить оптимизацию")
    print()

    # Get user input
    try:
        choice = input("Выбор [1-5/S/Q]: ").strip().lower()

        if choice == 'q':
            return None
        elif choice == 's':
            return -1
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                return idx
            else:
                print("Неверный номер")
                return -1
        else:
            return -1

    except (KeyboardInterrupt, EOFError):
        print("\n")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate DJ set plans from profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick plan
  python scripts/generate_set.py --dj "Josh Baker" --duration 60

  # Verified plan with ML analysis
  python scripts/generate_set.py --dj "Josh Baker" --duration 60 --verify

  # Export to M3U playlist
  python scripts/generate_set.py --dj "Josh Baker" --duration 60 --verify --save "Friday Night"

  # List available profiles
  python scripts/generate_set.py --list-profiles
"""
    )

    parser.add_argument(
        "--dj",
        type=str,
        help="DJ name to base style on"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Target set duration in minutes (default: 60)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run full ML analysis on tracks (slower but better)"
    )
    parser.add_argument(
        "--save",
        type=str,
        metavar="NAME",
        help="Export to playlist with given name"
    )
    parser.add_argument(
        "--export-json",
        type=str,
        metavar="PATH",
        help="Export plan to JSON file"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available DJ profiles"
    )
    parser.add_argument(
        "--no-reorder",
        action="store_true",
        help="Don't optimize track order during verification"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed track information"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Interactively optimize weak transitions"
    )
    parser.add_argument(
        "--auto-optimize",
        action="store_true",
        help="Automatically optimize weak transitions (no interaction)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Score threshold for weak transitions (default: 0.6)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results (default: random each time)"
    )

    args = parser.parse_args()

    # Initialize cache
    cache = CacheRepository()

    # List profiles mode
    if args.list_profiles:
        list_profiles(cache)
        return 0

    # Generate mode requires DJ name
    if not args.dj:
        parser.print_help()
        print("\nError: --dj is required for set generation")
        return 1

    # Initialize pipeline with optional seed
    pipeline = SetGeneratorPipeline(cache_repo=cache, seed=args.seed)

    # Show seed for reproducibility
    if args.seed is not None:
        print(f"Using seed: {args.seed}")

    # Phase 1: Generate quick plan
    print(f"\n[Phase 1] Generating draft plan for {args.dj}...")
    start_time = time.time()

    plan = pipeline.generate_plan(
        dj_name=args.dj,
        duration_min=args.duration,
    )

    elapsed = time.time() - start_time
    print(f"Draft plan generated in {elapsed:.1f}s")

    if plan.n_tracks == 0:
        print("\nNo tracks found! Check that:")
        print("  1. Rekordbox database is accessible")
        print("  2. DJ profile exists (run --list-profiles)")
        print("  3. BPM range in profile matches your library")
        return 1

    # Phase 2: Verify with ML (optional or interactive)
    run_verify = args.verify

    if not run_verify:
        # Show draft plan first
        print_plan_summary(plan, verbose=args.verbose)

        # Ask user if they want to run ML analysis
        print("\n" + "=" * 60)
        print("Запустить ML анализ треков? (улучшит подбор переходов)")
        print("  - Анализ займёт ~10-30 сек на трек")
        print("  - Будет определён жанр, энергия, точки микса")
        print("=" * 60)

        try:
            response = input("\nЗапустить анализ? [y/N]: ").strip().lower()
            run_verify = response in ('y', 'yes', 'д', 'да')
        except (KeyboardInterrupt, EOFError):
            print("\n")
            run_verify = False

    if run_verify:
        print(f"\n[Phase 2] ML Analysis")
        print("-" * 60)
        start_time = time.time()

        # Progress callback with realtime status
        def show_progress(current, total, track_title, stage):
            # Stage indicators
            stage_icons = {
                "loading": "📂",
                "cached": "💾",
                "analyzing": "🔬",
                "done": "✅",
                "error": "❌",
                "optimizing": "🔄",
                "complete": "🎉",
            }
            icon = stage_icons.get(stage, "•")

            # Progress bar
            bar_width = 30
            filled = int(bar_width * current / total)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Status text
            if stage == "cached":
                status = f"{icon} [{current}/{total}] {track_title} (cached)"
            elif stage == "analyzing":
                status = f"{icon} [{current}/{total}] {track_title}..."
            elif stage == "done":
                status = f"{icon} [{current}/{total}] {track_title}"
            elif stage == "error":
                status = f"{icon} [{current}/{total}] {track_title} (failed)"
            elif stage == "optimizing":
                status = f"{icon} Optimizing track order..."
            elif stage == "complete":
                status = f"{icon} Analysis complete!"
            else:
                status = f"• [{current}/{total}] {track_title}"

            # Print with carriage return for updating line
            print(f"\r{bar} {status:<50}", end="", flush=True)
            if stage in ("done", "cached", "error", "optimizing", "complete"):
                print()  # New line after each track/stage

        plan = pipeline.verify_and_optimize(
            plan,
            reorder=not args.no_reorder,
            progress_callback=show_progress,
        )

        print()  # Final newline
        elapsed = time.time() - start_time
        print(f"\nVerification completed in {elapsed:.1f}s")
        print("-" * 60)

        # Print updated results after verification
        print_plan_summary(plan, verbose=args.verbose)
    elif args.verify:
        # --verify flag was passed, plan already printed above
        pass

    # Phase 3: Interactive optimization (optional)
    if args.optimize or args.auto_optimize:
        # Find weak transitions
        weak = pipeline.find_weak_transitions(plan, args.threshold)

        if not weak:
            print(f"\n✅ Все переходы хорошие (score >= {args.threshold})")
        else:
            print(f"\n⚠️  Найдено {len(weak)} слабых переходов (score < {args.threshold})")

            if args.optimize:
                # Interactive mode
                print("\nЗапуск интерактивной оптимизации...")
                print("Для каждого слабого перехода будут предложены варианты замены.\n")

                plan = pipeline.interactive_optimize(
                    plan,
                    threshold=args.threshold,
                    callback=interactive_optimize_callback,
                )

                print("\n" + "=" * 60)
                print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
                print("=" * 60)
                print_plan_summary(plan, verbose=args.verbose)

            else:
                # Auto mode
                print("\nАвтоматическая оптимизация...")
                plan = pipeline.interactive_optimize(
                    plan,
                    threshold=args.threshold,
                    auto_accept_threshold=0.75,
                )
                print("\n✅ Автоматическая оптимизация завершена")
                print_plan_summary(plan, verbose=args.verbose)

    # Export to JSON
    if args.export_json:
        json_path = Path(args.export_json)
        json_path.write_text(plan.to_json(), encoding='utf-8')
        print(f"Plan exported to {json_path}")

    # Export to playlist
    if args.save:
        success = pipeline.export_to_rekordbox(plan, args.save)
        if success:
            print(f"Playlist '{args.save}' created successfully!")
        else:
            print("Failed to create playlist")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
