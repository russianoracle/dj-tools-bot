#!/usr/bin/env python3
"""
View DJ Profile from Database

Displays aggregated profile for a DJ based on all analyzed sets.

Usage:
    python scripts/view_dj_profile.py "Nina Kraviz"
    python scripts/view_dj_profile.py "Nina Kraviz" --json output.json
    python scripts/view_dj_profile.py --list
"""

import sys
import json
import argparse
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cache import CacheRepository


def format_time(timestamp: float) -> str:
    """Format timestamp as readable date."""
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp))


def print_dj_profile(dj_name: str, cache_repo: CacheRepository, output_json: str = None):
    """Print detailed DJ profile."""

    # Get all profiles to find this DJ
    all_profiles = cache_repo.get_all_dj_profiles_info()
    dj_meta = next((p for p in all_profiles if p['dj_name'] == dj_name), None)

    if not dj_meta:
        print(f"Error: No profile found for DJ '{dj_name}'")
        print()
        print("Available DJs:")
        list_all_djs(cache_repo)
        sys.exit(1)

    # Get full profile via CacheRepository
    cached_profile = cache_repo.get_dj_profile(dj_name)
    if not cached_profile:
        print(f"Error: Could not load profile data for '{dj_name}'")
        sys.exit(1)

    profile_data = cached_profile.to_dict()

    # Print header
    print()
    print(f"╔═══════════════════════════════════════════════════════════╗")
    print(f"║          DJ PROFILE: {dj_name.upper():<35} ║")
    print(f"╠═══════════════════════════════════════════════════════════╣")
    print(f"║  Total Sets Analyzed: {dj_meta['n_sets']:<35} ║")
    print(f"║  Total Hours:         {dj_meta['total_hours']:.1f}h{' ' * 32}║")
    print(f"║  Last Updated:        {format_time(dj_meta['updated_at']):<35} ║")
    print(f"╚═══════════════════════════════════════════════════════════╝")
    print()

    # Energy Arc Profile
    if 'energy_arc' in profile_data:
        arc = profile_data['energy_arc']
        print("Energy Arc Profile:")
        print(f"  Arc Shape:           {arc.get('arc_shape', 'unknown')}")
        print(f"  Opening Energy:      {arc.get('opening_energy', 0.0):.2f}")
        print(f"  Peak Energy:         {arc.get('peak_energy', 0.0):.2f}")
        print(f"  Closing Energy:      {arc.get('closing_energy', 0.0):.2f}")
        print(f"  Energy Variance:     {arc.get('energy_variance', 0.0):.3f}")
        print(f"  Peak Timing:         {arc.get('peak_timing_normalized', 0.0)*100:.1f}%")
        print()

    # Drop Pattern Profile
    if 'drop_pattern' in profile_data:
        drops = profile_data['drop_pattern']
        print("Drop Pattern Profile:")
        print(f"  Style:               {drops.get('drop_clustering', 'unknown')}")
        print(f"  Drops per Hour:      {drops.get('drops_per_hour', 0.0):.1f}")
        print(f"  Avg Drop Magnitude:  {drops.get('avg_drop_magnitude', 0.0):.2f}")
        print(f"  Buildups Detected:   {drops.get('buildup_count', 0)}")
        print()

    # Tempo Distribution Profile
    if 'tempo_distribution' in profile_data:
        tempo = profile_data['tempo_distribution']
        print("Tempo Distribution Profile:")
        print(f"  Mean Tempo:          {tempo.get('tempo_mean', 0.0):.1f} BPM")
        print(f"  Tempo Range:         {tempo.get('tempo_min', 0):.0f}-{tempo.get('tempo_max', 0):.0f} BPM")
        print(f"  Dominant Tempo:      {tempo.get('dominant_tempo', 0)} BPM")
        print(f"  Tempo Std Dev:       {tempo.get('tempo_std', 0.0):.1f} BPM")
        print()

    # Key Analysis Profile
    if 'key_analysis' in profile_data:
        key = profile_data['key_analysis']
        print("Key Analysis (Harmonic Mixing):")
        print(f"  Dominant Key:        {key.get('dominant_key', 'unknown')}")
        print(f"  Dominant Camelot:    {key.get('dominant_camelot', 'N/A')}")
        stability = key.get('key_stability', 0)
        if isinstance(stability, (int, float)):
            print(f"  Key Stability:       {stability:.0%}")
        print(f"  Key Changes:         {key.get('key_changes', 0)}")
        # Show camelot flow sample
        ct = key.get('camelot_trajectory', [])
        if ct and len(ct) > 0:
            step = max(1, len(ct) // 8)
            sample = []
            for x in ct[::step][:8]:
                if isinstance(x, str):
                    sample.append(x)
                elif isinstance(x, list) and len(x) > 0:
                    sample.append(str(x[0]))
            if sample:
                print(f"  Camelot Flow:        {' → '.join(sample)}")
        print()

    # Genre Profile
    if 'genre' in profile_data:
        genre = profile_data['genre']
        print("Genre & Mood:")
        print(f"  Genre:               {genre.get('genre', 'unknown')}")
        subgenre = genre.get('subgenre')
        if subgenre:
            print(f"  Subgenre:            {subgenre}")
        dj_cat = genre.get('dj_category')
        if dj_cat:
            print(f"  DJ Category:         {dj_cat}")
        conf = genre.get('confidence', 0)
        if isinstance(conf, (int, float)):
            print(f"  Confidence:          {conf:.0%}")
        mood_tags = genre.get('mood_tags', [])
        if mood_tags:
            # Handle both list of strings and list of [tag, score] pairs
            tags = []
            for t in mood_tags[:5]:
                if isinstance(t, str):
                    tags.append(t)
                elif isinstance(t, list) and len(t) > 0:
                    tags.append(str(t[0]))
            if tags:
                print(f"  Mood Tags:           {', '.join(tags)}")
        print()

    # Metadata
    if 'file_name' in profile_data:
        print("Latest Set:")
        print(f"  File: {profile_data['file_name']}")
        if 'venue' in profile_data and profile_data['venue']:
            print(f"  Venue: {profile_data['venue']}")
        if 'event' in profile_data and profile_data['event']:
            print(f"  Event: {profile_data['event']}")
        if 'date' in profile_data and profile_data['date']:
            print(f"  Date: {profile_data['date']}")
        print()

    # Export to JSON if requested
    if output_json:
        export_data = {
            'dj_name': dj_name,
            'metadata': dj_meta,
            'profile': profile_data
        }
        with open(output_json, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✓ Profile exported to: {output_json}")
        print()


def list_all_djs(cache_repo: CacheRepository):
    """List all DJs in database."""
    profiles = cache_repo.get_all_dj_profiles_info()

    if not profiles:
        print("No DJ profiles found in database.")
        print()
        print("To create a profile, analyze a set with:")
        print("  python scripts/profile_dj_set.py <file> --dj \"DJ Name\" --save-to-db")
        return

    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                    DJ PROFILES DATABASE                   ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    print(f"{'DJ Name':<30} {'Sets':<8} {'Hours':<10} {'Last Updated':<20}")
    print("─" * 70)

    for profile in profiles:
        dj_name = profile['dj_name']
        n_sets = profile['n_sets']
        hours = profile['total_hours']
        updated = format_time(profile['updated_at'])

        print(f"{dj_name:<30} {n_sets:<8} {hours:<10.1f} {updated:<20}")

    print()
    print(f"Total: {len(profiles)} DJ(s)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='View DJ profile from database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View specific DJ profile
  python scripts/view_dj_profile.py "Nina Kraviz"

  # Export to JSON
  python scripts/view_dj_profile.py "Nina Kraviz" --json nina-kraviz.json

  # List all DJs
  python scripts/view_dj_profile.py --list
        """
    )
    parser.add_argument('dj_name', nargs='?', help='DJ name to view')
    parser.add_argument('--list', action='store_true',
                        help='List all DJs in database')
    parser.add_argument('--json', dest='json_output',
                        help='Export profile to JSON file')
    parser.add_argument('--cache-dir', default='cache',
                        help='Cache directory (default: cache)')

    args = parser.parse_args()

    # Create cache repository
    cache_repo = CacheRepository(cache_dir=args.cache_dir)

    if args.list:
        list_all_djs(cache_repo)
    elif args.dj_name:
        print_dj_profile(args.dj_name, cache_repo, args.json_output)
    else:
        parser.print_help()
        print()
        print("Hint: Use --list to see all available DJs")


if __name__ == '__main__':
    main()
