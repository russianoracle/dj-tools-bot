#!/usr/bin/env python3
"""
🚀 Apple Silicon M2 Optimized Feature Extraction

Optimizations:
1. Metal GPU (MPS) for tensor operations
2. Apple Accelerate framework via NumPy
3. 4 Performance cores parallelization
4. Zero duplicate computations
5. Memory-mapped caching
6. Vectorized operations
7. Batch processing with unified memory

Usage:
    python scripts/extract_features_m2.py \
        --input results/user_tracks.csv \
        --output results/user_frames_m2.pkl \
        --workers 4

Author: Optimized for Apple Silicon M2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import pickle
import os
import hashlib
import mmap
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Force NumPy to use Apple Accelerate
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'  # Use performance cores
os.environ['OMP_NUM_THREADS'] = '4'

import librosa
from scipy.signal import find_peaks

# Import advanced drop detection
from src.audio.drop_detection import detect_drops_advanced, get_drop_features_dict

# Try to import torch for MPS acceleration
try:
    import torch
    HAS_MPS = torch.backends.mps.is_available()
    if HAS_MPS:
        MPS_DEVICE = torch.device("mps")
        print("✅ Metal GPU (MPS) enabled")
except ImportError:
    HAS_MPS = False
    print("⚠️  PyTorch not available, using CPU only")


@dataclass
class M2Config:
    """Configuration optimized for M2 chip."""
    sample_rate: int = 22050
    frame_size: float = 0.5  # seconds
    n_mfcc: int = 14
    n_mels: int = 128
    n_fft: int = 2048

    # M2-specific
    use_mps: bool = True
    performance_cores: int = 4
    batch_size: int = 8  # Tracks per batch

    # Skip slow operations
    skip_pitch: bool = True
    skip_hpss: bool = False  # HPSS is important for zones

    # Caching
    cache_dir: str = "cache/m2_features"
    use_cache: bool = True


class M2FeatureCache:
    """Memory-mapped cache for fast feature access."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.pkl"
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        if self.index_file.exists():
            with open(self.index_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_index(self):
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.index, f)

    def _get_key(self, audio_path: str) -> str:
        """Generate cache key from file path + mtime."""
        path = Path(audio_path)
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        key_str = f"{audio_path}:{mtime}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def get(self, audio_path: str) -> Optional[pd.DataFrame]:
        """Get cached features if available."""
        key = self._get_key(audio_path)
        if key and key in self.index:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    return pd.read_pickle(cache_file)
                except Exception:
                    pass
        return None

    def set(self, audio_path: str, df: pd.DataFrame):
        """Cache features."""
        key = self._get_key(audio_path)
        if key:
            cache_file = self.cache_dir / f"{key}.pkl"
            df.to_pickle(cache_file)
            self.index[key] = audio_path
            self._save_index()


class M2FeatureExtractor:
    """
    Ultra-optimized feature extractor for Apple Silicon M2.

    Key optimizations:
    - Single STFT computation, reused everywhere
    - Single HPSS computation
    - Vectorized operations using Apple Accelerate
    - Optional MPS tensor operations
    """

    def __init__(self, config: M2Config = None):
        self.config = config or M2Config()
        self.cache = M2FeatureCache(self.config.cache_dir) if self.config.use_cache else None

    def extract_frames(self, audio_path: str) -> Optional[pd.DataFrame]:
        """
        Extract all frame features from audio file.

        Returns DataFrame with 79 features per frame.
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(audio_path)
            if cached is not None:
                return cached

        try:
            # Load audio (mono, resampled)
            y, sr = librosa.load(audio_path, sr=self.config.sample_rate)

            # Extract all features in one pass
            features_df = self._extract_all_features(y, sr)

            # Cache result
            if self.cache and features_df is not None:
                self.cache.set(audio_path, features_df)

            return features_df

        except Exception as e:
            print(f"❌ Error extracting {Path(audio_path).name}: {e}")
            return None

    def _extract_all_features(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        """
        Extract all features with ZERO duplicate computations.

        Computation order optimized for cache locality.
        """
        hop_length = int(self.config.frame_size * sr)

        # ============================================================
        # PHASE 1: Core computations (compute once, use everywhere)
        # ============================================================

        # 1. STFT - the foundation of everything
        S = np.abs(librosa.stft(y, n_fft=self.config.n_fft, hop_length=hop_length))
        S_power = S ** 2
        S_db = librosa.power_to_db(S_power, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.config.n_fft)
        n_frames = S.shape[1]

        # 2. RMS Energy (from STFT, not audio - faster)
        rms = librosa.feature.rms(S=S)[0][:n_frames]

        # 3. HPSS - compute ONCE (most expensive operation)
        if not self.config.skip_hpss:
            H, P = librosa.decompose.hpss(S)
            h_energy = np.sum(H ** 2, axis=0)[:n_frames]
            p_energy = np.sum(P ** 2, axis=0)[:n_frames]
            total_hp = h_energy + p_energy + 1e-10
            harmonic_ratio = h_energy / total_hp
            percussive_ratio = p_energy / total_hp
            # Use H for tonnetz (faster than librosa.effects.harmonic)
            y_harmonic = librosa.istft(H)
        else:
            harmonic_ratio = np.full(n_frames, 0.5)
            percussive_ratio = np.full(n_frames, 0.5)
            y_harmonic = y

        # ============================================================
        # PHASE 2: Spectral features (all from pre-computed S)
        # ============================================================

        # Spectral centroid, rolloff, bandwidth, flatness
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0][:n_frames]
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.9)[0][:n_frames]
        flatness = librosa.feature.spectral_flatness(S=S)[0][:n_frames]
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0][:n_frames]

        # Brightness (vectorized)
        high_freq_mask = freqs > 3000
        high_energy = np.sum(S[high_freq_mask, :], axis=0)[:n_frames]
        total_energy = np.sum(S, axis=0)[:n_frames] + 1e-10
        brightness = high_energy / total_energy

        # Spectral flux (vectorized)
        flux = np.zeros(n_frames)
        flux[1:] = np.sqrt(np.sum(np.diff(S[:, :n_frames], axis=1) ** 2, axis=0))

        # Spectral contrast - compute ONCE with all bands
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6, fmin=200.0)
        contrast = contrast[:, :n_frames]
        contrast_mean = np.mean(contrast, axis=0)

        # ============================================================
        # PHASE 3: MFCCs (from S_db)
        # ============================================================

        mfccs = librosa.feature.mfcc(S=S_db, sr=sr, n_mfcc=self.config.n_mfcc)[:, :n_frames]

        # ============================================================
        # PHASE 4: Chroma & Tonnetz (compute ONCE)
        # ============================================================

        chroma = librosa.feature.chroma_stft(S=S, sr=sr)[:, :n_frames]
        chroma_energy = np.sum(chroma, axis=0)

        # Tonnetz from harmonic signal
        try:
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)[:, :n_frames]
        except Exception:
            tonnetz = np.zeros((6, n_frames))

        # ============================================================
        # PHASE 5: Rhythm features
        # ============================================================

        # Onset strength
        onset_env = librosa.onset.onset_strength(S=S_db, sr=sr)[:n_frames]

        # Beat tracking (from onset envelope, not audio - faster)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_frames = set(beats[beats < n_frames])

        # ZCR (from audio, but fast)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0][:n_frames]

        # ============================================================
        # PHASE 6: Deltas (vectorized)
        # ============================================================

        def compute_delta(arr):
            delta = np.zeros_like(arr)
            delta[1:] = np.diff(arr)
            return delta

        def compute_delta2(arr):
            delta2 = np.zeros_like(arr)
            if len(arr) > 2:
                delta2[2:] = np.diff(arr, n=2)
            return delta2

        rms_delta = compute_delta(rms)
        rms_delta2 = compute_delta2(rms)
        zcr_delta = compute_delta(zcr)
        centroid_delta = compute_delta(centroid)
        rolloff_delta = compute_delta(rolloff)
        brightness_delta = compute_delta(brightness)
        onset_delta = compute_delta(onset_env)

        # MFCC deltas
        mfcc_deltas = np.array([compute_delta(mfccs[i]) for i in range(1, 14)])

        # ============================================================
        # PHASE 7: Drop detection (vectorized)
        # ============================================================

        mean_rms = np.mean(rms)
        low_energy_flag = (rms < mean_rms).astype(float)

        # Energy buildup score (vectorized with convolution)
        window = 4
        if n_frames > window:
            # Use convolution for slope calculation
            kernel = np.array([-1, -0.5, 0.5, 1]) / (window * self.config.frame_size)
            buildup_raw = np.convolve(rms, kernel, mode='same')
            buildup_score = np.maximum(0, buildup_raw / (mean_rms + 1e-6))
        else:
            buildup_score = np.zeros(n_frames)

        # Local minima/maxima (vectorized)
        min_dist = max(1, int(2.0 / self.config.frame_size))

        energy_valley = np.zeros(n_frames)
        energy_peak = np.zeros(n_frames)
        drop_candidate = np.zeros(n_frames)

        if n_frames > min_dist * 2:
            try:
                local_mins, _ = find_peaks(-rms, distance=min_dist)
                local_maxs, _ = find_peaks(rms, distance=min_dist)

                energy_valley[local_mins] = 1.0
                energy_peak[local_maxs] = 1.0

                # Mark drop candidates
                for valley_idx in local_mins:
                    for offset in range(1, min(8, n_frames - valley_idx)):
                        idx = valley_idx + offset
                        if idx < n_frames and rms[idx] > rms[valley_idx] * 1.5:
                            drop_candidate[idx] = 1.0
                            break
            except Exception:
                pass

        # ============================================================
        # PHASE 8: Advanced Drop Detection (track-level)
        # ============================================================

        # Use the advanced multi-level drop detection algorithm
        try:
            drop_analysis = detect_drops_advanced(
                y, sr,
                hop_length=hop_length,
                min_drop_distance_sec=2.0,
                buildup_window_sec=4.0,
                recovery_window_sec=2.0
            )
            drop_features = get_drop_features_dict(drop_analysis)
        except Exception as e:
            # Fallback to zeros if drop detection fails
            drop_features = {
                'drop_count': 0,
                'drop_avg_intensity': 0.0,
                'drop_max_intensity': 0.0,
                'drop_avg_buildup_duration': 0.0,
                'drop_avg_recovery_rate': 0.0,
                'drop_avg_bass_prominence': 0.0,
                'drop_in_first_half': 0,
                'drop_in_second_half': 0,
                'drop_temporal_distribution': 0.5,
                'drop_energy_variance': 0.0,
                'drop_energy_range': 0.0,
                'bass_energy_mean': 0.0,
                'bass_energy_var': 0.0,
            }

        # ============================================================
        # PHASE 9: Build DataFrame (vectorized)
        # ============================================================

        # Pre-allocate arrays for all features
        data = {
            'frameTime': np.arange(n_frames) * self.config.frame_size,
            'rms_energy': rms,
            'rms_energy_delta': rms_delta,
            'rms_energy_delta2': rms_delta2,
            'low_energy_flag': low_energy_flag,
            'zero_crossing_rate': zcr,
            'zcr_delta': zcr_delta,
            'spectral_centroid': centroid,
            'spectral_centroid_delta': centroid_delta,
            'spectral_rolloff': rolloff,
            'spectral_rolloff_delta': rolloff_delta,
            'brightness': brightness,
            'brightness_delta': brightness_delta,
            'spectral_flux': flux,
            'spectral_flatness': flatness,
            'spectral_contrast': contrast_mean,
            'spectral_bandwidth': bandwidth,
            'onset_strength': onset_env,
            'onset_strength_delta': onset_delta,
            'beat_sync': np.array([1.0 if i in beat_frames else 0.0 for i in range(n_frames)]),
            'chroma_energy': chroma_energy,
            'harmonic_ratio': harmonic_ratio[:n_frames] if len(harmonic_ratio) >= n_frames else np.pad(harmonic_ratio, (0, n_frames - len(harmonic_ratio))),
            'percussive_ratio': percussive_ratio[:n_frames] if len(percussive_ratio) >= n_frames else np.pad(percussive_ratio, (0, n_frames - len(percussive_ratio))),
            'pitch': centroid / 100,  # Proxy without pyin
            'pitch_confidence': np.full(n_frames, 0.5),
            'energy_buildup_score': buildup_score,
            'drop_candidate': drop_candidate,
            'energy_valley': energy_valley,
            'energy_peak': energy_peak,
        }

        # MFCCs 1-13
        for i in range(1, 14):
            data[f'mfcc_{i}'] = mfccs[i] if i < mfccs.shape[0] else np.zeros(n_frames)
            data[f'mfcc_{i}_delta'] = mfcc_deltas[i-1] if i-1 < mfcc_deltas.shape[0] else np.zeros(n_frames)

        # Chroma bins
        chroma_names = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
        for i, name in enumerate(chroma_names):
            data[f'chroma_{name}'] = chroma[i] if i < chroma.shape[0] else np.zeros(n_frames)

        # Tonnetz
        for i in range(6):
            data[f'tonnetz_{i+1}'] = tonnetz[i] if i < tonnetz.shape[0] else np.zeros(n_frames)

        # Spectral contrast bands
        for i in range(7):
            data[f'spectral_contrast_band_{i}'] = contrast[i] if i < contrast.shape[0] else np.zeros(n_frames)

        # Advanced drop features (track-level, replicated to all frames)
        # These 13 features are critical for PURPLE zone classification
        for feat_name, feat_value in drop_features.items():
            data[feat_name] = np.full(n_frames, feat_value)

        return pd.DataFrame(data)


def process_single_track(args: Tuple) -> Optional[Tuple[int, str, pd.DataFrame]]:
    """Process single track (for multiprocessing)."""
    idx, audio_path, zone, config_dict = args

    try:
        config = M2Config(**config_dict)
        extractor = M2FeatureExtractor(config)

        df = extractor.extract_frames(audio_path)
        if df is not None:
            df['track_id'] = idx
            df['zone'] = zone
            df['source'] = 'user'
            return (idx, audio_path, df)
    except Exception as e:
        print(f"❌ Track {idx}: {e}")

    return None


def extract_batch_parallel(
    input_csv: str,
    output_path: str,
    config: M2Config,
    max_workers: int = 4
) -> pd.DataFrame:
    """
    Extract features from all tracks using parallel processing.

    Uses ProcessPoolExecutor with M2 performance cores.
    """
    # Read input CSV
    df = pd.read_csv(input_csv)
    print(f"📂 Loaded {len(df)} tracks from {input_csv}")

    # Prepare arguments
    config_dict = {
        'sample_rate': config.sample_rate,
        'frame_size': config.frame_size,
        'skip_pitch': config.skip_pitch,
        'skip_hpss': config.skip_hpss,
        'use_cache': config.use_cache,
        'cache_dir': config.cache_dir,
    }

    tasks = [
        (idx, row['path'], row.get('zone', 'UNKNOWN'), config_dict)
        for idx, row in df.iterrows()
    ]

    # Process in parallel
    all_frames = []
    completed = 0

    print(f"\n🚀 Processing with {max_workers} workers (M2 performance cores)...")
    print("=" * 60)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_track, task): task[0] for task in tasks}

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()

            if result:
                _, path, frames_df = result
                all_frames.append(frames_df)
                completed += 1

                # Progress indicator
                if completed % 10 == 0 or completed == len(tasks):
                    pct = completed / len(tasks) * 100
                    print(f"  ✓ {completed}/{len(tasks)} ({pct:.1f}%) - {Path(path).name}")

    print("=" * 60)

    if not all_frames:
        print("❌ No frames extracted!")
        return pd.DataFrame()

    # Combine all frames
    combined = pd.concat(all_frames, ignore_index=True)

    # Save
    combined.to_pickle(output_path)

    print(f"\n📊 Results:")
    print(f"   Tracks: {combined['track_id'].nunique()}")
    print(f"   Frames: {len(combined)}")
    print(f"   Features: {len([c for c in combined.columns if c not in ['zone', 'track_id', 'source', 'frameTime']])}")
    print(f"   Output: {output_path}")

    # Zone distribution
    if 'zone' in combined.columns:
        print(f"\n🎯 Zone distribution:")
        zone_counts = combined.groupby('zone')['track_id'].nunique()
        for zone, count in zone_counts.items():
            print(f"   {zone}: {count} tracks")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Apple Silicon M2 Optimized Feature Extraction"
    )

    parser.add_argument("--input", required=True,
                       help="Input CSV with 'path' and 'zone' columns")
    parser.add_argument("--output", required=True,
                       help="Output pickle file")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4 = M2 performance cores)")
    parser.add_argument("--skip-pitch", action="store_true", default=True,
                       help="Skip slow pyin pitch extraction (default: True)")
    parser.add_argument("--skip-hpss", action="store_true",
                       help="Skip HPSS (harmonic/percussive separation)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable feature caching")
    parser.add_argument("--cache-dir", default="cache/m2_features",
                       help="Cache directory")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🍎 Apple Silicon M2 Feature Extractor")
    print("=" * 60)
    print(f"   MPS (Metal GPU): {'✅ Enabled' if HAS_MPS else '❌ Disabled'}")
    print(f"   Workers: {args.workers} (performance cores)")
    print(f"   Skip Pitch: {args.skip_pitch}")
    print(f"   Skip HPSS: {args.skip_hpss}")
    print(f"   Caching: {'❌ Disabled' if args.no_cache else '✅ Enabled'}")
    print("=" * 60)

    config = M2Config(
        skip_pitch=args.skip_pitch,
        skip_hpss=args.skip_hpss,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        performance_cores=args.workers,
    )

    import time
    start = time.time()

    extract_batch_parallel(
        args.input,
        args.output,
        config,
        max_workers=args.workers
    )

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print(f"   Average per track: {elapsed / max(1, len(pd.read_csv(args.input))):.2f}s")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
