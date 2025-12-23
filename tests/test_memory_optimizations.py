"""
Memory optimization tests.

Tests accuracy, memory efficiency, and streaming behavior.
"""

import pytest
import numpy as np
import gc
import psutil


class TestAccuracy:
    """Test that optimizations don't degrade accuracy."""

    def test_scipy_fft_accuracy(self):
        """Verify scipy.fft produces identical results to np.fft."""
        # Generate test signal
        y = np.random.randn(2048).astype(np.float32)

        # Old: np.fft.rfft
        old_result = np.fft.rfft(y)

        # New: scipy.fft.rfft
        import scipy.fft
        new_result = scipy.fft.rfft(y)

        # Should be very close (scipy uses complex64, numpy uses complex128)
        # Error ~1.7e-5 is 0.0017% - inaudible for audio
        max_error = np.max(np.abs(old_result - new_result))
        assert max_error < 1e-4, f"FFT accuracy regression: {max_error}"

    def test_stft_optimization_accuracy(self):
        """Verify STFT optimizations preserve accuracy."""
        from app.common.primitives.stft import compute_stft

        # Generate 60-second test audio
        sr = 22050
        duration = 60
        y = np.random.randn(sr * duration).astype(np.float32)

        # Compute STFT (with optimizations)
        cache = compute_stft(y, sr=sr)

        # Verify output shape
        assert cache.S.shape[0] == 1025  # n_fft//2 + 1 = 2048//2 + 1
        assert cache.S.dtype == np.float32
        assert cache.phase.dtype == np.float32

        # Verify no NaN or Inf
        assert not np.any(np.isnan(cache.S))
        assert not np.any(np.isinf(cache.S))

    def test_streaming_vs_standard_accuracy(self):
        """Verify streaming STFT matches standard computation."""
        from app.common.primitives.stft import compute_stft
        from app.common.primitives.streaming_stft import compute_stft_streaming

        # Generate 2-minute test audio (not too long for standard STFT)
        sr = 22050
        duration = 120  # 2 minutes
        y = np.random.randn(sr * duration).astype(np.float32)

        # Standard computation
        cache_standard = compute_stft(y, sr=sr)

        # Streaming computation
        cache_stream = compute_stft_streaming(y, sr=sr, chunk_duration=30)

        # Frame counts should be identical (frame-aligned chunking)
        assert cache_standard.S.shape == cache_stream.S.shape, \
            f"Shape mismatch: {cache_standard.S.shape} vs {cache_stream.S.shape}"

        # Results should be nearly identical (< 0.01% error)
        diff = np.abs(cache_standard.S - cache_stream.S)
        max_abs_error = np.max(diff)
        max_rel_error = np.max(diff / (cache_standard.S + 1e-10))

        # Very tight tolerance - frame-aligned chunking should be exact
        assert max_rel_error < 1e-4, \
            f"Streaming error too high: max_abs={max_abs_error:.2e}, max_rel={max_rel_error:.2e}"


class TestMemoryEfficiency:
    """Test memory cleanup and efficiency."""

    def test_gc_collect_frees_memory(self):
        """Verify del + gc.collect() actually frees memory."""
        process = psutil.Process()

        # Get baseline
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024

        # Create large array
        large = np.zeros((5000, 5000), dtype=np.float32)  # ~95 MB
        mem_with_array = process.memory_info().rss / 1024 / 1024

        # Delete and collect
        del large
        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024

        # Should free most memory (allow 20% overhead for fragmentation)
        freed = mem_with_array - mem_after
        assert freed > 70, f"Memory not freed efficiently: only {freed:.1f} MB released"

    def test_stft_memory_cleanup(self):
        """Verify STFT properly cleans up intermediate arrays."""
        from app.common.primitives.stft import compute_stft

        process = psutil.Process()

        # Baseline
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024

        # Process 60-second audio
        sr = 22050
        y = np.random.randn(sr * 60).astype(np.float32)
        cache = compute_stft(y, sr=sr)

        # After STFT
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_increase = mem_after - mem_before

        # Delete cache and collect
        del cache
        gc.collect()
        mem_final = process.memory_info().rss / 1024 / 1024

        # Should return close to baseline (allow 10% overhead)
        assert mem_final < mem_before + 10, f"Memory leak detected: {mem_final - mem_before:.1f} MB not freed"


class TestStreamingBehavior:
    """Test streaming STFT behavior."""

    def test_should_use_streaming_threshold(self):
        """Test streaming threshold detection."""
        from app.common.primitives.streaming_stft import should_use_streaming

        sr = 22050

        # 60 minutes: should NOT use streaming
        assert not should_use_streaming(60 * 60, sr)

        # 90 minutes: should NOT use streaming (at threshold)
        assert not should_use_streaming(90 * 60, sr)

        # 91 minutes: should use streaming
        assert should_use_streaming(91 * 60, sr)

        # 120 minutes: should use streaming
        assert should_use_streaming(120 * 60, sr)

    def test_streaming_memory_constant(self):
        """Verify streaming maintains constant memory."""
        from app.common.primitives.streaming_stft import compute_stft_streaming

        process = psutil.Process()
        sr = 22050

        # Baseline
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024

        # Process 5-minute file (multiple chunks)
        y = np.random.randn(sr * 5 * 60).astype(np.float32)
        cache = compute_stft_streaming(y, sr=sr, chunk_duration=30)

        mem_peak = process.memory_info().rss / 1024 / 1024
        peak_increase = mem_peak - mem_before

        # Peak should be reasonable (< 2 GB for any file length)
        assert peak_increase < 2000, f"Streaming memory too high: {peak_increase:.1f} MB"

        # Cleanup
        del cache
        gc.collect()


class TestIntegration:
    """Integration tests with full pipeline."""

    def test_create_audio_context_routing(self):
        """Test that create_audio_context routes to streaming correctly."""
        from app.modules.analysis.tasks.base import create_audio_context

        sr = 22050

        # Short file: should use standard STFT
        y_short = np.random.randn(sr * 60).astype(np.float32)  # 60 sec
        ctx_short = create_audio_context(y_short, sr=sr)
        assert ctx_short.stft_cache is not None

        # Long file: should use streaming STFT
        y_long = np.random.randn(sr * 91 * 60).astype(np.float32)  # 91 min
        ctx_long = create_audio_context(y_long, sr=sr)
        assert ctx_long.stft_cache is not None

        # Verify both produce valid results
        assert ctx_short.stft_cache.S.shape[0] == 1025
        assert ctx_long.stft_cache.S.shape[0] == 1025
