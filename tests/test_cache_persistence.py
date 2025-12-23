"""
Test cache persistence configuration.

Verifies that cache uses DATA_DIR environment variable for production persistence.
"""

import os
import pytest
from pathlib import Path


class TestCachePersistence:
    """Test that cache uses DATA_DIR for persistence between deployments."""

    def test_cache_repository_uses_data_dir_env(self, tmp_path):
        """CacheRepository should use DATA_DIR environment variable by default."""
        from app.core.cache.repository import CacheRepository

        # Set DATA_DIR environment variable to tmp directory
        test_data_dir = str(tmp_path / 'data')
        os.environ['DATA_DIR'] = test_data_dir

        try:
            # Create instance without arguments (singleton pattern)
            # Must reset singleton first
            CacheRepository._instance = None
            repo = CacheRepository.get_instance()

            # Should use DATA_DIR from environment
            assert repo.cache_dir == Path(test_data_dir)
            assert str(repo._manager.predictions_db) == f'{test_data_dir}/predictions.db'

        finally:
            # Cleanup
            CacheRepository._instance = None
            if 'DATA_DIR' in os.environ:
                del os.environ['DATA_DIR']

    def test_cache_repository_uses_custom_path_if_specified(self, tmp_path):
        """CacheRepository should use custom path if explicitly provided."""
        from app.core.cache.repository import CacheRepository

        # Reset singleton
        CacheRepository._instance = None

        custom_cache_dir = str(tmp_path / 'custom_cache')

        # Create with custom path
        repo = CacheRepository.get_instance(cache_dir=custom_cache_dir)

        # Should use custom path, not DATA_DIR
        assert repo.cache_dir == Path(custom_cache_dir)

        # Cleanup
        CacheRepository._instance = None

    def test_cache_repository_falls_back_to_cache_if_no_data_dir(self):
        """CacheRepository should fall back to 'cache' if DATA_DIR not set."""
        from app.core.cache.repository import CacheRepository

        # Ensure DATA_DIR is not set
        if 'DATA_DIR' in os.environ:
            del os.environ['DATA_DIR']

        # Reset singleton
        CacheRepository._instance = None

        # Create instance
        repo = CacheRepository.get_instance()

        # Should use default 'cache'
        assert repo.cache_dir == Path('cache')

        # Cleanup
        CacheRepository._instance = None

    def test_production_scenario_simulation(self, tmp_path):
        """
        Simulate production scenario:
        - DATA_DIR=/data (Docker volume)
        - predictions.db created in /data (persists between deployments)
        """
        from app.core.cache.repository import CacheRepository

        # Simulate production environment with tmp directory
        test_data_dir = str(tmp_path / 'data')
        os.environ['DATA_DIR'] = test_data_dir

        try:
            # Reset singleton
            CacheRepository._instance = None

            # Get instance (as happens on worker startup)
            repo = CacheRepository.get_instance()

            # Verify paths point to volume-backed directory
            assert repo.cache_dir == Path(test_data_dir)
            assert str(repo._manager.predictions_db) == f'{test_data_dir}/predictions.db'
            assert str(repo._manager.stft_dir) == f'{test_data_dir}/stft'
            assert str(repo._manager.features_dir) == f'{test_data_dir}/features'

            # These paths are in Docker volume, so they persist across deployments!

        finally:
            # Cleanup
            CacheRepository._instance = None
            if 'DATA_DIR' in os.environ:
                del os.environ['DATA_DIR']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])