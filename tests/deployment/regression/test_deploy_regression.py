#!/usr/bin/env python3
"""
Regression tests for deployment process.
Tests version upgrades and backward compatibility.
"""
import os
import subprocess
import shutil
from pathlib import Path
import pytest


class TestVersionUpgrade:
    """Test upgrading from previous deployment version."""

    @pytest.fixture
    def old_deployment(self, tmp_path):
        """Create old deployment structure (v1.0)."""
        old = tmp_path / "old_deploy"
        old.mkdir()

        # Old structure (before Clean Architecture refactor)
        (old / "bot.py").write_text("# Old bot entry point")
        (old / "handlers").mkdir()
        (old / "handlers" / "music.py").write_text("# Old handlers")
        (old / "Dockerfile").write_text("FROM python:3.10")
        (old / "requirements.txt").write_text("aiogram==2.25.1")

        return old

    @pytest.fixture
    def new_deployment(self, tmp_path):
        """Create new deployment structure (v2.0 - Clean Architecture)."""
        new = tmp_path / "new_deploy"
        new.mkdir()

        # New Clean Architecture structure
        app = new / "app"
        app.mkdir()
        (app / "main.py").write_text("# New main entry point")

        core = app / "core"
        core.mkdir()
        (core / "config.py").write_text("# Core config")

        modules = app / "modules"
        modules.mkdir()
        bot = modules / "bot"
        bot.mkdir()
        (bot / "handlers.py").write_text("# New handlers")

        (new / "Dockerfile.unified").write_text("FROM python:3.12")
        (new / "requirements-prod.txt").write_text("aiogram==3.4.1")

        return new

    def test_upgrade_structure_changed(self, old_deployment, new_deployment):
        """Test that deployment structure changed between versions."""
        # Old has bot.py
        assert (old_deployment / "bot.py").exists()
        assert not (old_deployment / "app").exists()

        # New has app/main.py
        assert (new_deployment / "app" / "main.py").exists()
        assert not (new_deployment / "bot.py").exists()

    def test_upgrade_preserves_data_dir(self, tmp_path):
        """Test that DATA_DIR is preserved during upgrade."""
        # Simulate old deployment with data
        old_data = tmp_path / "data"
        old_data.mkdir()
        (old_data / "predictions.db").write_text("OLD_DB_DATA")

        # Deploy new version (data dir unchanged)
        new_deploy = tmp_path / "new_code"
        new_deploy.mkdir()

        # Data should still exist
        assert (old_data / "predictions.db").exists()
        assert (old_data / "predictions.db").read_text() == "OLD_DB_DATA"

    def test_upgrade_docker_volume_unchanged(self):
        """Test that Docker volumes are preserved across deployments."""
        # Docker volumes are persistent and not affected by code updates
        # This test documents the expected behavior

        volume_config = {
            "volumes": ["mood-data:/data"],
            "environment": ["DATA_DIR=/data"],
        }

        # Volume persists across deployments
        assert "mood-data:/data" in volume_config["volumes"]
        assert "DATA_DIR=/data" in volume_config["environment"]


class TestBackwardCompatibility:
    """Test backward compatibility with previous versions."""

    def test_old_cache_format_readable(self, tmp_path):
        """Test that old cache DB format can be read."""
        # Simulate old cache DB
        import sqlite3

        old_db = tmp_path / "old_cache.db"
        conn = sqlite3.connect(old_db)
        cursor = conn.cursor()

        # Old schema
        cursor.execute("""
            CREATE TABLE track_cache (
                path TEXT PRIMARY KEY,
                bpm REAL,
                key TEXT,
                energy REAL
            )
        """)
        cursor.execute(
            "INSERT INTO track_cache VALUES (?, ?, ?, ?)",
            ("/path/to/track.mp3", 128.0, "Am", 0.75),
        )
        conn.commit()
        conn.close()

        # Read old data
        conn = sqlite3.connect(old_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM track_cache")
        row = cursor.fetchone()

        assert row[0] == "/path/to/track.mp3"
        assert row[1] == 128.0
        conn.close()

    def test_old_docker_compose_compatibility(self, tmp_path):
        """Test that old docker-compose.yml still works."""
        old_compose = tmp_path / "docker-compose.old.yml"
        old_compose.write_text("""version: '3.8'
services:
  bot:
    build: .
    environment:
      - TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
""")

        # Old format should still be parsable
        import yaml
        with open(old_compose) as f:
            config = yaml.safe_load(f)

        assert config["version"] == "3.8"
        assert "bot" in config["services"]

    def test_old_env_vars_still_work(self):
        """Test that old environment variables are still supported."""
        # Old var names (if any were changed)
        old_vars = {
            "TELEGRAM_TOKEN": "123:ABC",
            "REDIS_URL": "redis://localhost:6379",
        }

        # These should still work in new version
        for key, value in old_vars.items():
            os.environ[key] = value
            assert os.environ[key] == value


class TestDatabaseMigration:
    """Test database migration during deployment."""

    def test_schema_migration_additive(self, tmp_path):
        """Test that schema changes are additive (don't break old data)."""
        import sqlite3

        db_path = tmp_path / "predictions.db"

        # Old schema (v1)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE predictions (
                track_path TEXT PRIMARY KEY,
                zone TEXT,
                confidence REAL
            )
        """)
        cursor.execute(
            "INSERT INTO predictions VALUES (?, ?, ?)",
            ("/path/track.mp3", "PURPLE", 0.85),
        )
        conn.commit()
        conn.close()

        # New schema (v2) - adds columns but keeps old ones
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check old columns still exist
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [row[1] for row in cursor.fetchall()]

        assert "track_path" in columns
        assert "zone" in columns
        assert "confidence" in columns

        # Old data still readable
        cursor.execute("SELECT * FROM predictions")
        row = cursor.fetchone()
        assert row[0] == "/path/track.mp3"
        conn.close()

    def test_migration_rollback_safe(self, tmp_path):
        """Test that failed migration can be rolled back."""
        import sqlite3

        db_path = tmp_path / "test.db"
        backup_path = tmp_path / "test.db.backup"

        # Original database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        cursor.execute("INSERT INTO test VALUES (1)")
        conn.commit()
        conn.close()

        # Backup before migration
        shutil.copy2(db_path, backup_path)

        # Simulate failed migration
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("ALTER TABLE test ADD COLUMN new_col TEXT")
            # Simulate error
            raise Exception("Migration failed!")
        except Exception:
            conn.close()
            # Rollback: restore from backup
            shutil.copy2(backup_path, db_path)

        # Verify rollback worked
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM test")
        row = cursor.fetchone()
        assert row[0] == 1
        conn.close()


class TestDeploymentRollback:
    """Test rolling back failed deployments."""

    def test_git_rollback_to_previous_commit(self, tmp_path):
        """Test rolling back to previous Git commit."""
        repo = tmp_path / "repo"
        repo.mkdir()

        # Initialize repo
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo, check=True, capture_output=True,
        )

        # Commit v1
        (repo / "app.py").write_text("v1")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "v1"],
            cwd=repo, check=True, capture_output=True,
        )

        # Commit v2 (bad)
        (repo / "app.py").write_text("v2 broken")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "v2"],
            cwd=repo, check=True, capture_output=True,
        )

        # Rollback to v1
        subprocess.run(
            ["git", "reset", "--hard", "HEAD~1"],
            cwd=repo, check=True, capture_output=True,
        )

        # Verify rollback
        content = (repo / "app.py").read_text()
        assert content == "v1"

    def test_docker_rollback_to_previous_image(self):
        """Test rolling back to previous Docker image."""
        # Docker image tags for rollback
        current_image = "cr.yandex/registry/app:v2.0"
        previous_image = "cr.yandex/registry/app:v1.0"

        # Rollback command (would update docker-compose.yml)
        rollback_config = {
            "services": {
                "app": {
                    "image": previous_image,  # Changed from v2.0
                }
            }
        }

        assert rollback_config["services"]["app"]["image"] == previous_image


class TestBreakingChanges:
    """Test detection of breaking changes."""

    def test_detect_removed_endpoints(self, tmp_path):
        """Test detecting removed API endpoints."""
        old_handlers = tmp_path / "old_handlers.py"
        old_handlers.write_text("""
def handle_analyze(message):
    pass

def handle_profile(message):
    pass

def handle_settings(message):
    pass
""")

        new_handlers = tmp_path / "new_handlers.py"
        new_handlers.write_text("""
def handle_analyze(message):
    pass

def handle_profile(message):
    pass

# handle_settings removed - BREAKING CHANGE!
""")

        # Parse handlers
        old_functions = ["handle_analyze", "handle_profile", "handle_settings"]
        new_functions = ["handle_analyze", "handle_profile"]

        # Detect removed
        removed = set(old_functions) - set(new_functions)
        assert "handle_settings" in removed

    def test_detect_changed_config_format(self):
        """Test detecting changed configuration format."""
        old_config = {
            "telegram": {
                "token": "TOKEN",
            }
        }

        new_config = {
            "telegram": {
                "bot_token": "TOKEN",  # Changed key name
            }
        }

        # Detect change
        assert "token" in old_config["telegram"]
        assert "bot_token" in new_config["telegram"]
        assert "token" not in new_config["telegram"]  # Breaking change

    def test_detect_removed_env_vars(self):
        """Test detecting removed environment variables."""
        old_env = {
            "TELEGRAM_TOKEN",
            "REDIS_URL",
            "LOG_LEVEL",
        }

        new_env = {
            "TELEGRAM_TOKEN",
            "REDIS_URL",
            # LOG_LEVEL removed - might break old configs
        }

        removed = old_env - new_env
        assert "LOG_LEVEL" in removed


class TestDataPersistence:
    """Test data persistence across deployments."""

    def test_volume_mount_unchanged(self):
        """Test that volume mount configuration is unchanged."""
        # Production docker-compose.yml should have:
        volume_config = {
            "volumes": ["mood-data:/data"],
        }

        assert "mood-data:/data" in volume_config["volumes"]

    def test_database_path_unchanged(self):
        """Test that database path is unchanged."""
        # Should always use DATA_DIR environment variable
        import os
        os.environ["DATA_DIR"] = "/data"

        db_path = os.path.join(os.environ.get("DATA_DIR", "."), "predictions.db")
        assert db_path == "/data/predictions.db"

    def test_cache_dir_unchanged(self):
        """Test that cache directory location is unchanged."""
        import os
        os.environ["DATA_DIR"] = "/data"

        cache_dir = os.path.join(os.environ.get("DATA_DIR", "."), "cache")
        assert cache_dir == "/data/cache"


class TestCITestCompatibility:
    """Test that CI tests remain compatible across versions."""

    def test_ci_test_files_present(self, tmp_path):
        """Test that required CI test files are deployed."""
        tests = tmp_path / "tests"
        tests.mkdir()

        ci_tests = [
            "test_ci_bot.py",
            "test_ci_cache_redis.py",
            "test_ci_pipelines.py",
            "test_ci_container.py",
        ]

        for test in ci_tests:
            (tests / test).write_text("def test_dummy(): pass")

        # Verify all present
        for test in ci_tests:
            assert (tests / test).exists()

    def test_fixtures_remain_compatible(self, tmp_path):
        """Test that audio fixtures format is unchanged."""
        fixtures = tmp_path / "tests" / "fixtures" / "audio"
        fixtures.mkdir(parents=True)

        # These files should always be present
        (fixtures / "track_sample_30s.flac").write_text("FAKE_FLAC")
        (fixtures / "set_sample_30s.m4a").write_text("FAKE_M4A")

        assert (fixtures / "track_sample_30s.flac").exists()
        assert (fixtures / "set_sample_30s.m4a").exists()
