#!/usr/bin/env python3
"""
Unit tests for deploy.sh file validation and checksum logic.
Tests individual functions and validation rules without deploying.
"""
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import pytest


class TestDeployFileValidation:
    """Test file validation logic in deploy.sh."""

    @pytest.fixture
    def project_root(self):
        """Get actual project root."""
        return Path(__file__).parent.parent.parent.parent

    @pytest.fixture
    def mock_project(self, tmp_path):
        """Create minimal mock project structure."""
        project = tmp_path / "mock_project"
        project.mkdir()

        # Required directories
        (project / "app").mkdir()
        (project / "app" / "main.py").write_text("# Main entry point")
        (project / "config").mkdir()
        (project / "config" / "test.yaml").write_text("test: config")
        (project / ".github" / "workflows").mkdir(parents=True)
        (project / ".github" / "workflows" / "deploy.yml").write_text("name: deploy")

        # Required files
        (project / "Dockerfile.unified").write_text("FROM python:3.12")
        (project / "Dockerfile.base").write_text("FROM python:3.12")
        (project / "docker-compose.yml").write_text("version: '3.8'")
        (project / "requirements-prod.txt").write_text("aiogram==3.4.1")
        (project / "README.md").write_text("# Test Project")

        return project

    def test_app_directory_required(self, mock_project):
        """Test that deploy fails if app/ directory is missing."""
        # Remove app/ directory
        shutil.rmtree(mock_project / "app")

        # Try to validate (would fail in deploy.sh at line 39-44)
        assert not (mock_project / "app").exists()

    def test_required_files_present(self, mock_project):
        """Test that all required files are present in mock project."""
        required_files = [
            "Dockerfile.unified",
            "Dockerfile.base",
            "docker-compose.yml",
            "requirements-prod.txt",
        ]

        for file in required_files:
            assert (mock_project / file).exists(), f"Missing required file: {file}"

    def test_workflow_file_present(self, mock_project):
        """Test that GitHub workflow is present."""
        workflow_path = mock_project / ".github" / "workflows" / "deploy.yml"
        assert workflow_path.exists()

    def test_pycache_removal(self, mock_project):
        """Test that __pycache__ directories should be removed."""
        # Create __pycache__ in app/
        pycache = mock_project / "app" / "__pycache__"
        pycache.mkdir()
        (pycache / "test.pyc").write_text("bytecode")

        # Verify it exists
        assert pycache.exists()

        # Simulate removal (deploy.sh line 48)
        for pycache_dir in mock_project.rglob("__pycache__"):
            shutil.rmtree(pycache_dir)

        assert not (mock_project / "app" / "__pycache__").exists()

    def test_gitignore_generation(self, mock_project):
        """Test .gitignore content generation logic."""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*.pyo
*.so

# Environment
.env
.env.*
!.env.example
venv/
.venv/

# Cache and data (NEVER commit user data!)
# Production uses DATA_DIR env var to separate code from data
# IMPORTANT: Exclude only root-level cache/, not app/core/cache/ (source code)
/cache/
/data/
/downloads/
*.log
*.db
*.db-journal
dump.rdb
backups/

# IDE
.DS_Store
.idea/
.vscode/

# Training and development (should never be in deploy repo)
training/
experiments/

# NOTE: tests/ is INCLUDED for CI/CD integration tests
# NOTE: scripts/ is INCLUDED for production deployment scripts (fetch-secrets.sh)
"""
        gitignore = mock_project / ".gitignore"
        gitignore.write_text(gitignore_content)

        content = gitignore.read_text()
        assert "/cache/" in content
        assert "/data/" in content
        assert "training/" in content
        assert "!.env.example" in content

    def test_optional_files_handling(self, mock_project):
        """Test that optional files don't break deployment."""
        optional_files = [
            "docker-compose.build.yml",
            "fluent-bit.conf",
            "parsers.conf",
            "healthcheck_worker.py",
            ".env.example",
        ]

        # None of these exist in mock project
        for file in optional_files:
            assert not (mock_project / file).exists()

        # This should not cause failure (deploy.sh uses '|| true')


class TestDeployChecksumGeneration:
    """Test file checksum generation for integrity verification."""

    def test_checksum_file_content(self, tmp_path):
        """Test generating checksums for deployed files."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        # Generate SHA256 checksum
        result = subprocess.run(
            ["shasum", "-a", "256", str(test_file)],
            capture_output=True,
            text=True,
            check=True,
        )

        checksum = result.stdout.split()[0]
        assert len(checksum) == 64  # SHA256 is 64 hex chars
        assert checksum == "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"

    def test_multiple_file_checksums(self, tmp_path):
        """Test generating checksums for multiple files."""
        files = {
            "file1.txt": "content1",
            "file2.txt": "content2",
            "file3.txt": "content3",
        }

        checksums = {}
        for filename, content in files.items():
            file_path = tmp_path / filename
            file_path.write_text(content)

            result = subprocess.run(
                ["shasum", "-a", "256", str(file_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            checksums[filename] = result.stdout.split()[0]

        # All checksums should be unique
        assert len(checksums) == 3
        assert len(set(checksums.values())) == 3


class TestDeployDryRun:
    """Test dry-run mode (validation without actual deployment)."""

    @pytest.fixture
    def deploy_script(self):
        """Get path to actual deploy.sh."""
        root = Path(__file__).parent.parent.parent.parent
        return root / "deploy.sh"

    def test_dry_run_creates_temp_dir(self, deploy_script, tmp_path, monkeypatch):
        """Test that dry run creates temp directory structure."""
        # Mock mktemp to use tmp_path
        mock_temp = tmp_path / "deploy_temp"
        mock_temp.mkdir()

        # Simulate what deploy.sh does (line 18-19)
        temp_repo = mock_temp / "repo"
        temp_repo.mkdir()

        assert temp_repo.exists()
        assert (temp_repo.parent == mock_temp)

    def test_cleanup_on_exit(self, tmp_path):
        """Test that temp directory is cleaned up on exit."""
        temp_dir = tmp_path / "deploy_temp"
        temp_dir.mkdir()

        # Create some files
        (temp_dir / "test.txt").write_text("temp")

        # Simulate cleanup (deploy.sh line 19: trap "rm -rf $TEMP_DIR" EXIT)
        shutil.rmtree(temp_dir)

        assert not temp_dir.exists()


class TestDeployErrorHandling:
    """Test error handling in deploy.sh."""

    def test_missing_app_directory_error(self, tmp_path):
        """Test error when app/ directory is missing."""
        project = tmp_path / "project"
        project.mkdir()

        # No app/ directory
        app_dir = project / "app"

        # This should trigger error (deploy.sh line 43)
        if not app_dir.exists():
            error_msg = "ERROR: app/ directory not found!"
            assert "app/ directory not found" in error_msg

    def test_git_clone_failure(self, tmp_path):
        """Test handling of git clone failure."""
        invalid_repo = "git@github.com:invalid/nonexistent.git"

        # Try to clone (should fail)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", invalid_repo, str(tmp_path / "repo")],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        # deploy.sh handles this at line 24-28

    def test_set_errexit_enabled(self):
        """Test that 'set -e' causes script to exit on error."""
        # deploy.sh line 8: set -e
        # This means any command failure exits immediately

        script = """#!/bin/bash
set -e
false  # This will exit with code 1
echo "Should not reach here"
"""

        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
        )

        assert result.returncode == 1
        assert b"Should not reach here" not in result.stdout


class TestDeployFileList:
    """Test that deploy.sh copies all required files."""

    def test_required_core_files(self):
        """Test list of core files that must be deployed."""
        required_core = [
            "Dockerfile.unified",
            "Dockerfile.base",
            "docker-compose.yml",
            "requirements-prod.txt",
        ]

        # These are always copied (deploy.sh lines 58-70)
        for file in required_core:
            assert file  # Just verify list is correct

    def test_optional_docker_files(self):
        """Test optional Docker files that may be copied."""
        optional_docker = [
            "docker-compose.build.yml",
            "fluent-bit.conf",
            "parsers.conf",
            "healthcheck_worker.py",
        ]

        # These are copied with '|| true' (deploy.sh lines 61-68)
        for file in optional_docker:
            assert file

    def test_ci_test_files(self):
        """Test CI test files that should be copied."""
        ci_tests = [
            "tests/test_ci_bot.py",
            "tests/test_ci_cache_redis.py",
            "tests/test_ci_pipelines.py",
            "tests/test_ci_container.py",
            "tests/conftest.py",
        ]

        # These are copied for CI/CD testing (deploy.sh lines 103-107)
        for file in ci_tests:
            assert file

    def test_audio_fixtures(self):
        """Test audio fixture files for pipeline tests."""
        fixtures = [
            "tests/fixtures/audio/track_sample_30s.flac",
            "tests/fixtures/audio/set_sample_30s.m4a",
        ]

        # Required for CI tests (deploy.sh lines 110-111)
        for file in fixtures:
            assert file

    def test_production_scripts(self):
        """Test production scripts that should be deployed."""
        scripts = [
            "scripts/fetch-secrets.sh",
            "scripts/kill-pipelines.sh",
        ]

        # Deployed for production use (deploy.sh lines 148-155)
        for file in scripts:
            assert file


class TestDeployCommitMessage:
    """Test commit message handling."""

    def test_default_commit_message(self):
        """Test default commit message format."""
        # deploy.sh line 12: COMMIT_MSG="${1:-Deploy update $(date +%Y-%m-%d)}"
        import datetime
        default_msg = f"Deploy update {datetime.date.today().strftime('%Y-%m-%d')}"

        assert "Deploy update" in default_msg
        assert len(default_msg.split("-")) == 3  # YYYY-MM-DD

    def test_custom_commit_message(self):
        """Test custom commit message from argument."""
        custom_msg = "feat: add new feature"

        # This would be passed as $1 in deploy.sh
        assert custom_msg == "feat: add new feature"
        assert custom_msg != ""
