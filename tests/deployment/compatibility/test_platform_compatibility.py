#!/usr/bin/env python3
"""
Platform compatibility tests for deployment.
Tests deployment on M2 macOS (dev) vs x86_64 Linux (prod).
"""
import os
import platform
import subprocess
import tempfile
from pathlib import Path
import pytest


class TestPlatformDetection:
    """Test platform detection and environment setup."""

    def test_detect_current_platform(self):
        """Test detecting current platform."""
        system = platform.system()
        machine = platform.machine()

        assert system in ["Darwin", "Linux", "Windows"]
        assert machine in ["arm64", "x86_64", "aarch64", "AMD64"]

    def test_macos_arm64_detection(self):
        """Test M2 Apple Silicon detection."""
        system = platform.system()
        machine = platform.machine()

        if system == "Darwin" and machine == "arm64":
            # Running on M2 Mac
            assert True
        else:
            pytest.skip("Not running on M2 macOS")

    def test_linux_x86_64_detection(self):
        """Test x86_64 Linux detection."""
        system = platform.system()
        machine = platform.machine()

        if system == "Linux" and machine == "x86_64":
            # Running on Linux x86_64
            assert True
        else:
            pytest.skip("Not running on x86_64 Linux")


class TestDockerEnvironment:
    """Test Docker environment across platforms."""

    def test_docker_available(self):
        """Test that Docker is available on the system."""
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            assert "Docker version" in result.stdout
        else:
            pytest.skip("Docker not available")

    def test_docker_platform_macos(self):
        """Test Docker Desktop on macOS (M2)."""
        if platform.system() != "Darwin":
            pytest.skip("Not running on macOS")

        # Check Docker Desktop is running
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Docker Desktop for Mac
            assert "Operating System" in result.stdout or "OSType" in result.stdout

    def test_docker_platform_linux(self):
        """Test Docker CE on Linux."""
        if platform.system() != "Linux":
            pytest.skip("Not running on Linux")

        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Docker CE on Linux
            assert "linux" in result.stdout.lower()


class TestPathCompatibility:
    """Test path handling across platforms."""

    def test_tmp_path_creation(self, tmp_path):
        """Test temporary path creation works on all platforms."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello")

        assert test_file.exists()
        assert test_file.read_text() == "Hello"

    def test_absolute_paths_macos(self):
        """Test macOS path format."""
        if platform.system() != "Darwin":
            pytest.skip("Not running on macOS")

        # macOS uses /Users/username
        home = Path.home()
        assert str(home).startswith("/Users/")

    def test_absolute_paths_linux(self):
        """Test Linux path format."""
        if platform.system() != "Linux":
            pytest.skip("Not running on Linux")

        # Linux uses /home/username or /root
        home = Path.home()
        assert str(home).startswith("/home/") or str(home) == "/root"

    def test_path_separator_universal(self):
        """Test that Path uses correct separator."""
        test_path = Path("foo") / "bar" / "baz.txt"

        # Path should use correct separator for platform
        assert "foo" in str(test_path)
        assert "bar" in str(test_path)
        assert "baz.txt" in str(test_path)


class TestShellCommandCompatibility:
    """Test shell command compatibility across platforms."""

    def test_bash_available(self):
        """Test that bash is available on all platforms."""
        result = subprocess.run(
            ["bash", "--version"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "bash" in result.stdout.lower()

    def test_mktemp_command_macos(self):
        """Test mktemp on macOS."""
        if platform.system() != "Darwin":
            pytest.skip("Not running on macOS")

        # macOS mktemp requires template
        result = subprocess.run(
            ["mktemp", "-d"],
            capture_output=True,
            text=True,
            check=True,
        )

        temp_dir = result.stdout.strip()
        assert Path(temp_dir).exists()

        # Cleanup
        subprocess.run(["rm", "-rf", temp_dir], check=True)

    def test_mktemp_command_linux(self):
        """Test mktemp on Linux."""
        if platform.system() != "Linux":
            pytest.skip("Not running on Linux")

        result = subprocess.run(
            ["mktemp", "-d"],
            capture_output=True,
            text=True,
            check=True,
        )

        temp_dir = result.stdout.strip()
        assert Path(temp_dir).exists()

        # Cleanup
        subprocess.run(["rm", "-rf", temp_dir], check=True)

    def test_find_command_compatibility(self):
        """Test find command works the same on all platforms."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_dir = Path(tmpdir)
            (test_dir / "__pycache__").mkdir()
            (test_dir / "__pycache__" / "test.pyc").write_text("bytecode")
            (test_dir / "app").mkdir()
            (test_dir / "app" / "__pycache__").mkdir()

            # Find all __pycache__ directories
            result = subprocess.run(
                ["find", str(test_dir), "-type", "d", "-name", "__pycache__"],
                capture_output=True,
                text=True,
                check=True,
            )

            # Should find 2 __pycache__ directories
            found = [line for line in result.stdout.strip().split("\n") if line]
            assert len(found) == 2


class TestGitCompatibility:
    """Test Git operations across platforms."""

    def test_git_available(self):
        """Test that Git is available."""
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "git version" in result.stdout

    def test_git_clone_depth(self):
        """Test git clone --depth works on all platforms."""
        # This is used in deploy.sh line 243 and scripts/deploy_to_bot.sh line 71
        # Test that --depth option is recognized (non-zero exit but option parsed)
        result = subprocess.run(
            ["git", "clone", "--depth", "1"],
            capture_output=True,
            text=True,
        )

        # Should fail with usage error (missing repo), not "unknown option"
        assert "unknown option" not in result.stderr.lower()
        assert "unrecognized option" not in result.stderr.lower()

    def test_git_config(self, tmp_path):
        """Test git config works on all platforms."""
        repo = tmp_path / "test_repo"
        repo.mkdir()

        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

        # Set config
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        # Read config
        result = subprocess.run(
            ["git", "config", "user.name"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == "Test User"


class TestDockerBuildCompatibility:
    """Test Docker build compatibility (buildx for multi-platform)."""

    def test_docker_buildx_available(self):
        """Test that docker buildx is available."""
        result = subprocess.run(
            ["docker", "buildx", "version"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            assert "buildx" in result.stdout.lower()
        else:
            pytest.skip("docker buildx not available")

    def test_docker_platform_flag(self):
        """Test that --platform flag is supported."""
        # Test that platform flag is recognized
        result = subprocess.run(
            ["docker", "build", "--help"],
            capture_output=True,
            text=True,
        )

        assert "--platform" in result.stdout

    @pytest.mark.slow
    def test_build_for_linux_amd64_from_macos(self, tmp_path):
        """Test building linux/amd64 image from macOS M2."""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            pytest.skip("Test requires M2 macOS")

        # Create minimal Dockerfile
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM alpine:latest\nRUN echo 'Hello'")

        # Build for linux/amd64
        result = subprocess.run(
            [
                "docker", "build",
                "--platform", "linux/amd64",
                "-t", "test-amd64",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        # Should succeed with buildx/emulation
        if result.returncode == 0:
            assert "Successfully built" in result.stdout or "Successfully tagged" in result.stdout

            # Cleanup
            subprocess.run(["docker", "rmi", "test-amd64"], capture_output=True)


class TestEnvironmentVariables:
    """Test environment variable handling across platforms."""

    def test_env_var_expansion(self):
        """Test environment variable expansion in shell."""
        os.environ["TEST_VAR"] = "test_value"

        result = subprocess.run(
            ["bash", "-c", "echo $TEST_VAR"],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == "test_value"

    def test_data_dir_env_var(self):
        """Test DATA_DIR environment variable."""
        os.environ["DATA_DIR"] = "/data"

        # Verify it's set
        assert os.environ.get("DATA_DIR") == "/data"

    def test_docker_env_file_compatibility(self, tmp_path):
        """Test .env file format works on all platforms."""
        env_file = tmp_path / ".env"
        env_file.write_text("""# Environment variables
TELEGRAM_TOKEN=123456:ABC-DEF
REDIS_URL=redis://localhost:6379
DATA_DIR=/data
""")

        # Read and parse
        env_vars = {}
        for line in env_file.read_text().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                env_vars[key] = value

        assert env_vars["TELEGRAM_TOKEN"] == "123456:ABC-DEF"
        assert env_vars["DATA_DIR"] == "/data"


class TestFilePermissions:
    """Test file permissions across platforms."""

    def test_chmod_script_executable(self, tmp_path):
        """Test making scripts executable with chmod."""
        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\necho test")

        # Make executable
        os.chmod(script, 0o755)

        # Verify
        assert os.access(script, os.X_OK)

    def test_script_execution_macos(self, tmp_path):
        """Test script execution on macOS."""
        if platform.system() != "Darwin":
            pytest.skip("Not running on macOS")

        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\necho 'macOS test'")
        os.chmod(script, 0o755)

        result = subprocess.run(
            [str(script)],
            capture_output=True,
            text=True,
            check=True,
        )

        assert "macOS test" in result.stdout

    def test_script_execution_linux(self, tmp_path):
        """Test script execution on Linux."""
        if platform.system() != "Linux":
            pytest.skip("Not running on Linux")

        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\necho 'Linux test'")
        os.chmod(script, 0o755)

        result = subprocess.run(
            [str(script)],
            capture_output=True,
            text=True,
            check=True,
        )

        assert "Linux test" in result.stdout
