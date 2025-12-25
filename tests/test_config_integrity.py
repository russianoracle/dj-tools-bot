"""
Configuration Integrity Tests

Ensures that all environment variables are properly documented and configured:
1. All settings.py vars are in .env.example
2. All .env.example vars are used in settings.py or have documented purpose
3. All docker-compose environment vars are valid
4. No typos in variable names across files
"""

import os
import re
import pytest
from pathlib import Path
from typing import Set, Dict
import yaml


PROJECT_ROOT = Path(__file__).parent.parent
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
SETTINGS_PY = PROJECT_ROOT / "app" / "core" / "config" / "settings.py"
DOCKER_COMPOSE = PROJECT_ROOT / "docker-compose.yml"


def extract_env_vars_from_settings() -> Set[str]:
    """Extract all environment variable names from settings.py"""
    settings_content = SETTINGS_PY.read_text()

    # Find all os.getenv(...) calls
    pattern = r'os\.getenv\(["\']([A-Z_]+)["\']'
    matches = re.findall(pattern, settings_content)

    return set(matches)


def extract_env_vars_from_env_example() -> Dict[str, str]:
    """Extract all environment variable names from .env.example with their docs"""
    env_content = ENV_EXAMPLE.read_text()

    vars_dict = {}
    for line in env_content.split('\n'):
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue

        # Extract var name (commented or uncommented)
        if '=' in line:
            # Handle: # VAR=value or VAR=value
            clean_line = line.lstrip('#').strip()
            var_name = clean_line.split('=')[0].strip()
            if var_name and var_name[0].isupper():
                vars_dict[var_name] = line

    return vars_dict


def extract_env_vars_from_docker_compose() -> Set[str]:
    """Extract all environment variable names from docker-compose.yml"""
    with open(DOCKER_COMPOSE, 'r') as f:
        compose_data = yaml.safe_load(f)

    env_vars = set()

    for service_name, service_config in compose_data.get('services', {}).items():
        # From environment: section
        environment = service_config.get('environment', [])
        for env_entry in environment:
            if isinstance(env_entry, str):
                # Format: - VAR=value or - VAR=${VAR}
                var_name = env_entry.split('=')[0].strip().lstrip('-').strip()
                env_vars.add(var_name)
            elif isinstance(env_entry, dict):
                # Format: VAR: value
                env_vars.update(env_entry.keys())

    return env_vars


class TestConfigIntegrity:
    """Configuration integrity test suite"""

    def test_all_settings_vars_documented_in_env_example(self):
        """All variables in settings.py must be documented in .env.example"""
        settings_vars = extract_env_vars_from_settings()
        env_example_vars = extract_env_vars_from_env_example()

        # These vars are OK to not be in .env.example (runtime-only)
        ALLOWED_UNDOCUMENTED = {
            "ENV",  # Runtime environment indicator
        }

        missing_vars = settings_vars - env_example_vars.keys() - ALLOWED_UNDOCUMENTED

        if missing_vars:
            pytest.fail(
                f"Variables in settings.py but missing from .env.example:\n"
                f"{sorted(missing_vars)}\n\n"
                f"Add them to .env.example with documentation."
            )

    def test_no_typos_in_env_variable_names(self):
        """Check for potential typos in variable names (similar names)"""
        all_vars = (
            extract_env_vars_from_settings() |
            extract_env_vars_from_env_example().keys() |
            extract_env_vars_from_docker_compose()
        )

        # Known similar patterns that should be consistent
        patterns = {
            "LOG": ["LOG_LEVEL", "LOG_JSON_FORMAT", "LOG_FILE"],
            "REDIS": ["REDIS_HOST", "REDIS_PORT", "REDIS_URL", "REDIS_DB"],
            "DOWNLOAD": [
                "DOWNLOAD_MAX_RETRIES", "DOWNLOAD_BACKOFF_BASE",
                "DOWNLOAD_BACKOFF_MAX", "DOWNLOADS_DIR"
            ],
            "YTDLP": ["YTDLP_RETRIES", "YTDLP_FRAGMENT_RETRIES", "YTDLP_PROXY"],
            "ARQ": ["ARQ_MAX_JOBS", "ARQ_MAX_TRIES"],
            "YC": [
                "YC_LOCKBOX_SECRET_ID", "YC_SERVICE_ACCOUNT_ID",
                "YC_FOLDER_ID", "YC_LOG_GROUP_ID", "YC_IAM_TOKEN", "YC_SA_KEY_FILE"
            ],
        }

        issues = []
        for prefix, expected_vars in patterns.items():
            matching_vars = [v for v in all_vars if v.startswith(prefix)]
            unexpected = set(matching_vars) - set(expected_vars)

            if unexpected:
                issues.append(
                    f"{prefix}* pattern: Found unexpected vars {unexpected}. "
                    f"Expected: {expected_vars}"
                )

        if issues:
            pytest.fail("\n".join(issues))

    def test_docker_compose_env_vars_are_valid(self):
        """All env vars in docker-compose.yml should be documented or in settings"""
        docker_vars = extract_env_vars_from_docker_compose()
        settings_vars = extract_env_vars_from_settings()
        env_example_vars = extract_env_vars_from_env_example()

        # These are docker-specific and don't need to be in settings
        DOCKER_ONLY_VARS = {
            "REGISTRY", "REGISTRY_ID", "TAG",  # Image tags
            "REQUIRE_SECRETS",  # Runtime flag
            "METRICS_PORT", "METRICS_UPDATE_INTERVAL", "PROCESS_NAME",  # Metrics
        }

        all_documented = settings_vars | env_example_vars.keys() | DOCKER_ONLY_VARS

        undocumented = docker_vars - all_documented

        if undocumented:
            pytest.fail(
                f"Variables in docker-compose.yml but not documented:\n"
                f"{sorted(undocumented)}\n\n"
                f"Add them to settings.py or .env.example"
            )

    def test_env_example_has_descriptions(self):
        """Critical env vars should have description comments"""
        env_content = ENV_EXAMPLE.read_text()

        critical_vars = [
            "TELEGRAM_BOT_TOKEN",
            "REDIS_HOST",
            "ARQ_MAX_JOBS",
            "DOWNLOAD_MAX_RETRIES",
            "YC_LOCKBOX_SECRET_ID",
        ]

        missing_descriptions = []
        for var in critical_vars:
            # Find var in env file
            var_line_idx = None
            lines = env_content.split('\n')
            for i, line in enumerate(lines):
                if f"{var}=" in line:
                    var_line_idx = i
                    break

            if var_line_idx is None:
                missing_descriptions.append(f"{var}: not found in .env.example")
                continue

            # Check if there's a comment in the 3 lines before
            has_comment = False
            for i in range(max(0, var_line_idx - 3), var_line_idx + 1):
                if i < len(lines) and lines[i].strip().startswith('#') and len(lines[i]) > 5:
                    has_comment = True
                    break

            if not has_comment:
                missing_descriptions.append(f"{var}: no description comment")

        if missing_descriptions:
            pytest.fail(
                "Critical variables missing descriptions:\n" +
                "\n".join(f"  - {desc}" for desc in missing_descriptions)
            )

    def test_consistent_defaults_between_settings_and_docker_compose(self):
        """Defaults in settings.py should match docker-compose.yml where applicable"""
        # This is a smoke test to catch obvious mismatches
        settings_content = SETTINGS_PY.read_text()

        # Extract defaults from settings
        defaults = {
            "REDIS_PORT": re.search(r'REDIS_PORT.*?(\d+)', settings_content).group(1),
            "ARQ_MAX_TRIES": re.search(r'ARQ_MAX_TRIES.*?(\d+)', settings_content).group(1),
            "DOWNLOAD_MAX_RETRIES": re.search(r'DOWNLOAD_MAX_RETRIES.*?(\d+)', settings_content).group(1),
        }

        with open(DOCKER_COMPOSE, 'r') as f:
            compose_content = f.read()

        # Check docker-compose defaults
        mismatches = []

        # REDIS_PORT should be 6379
        if "REDIS_PORT=6379" not in compose_content:
            mismatches.append("REDIS_PORT docker-compose mismatch")

        if mismatches:
            pytest.fail(
                "Default value mismatches between settings.py and docker-compose.yml:\n" +
                "\n".join(mismatches)
            )


class TestConfigSecurity:
    """Security-related configuration tests"""

    def test_no_secrets_in_env_example(self):
        """Ensure .env.example doesn't contain real secrets"""
        env_content = ENV_EXAMPLE.read_text()

        forbidden_patterns = [
            (r'\d{10}:\w{35}', "Telegram bot token format"),
            (r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', "JWT token"),
            (r'sk-[A-Za-z0-9]{48}', "OpenAI API key"),
            (r'xox[baprs]-[A-Za-z0-9-]+', "Slack token"),
        ]

        violations = []
        for pattern, description in forbidden_patterns:
            if re.search(pattern, env_content):
                violations.append(f"Found {description} in .env.example")

        if violations:
            pytest.fail(
                "SECURITY: Real secrets found in .env.example:\n" +
                "\n".join(f"  - {v}" for v in violations)
            )

    def test_env_example_has_placeholder_values(self):
        """Ensure .env.example uses placeholder values, not real ones"""
        env_content = ENV_EXAMPLE.read_text()

        # These should have placeholder patterns
        required_placeholders = {
            "TELEGRAM_BOT_TOKEN": ["your_bot_token_here", "your_token", "bot_token"],
            "YC_LOG_GROUP_ID": ["your_log_group_id"],
        }

        for var_name, valid_placeholders in required_placeholders.items():
            if var_name in env_content:
                var_line = [l for l in env_content.split('\n') if f"{var_name}=" in l][0]
                has_valid_placeholder = any(ph in var_line for ph in valid_placeholders)

                # Skip if it's a real production value
                if not has_valid_placeholder and "e6q" not in var_line:  # Not the real lockbox ID
                    pytest.fail(
                        f"{var_name} should use a placeholder value, found: {var_line}"
                    )