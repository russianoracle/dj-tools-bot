"""Centralized logging configuration management."""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional


class LoggingConfig:
    """Centralized logging configuration for all components."""

    _instance: Optional['LoggingConfig'] = None
    _config: Dict = {}

    def __init__(self, config_path: Optional[str] = None):
        """Initialize logging configuration.

        Args:
            config_path: Path to logging-config.yaml (default: project root)
        """
        if config_path is None:
            # Try to find config in project root
            current = Path(__file__).parent
            for _ in range(5):  # Search up to 5 levels up
                config_file = current / "logging-config.yaml"
                if config_file.exists():
                    config_path = str(config_file)
                    break
                current = current.parent

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Fallback to defaults if no config file
            self._config = {
                'default_level': 'INFO',
                'components': {},
                'frameworks': {},
                'modules': {}
            }

    @classmethod
    def get_instance(cls) -> 'LoggingConfig':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_level(self, component: str = 'default') -> str:
        """Get log level for a component.

        Args:
            component: Component name (bot, worker, fluent-bit, etc)

        Returns:
            Log level string (DEBUG, INFO, WARNING, ERROR)
        """
        # Environment variable override (highest priority)
        env_var = f"LOG_LEVEL_{component.upper().replace('-', '_')}"
        if env_level := os.getenv(env_var):
            return env_level.upper()

        # Generic LOG_LEVEL for backward compatibility
        if component == 'default' and (env_level := os.getenv('LOG_LEVEL')):
            return env_level.upper()

        # Component-specific config
        if component in self._config.get('components', {}):
            comp_cfg = self._config['components'][component]
            if isinstance(comp_cfg, dict) and 'level' in comp_cfg:
                return comp_cfg['level'].upper()
            elif isinstance(comp_cfg, str):
                return comp_cfg.upper()

        # Default level from config
        return self._config.get('default_level', 'INFO').upper()

    def get_json_format(self, component: str = 'default') -> bool:
        """Get JSON format flag for a component.

        Args:
            component: Component name

        Returns:
            True if JSON format should be used
        """
        # Environment variable override
        env_var = f"LOG_JSON_FORMAT_{component.upper().replace('-', '_')}"
        if env_json := os.getenv(env_var):
            return env_json.lower() in ('true', '1', 'yes')

        # Generic LOG_JSON_FORMAT for backward compatibility
        if component == 'default' and (env_json := os.getenv('LOG_JSON_FORMAT')):
            return env_json.lower() in ('true', '1', 'yes')

        # Component-specific config
        if component in self._config.get('components', {}):
            comp_cfg = self._config['components'][component]
            if isinstance(comp_cfg, dict):
                return comp_cfg.get('json_format', True)

        return True

    def get_module_level(self, module_name: str) -> Optional[str]:
        """Get log level for a specific Python module.

        Args:
            module_name: Fully qualified module name (e.g., 'app.services.arq_worker')

        Returns:
            Log level or None if not configured
        """
        modules_cfg = self._config.get('modules', {})
        if module_name in modules_cfg:
            return modules_cfg[module_name].upper()
        return None

    def get_framework_level(self, framework: str) -> Optional[str]:
        """Get log level for a framework.

        Args:
            framework: Framework name (aiogram, arq, etc)

        Returns:
            Log level or None if not configured
        """
        frameworks_cfg = self._config.get('frameworks', {})
        if framework in frameworks_cfg:
            return frameworks_cfg[framework].upper()
        return None


def get_logging_config() -> LoggingConfig:
    """Get singleton logging configuration instance."""
    return LoggingConfig.get_instance()
