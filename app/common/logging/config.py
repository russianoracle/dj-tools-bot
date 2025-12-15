"""Configuration management for mood classifier."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for mood classifier."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to custom config file. If None, uses default config.
        """
        self._config: Dict[str, Any] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Use default config
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "default_config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Expand home directory in paths
        self._expand_paths()

    def _expand_paths(self) -> None:
        """Expand ~ in file paths."""
        if 'performance' in self._config and 'cache_dir' in self._config['performance']:
            cache_dir = self._config['performance']['cache_dir']
            self._config['performance']['cache_dir'] = os.path.expanduser(cache_dir)

        if 'logging' in self._config and 'log_file' in self._config['logging']:
            log_file = self._config['logging']['log_file']
            self._config['logging']['log_file'] = os.path.expanduser(log_file)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'audio.sample_rate')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self, output_path: str) -> None:
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    @property
    def audio(self) -> Dict[str, Any]:
        """Get audio processing settings."""
        return self._config.get('audio', {})

    @property
    def features(self) -> Dict[str, Any]:
        """Get feature extraction settings."""
        return self._config.get('features', {})

    @property
    def classification(self) -> Dict[str, Any]:
        """Get classification settings."""
        return self._config.get('classification', {})

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata settings."""
        return self._config.get('metadata', {})

    @property
    def performance(self) -> Dict[str, Any]:
        """Get performance settings."""
        return self._config.get('performance', {})

    @property
    def gui(self) -> Dict[str, Any]:
        """Get GUI settings."""
        return self._config.get('gui', {})

    @property
    def zones(self) -> Dict[str, Any]:
        """Get zone definitions."""
        return self._config.get('zones', {})

    @property
    def export(self) -> Dict[str, Any]:
        """Get export settings."""
        return self._config.get('export', {})

    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging settings."""
        return self._config.get('logging', {})


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Path to custom config file (only used on first call)

    Returns:
        Config instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_path)

    return _config_instance
