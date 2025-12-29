"""Configuration management utilities.

This module provides utilities for loading and managing configuration files.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration object with dict-like and attribute access."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        self._convert_to_nested_config(config_dict)

    def _convert_to_nested_config(self, config_dict: Dict[str, Any]) -> None:
        """Convert nested dictionaries to Config objects.

        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Get config value by key.

        Args:
            key: Configuration key

        Returns:
            Configuration value
        """
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config.

        Args:
            key: Configuration key

        Returns:
            True if key exists, False otherwise
        """
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)
