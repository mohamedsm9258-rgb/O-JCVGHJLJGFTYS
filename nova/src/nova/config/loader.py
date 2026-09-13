"""Configuration loading utilities."""

from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel

from .schemas import AnimationConfig, ModelConfig, MotionConfig, GenerationConfig, ValidationConfig, ExportConfig


def load_config(path: Path | str) -> dict[str, Any]:
    """Load YAML config file."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two config dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def create_animation_config(
    start_image: Path | str,
    end_image: Path | str,
    motion_prompt: str,
    config_path: Optional[Path | str] = None,
    **overrides,
) -> AnimationConfig:
    """
    Create AnimationConfig from base config + overrides.

    Args:
        start_image: Path to start frame
        end_image: Path to end frame
        motion_prompt: Natural language motion description
        config_path: Optional path to YAML config file
        **overrides: Any config field overrides

    Returns:
        Validated AnimationConfig
    """
    # Load base config
    base_config = {}
    if config_path:
        base_config = load_config(config_path)

    # Apply overrides
    config_dict = merge_configs(base_config, overrides)

    # Ensure required fields
    config_dict.setdefault("start_image", start_image)
    config_dict.setdefault("end_image", end_image)
    config_dict.setdefault("motion_prompt", motion_prompt)

    # Create and validate
    return AnimationConfig(**config_dict)


def load_model_config(config_path: Optional[Path | str] = None) -> ModelConfig:
    """Load ModelConfig from YAML."""
    if config_path:
        data = load_config(config_path)
        return ModelConfig(**data.get("models", {}))
    return ModelConfig()


def load_motion_config(config_path: Optional[Path | str] = None) -> MotionConfig:
    """Load MotionConfig from YAML."""
    if config_path:
        data = load_config(config_path)
        return MotionConfig(**data.get("motion", {}))
    return MotionConfig()


def load_generation_config(config_path: Optional[Path | str] = None) -> GenerationConfig:
    """Load GenerationConfig from YAML."""
    if config_path:
        data = load_config(config_path)
        return GenerationConfig(**data.get("generation", {}))
    return GenerationConfig()


def load_validation_config(config_path: Optional[Path | str] = None) -> ValidationConfig:
    """Load ValidationConfig from YAML."""
    if config_path:
        data = load_config(config_path)
        return ValidationConfig(**data.get("validation", {}))
    return ValidationConfig()


def load_export_config(config_path: Optional[Path | str] = None) -> ExportConfig:
    """Load ExportConfig from YAML."""
    if config_path:
        data = load_config(config_path)
        return ExportConfig(**data.get("export", {}))
    return ExportConfig()