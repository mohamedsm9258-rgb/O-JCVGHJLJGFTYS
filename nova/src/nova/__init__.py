"""
Nova: AI-Assisted 2D Anime Animation System

Progressive frame-by-frame animation construction from Start State to Target State
with human creative control, hybrid CV+generative pipeline, and audio-driven motion.
"""

__version__ = "0.1.0"
__author__ = "Nova Team"
__description__ = "AI-Assisted 2D Anime Animation System"

from .config import (
    AnimationConfig,
    AudioTrack,
    AudioRole,
    MotionConfig,
    GenerationConfig,
    ValidationConfig,
    ExportConfig,
    ModelConfig,
)
from .config.loader import create_animation_config, load_config
from .assets import (
    load_image,
    validate_image_pair,
    image_to_tensor,
    tensor_to_image,
    save_image,
    create_side_by_side,
    create_onion_skin,
)

__all__ = [
    "__version__",
    "AnimationConfig",
    "AudioTrack",
    "AudioRole",
    "MotionConfig",
    "GenerationConfig",
    "ValidationConfig",
    "ExportConfig",
    "ModelConfig",
    "create_animation_config",
    "load_config",
    "load_image",
    "validate_image_pair",
    "image_to_tensor",
    "tensor_to_image",
    "save_image",
    "create_side_by_side",
    "create_onion_skin",
]