"""Pydantic schemas for Nova configuration and data models."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class AudioRole(str, Enum):
    """User-defined role for an audio track."""

    CHARACTER_VOICE = "character_voice"
    BACKGROUND_MUSIC = "background_music"
    SOUND_EFFECTS = "sound_effects"
    REFERENCE_TIMING = "reference_timing"
    CUSTOM = "custom"


class AudioTrack(BaseModel):
    """Single audio input with user-defined role."""

    path: Path
    role: AudioRole
    character_id: Optional[str] = None  # For multi-character voice tracks
    weight: float = Field(default=1.0, ge=0.0, le=2.0)  # Influence weight
    offset_seconds: float = Field(default=0.0)  # Time offset from frame 0

    @field_validator("path", mode="before")
    @classmethod
    def _resolve_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


class ModelConfig(BaseModel):
    """Model selection and parameters."""

    # Base diffusion model
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    inpainting_model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix"

    # ControlNet
    controlnet_openpose: str = "thibaud/controlnet-sdxl-openpose"
    controlnet_depth: str = "diffusers/controlnet-depth-sdxl-1.0"
    controlnet_canny: str = "diffusers/controlnet-canny-sdxl-1.0"

    # IP-Adapter
    ip_adapter_plus: str = "h94/IP-Adapter"
    ip_adapter_face: str = "h94/IP-Adapter/face_id"

    # Device & precision
    device: str = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    use_xformers: bool = True
    enable_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False

    # VAE tiling for high-res
    vae_tiling: bool = True
    vae_tile_size: int = 512

    # LoRA
    lora_dir: Path = Path("./models/loras")
    default_lora_scale: float = 0.8


class MotionConfig(BaseModel):
    """Motion planning configuration."""

    # Keyframe interpolation
    interpolation: Literal["linear", "ease_in", "ease_out", "ease_in_out", "custom"] = "ease_in_out"
    custom_curve: Optional[list[float]] = None  # 0-1 values for custom easing

    # Per-joint overrides (joint_name -> curve_type)
    joint_curves: dict[str, str] = Field(default_factory=dict)

    # Audio-driven motion
    audio_sync_enabled: bool = True
    beat_snap_enabled: bool = True
    beat_snap_tolerance_frames: int = 1
    phoneme_to_viseme_enabled: bool = True
    prosody_to_intensity_enabled: bool = True

    # Motion constraints
    max_joint_delta_per_frame: float = 0.15  # Normalized 0-1
    preserve_face: bool = True
    preserve_hands: bool = False  # Hands often need to move


class GenerationConfig(BaseModel):
    """Frame generation parameters."""

    # Resolution
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)

    # Sampling
    sampler: Literal["euler_a", "euler", "dpmpp_2m", "dpmpp_2m_sde", "ddim", "unipc"] = "euler_a"
    steps: int = Field(default=25, ge=10, le=100)
    cfg_scale: float = Field(default=5.5, ge=1.0, le=20.0)
    denoising_strength: float = Field(default=0.7, ge=0.1, le=1.0)

    # Conditioning scales
    controlnet_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    controlnet_depth_scale: float = Field(default=0.6, ge=0.0, le=2.0)
    controlnet_canny_scale: float = Field(default=0.4, ge=0.0, le=2.0)
    ip_adapter_scale: float = Field(default=0.6, ge=0.0, le=2.0)
    ip_adapter_face_scale: float = Field(default=0.4, ge=0.0, le=2.0)

    # Latent initialization
    use_prev_latent_init: bool = True
    prev_latent_weight: float = Field(default=0.8, ge=0.0, le=1.0)

    # Seeds
    seed: Optional[int] = None
    seed_per_frame: bool = False

    # Quality modes
    draft_mode: bool = False
    draft_steps: int = 12
    draft_cfg_scale: float = 4.5

    # Warping
    warp_enabled: bool = True
    warp_method: Literal["delaunay", "raft", "sparse_flow"] = "delaunay"
    warp_strength: float = Field(default=1.0, ge=0.0, le=2.0)

    # Inpainting refinement
    refine_enabled: bool = True
    refine_mask_dilation: int = 8
    refine_mask_blur: int = 4


class ValidationConfig(BaseModel):
    """Frame validation thresholds."""

    # Identity thresholds (higher = stricter)
    clip_similarity_min: float = Field(default=0.85, ge=0.0, le=1.0)
    dino_similarity_min: float = Field(default=0.80, ge=0.0, le=1.0)
    arcface_similarity_min: float = Field(default=0.75, ge=0.0, le=1.0)

    # Temporal coherence
    ssim_min: float = Field(default=0.90, ge=0.0, le=1.0)
    lpips_max: float = Field(default=0.10, ge=0.0, le=1.0)
    flow_consistency_min: float = Field(default=0.90, ge=0.0, le=1.0)

    # Pose accuracy
    pose_keypoint_max_dist: float = Field(default=0.05, ge=0.0, le=1.0)  # Normalized

    # Artifacts
    artifact_score_max: float = Field(default=0.3, ge=0.0, le=1.0)

    # Audio sync
    lip_sync_min: float = Field(default=0.70, ge=0.0, le=1.0)
    beat_sync_max_offset_frames: int = 1

    # Auto-retry
    max_retries: int = 3
    retry_denoising_increment: float = 0.05
    retry_ip_adapter_increment: float = 0.1


class ExportConfig(BaseModel):
    """Output export settings."""

    # Formats
    formats: list[Literal["png", "mp4", "webm", "gif", "exr"]] = Field(default=["png", "mp4"])

    # Video encoding
    video_codec: Literal["h264", "h265", "vp9", "av1"] = "h264"
    video_crf: int = Field(default=18, ge=0, le=51)
    video_preset: str = "slow"
    video_pixel_format: str = "yuv420p"

    # GIF
    gif_fps: int = 12
    gif_colors: int = 256

    # Metadata
    include_metadata_json: bool = True
    metadata_per_frame: bool = True


class AnimationConfig(BaseModel):
    """Complete animation job configuration."""

    # Core inputs
    start_image: Path
    end_image: Path
    motion_prompt: str

    # Timing
    fps: int = Field(default=24, ge=1, le=60)
    frame_count: int = Field(default=48, ge=2, le=500)
    duration_seconds: Optional[float] = None  # Alternative to frame_count

    # Audio
    audio_tracks: list[AudioTrack] = Field(default_factory=list)

    # Sub-configs
    models: ModelConfig = Field(default_factory=ModelConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    # Output
    output_dir: Path = Path("./output")
    project_name: str = "animation"

    # Runtime
    device: str = "cuda"
    verbose: bool = True

    @field_validator("start_image", "end_image", "output_dir", mode="before")
    @classmethod
    def _resolve_paths(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    @field_validator("duration_seconds")
    @classmethod
    def _compute_frame_count(cls, v: Optional[float], info) -> Optional[float]:
        if v is not None and "frame_count" not in info.data:
            # Will be computed in model_post_init
            pass
        return v

    def model_post_init(self, __context: Any) -> None:
        if self.duration_seconds is not None:
            self.frame_count = int(self.duration_seconds * self.fps)
        self.output_dir = self.output_dir / self.project_name
        self.output_dir.mkdir(parents=True, exist_ok=True)