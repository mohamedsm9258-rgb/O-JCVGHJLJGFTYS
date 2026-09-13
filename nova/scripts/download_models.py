#!/usr/bin/env python
"""
Model download and cache management script for Nova.

Downloads all required models to local cache for offline/Colab use.
Run once before first training/inference.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nova.config import ModelConfig
from huggingface_hub import snapshot_download, hf_hub_download
from loguru import logger


MODEL_SPECS = {
    "sdxl_base": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "allow_patterns": ["*.safetensors", "*.json", "*.fp16.safetensors"],
    },
    "sdxl_inpaint": {
        "repo_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "allow_patterns": ["*.safetensors", "*.json", "*.fp16.safetensors"],
    },
    "sdxl_vae": {
        "repo_id": "madebyollin/sdxl-vae-fp16-fix",
        "allow_patterns": ["*.safetensors", "*.json"],
    },
    "controlnet_openpose": {
        "repo_id": "thibaud/controlnet-sdxl-openpose",
        "allow_patterns": ["*.safetensors", "*.json", "*.yaml"],
    },
    "controlnet_depth": {
        "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
        "allow_patterns": ["*.safetensors", "*.json"],
    },
    "controlnet_canny": {
        "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
        "allow_patterns": ["*.safetensors", "*.json"],
    },
    "ip_adapter": {
        "repo_id": "h94/IP-Adapter",
        "allow_patterns": [
            "sdxl_models/*",
            "models/image_encoder/*",
            "ip-adapter-plus_sdxl_vit-h.safetensors",
            "ip-adapter-plus-face_sdxl_vit-h.safetensors",
        ],
    },
    "depth_anything_v2": {
        "repo_id": "depth-anything/Depth-Anything-V2-Large",
        "allow_patterns": ["*.pth", "*.json", "*.py"],
    },
    "raft": {
        "repo_id": "fvcore/raft-things.pth",
        "filename": "raft-things.pth",
    },
    "whisperx": {
        "repo_id": "m-bain/whisperx-vad",
        "allow_patterns": ["*"],
    },
}


def setup_cache_dir(cache_dir: Path) -> Path:
    """Create and return cache directory."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    (cache_dir / "torch").mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory: {cache_dir}")
    return cache_dir


def download_model(repo_id: str, cache_dir: Path, **kwargs) -> Path:
    """Download a model from Hugging Face Hub."""
    local_dir = cache_dir / repo_id.replace("/", "--")
    logger.info(f"Downloading {repo_id} -> {local_dir}")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            **kwargs,
        )
        logger.success(f"Downloaded: {repo_id}")
        return Path(path)
    except Exception as e:
        logger.error(f"Failed to download {repo_id}: {e}")
        raise


def download_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    """Download a single file from Hugging Face Hub."""
    logger.info(f"Downloading {filename} from {repo_id}")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            resume_download=True,
        )
        logger.success(f"Downloaded: {filename}")
        return Path(path)
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        raise


def download_mediapipe_models(cache_dir: Path) -> None:
    """MediaPipe models are auto-downloaded on first use, but we can pre-fetch."""
    logger.info("MediaPipe models will auto-download on first use")
    # MediaPipe stores in ~/.mediapipe - we can't easily pre-download


def download_whisperx_models(cache_dir: Path) -> None:
    """WhisperX models auto-download. Just ensure cache dir is set."""
    logger.info("WhisperX models will auto-download on first use")


def verify_downloads(cache_dir: Path) -> dict[str, bool]:
    """Verify all expected model files exist."""
    results = {}

    # Check key files
    checks = {
        "sdxl_base": cache_dir / "stabilityai--stable-diffusion-xl-base-1.0" / "model_index.json",
        "sdxl_inpaint": cache_dir / "diffusers--stable-diffusion-xl-1.0-inpainting-0.1" / "model_index.json",
        "sdxl_vae": cache_dir / "madebyollin--sdxl-vae-fp16-fix" / "diffusion_pytorch_model.fp16.safetensors",
        "controlnet_openpose": cache_dir / "thibaud--controlnet-sdxl-openpose" / "diffusion_pytorch_model.safetensors",
        "controlnet_depth": cache_dir / "diffusers--controlnet-depth-sdxl-1.0" / "diffusion_pytorch_model.safetensors",
        "controlnet_canny": cache_dir / "diffusers--controlnet-canny-sdxl-1.0" / "diffusion_pytorch_model.safetensors",
        "ip_adapter_plus": cache_dir / "h94--IP-Adapter" / "sdxl_models" / "ip-adapter-plus_sdxl_vit-h.safetensors",
        "ip_adapter_face": cache_dir / "h94--IP-Adapter" / "sdxl_models" / "ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "ip_adapter_image_encoder": cache_dir / "h94--IP-Adapter" / "models" / "image_encoder" / "model.safetensors",
    }

    for name, path in checks.items():
        exists = path.exists()
        results[name] = exists
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {name}: {path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Download Nova models")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./models"),
        help="Cache directory for models (default: ./models)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Model names to skip (e.g., sdxl_base controlnet_openpose)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    cache_dir = setup_cache_dir(args.cache_dir)

    if args.list:
        print("Available models:")
        for name in MODEL_SPECS:
            print(f"  {name}")
        return 0

    if args.verify_only:
        logger.info("Verifying existing downloads...")
        results = verify_downloads(cache_dir)
        all_ok = all(results.values())
        if all_ok:
            logger.success("All models verified!")
        else:
            missing = [k for k, v in results.items() if not v]
            logger.error(f"Missing: {missing}")
        return 0 if all_ok else 1

    # Download each model
    skipped = set(args.skip)
    failed = []

    for name, spec in MODEL_SPECS.items():
        if name in skipped:
            logger.info(f"Skipping {name}")
            continue

        try:
            if "filename" in spec:
                download_file(spec["repo_id"], spec["filename"], cache_dir)
            else:
                download_model(spec["repo_id"], cache_dir, allow_patterns=spec.get("allow_patterns"))
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            failed.append(name)

    # Handle special cases
    if "raft" not in skipped:
        try:
            download_file("fvcore/raft-things.pth", "raft-things.pth", cache_dir)
        except Exception as e:
            logger.warning(f"RAFT download failed (may need manual): {e}")

    # Auto-download models
    download_mediapipe_models(cache_dir)
    download_whisperx_models(cache_dir)

    # Verify
    logger.info("\nVerifying downloads...")
    results = verify_downloads(cache_dir)

    if failed:
        logger.error(f"Failed downloads: {failed}")
        return 1

    if all(results.values()):
        logger.success("All models downloaded and verified!")
        return 0
    else:
        missing = [k for k, v in results.items() if not v]
        logger.warning(f"Some models may be incomplete: {missing}")
        return 0


if __name__ == "__main__":
    sys.exit(main())