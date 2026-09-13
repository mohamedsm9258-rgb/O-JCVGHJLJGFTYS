"""Image loading, validation, and display utilities for Nova."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from loguru import logger


def load_image(
    path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    mode: str = "RGB",
) -> Image.Image:
    """
    Load an image from disk with optional resizing.

    Args:
        path: Path to image file
        target_size: Optional (width, height) to resize to
        mode: PIL mode ('RGB', 'RGBA', 'L')

    Returns:
        PIL Image
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    img = img.convert(mode)

    if target_size and img.size != target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        logger.debug(f"Resized {path.name} to {target_size}")

    return img


def validate_image_pair(
    start_path: Union[str, Path],
    end_path: Union[str, Path],
    require_same_size: bool = True,
    require_same_mode: bool = True,
) -> Tuple[Image.Image, Image.Image]:
    """
    Load and validate a start/end image pair.

    Args:
        start_path: Path to start frame
        end_path: Path to end frame
        require_same_size: Both images must have same dimensions
        require_same_mode: Both images must have same color mode

    Returns:
        Tuple of (start_image, end_image) as PIL Images
    """
    start_img = load_image(start_path)
    end_img = load_image(end_path)

    if require_same_size and start_img.size != end_img.size:
        raise ValueError(
            f"Start and end images must have same size. "
            f"Start: {start_img.size}, End: {end_img.size}"
        )

    if require_same_mode and start_img.mode != end_img.mode:
        raise ValueError(
            f"Start and end images must have same mode. "
            f"Start: {start_img.mode}, End: {end_img.mode}"
        )

    logger.info(f"Validated image pair: {start_img.size} {start_img.mode}")
    return start_img, end_img


def image_to_tensor(
    img: Image.Image,
    normalize: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert PIL Image to PyTorch tensor [C, H, W] in range [0, 1] or [-1, 1].

    Args:
        img: PIL Image
        normalize: If True, normalize to [-1, 1] for diffusion models
        device: Target device

    Returns:
        Tensor of shape [C, H, W]
    """
    arr = np.array(img).astype(np.float32) / 255.0  # [H, W, C] in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]

    if normalize:
        tensor = tensor * 2.0 - 1.0  # [-1, 1]

    if device:
        tensor = tensor.to(device)

    return tensor


def tensor_to_image(
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> Image.Image:
    """
    Convert PyTorch tensor [C, H, W] to PIL Image.

    Args:
        tensor: Tensor in range [-1, 1] or [0, 1]
        denormalize: If True, assume [-1, 1] and convert to [0, 1]

    Returns:
        PIL Image
    """
    if denormalize:
        tensor = (tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    tensor = tensor.clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_image(
    img: Union[Image.Image, torch.Tensor, np.ndarray],
    path: Union[str, Path],
    quality: int = 95,
) -> Path:
    """
    Save image to disk.

    Args:
        img: PIL Image, tensor [C, H, W], or numpy array [H, W, C]
        path: Output path
        quality: JPEG/WebP quality (1-100)

    Returns:
        Path to saved file
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(img, torch.Tensor):
        img = tensor_to_image(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))

    img.save(path, quality=quality)
    logger.debug(f"Saved image to {path}")
    return path


def create_side_by_side(
    images: list[Image.Image],
    labels: Optional[list[str]] = None,
    spacing: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    label_height: int = 40,
    font_size: int = 16,
) -> Image.Image:
    """
    Create a horizontal side-by-side comparison image.

    Args:
        images: List of PIL Images (all same size)
        labels: Optional labels for each image
        spacing: Horizontal spacing between images
        background_color: Background RGB color
        label_height: Height of label area at top
        font_size: Font size for labels

    Returns:
        Combined PIL Image
    """
    if not images:
        raise ValueError("At least one image required")

    # Validate all same size
    first_size = images[0].size
    for i, img in enumerate(images):
        if img.size != first_size:
            raise ValueError(f"Image {i} size {img.size} != first {first_size}")

    w, h = first_size
    n = len(images)
    total_w = n * w + (n - 1) * spacing
    total_h = h + (label_height if labels else 0)

    canvas = Image.new("RGB", (total_w, total_h), background_color)

    for i, img in enumerate(images):
        x = i * (w + spacing)
        y = label_height if labels else 0
        canvas.paste(img, (x, y))

    if labels:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        for i, label in enumerate(labels):
            x = i * (w + spacing) + w // 2
            y = label_height // 2
            bbox = draw.textbbox((x, y), label, font=font, anchor="mm")
            draw.text((x, y), label, font=font, fill=(0, 0, 0), anchor="mm")

    return canvas


def create_frame_grid(
    images: list[Image.Image],
    cols: int = 6,
    spacing: int = 5,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Create a grid of frames for sequence overview.

    Args:
        images: List of PIL Images (all same size)
        cols: Number of columns
        spacing: Spacing between frames
        background_color: Background RGB color

    Returns:
        Combined PIL Image grid
    """
    if not images:
        raise ValueError("At least one image required")

    first_size = images[0].size
    for i, img in enumerate(images):
        if img.size != first_size:
            raise ValueError(f"Image {i} size {img.size} != first {first_size}")

    w, h = first_size
    n = len(images)
    rows = (n + cols - 1) // cols

    total_w = cols * w + (cols - 1) * spacing
    total_h = rows * h + (rows - 1) * spacing

    canvas = Image.new("RGB", (total_w, total_h), background_color)

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * (w + spacing)
        y = row * (h + spacing)
        canvas.paste(img, (x, y))

    return canvas


def create_onion_skin(
    prev_frame: Image.Image,
    curr_frame: Image.Image,
    next_frame: Optional[Image.Image] = None,
    prev_alpha: float = 0.3,
    next_alpha: float = 0.3,
    curr_alpha: float = 1.0,
) -> Image.Image:
    """
    Create onion-skinning visualization: previous (ghosted) + current + next (ghosted).

    Args:
        prev_frame: Previous frame
        curr_frame: Current frame (full opacity)
        next_frame: Optional next frame
        prev_alpha: Opacity for previous frame
        next_alpha: Opacity for next frame
        curr_alpha: Opacity for current frame

    Returns:
        Combined PIL Image with onion skinning
    """
    if prev_frame.size != curr_frame.size:
        raise ValueError("All frames must have same size")

    # Convert to RGBA for alpha blending
    prev_rgba = prev_frame.convert("RGBA")
    curr_rgba = curr_frame.convert("RGBA")

    # Apply alpha to previous
    prev_data = np.array(prev_rgba)
    prev_data[:, :, 3] = (prev_data[:, :, 3] * prev_alpha).astype(np.uint8)
    prev_rgba = Image.fromarray(prev_data, "RGBA")

    # Apply alpha to current
    curr_data = np.array(curr_rgba)
    curr_data[:, :, 3] = (curr_data[:, :, 3] * curr_alpha).astype(np.uint8)
    curr_rgba = Image.fromarray(curr_data, "RGBA")

    # Composite: prev -> curr
    result = Image.alpha_composite(prev_rgba, curr_rgba)

    if next_frame:
        if next_frame.size != curr_frame.size:
            raise ValueError("Next frame must have same size")
        next_rgba = next_frame.convert("RGBA")
        next_data = np.array(next_rgba)
        next_data[:, :, 3] = (next_data[:, :, 3] * next_alpha).astype(np.uint8)
        next_rgba = Image.fromarray(next_data, "RGBA")
        result = Image.alpha_composite(result, next_rgba)

    return result.convert("RGB")


def extract_patches(
    img: Image.Image,
    patch_size: int = 256,
    stride: Optional[int] = None,
) -> list[Tuple[Image.Image, Tuple[int, int]]]:
    """
    Extract overlapping patches from image for high-res processing.

    Args:
        img: Source image
        patch_size: Size of square patches
        stride: Stride between patches (default: patch_size // 2)

    Returns:
        List of (patch_image, (x, y)) tuples
    """
    if stride is None:
        stride = patch_size // 2

    w, h = img.size
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append((patch, (x, y)))

    # Handle right/bottom edges if not covered
    if (w - patch_size) % stride != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append((patch, (x, y)))

    if (h - patch_size) % stride != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append((patch, (x, y)))

    return patches


def stitch_patches(
    patches: list[Tuple[Image.Image, Tuple[int, int]]],
    canvas_size: Tuple[int, int],
    patch_size: int = 256,
    blend: bool = True,
) -> Image.Image:
    """
    Stitch patches back into full image with optional feathered blending.

    Args:
        patches: List of (patch_image, (x, y)) tuples
        canvas_size: Output (width, height)
        patch_size: Patch size used for extraction
        blend: Use feathered blending at overlaps

    Returns:
        Stitched PIL Image
    """
    canvas = Image.new("RGB", canvas_size, (0, 0, 0))
    weight_map = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
    canvas_arr = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)

    for patch, (x, y) in patches:
        patch_arr = np.array(patch).astype(np.float32) / 255.0

        if blend:
            # Create feathered weight mask
            feather = min(16, patch_size // 8)
            weight = np.ones((patch_size, patch_size), dtype=np.float32)
            # Fade edges
            for i in range(feather):
                alpha = i / feather
                weight[i, :] *= alpha
                weight[-1-i, :] *= alpha
                weight[:, i] *= alpha
                weight[:, -1-i] *= alpha
        else:
            weight = np.ones((patch_size, patch_size), dtype=np.float32)

        # Accumulate
        canvas_arr[y:y+patch_size, x:x+patch_size] += patch_arr * weight[:, :, None]
        weight_map[y:y+patch_size, x:x+patch_size] += weight

    # Normalize
    weight_map[weight_map == 0] = 1.0
    result_arr = (canvas_arr / weight_map[:, :, None] * 255).astype(np.uint8)

    return Image.fromarray(result_arr)