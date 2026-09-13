"""Tests for image utilities."""

import pytest
from PIL import Image
import numpy as np
import torch

from nova.assets.images import (
    load_image,
    validate_image_pair,
    image_to_tensor,
    tensor_to_image,
    save_image,
    create_side_by_side,
    create_onion_skin,
    extract_patches,
    stitch_patches,
)


def create_test_image(color: tuple = (255, 0, 0), size: tuple = (256, 256)) -> Image.Image:
    """Create a solid color test image."""
    return Image.new("RGB", size, color)


def test_load_image(tmp_path):
    """Test loading an image."""
    img = create_test_image((100, 150, 200))
    path = tmp_path / "test.png"
    img.save(path)

    loaded = load_image(path)
    assert loaded.size == size
    assert loaded.mode == "RGB"


def test_validate_image_pair_same_size():
    """Test validation passes for same-size images."""
    img1 = create_test_image((255, 0, 0))
    img2 = create_test_image((0, 255, 0))

    # Should not raise
    validate_image_pair(img1, img2)


def test_validate_image_pair_different_size():
    """Test validation fails for different sizes."""
    img1 = create_test_image((255, 0, 0), (256, 256))
    img2 = create_test_image((0, 255, 0), (512, 512))

    with pytest.raises(ValueError, match="same size"):
        validate_image_pair(img1, img2)


def test_image_to_tensor():
    """Test PIL to tensor conversion."""
    img = create_test_image((128, 128, 128), (64, 64))
    tensor = image_to_tensor(img, normalize=False)

    assert tensor.shape == (3, 64, 64)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0

    # Test normalization
    tensor_norm = image_to_tensor(img, normalize=True)
    assert tensor_norm.min() >= -1.0 and tensor_norm.max() <= 1.0


def test_tensor_to_image():
    """Test tensor to PIL conversion."""
    tensor = torch.rand(3, 64, 64) * 2 - 1  # [-1, 1]
    img = tensor_to_image(tensor, denormalize=True)

    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_round_trip():
    """Test image -> tensor -> image round trip."""
    original = create_test_image((200, 100, 50), (128, 128))
    tensor = image_to_tensor(original, normalize=True)
    reconstructed = tensor_to_image(tensor, denormalize=True)

    # Check pixel values close (allow small floating point differences)
    orig_arr = np.array(original).astype(np.float32)
    recon_arr = np.array(reconstructed).astype(np.float32)
    diff = np.abs(orig_arr - recon_arr).max()
    assert diff < 2.0  # Allow small rounding differences


def test_create_side_by_side():
    """Test side-by-side comparison creation."""
    img1 = create_test_image((255, 0, 0), (100, 100))
    img2 = create_test_image((0, 255, 0), (100, 100))

    combined = create_side_by_side([img1, img2], labels=["Red", "Green"], spacing=10)

    assert combined.size == (210, 140)  # 2*100 + 10 spacing, 100 + 40 label height


def test_create_onion_skin():
    """Test onion skinning visualization."""
    prev = create_test_image((255, 0, 0), (100, 100))
    curr = create_test_image((0, 255, 0), (100, 100))
    next_frame = create_test_image((0, 0, 255), (100, 100))

    onion = create_onion_skin(prev, curr, next_frame, prev_alpha=0.5, next_alpha=0.5)

    assert isinstance(onion, Image.Image)
    assert onion.size == (100, 100)
    assert onion.mode == "RGB"


def test_extract_patches():
    """Test patch extraction."""
    img = create_test_image((128, 128, 128), (512, 512))
    patches = extract_patches(img, patch_size=256, stride=128)

    # Should get 3x3 = 9 patches for 512 with stride 128
    assert len(patches) >= 9
    for patch, (x, y) in patches:
        assert patch.size == (256, 256)


def test_stitch_patches():
    """Test patch stitching."""
    img = create_test_image((100, 150, 200), (512, 512))
    patches = extract_patches(img, patch_size=256, stride=128)

    stitched = stitch_patches(patches, (512, 512), patch_size=256, blend=True)

    assert stitched.size == (512, 512)
    # Check content preserved
    orig_arr = np.array(img).astype(np.float32)
    stitched_arr = np.array(stitched).astype(np.float32)
    diff = np.abs(orig_arr - stitched_arr).mean()
    assert diff < 1.0  # Blending introduces small differences


def test_save_image(tmp_path):
    """Test saving images in various formats."""
    img = create_test_image((50, 100, 150))

    # PIL Image
    path1 = save_image(img, tmp_path / "test1.png")
    assert path1.exists()

    # Tensor
    tensor = image_to_tensor(img, normalize=True)
    path2 = save_image(tensor, tmp_path / "test2.png")
    assert path2.exists()

    # Numpy array
    arr = np.array(img)
    path3 = save_image(arr, tmp_path / "test3.png")
    assert path3.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])