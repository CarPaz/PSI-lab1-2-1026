"""Only generated local files are used; no sample-image downloads in CI."""

import numpy as np
import pytest
from skimage import io

from psi_lab.lab1.datos import load_image, periodic_pattern, synthetic_image, zone_plate


def test_synthetic_image_is_deterministic_and_bounded():
    a = synthetic_image((41, 57))
    np.testing.assert_array_equal(a, synthetic_image((41, 57)))
    assert a.shape == (41, 57)
    assert 0 <= a.min() < a.max() <= 1


def test_local_grayscale_load_uses_dtype_range(tmp_path):
    pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / "generated.png"
    io.imsave(path, pixels, check_contrast=False)
    np.testing.assert_allclose(load_image(path), pixels / 255)
    assert load_image(path, (12, 18)).shape == (12, 18)


def test_rgba_load_composites_transparency_on_white(tmp_path):
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    path = tmp_path / "transparent.png"
    io.imsave(path, pixels, check_contrast=False)
    np.testing.assert_allclose(load_image(path), 1)


def test_missing_real_image_is_not_replaced_with_synthetic_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "absent.png")


def test_zone_plate_matches_formula_and_symmetry():
    size = 33
    image = zone_plate(size)
    assert image.shape == (size, size)
    assert image[16, 16] == 1
    assert image[16, 20] == pytest.approx(0.5 * (1 + np.cos(np.pi * 16 / size)))
    np.testing.assert_allclose(image, image.T)
    assert 0 <= image.min() <= image.max() <= 1


@pytest.mark.parametrize("operation", [
    lambda: synthetic_image(0),
    lambda: zone_plate((32, 48)),
    lambda: periodic_pattern((32, 32), 0.2, 0.6, 0),
    lambda: periodic_pattern((32, 32), -0.2, 0.1, 0),
])
def test_invalid_generation_parameters(operation):
    with pytest.raises(ValueError):
        operation()
