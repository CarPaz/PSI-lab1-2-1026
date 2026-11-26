"""Load photographs and generate deterministic data for the image experiments.

Images are grayscale floating-point arrays in [0, 1]. Input photographs belong
in the ignored data/ directory; experiments write their results under out/.
"""

from pathlib import Path

import numpy as np


def _shape(size):
    """Accept a positive square size or a (rows, columns) pair."""
    if isinstance(size, (int, np.integer)):
        size = (size, size)
    if len(size) != 2 or any(int(n) != n or n < 2 for n in size):
        raise ValueError("size must contain two integer dimensions >= 2")
    return tuple(int(n) for n in size)


def load_image(source="camera", size=None):
    """Load skimage's camera/astronaut or a local image, optionally resized.

    Integer pixels are converted using their dtype's full range, not the
    particular image's minimum and maximum. RGBA is composited on white.
    Resizing is an auxiliary operation with anti-aliasing; the deliberate
    decimation experiment does NOT use this loader to downsample its input.
    """
    from skimage import color, data, io, transform, util

    source = str(source)
    if source in {"camera", "astronaut"}:
        image = getattr(data, source)()
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        image = io.imread(path)
    image = util.img_as_float(image)
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)
        if image.shape[-1] == 1:
            image = image[..., 0]
        else:
            image = color.rgb2gray(image)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("Expected a 2D image or an RGB/RGBA photograph")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Loaded pixels must be finite and in [0, 1]")
    if size is not None:
        image = transform.resize(image, _shape(size), anti_aliasing=True,
                                 preserve_range=True)
    return np.asarray(image, dtype=float)


def synthetic_image(size=128):
    """An offline test image with smooth regions, edges and asymmetric objects."""
    rows, cols = _shape(size)
    y, x = np.meshgrid(np.linspace(-1, 1, rows), np.linspace(-1, 1, cols),
                       indexing="ij")
    image = 0.2 + 0.1 * x + 0.1 * y
    image += 0.35 * np.exp(-((x + 0.35) ** 2 + (y + 0.2) ** 2) / 0.09)
    image += 0.25 * ((x > 0.1) & (x < 0.65) & (y > -0.5) & (y < 0.4))
    image += 0.12 * ((x + 0.4) ** 2 + (y - 0.5) ** 2 < 0.15 ** 2)
    return image


def periodic_pattern(shape, amplitude, u, v, phase=0.0):
    """A*cos(2*pi*(u*x + v*y) + phase), with u,v in cycles/pixel.

    u is horizontal and v vertical. For exact DFT-bin peaks, u*columns and
    v*rows must be integers. Frequencies must lie in [-0.5, 0.5).
    """
    rows, cols = _shape(shape)
    if not np.isfinite([amplitude, u, v, phase]).all() or amplitude < 0:
        raise ValueError("Parameters must be finite and amplitude nonnegative")
    if not (-0.5 <= u < 0.5 and -0.5 <= v < 0.5):
        raise ValueError("u and v must be in [-0.5, 0.5) cycles/pixel")
    y, x = np.indices((rows, cols))
    return amplitude * np.cos(2 * np.pi * (u * x + v * y) + phase)


def zone_plate(size=256):
    """M x M zone plate: (1 + cos(pi*(x**2 + y**2)/M))/2.

    Pixel coordinates are centered; local directional frequencies are x/M
    and y/M cycles/pixel and therefore grow toward the image boundaries.
    """
    rows, cols = _shape(size)
    if rows != cols:
        raise ValueError("The zone plate requires a square image")
    coords = np.arange(rows) - (rows - 1) / 2
    y, x = np.meshgrid(coords, coords, indexing="ij")
    return 0.5 * (1 + np.cos(np.pi * (x * x + y * y) / rows))

