"""Case C: Fourier transforms, periodic interference and spatial aliasing.

Calculation spectra use unshifted NumPy order. Only displays and notch masks
use fftshift order. Peaks are (row, column) offsets from shifted DC in DFT
bins, NOT (u, v) cycles/pixel. No image is clipped during processing.
"""

import numpy as np

from psi_lab.lab1.datos import periodic_pattern
from psi_lab.lab1.filtros import fir_pasabajos


def _array2d(value):
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) < 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite 2D array with dimensions >= 2")
    return array


def _image(value):
    array = _array2d(value)
    if np.iscomplexobj(array):
        raise ValueError("Expected a real-valued image")
    return array.astype(float, copy=False)


def dft2d(x):
    """2D DFT by 1D transforms across rows, then columns; unshifted output."""
    x = _array2d(x)
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0)


def idft2d(X):
    """Inverse of an unshifted spectrum, returning complex pixel values."""
    X = _array2d(X)
    return np.fft.ifft(np.fft.ifft(X, axis=0), axis=1)


def magnitud_log(X):
    """Centered log(1 + |X|), ONLY for plotting, never reconstruction."""
    return np.log1p(np.abs(np.fft.fftshift(_array2d(X))))


def fase(X, *, centrada=False):
    """Phase in radians, unshifted by default to match reconstruir.

    Use centrada=True ONLY for a plot alongside magnitud_log.
    """
    X = _array2d(X)
    return np.angle(np.fft.fftshift(X) if centrada else X)


def reconstruir(magnitud, fase, *, centrada=False):
    """Reconstruct using LINEAR magnitude and phase with matching order.

    Both arguments are unshifted by default. If both are centered, pass
    centrada=True to undo the shift before inversion (also for odd sizes).
    """
    magnitude = _image(magnitud)
    phase = _image(fase)
    if magnitude.shape != phase.shape or np.any(magnitude < 0):
        raise ValueError("Magnitude must be nonnegative and match phase shape")
    spectrum = magnitude * np.exp(1j * phase)
    if centrada:
        spectrum = np.fft.ifftshift(spectrum)
    result = idft2d(spectrum)
    if np.max(np.abs(result.imag)) > 1e-8 * max(1.0, np.max(np.abs(result))):
        raise ValueError("Magnitude and phase do not describe a real image")
    return result.real


def swap_phases(first, second):
    """Return magnitude(first)+phase(second) and its converse, without clipping."""
    first, second = _image(first), _image(second)
    if first.shape != second.shape:
        raise ValueError("Phase exchange requires images with the same shape")
    a, b = dft2d(first), dft2d(second)
    return reconstruir(np.abs(a), fase(b)), reconstruir(np.abs(b), fase(a))


def add_periodic_noise(image, interferences):
    """Add (amplitude, u, v) components and return (corrupted, total_noise).

    Frequencies are cycles/pixel. Values outside [0, 1] are retained, avoiding
    harmonics and distorted error measurements caused by saturation.
    """
    image = _image(image)
    noise = np.zeros_like(image)
    for amplitude, u, v in interferences:
        noise += periodic_pattern(image.shape, amplitude, u, v)
    return image + noise, noise


def _distance_squared(shape, point):
    """Wrapped distance to a signed-bin point on the centered DFT grid."""
    rows, cols = shape
    y = np.arange(rows)[:, None] - rows // 2
    x = np.arange(cols)[None, :] - cols // 2
    dy = (y - point[0] + rows / 2) % rows - rows / 2
    dx = (x - point[1] + cols / 2) % cols - cols / 2
    return dy * dy + dx * dx


def _opposite(point, shape):
    """Conjugate partner, including odd dimensions and Nyquist bins."""
    return tuple(int((-p + n // 2) % n - n // 2) for p, n in zip(point, shape))


def detect_periodic_peaks(spectrum, *, dc_radius=8, threshold_ratio=12.0,
                          min_distance=5, max_pairs=4):
    """Find isolated conjugate peaks automatically, returning signed bin pairs.

    A radial median removes the natural low-to-high-frequency image envelope.
    A candidate must be a 3x3 local maximum, exceed that envelope by
    threshold_ratio, and exceed 1% of the largest non-DC coefficient. Greedy
    nonmaximum suppression keeps separate pairs at least min_distance apart.
    The low-frequency disk is protected. No injected frequencies are supplied.
    Results are candidates: natural textures can also produce spectral peaks.
    """
    spectrum = _array2d(spectrum)
    if (not np.isfinite([dc_radius, threshold_ratio, min_distance]).all()
            or dc_radius < 0 or threshold_ratio <= 1 or min_distance < 1
            or not isinstance(max_pairs, (int, np.integer)) or max_pairs < 1):
        raise ValueError("Invalid peak detector parameters")
    amplitude = np.abs(np.fft.fftshift(spectrum))
    if amplitude.max() == 0:
        return np.empty((0, 2), dtype=int)
    rows, cols = amplitude.shape
    y, x = np.indices(amplitude.shape)
    radius = np.hypot(y - rows // 2, x - cols // 2)
    rings = np.rint(radius).astype(int)
    log_amplitude = np.log(np.maximum(amplitude, amplitude.max() * 1e-14))
    baseline = np.array([np.median(log_amplitude[rings == r])
                         for r in range(rings.max() + 1)])
    score = log_amplitude - baseline[rings]
    valid = radius > dc_radius
    if not valid.any():
        return np.empty((0, 2), dtype=int)
    valid &= amplitude >= max(amplitude[valid].max() * 0.01,
                              amplitude.max() * 1e-10)
    valid &= score > np.log(threshold_ratio)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                valid &= amplitude >= np.roll(amplitude, (dy, dx), (0, 1))
    candidates = np.argwhere(valid)
    order = sorted(candidates.tolist(), key=lambda p: (-score[tuple(p)], p))
    selected = []
    pair_count = 0
    for row, col in order:
        point = (row - rows // 2, col - cols // 2)
        partner = _opposite(point, spectrum.shape)
        if any(_distance_squared(spectrum.shape, p)[row, col] < min_distance ** 2
               for p in selected):
            continue
        selected.append(point)
        if partner != point:
            selected.append(partner)
        pair_count += 1
        if pair_count == max_pairs:
            break
    return np.asarray(selected, dtype=int).reshape(-1, 2)


def notch_mask(shape, peaks, *, radius=2.0, kind="ideal"):
    """Centered real mask with conjugate symmetry and preserved DC.

    Ideal: zero inside radius bins. Gaussian: product of
    1-exp(-distance**2/(2*radius**2)) for each peak and its partner.
    radius is in DFT bins; fractional coordinates are supported for off-bin
    peaks supplied manually. Frequencies wrap around the DFT boundaries.
    """
    if len(shape) != 2 or any(int(n) != n or n < 2 for n in shape):
        raise ValueError("shape must contain two integer dimensions >= 2")
    shape = tuple(int(n) for n in shape)
    if kind not in {"ideal", "gaussian"} or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Choose ideal/gaussian and a positive radius")
    points = np.asarray(peaks, dtype=float)
    if points.size == 0:
        return np.ones(shape)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("peaks must have shape (n, 2) and finite coordinates")
    centers = set()
    for point in points:
        point = tuple((p + n // 2) % n - n // 2 for p, n in zip(point, shape))
        opposite = tuple((-p + n // 2) % n - n // 2
                         for p, n in zip(point, shape))
        centers.update((point, opposite))
    mask = np.ones(shape)
    dc = (shape[0] // 2, shape[1] // 2)
    for point in sorted(centers):
        d2 = _distance_squared(shape, point)
        if d2[dc] <= radius ** 2:
            raise ValueError("Notches must not cover DC; reduce radius")
        if kind == "ideal":
            mask[d2 <= radius ** 2] = 0.0
        else:
            mask *= -np.expm1(-d2 / (2 * radius ** 2))
    mask[dc] = 1.0
    return mask


def apply_notch(image, peaks, *, radius=2.0, kind="ideal"):
    """Filter a real image with the centered notch mask, undoing the shift."""
    spectrum = dft2d(_image(image))
    mask = notch_mask(spectrum.shape, peaks, radius=radius, kind=kind)
    return idft2d(np.fft.ifftshift(np.fft.fftshift(spectrum) * mask)).real


def separable_filter(image, kernel):
    """Direct convolution with a 1D kernel across rows then columns.

    Reflect padding avoids artificial dark borders; a symmetric odd kernel
    gives centered output with no pixel displacement. No scipy filter is used.
    """
    image = _image(image)
    kernel = np.asarray(kernel, dtype=float)
    if (kernel.ndim != 1 or kernel.size % 2 != 1 or not np.isfinite(kernel).all()
            or not np.allclose(kernel, kernel[::-1], rtol=0, atol=1e-12)):
        raise ValueError("Expected a finite, symmetric, odd-length kernel")
    half = kernel.size // 2
    result = image
    for axis in (1, 0):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (half, half)
        padded = np.pad(result, padding, mode="reflect")
        result = np.zeros_like(result)
        for j, coefficient in enumerate(kernel[::-1]):
            slices = [slice(None), slice(None)]
            slices[axis] = slice(j, j + image.shape[axis])
            result += coefficient * padded[tuple(slices)]
    return result


def gaussian_lowpass(image, sigma=1.0):
    """Spatial Gaussian convolution with a separable kernel truncated at 3 sigma."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    half = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-half, half + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return separable_filter(image, kernel / kernel.sum())


def psnr(reference, estimate, data_range=1.0):
    """Own PSNR: 10*log10(data_range**2 / mean(square(error))), in dB.

    Equal images give +inf. No clipping or per-image normalization is applied.
    Use fixed data_range=1 even if noisy pixels fall outside [0, 1].
    """
    reference, estimate = _image(reference), _image(estimate)
    if reference.shape != estimate.shape:
        raise ValueError("Metric inputs must have equal shapes")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be positive and finite")
    mse = np.mean((reference - estimate) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))


def image_metrics(reference, estimate):
    """Own PSNR plus skimage SSIM, on unmodified pixels with fixed range 1."""
    from skimage.metrics import structural_similarity

    reference, estimate = _image(reference), _image(estimate)
    score = psnr(reference, estimate)
    if min(reference.shape) < 3:
        raise ValueError("SSIM requires dimensions >= 3")
    window = min(7, min(reference.shape))
    window -= 1 - window % 2
    return {"psnr_db": score, "ssim": float(structural_similarity(
        reference, estimate, data_range=1.0, win_size=window))}


def anti_alias_filter(image, factor, n_coef=65, cutoff_ratio=0.8):
    """FIR from the common part, fs=1 cycle/pixel, with a transition guard band.

    After decimation by k the old-grid Nyquist is 1/(2*k). The cutoff is
    cutoff_ratio/(2*k). A finite FIR attenuates rather than perfectly removes
    the out-of-band components. The same kernel is applied along both axes.
    """
    if (not isinstance(factor, (int, np.integer)) or factor < 2
            or not isinstance(n_coef, (int, np.integer)) or n_coef < 3
            or n_coef % 2 != 1 or not 0 < cutoff_ratio < 1):
        raise ValueError("Use factor >= 2, odd n_coef >= 3, 0 < cutoff_ratio < 1")
    kernel = fir_pasabajos(cutoff_ratio / (2 * factor), 1.0, n_coef, "hamming")
    return separable_filter(image, kernel)


def downsample(image, factor, *, anti_alias=True, n_coef=65, cutoff_ratio=0.8):
    """Keep one pixel in k on both axes, optionally after our separable FIR."""
    image = _image(image)
    if not isinstance(factor, (int, np.integer)) or factor < 2:
        raise ValueError("factor must be an integer >= 2")
    if min(image.shape) < 2 * factor:
        raise ValueError("Image is too small for this reduction factor")
    if anti_alias:
        image = anti_alias_filter(image, factor, n_coef, cutoff_ratio)
    return image[::factor, ::factor]
