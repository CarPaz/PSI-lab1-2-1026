import numpy as np
import pytest

from psi_lab.lab1.caso_imagen import (
    add_periodic_noise,
    apply_notch,
    detect_periodic_peaks,
    dft2d,
    downsample,
    fase,
    gaussian_lowpass,
    idft2d,
    image_metrics,
    notch_mask,
    psnr,
    reconstruir,
    separable_filter,
    swap_phases,
)
from psi_lab.lab1.datos import periodic_pattern, synthetic_image


def test_dft2d_coincide_con_fft2():
    rng = np.random.default_rng(0)
    for shape in [(8, 8), (5, 7), (16, 16)]:
        x = rng.normal(size=shape)
        np.testing.assert_allclose(dft2d(x), np.fft.fft2(x), atol=1e-9)


def test_idft2d_recupera_la_imagen():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 12))
    np.testing.assert_allclose(np.real(idft2d(dft2d(x))), x, atol=1e-9)


@pytest.mark.parametrize("shape", [(9, 13), (10, 12)])
def test_phase_and_magnitude_reconstruct_in_both_orders(shape):
    image = np.random.default_rng(12).normal(size=shape)
    spectrum = dft2d(image)
    np.testing.assert_allclose(reconstruir(np.abs(spectrum), fase(spectrum)), image,
                               atol=1e-12)
    np.testing.assert_allclose(
        reconstruir(np.abs(np.fft.fftshift(spectrum)), fase(spectrum, centrada=True),
                    centrada=True), image, atol=1e-12)


def test_phase_exchange_keeps_the_donors_magnitude_and_phase():
    a = synthetic_image((15, 19))
    b = np.random.default_rng(3).uniform(size=a.shape)
    ab, ba = swap_phases(a, b)
    np.testing.assert_allclose(dft2d(ab), np.abs(dft2d(a)) * np.exp(1j * fase(dft2d(b))),
                               atol=1e-11)
    np.testing.assert_allclose(dft2d(ba), np.abs(dft2d(b)) * np.exp(1j * fase(dft2d(a))),
                               atol=1e-11)


def test_cosine_has_exactly_two_conjugate_peaks():
    shape = (96, 80)
    image = periodic_pattern(shape, 0.3, 11 / 80, 7 / 96)
    spectrum = np.abs(np.fft.fftshift(dft2d(image)))
    actual = set(map(tuple, np.argwhere(spectrum > 1e-8)))
    assert actual == {(48 + 7, 40 + 11), (48 - 7, 40 - 11)}
    np.testing.assert_allclose(spectrum[spectrum > 1e-8], 0.3 * 96 * 80 / 2)


def test_detector_finds_both_peaks_without_known_frequency():
    image = 0.5 + periodic_pattern((96, 80), 0.2, 11 / 80, 7 / 96)
    actual = set(map(tuple, detect_periodic_peaks(dft2d(image))))
    assert actual == {(7, 11), (-7, -11)}


@pytest.mark.parametrize("value", [0.0, 0.4])
def test_constant_image_has_no_interference_peaks(value):
    assert detect_periodic_peaks(dft2d(np.full((64, 64), value))).shape == (0, 2)


@pytest.mark.parametrize("shape", [(63, 79), (64, 80)])
@pytest.mark.parametrize("kind", ["ideal", "gaussian"])
def test_notch_keeps_conjugate_symmetry_and_dc(shape, kind):
    mask = notch_mask(shape, [(13, -17)], kind=kind)
    unshifted = np.fft.ifftshift(mask)
    partner = unshifted[np.ix_((-np.arange(shape[0])) % shape[0],
                               (-np.arange(shape[1])) % shape[1])]
    np.testing.assert_allclose(unshifted, partner, atol=1e-13)
    assert unshifted[0, 0] == 1
    assert np.all((mask >= 0) & (mask <= 1))


@pytest.mark.parametrize("kind", ["ideal", "gaussian"])
def test_notch_removes_exact_interference_and_preserves_mean(kind):
    clean = np.full((96, 80), 0.5)
    noisy, _ = add_periodic_noise(clean, [(0.2, 11 / 80, 7 / 96)])
    peaks = detect_periodic_peaks(dft2d(noisy))
    restored = apply_notch(noisy, peaks, kind=kind)
    np.testing.assert_allclose(restored, clean, atol=1e-13)
    assert restored.mean() == pytest.approx(noisy.mean(), abs=1e-13)


def test_no_peaks_means_identity_filter():
    image = synthetic_image(31)
    np.testing.assert_allclose(apply_notch(image, []), image, atol=1e-13)


def test_noise_is_not_clipped_or_normalized():
    clean = np.ones((32, 32))
    noisy, noise = add_periodic_noise(clean, [(0.4, 0.25, 0)])
    assert noisy.max() > 1
    np.testing.assert_allclose(noisy, clean + noise)


def test_spatial_separable_convolution_has_correct_impulse_and_no_shift():
    image = np.zeros((21, 23))
    image[10, 11] = 1
    kernel = np.array([0.25, 0.5, 0.25])
    output = separable_filter(image, kernel)
    np.testing.assert_allclose(output[9:12, 10:13], np.outer(kernel, kernel))
    assert np.unravel_index(np.argmax(output), output.shape) == (10, 11)
    assert output.sum() == pytest.approx(1)


def test_gaussian_reflect_boundary_preserves_constant_image():
    image = np.full((15, 17), 0.7)
    np.testing.assert_allclose(gaussian_lowpass(image, 1.4), image, atol=1e-14)


def test_own_psnr_known_error_and_identical_images():
    image = np.zeros((12, 13))
    assert psnr(image, image + 0.1) == pytest.approx(20)
    assert psnr(image, image) == float("inf")
    # Out-of-range estimates retain their true error.
    assert psnr(image, image + 2) == pytest.approx(-20 * np.log10(2))


def test_ssim_identity():
    image = synthetic_image(32)
    assert image_metrics(image, image)["ssim"] == pytest.approx(1)


@pytest.mark.parametrize("factor", [2, 4])
def test_anti_alias_fir_rejects_a_tone_that_would_fold(factor):
    image = 0.5 + periodic_pattern((256, 256), 0.3, 0.375, 0)
    raw = downsample(image, factor, anti_alias=False)
    filtered = downsample(image, factor)
    assert np.std(filtered[10:-10, 10:-10]) < np.std(raw[10:-10, 10:-10]) / 100
    assert filtered.shape == raw.shape
    np.testing.assert_allclose(raw, image[::factor, ::factor])


@pytest.mark.parametrize("factor", [2, 4])
def test_anti_alias_filter_preserves_dc(factor):
    result = downsample(np.full((64, 64), 0.4), factor)
    np.testing.assert_allclose(result, 0.4, atol=1e-14)


@pytest.mark.parametrize("operation", [
    lambda: dft2d([1, 2, 3]),
    lambda: dft2d([[1, np.nan], [2, 3]]),
    lambda: swap_phases(np.ones((4, 4)), np.ones((5, 5))),
    lambda: reconstruir(-np.ones((4, 4)), np.zeros((4, 4))),
    lambda: notch_mask((32, 32), [(0, 0)]),
    lambda: notch_mask((32, 32), [(8, 8)], radius=0),
    lambda: notch_mask((32, 32), [(8, 8)], kind="unknown"),
    lambda: psnr(np.ones((4, 4)), np.ones((5, 5))),
    lambda: gaussian_lowpass(np.ones((4, 4)), sigma=0),
    lambda: downsample(np.ones((16, 16)), 1),
    lambda: downsample(np.ones((16, 16)), 2, n_coef=64),
])
def test_invalid_parameters_raise_clear_errors(operation):
    with pytest.raises(ValueError):
        operation()
