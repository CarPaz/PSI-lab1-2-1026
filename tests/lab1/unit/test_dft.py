import numpy as np
import pytest

from psi_lab.lab1.dft import dft, espectro, idft

rng = np.random.default_rng(1)


@pytest.mark.parametrize("n", [1, 8, 31, 64])
def test_dft_coincide_con_numpy(n):
    x = rng.normal(size=n)
    np.testing.assert_allclose(dft(x), np.fft.fft(x), atol=1e-9)


def test_idft_recupera_la_senal():
    x = rng.normal(size=50) + 1j * rng.normal(size=50)
    np.testing.assert_allclose(idft(dft(x)), x, atol=1e-9)


def test_simetria_hermitica_para_senal_real():
    x = rng.normal(size=40)
    X = dft(x)
    k = np.arange(1, 40)
    np.testing.assert_allclose(X[40 - k], np.conj(X[k]), atol=1e-9)


def test_parseval():
    x = rng.normal(size=64)
    X = dft(x)
    assert np.sum(np.abs(x) ** 2) == pytest.approx(np.sum(np.abs(X) ** 2) / x.size)


@pytest.mark.parametrize("ventana", ["rect", "hann", "hamming"])
def test_espectro_recupera_amplitud_y_frecuencia(ventana):
    fs, f, a = 1000, 50, 2.0
    t = np.arange(1000) / fs
    freqs, amp = espectro(a * np.sin(2 * np.pi * f * t), fs, ventana=ventana)
    assert freqs[np.argmax(amp)] == pytest.approx(f)
    assert amp.max() == pytest.approx(a, rel=1e-6)


def test_hann_reduce_la_fuga():
    # Tono fuerte que no cae justo en un bin: lejos de el, Hann fuga mucho menos.
    fs = 1000
    t = np.arange(500) / fs
    x = np.sin(2 * np.pi * 50.5 * t)
    freqs, a_rect = espectro(x, fs, "rect")
    _, a_hann = espectro(x, fs, "hann")
    lejos = freqs > 150
    assert a_hann[lejos].max() < a_rect[lejos].max() / 10
