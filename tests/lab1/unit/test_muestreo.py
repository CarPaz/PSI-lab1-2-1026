import numpy as np
import pytest

from psi_lab.lab1.dft import espectro
from psi_lab.lab1.muestreo import frecuencia_alias, muestrear


def test_muestrear_instantes_y_largo():
    t, x = muestrear(lambda t: np.ones_like(t), fs=10, duracion=2)
    assert t.size == x.size == 20
    assert t[1] - t[0] == pytest.approx(0.1)


@pytest.mark.parametrize("f, esperado", [(3, 3), (7, 3), (13, 3), (23, 3), (5, 5)])
def test_frecuencia_alias_casos_del_enunciado(f, esperado):
    assert frecuencia_alias(f, 10) == pytest.approx(esperado)


def test_dos_frecuencias_mismas_muestras():
    # 3 Hz y 13 Hz muestreadas a 10 Hz dan exactamente las mismas muestras.
    _, x1 = muestrear(lambda t: np.cos(2 * np.pi * 3 * t), 10, 2)
    _, x2 = muestrear(lambda t: np.cos(2 * np.pi * 13 * t), 10, 2)
    np.testing.assert_allclose(x1, x2, atol=1e-9)


def test_pico_del_espectro_cae_en_frecuencia_alias():
    rng = np.random.default_rng(0)
    for _ in range(10):
        fs = float(rng.integers(50, 200))
        f = float(rng.integers(1, 500))
        if np.isclose(np.mod(f, fs / 2), 0):
            continue  # DC o Nyquist: el seno se anula en las muestras
        _, x = muestrear(lambda t, f=f: np.cos(2 * np.pi * f * t), fs, 1.0)
        freqs, amp = espectro(x, fs)
        assert freqs[np.argmax(amp)] == pytest.approx(frecuencia_alias(f, fs), abs=1.0)
