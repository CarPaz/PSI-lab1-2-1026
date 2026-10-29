import numpy as np

from psi_lab.lab1.convolucion import convolucion_directa, convolucion_fft

rng = np.random.default_rng(2)


def test_ambas_coinciden_con_numpy():
    x, h = rng.normal(size=40), rng.normal(size=7)
    np.testing.assert_allclose(convolucion_directa(x, h), np.convolve(x, h), atol=1e-9)
    np.testing.assert_allclose(convolucion_fft(x, h), np.convolve(x, h), atol=1e-9)


def test_impulso_desplazado_desplaza_la_senal():
    x = rng.normal(size=20)
    h = np.zeros(5)
    h[3] = 1.0
    y = convolucion_directa(x, h)
    np.testing.assert_allclose(y[3 : 3 + x.size], x)


def test_conmutativa():
    x, h = rng.normal(size=15), rng.normal(size=9)
    np.testing.assert_allclose(convolucion_directa(x, h), convolucion_directa(h, x), atol=1e-12)


def test_sin_relleno_es_circular():
    x, h = rng.normal(size=16), rng.normal(size=4)
    circular = convolucion_fft(x, h, rellenar=False)
    lineal = np.convolve(x, h)
    # La cola de la lineal se "enrolla" sobre el comienzo.
    esperado = lineal[:16].copy()
    esperado[: lineal.size - 16] += lineal[16:]
    np.testing.assert_allclose(circular, esperado, atol=1e-9)
