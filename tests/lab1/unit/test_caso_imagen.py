import numpy as np

from psi_lab.lab1.caso_imagen import dft2d, idft2d


def test_dft2d_coincide_con_fft2():
    rng = np.random.default_rng(0)
    for shape in [(8, 8), (5, 7), (16, 16)]:
        x = rng.normal(size=shape)
        np.testing.assert_allclose(dft2d(x), np.fft.fft2(x), atol=1e-9)


def test_idft2d_recupera_la_imagen():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 12))
    np.testing.assert_allclose(np.real(idft2d(dft2d(x))), x, atol=1e-9)