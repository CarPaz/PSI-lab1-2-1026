import numpy as np
import pytest

from psi_lab.lab1.filtros import (
    filtrar,
    fir_pasaaltos,
    fir_pasabajos,
    fir_pasabanda,
    fir_rechazabanda,
    respuesta_frecuencia,
)

FS = 1000


def _db(H):
    return 20 * np.log10(np.abs(H) + 1e-12)


def test_pasabajos_paso_y_rechazo():
    h = fir_pasabajos(100, FS, 101, "hamming")
    f, H = respuesta_frecuencia(h, FS)
    assert np.abs(H[f <= 70]) == pytest.approx(1, abs=0.01)
    assert _db(H[f >= 140]).max() < -40


def test_coeficientes_simetricos_fase_lineal():
    for h in [
        fir_pasabajos(100, FS, 51),
        fir_pasaaltos(100, FS, 51),
        fir_pasabanda(50, 150, FS, 51),
        fir_rechazabanda(50, 150, FS, 51),
    ]:
        np.testing.assert_allclose(h, h[::-1], atol=1e-12)


def test_pasaaltos_quita_dc_y_deja_alta():
    f, H = respuesta_frecuencia(fir_pasaaltos(100, FS, 101), FS)
    assert abs(H[0]) < 0.01
    assert np.abs(H[f >= 140]) == pytest.approx(1, abs=0.01)


def test_pasabanda_y_rechazabanda():
    f, Hb = respuesta_frecuencia(fir_pasabanda(100, 200, FS, 201), FS)
    _, Hr = respuesta_frecuencia(fir_rechazabanda(100, 200, FS, 201), FS)
    centro = (f > 130) & (f < 170)
    assert np.abs(Hb[centro]) == pytest.approx(1, abs=0.01)
    assert np.abs(Hr[centro]).max() < 0.01


def test_filtrar_no_desplaza_un_pulso():
    x = np.zeros(300)
    x[150] = 1.0
    y = filtrar(x, fir_pasabajos(100, FS, 51))
    assert y.size == x.size
    assert np.argmax(y) == 150


def test_n_coef_par_se_rechaza():
    with pytest.raises(ValueError):
        fir_pasabajos(100, FS, 50)
