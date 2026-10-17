import numpy as np
import pytest
from psi_lab.senales import espectro, senoidal
def test_numero_de_muestras():
    t, x = senoidal(f=5, fs=100, duracion=2)
    assert t.size == x.size == 200

@pytest.mark.parametrize("f", [10.0, 50.0, 120.0])
def test_pico_en_la_frecuencia_correcta(f):
    _, x = senoidal(f=f, fs=1000, duracion=1)
    freqs, amp = espectro(x, fs=1000)
    assert freqs[np.argmax(amp)] == pytest.approx(f)

def test_amplitud_recuperada():
    _, x = senoidal(f=50, fs=1000, duracion=1, amplitud=3.0)
    _, amp = espectro(x, fs=1000)
    assert amp.max() == pytest.approx(3.0, rel=1e-6)

from psi_lab.senales import cuadrada, espectro

def test_armonicos_onda_cuadrada():
    f = 10.0
    fs = 1000.0
    duracion = 2.0
    t, x = cuadrada(f, fs, duracion)
    freqs, amp = espectro(x, fs)

    def obtener_amplitud(f_buscada):
        idx = np.argmin(np.abs(freqs - f_buscada))
        return amp[idx]

# Armónicos impares f, 3f, 5f tienen picos significativos
    assert obtener_amplitud(f) > 0.5
    assert obtener_amplitud(3 * f) > 0.1
    assert obtener_amplitud(5 * f) > 0.05

    # Armónicos pares 2f, 4f son cercanos a 0
    assert obtener_amplitud(2 * f) == pytest.approx(0.0, abs=0.05)
    assert obtener_amplitud(4 * f) == pytest.approx(0.0, abs=0.05)   