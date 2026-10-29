"""Parte comun 3: convolucion lineal directa y por FFT."""

import numpy as np

from psi_lab.lab1.dft import dft, idft


def convolucion_directa(x, h):
    """Convolucion lineal completa por definicion: y[n] = sum_k x[k] h[n - k].

    Largo del resultado: len(x) + len(h) - 1. Costo O(len(x) * len(h)).
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.zeros(x.size + h.size - 1)
    for k, xk in enumerate(x):
        y[k : k + h.size] += xk * h
    return y


def convolucion_fft(x, h, rellenar=True):
    """Misma convolucion lineal via el teorema de la convolucion: y = IDFT(X . H).

    Hay que rellenar con ceros hasta L = len(x) + len(h) - 1; si no
    (rellenar=False) la DFT hace convolucion circular y el final de la
    senal se "enrolla" sobre el comienzo.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    largo = x.size + h.size - 1
    n = largo if rellenar else max(x.size, h.size)
    xp = np.zeros(n)
    hp = np.zeros(n)
    xp[: x.size] = x
    hp[: h.size] = h
    y = np.real(idft(dft(xp) * dft(hp)))
    return y
