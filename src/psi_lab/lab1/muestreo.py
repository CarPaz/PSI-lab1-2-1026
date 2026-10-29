"""Parte comun 1: muestreo y aliasing."""

import numpy as np


def muestrear(fun, fs, duracion):
    """Evalua fun(t) en t = n / fs, n = 0..N-1, con N = round(fs * duracion).

    fun: funcion del tiempo [s] que acepta arreglos. fs [Hz], duracion [s].
    Retorna (t, x): instantes de muestreo [s] y valores muestreados.
    """
    n = round(fs * duracion)
    t = np.arange(n) / fs
    return t, np.asarray(fun(t), dtype=float)


def frecuencia_alias(f, fs):
    """Frecuencia aparente en [0, fs/2] de una senoidal de f Hz muestreada a fs Hz.

    Al muestrear, f y f + k*fs dan las mismas muestras (el espectro se repite cada fs),
    y para senales reales f y -f se ven igual. Entonces se lleva f al intervalo
    [0, fs) con el modulo y, si queda sobre fs/2, se refleja: fs - f.
    """
    r = np.mod(f, fs)
    alias = np.where(r > fs / 2, fs - r, r)
    return float(alias) if np.ndim(alias) == 0 else alias
