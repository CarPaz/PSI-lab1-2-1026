"""Parte comun 4: filtros FIR por el metodo de la ventana."""

import numpy as np

from psi_lab.lab1.convolucion import convolucion_directa
from psi_lab.lab1.dft import crear_ventana


def _sinc_enventanada(fc, fs, n_coef, ventana):
    """Respuesta al impulso ideal del pasabajos (sinc) truncada a n_coef y enventanada."""
    if n_coef % 2 == 0:
        raise ValueError("n_coef debe ser impar (filtro tipo I, retardo entero)")
    m = np.arange(n_coef) - (n_coef - 1) / 2
    fcn = fc / fs  # frecuencia de corte normalizada (ciclos por muestra)
    h = 2 * fcn * np.sinc(2 * fcn * m)
    return h * crear_ventana(ventana, n_coef)


def fir_pasabajos(fc, fs, n_coef, ventana="hamming"):
    """Sinc truncada y enventanada, normalizada a ganancia 1 en DC. fc y fs en Hz."""
    h = _sinc_enventanada(fc, fs, n_coef, ventana)
    return h / h.sum()


def _impulso(n_coef):
    d = np.zeros(n_coef)
    d[(n_coef - 1) // 2] = 1.0
    return d


def fir_pasaaltos(fc, fs, n_coef, ventana="hamming"):
    """Inversion espectral: impulso - pasabajos (deja pasar lo que el pasabajos quita)."""
    return _impulso(n_coef) - fir_pasabajos(fc, fs, n_coef, ventana)


def fir_pasabanda(f1, f2, fs, n_coef, ventana="hamming"):
    """Resta de pasabajos: pasabajos(f2) - pasabajos(f1), con f1 < f2."""
    return fir_pasabajos(f2, fs, n_coef, ventana) - fir_pasabajos(f1, fs, n_coef, ventana)


def fir_rechazabanda(f1, f2, fs, n_coef, ventana="hamming"):
    """Inversion espectral del pasabanda: impulso - pasabanda."""
    return _impulso(n_coef) - fir_pasabanda(f1, f2, fs, n_coef, ventana)


def respuesta_frecuencia(h, fs, n=4096):
    """Respuesta en frecuencia H(f) = sum_m h[m] exp(-2j pi f m / fs) en [0, fs/2].

    Retorna (freqs [Hz], H complejo) con n puntos.
    """
    h = np.asarray(h, dtype=float)
    freqs = np.linspace(0, fs / 2, n)
    m = np.arange(h.size)
    H = np.exp(-2j * np.pi * np.outer(freqs, m) / fs) @ h
    return freqs, H


def filtrar(x, h):
    """Aplica h a x (resultado del mismo largo que x), compensando el retardo (n_coef - 1)/2.

    La convolucion completa tiene largo len(x) + len(h) - 1 y llega atrasada
    (n_coef - 1)/2 muestras; se recorta desde ahi para que no haya desfase.
    """
    x = np.asarray(x, dtype=float)
    y = convolucion_directa(x, h)
    retardo = (len(h) - 1) // 2
    return y[retardo : retardo + x.size]
