"""Caso aplicado C: Fourier 2D, ruido periodico, filtros notch y aliasing."""

import numpy as np


def dft2d(x):
    """DFT 2D como DFT 1D por filas y luego por columnas.

    No se usa np.fft.fft2 en la implementacion (solo como referencia en las
    pruebas): la 2D se arma a partir de la 1D.
    """
    x = np.asarray(x)
    filas = np.fft.fft(x, axis=1)      # 1D sobre cada fila (axis=1 son las filas)
    return np.fft.fft(filas, axis=0)   # 1D sobre cada columna (axis=0 son las columnas)


def idft2d(X):
    """DFT 2D inversa: columnas y luego filas (orden inverso)."""
    X = np.asarray(X)
    columnas = np.fft.ifft(X, axis=0)
    return np.fft.ifft(columnas, axis=1)


def magnitud_log(X):
    """Magnitud logaritmica centrada (fftshift), para visualizar."""
    return np.log1p(np.abs(np.fft.fftshift(X)))


def fase(X):
    """Fase centrada (fftshift) del espectro, en radianes."""
    return np.angle(np.fft.fftshift(X))


def reconstruir(magnitud, fase):
    """Reconstruye con la IDFT 2D a partir de una magnitud y una fase dadas.

    Sirve para el experimento de intercambio de fases (que 'manda' la fase).
    """
    return np.real(idft2d(magnitud * np.exp(1j * fase)))