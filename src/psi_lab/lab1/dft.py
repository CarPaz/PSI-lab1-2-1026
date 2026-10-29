"""Parte comun 2: DFT por definicion y espectro con ventanas."""

import numpy as np


def dft(x):
    """DFT por definicion, O(N^2): X[k] = sum_n x[n] exp(-2j pi k n / N).

    Se arma la matriz W[k, n] = exp(-2j pi k n / N) y se multiplica por x.
    """
    x = np.asarray(x, dtype=complex)
    n_muestras = x.size
    n = np.arange(n_muestras)
    w = np.exp(-2j * np.pi * np.outer(n, n) / n_muestras)
    return w @ x


def idft(X):
    """DFT inversa por definicion: x[n] = (1/N) sum_k X[k] exp(+2j pi k n / N)."""
    X = np.asarray(X, dtype=complex)
    n_muestras = X.size
    k = np.arange(n_muestras)
    w = np.exp(2j * np.pi * np.outer(k, k) / n_muestras)
    return (w @ X) / n_muestras


def crear_ventana(nombre, n):
    """Ventana de largo n: "rect", "hann" o "hamming" (formulas simetricas)."""
    if nombre == "rect":
        return np.ones(n)
    if n == 1:
        return np.ones(1)
    m = np.arange(n)
    if nombre == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * m / (n - 1))
    if nombre == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * m / (n - 1))
    raise ValueError(f"ventana desconocida: {nombre}")


def espectro(x, fs, ventana="rect", nfft=None):
    """Espectro de amplitud de un lado de la senal real x muestreada a fs [Hz].

    ventana en {"rect", "hann", "hamming"}. nfft > len(x) rellena con ceros
    (mas puntos en el grafico, pero no mas resolucion: esa la da 1/T).
    La amplitud se divide por la suma de la ventana, asi un tono de amplitud A
    se ve con altura A en cualquier ventana. Retorna (freqs [Hz], amp).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    w = crear_ventana(ventana, n)
    nfft = n if nfft is None else int(nfft)
    xw = np.zeros(nfft)
    xw[:n] = x * w
    X = dft(xw)[: nfft // 2 + 1]
    amp = np.abs(X) / w.sum()
    # Lado unico: se duplica todo salvo DC (y Nyquist si nfft es par).
    amp[1 : (nfft + 1) // 2] *= 2
    freqs = np.arange(nfft // 2 + 1) * fs / nfft
    return freqs, amp
