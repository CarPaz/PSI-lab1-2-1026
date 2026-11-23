"""Genera las figuras y tablas de la parte comun del Laboratorio 1.

Cada funcion corresponde a una tarea del enunciado del lab . Al final en main se
ejecutan todas y se crean los .png y .csv 

Nota: np.fft se usa aqui solo como referencia para medir tiempos y como
referencia de la "FFT real", no dentro de las implementaciones de la parte
comun (que estan en src/psi_lab/lab1/).
"""

import csv
import time

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla (CI / Docker)

import matplotlib.pyplot as plt
import numpy as np

from psi_lab.lab1.convolucion import convolucion_directa, convolucion_fft
from psi_lab.lab1.dft import dft, espectro
from psi_lab.lab1.filtros import filtrar, fir_pasabajos, respuesta_frecuencia
from psi_lab.lab1.muestreo import frecuencia_alias, muestrear


def tarea1_muestreo():
    print("TAREA 1 : MUESTREO Y ALIASING")
    fs = 10
    frecuencias = [3, 7, 13, 23]

    # 1. Tabla de aliasing
    print(f"{'f [Hz]':<10} | {'Alias Teorico':<15} | {'Alias Medido'}")
    print("-" * 45)
    for f in frecuencias:
        _, x = muestrear(lambda t, f=f: np.cos(2 * np.pi * f * t), fs, duracion=2)
        freqs, amp = espectro(x, fs)
        f_medida = freqs[np.argmax(amp)]
        f_teorica = frecuencia_alias(f, fs)
        print(f"{f:<10} | {f_teorica:<15.2f} | {f_medida:.2f}")

    # 2. Grafico dos frecuencias distintas con las mismas muestras
    t_cont = np.linspace(0, 1, 1000)
    x3_cont = np.cos(2 * np.pi * 3 * t_cont)
    x13_cont = np.cos(2 * np.pi * 13 * t_cont)

    t_samp, x_samp = muestrear(lambda t: np.cos(2 * np.pi * 3 * t), fs, 1)

    plt.figure(figsize=(10, 4))
    plt.plot(t_cont, x3_cont, label="Continua 3 Hz", alpha=0.7)
    plt.plot(t_cont, x13_cont, label="Continua 13 Hz", linestyle="--", alpha=0.7)
    plt.stem(t_samp, x_samp, linefmt="k-", markerfmt="ko", basefmt="k-",
             label="Muestras a 10 Hz")
    plt.title("Aliasing: 3 Hz y 13 Hz muestreadas a 10 Hz")
    plt.xlabel("Tiempo [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tarea1_aliasing.png")
    plt.close()


def tarea2_dft():
    print("\nTAREA 2: DFT, RESOLUCION Y FUGA")

    # 1. Tiempos de ejecucion DFT vs FFT
    n_vals = [2 ** i for i in range(4, 13)]
    tiempos_dft = []
    tiempos_fft = []

    for n in n_vals:
        x = np.random.randn(n)

        t0 = time.perf_counter()
        dft(x)
        tiempos_dft.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        np.fft.fft(x)
        tiempos_fft.append(time.perf_counter() - t0)

    plt.figure(figsize=(8, 5))
    plt.loglog(n_vals, tiempos_dft, "o-", label="DFT propia O(N^2)")
    plt.loglog(n_vals, tiempos_fft, "s-", label="np.fft.fft O(N log N)")
    plt.title("Tiempo de ejecución: DFT vs FFT")
    plt.xlabel("Largo de la senal (N)")
    plt.ylabel("Tiempo [s]")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tarea2_tiempos.png")
    plt.close()

    # 2. Resolucion para distintos T
    fs = 1000
    duraciones = [0.25, 0.5, 1, 2]
    plt.figure(figsize=(12, 8))

    for i, T in enumerate(duraciones, 1):
        t = np.arange(int(fs * T)) / fs
        x = np.cos(2 * np.pi * 50 * t) + np.cos(2 * np.pi * 52 * t)
        freqs, amp = espectro(x, fs)

        plt.subplot(2, 2, i)
        plt.plot(freqs, amp)
        plt.xlim(45, 57)
        plt.title(f"T = {T} s (Δf = {1 / T:.2f} Hz)")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("tarea2_resolucion.png")
    plt.close()

    # 3. Fuga y ventanas (tono debil oculto)
    t = np.arange(int(fs * 1)) / fs
    x = np.cos(2 * np.pi * 50 * t) + 1e-3 * np.cos(2 * np.pi * 80 * t)

    freqs_rect, amp_rect = espectro(x, fs, ventana="rect")
    freqs_hann, amp_hann = espectro(x, fs, ventana="hann")

    amp_rect_db = 20 * np.log10(amp_rect + 1e-12)
    amp_hann_db = 20 * np.log10(amp_hann + 1e-12)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs_rect, amp_rect_db, label="Rectangular", alpha=0.8)
    plt.plot(freqs_hann, amp_hann_db, label="Hann", alpha=0.8)
    plt.xlim(40, 100)
    plt.ylim(-100, 5)
    plt.title("Fuga espectral: Tono de 50 Hz (amp 1) y 80 Hz (amp 1e-3)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tarea2_fuga_ventanas.png")
    plt.close()


def tarea2_zero_padding():
    print("\nTAREA 2 : RELLENO CON CEROS")
    fs = 1000
    T = 0.25
    t = np.arange(int(fs * T)) / fs          # N = 250 muestras
    x = np.cos(2 * np.pi * 50 * t) + np.cos(2 * np.pi * 52 * t)
    nfft = 8 * t.size

    freqs, amp = espectro(x, fs)             # nfft = N
    freqs_z, amp_z = espectro(x, fs, nfft=nfft)  # nfft = 8N

    # Conteo de picos locales en la zona de interes (verificacion numerica).
    def contar_picos(f, a):
        zona = (f >= 45) & (f <= 60)
        a = a[zona]
        return int(np.sum((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(freqs, amp)
    ax[0].set_title(f"Sin relleno (nfft = N = {t.size}), {contar_picos(freqs, amp)} pico(s)")
    ax[1].plot(freqs_z, amp_z)
    ax[1].set_title(f"Con relleno (nfft = 8N = {nfft}), {contar_picos(freqs_z, amp_z)} pico(s)")
    for a in ax:
        a.set_xlim(45, 60)
        a.set_xlabel("Frecuencia [Hz]")
        a.set_ylabel("Amplitud")
        a.grid(True)
    fig.suptitle("Relleno con ceros: los 2 tonos (50 y 52 Hz) siguen sin separarse")
    fig.tight_layout()
    fig.savefig("tarea2_zero_padding.png")
    plt.close()
    print("  Los dos tonos de 2 Hz de separacion (df = 1/T = 4 Hz) no se resuelven;")
    print("  el relleno solo interpola (apariencia), no mejora la resolucion.")


def _convolucion_fft_numpy(x, h):
    """Convolucion lineal via np.fft (solo referencia de tiempos)."""
    largo = x.size + h.size - 1
    xp = np.fft.fft(x, largo)
    hp = np.fft.fft(h, largo)
    return np.real(np.fft.ifft(xp * hp))


def tarea3_convolucion():
    print("\nTAREA 3: CONVOLUCION")

    # 2. Convolucion circular (sin relleno) vs lineal (correcta).
    x = np.arange(1, 9, dtype=float)      # [1, 2, ..., 8]
    h = np.array([1.0, 1.0, 1.0])          # promedio movil de 3

    lineal = convolucion_directa(x, h)
    circular = convolucion_fft(x, h, rellenar=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].stem(np.arange(lineal.size), lineal)
    ax[0].set_title(f"Lineal (relleno con ceros), largo {lineal.size}")
    ax[1].stem(np.arange(circular.size), circular)
    ax[1].set_title(f"Circular (sin relleno), largo {circular.size}")
    for a in ax:
        a.set_xlabel("n")
        a.set_ylabel("y[n]")
        a.grid(True)
    fig.suptitle("Sin relleno de ceros la cola se 'enrolla' sobre el comienzo")
    fig.tight_layout()
    fig.savefig("tarea3_convolucion_circular.png")
    plt.close()

    # 3. Tiempos en funcion del largo del filtro.
    n = 1024
    largos_h = [3, 7, 15, 31, 63, 127, 255, 511]
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)

    t_dir, t_fft_propia, t_fft_numpy = [], [], []
    for m in largos_h:
        h = rng.normal(size=m)
        t0 = time.perf_counter()
        convolucion_directa(x, h)
        t_dir.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        convolucion_fft(x, h)            # usa la DFT O(N^2) propia
        t_fft_propia.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _convolucion_fft_numpy(x, h)     # FFT real O(N log N)
        t_fft_numpy.append(time.perf_counter() - t0)

    plt.figure(figsize=(8, 5))
    plt.loglog(largos_h, t_dir, "o-", label="directa O(n*m)")
    plt.loglog(largos_h, t_fft_propia, "s--", label="fft propia (DFT O(N^2))")
    plt.loglog(largos_h, t_fft_numpy, "^-", label="FFT real (np.fft, O(N log N))")
    plt.xlabel("Largo del filtro (m)")
    plt.ylabel("Tiempo [s]")
    plt.title("Convolucion: directa vs FFT (senal de largo fijo n=1024)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tarea3_convolucion_tiempos.png")
    plt.close()


def _medir_filtro(h, fs):
    """Ancho de transicion (a -40 dB) y atenuacion en rechazo de un pasabajos.

    - Ancho de transicion: distancia entre el cruce de -3 dB (borde de la banda
      de paso) y el primer cruce de -40 dB.
    - Atenuacion en rechazo: nivel del primer lobulo lateral (primer maximo
      local de |H| despues del primer nulo), que es el pico de la banda de
      rechazo.
    """
    f, H = respuesta_frecuencia(h, fs)
    db = 20 * np.log10(np.abs(H) + 1e-12)

    cruce3 = np.flatnonzero(db < -3.0)
    if cruce3.size == 0:
        return np.nan, np.nan
    i3 = cruce3[0]
    f3 = f[i3]

    cruce40 = np.flatnonzero((f > f3) & (db < -40.0))
    f40 = f[cruce40[0]] if cruce40.size else np.nan
    ancho = f40 - f3 if not np.isnan(f40) else np.nan

    # Primer nulo y primer lobulo lateral (extremos locales de db).
    d = np.diff(db)
    i_min = np.flatnonzero((d[:-1] < 0) & (d[1:] > 0)) + 1  # minimos locales
    i_max = np.flatnonzero((d[:-1] > 0) & (d[1:] < 0)) + 1  # maximos locales
    nulos = i_min[i_min > i3]
    if nulos.size == 0:
        return ancho, np.nan
    i_nulo = nulos[0]
    lobulos = i_max[i_max > i_nulo]
    atenuacion = -db[lobulos[0]] if lobulos.size else np.nan
    return ancho, atenuacion


def tarea4_filtros():
    print("\nTAREA 4: FILTROS FIR POR VENTANA")
    fs = 1000
    fc = 100.0
    n_coefs = [21, 51, 101, 201]
    ventanas = ["rect", "hamming", "hann"]
    etiquetas = {"rect": "Rectangular", "hamming": "Hamming", "hann": "Hann"}

    # 2. |H(f)| en dB para cada (n_coef, ventana).
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for j, ventana in enumerate(ventanas):
        for n_coef in n_coefs:
            h = fir_pasabajos(fc, fs, n_coef, ventana)
            f, H = respuesta_frecuencia(h, fs)
            db = 20 * np.log10(np.abs(H) + 1e-12)
            ax[j].plot(f, db, label=f"n_coef={n_coef}")
        ax[j].axvline(fc, color="k", ls=":", lw=0.8)
        ax[j].set_title(etiquetas[ventana])
        ax[j].set_xlabel("Frecuencia [Hz]")
        ax[j].set_ylim(-90, 5)
        ax[j].grid(True)
        ax[j].legend(fontsize=7)
    ax[0].set_ylabel("|H(f)| [dB]")
    fig.suptitle(f"Pasabajos FIR (fc={fc:.0f} Hz, fs={fs} Hz): efecto de n_coef y ventana")
    fig.tight_layout()
    fig.savefig("tarea4_respuesta_filtros.png")
    plt.close()

    # Tabla: ancho de transicion y atenuacion en rechazo.
    print(f"{'Ventana':<12} | {'n_coef':<7} | {'Ancho transicion [Hz]':<22} | {'Atenuacion rechazo [dB]'}")
    print("-" * 70)
    filas = []
    for ventana in ventanas:
        for n_coef in n_coefs:
            h = fir_pasabajos(fc, fs, n_coef, ventana)
            ancho, atenuacion = _medir_filtro(h, fs)
            filas.append((etiquetas[ventana], n_coef, ancho, atenuacion))
            ancho_s = f"{ancho:.1f}" if not np.isnan(ancho) else "—"
            print(f"{etiquetas[ventana]:<12} | {n_coef:<7} | {ancho_s:<22} | {atenuacion:.1f}")

    with open("tarea4_tabla_filtros.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["ventana", "n_coef", "ancho_transicion_Hz", "atenuacion_rechazo_dB"])
        for ventana, n_coef, ancho, atenuacion in filas:
            w.writerow([ventana, n_coef, ancho, atenuacion])

    # 3. Chirp de 0 a 500 Hz filtrado con el pasabajos.
    tarea4_chirp(fs, fc)


def tarea4_chirp(fs=1000.0, fc=100.0):
    T = 2.0
    t = np.arange(int(fs * T)) / fs
    f_max = 500.0
    # Fase = 2*pi * integral de f(t) = 2*pi * (f_max/T) * t^2 / 2
    x = np.cos(np.pi * (f_max / T) * t ** 2)

    h = fir_pasabajos(fc, fs, n_coef=101, ventana="hamming")
    y = filtrar(x, h)

    # Instante en que la frecuencia instantanea cruza fc.
    t_corte = T * fc / f_max

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, x, lw=0.6, label="Chirp 0-500 Hz (entrada)")
    ax[0].plot(t, y, lw=1.0, label=f"Filtrado (pasabajos fc={fc:.0f} Hz)")
    ax[0].axvline(t_corte, color="k", ls=":", lw=0.8)
    ax[0].set_ylabel("Amplitud")
    ax[0].set_title(f"Chirp filtrado: antes de t={t_corte:.2f} s (f<{fc:.0f} Hz) pasa, despues se atenua")
    ax[0].legend()
    ax[0].grid(True)

    f_inst = (f_max / T) * t
    ax[1].plot(t, f_inst, label="Frecuencia instantanea")
    ax[1].axhline(fc, color="k", ls=":", lw=0.8)
    ax[1].axvline(t_corte, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("Tiempo [s]")
    ax[1].set_ylabel("Frecuencia [Hz]")
    ax[1].legend()
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig("tarea4_chirp_filtrado.png")
    plt.close()


if __name__ == "__main__":
    tarea1_muestreo()
    tarea2_dft()
    tarea2_zero_padding()
    tarea3_convolucion()
    tarea4_filtros()
    print("\nFiguras y tablas de la parte comun generadas con exito.")
