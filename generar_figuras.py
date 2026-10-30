import numpy as np  
import matplotlib.pyplot as plt
import time

from psi_lab.lab1.muestreo import muestrear, frecuencia_alias
from psi_lab.lab1.dft import dft, espectro

def tarea1_muestreo ():
    print("TAREA 1 : MUESTREO Y ALIASING")
    fs = 10
    frecuencias = [3, 7, 13, 23]

    #1. Tabña de aliasing
    print(f"{'f [Hz]':<10} | {'Alias Teórico':<15} | {'Alias Medido'}")
    print("-" * 45)
    for f in frecuencias:
        t, x= muestrear(lambda t, f=f: np.cos(2* np.pi * f * t), fs , duracion=2)
        freqs, amp= espectro(x, fs)
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
    plt.stem(t_samp, x_samp, linefmt="k-", markerfmt="ko", basefmt="k-", label="Muestras a 10 Hz")
    plt.title("Aliasing: 3 Hz y 13 Hz muestreadas a 10 Hz")
    plt.xlabel("Tiempo [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tarea1_aliasing.png")
    plt.close()

def tarea2_dft():
    print("\n TAREA 2: DFT, RESOLUCIÓN Y FUGA")
    
    # 1. Tiempos de ejecución DFT vs FFT
    N_vals = [2**i for i in range(4, 13)]
    tiempos_dft = []
    tiempos_fft = []
    
    for N in N_vals:
        x = np.random.randn(N)
        
        t0 = time.perf_counter()
        dft(x)
        tiempos_dft.append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        np.fft.fft(x)
        tiempos_fft.append(time.perf_counter() - t0)

    plt.figure(figsize=(8, 5))
    plt.loglog(N_vals, tiempos_dft, 'o-', label="DFT propia O(N^2)")
    plt.loglog(N_vals, tiempos_fft, 's-', label="np.fft.fft O(N log N)")
    plt.title("Tiempo de ejecución: DFT vs FFT")
    plt.xlabel("Largo de la señal (N)")
    plt.ylabel("Tiempo [s]")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tarea2_tiempos.png")
    plt.close()

    # 2. Resolución para distintos T
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
        plt.title(f"T = {T} s (Δf = {1/T:.2f} Hz)")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("tarea2_resolucion.png")
    plt.close()

    # 3. Fuga y ventanas (Tono débil oculto)
    t = np.arange(int(fs * 1)) / fs
    x = np.cos(2 * np.pi * 50 * t) + 1e-3 * np.cos(2 * np.pi * 80 * t)
    
    freqs_rect, amp_rect = espectro(x, fs, ventana="rect")
    freqs_hann, amp_hann = espectro(x, fs, ventana="hann")
    
    # Evitar log(0) sumando un epsilon
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

if __name__ == "__main__":
    tarea1_muestreo()
    tarea2_dft()
    print("Figuras de Tarea 1 y 2 generadas con éxito.")