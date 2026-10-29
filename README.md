# PSI-lab1-2-1026

[![CI](https://github.com/CarPaz/PSI-lab1-2-1026/actions/workflows/ci.yml/badge.svg)](https://github.com/CarPaz/PSI-lab1-2-1026/actions/workflows/ci.yml)

Laboratorios de Procesamiento de Señales e Imágenes (13318), USACH, 2-2026.

## Instalar y probar (sin Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -v                        # pruebas unitarias e integración
ruff check .                     # estilo
psi-demo -f 50 120 300 --ruido 0.8   # genera out/demo.png
```

## Con Docker

```bash
docker compose build
docker compose run --rm tests    # pytest dentro del contenedor
docker compose run --rm demo     # escribe out/demo.png
docker compose up notebook       # JupyterLab en http://localhost:8888
```

## Interfaz del paquete (`psi_lab.senales`)

| Función | Argumentos | Retorna |
|---|---|---|
| `senoidal(f, fs, duracion, amplitud=1.0, fase=0.0)` | f [Hz], fs [Hz], duracion [s], fase [rad] | `(t, x)`: instantes [s] y valores |
| `cuadrada(f, fs, duracion, amplitud=1.0)` | f [Hz], fs [Hz], duracion [s] | `(t, x)`: onda cuadrada ±amplitud |
| `espectro(x, fs)` | señal real x, fs [Hz] | `(freqs, amp)`: frecuencias [Hz] y amplitud de un lado |

Comando `psi-demo`: suma senoidales (`-f`), agrega ruido gaussiano (`--ruido`, `--semilla`), imprime los picos detectados y guarda la figura (`-o`, por defecto `out/demo.png`).

## Integración continua

En cada push a `main` y en cada pull request, GitHub Actions corre tres jobs en orden: `lint` (ruff), `test` (pytest en Python 3.11, 3.12 y 3.13) y `docker` (construye la imagen, corre las pruebas en el contenedor y sube `demo.png` como artifact).

## Laboratorio 1 · Parte común (`psi_lab.lab1`)

Solo NumPy; `np.fft` y `np.convolve` se usan únicamente en las pruebas como referencia.

| Módulo | Función | Argumentos | Retorna |
|---|---|---|---|
| `muestreo` | `muestrear(fun, fs, duracion)` | `fun(t)` vectorizada, fs [Hz], duracion [s] | `(t, x)`, N = round(fs·duracion) |
| `muestreo` | `frecuencia_alias(f, fs)` | f [Hz] (escalar o arreglo), fs [Hz] | frecuencia aparente en [0, fs/2] [Hz] |
| `dft` | `dft(x)` / `idft(X)` | arreglo de largo N | arreglo complejo de largo N (O(N²)) |
| `dft` | `espectro(x, fs, ventana="rect", nfft=None)` | ventana ∈ {rect, hann, hamming}; nfft ≥ len(x) rellena con ceros | `(freqs [Hz], amp)` de un lado, nfft//2+1 puntos |
| `convolucion` | `convolucion_directa(x, h)` / `convolucion_fft(x, h)` | arreglos 1D | convolución lineal, largo len(x)+len(h)−1 |
| `filtros` | `fir_pasabajos(fc, fs, n_coef, ventana="hamming")` | fc, fs [Hz], n_coef impar | coeficientes h (ganancia 1 en DC) |
| `filtros` | `fir_pasaaltos`, `fir_pasabanda(f1, f2, ...)`, `fir_rechazabanda(f1, f2, ...)` | igual que arriba | coeficientes h simétricos (fase lineal) |
| `filtros` | `respuesta_frecuencia(h, fs, n=4096)` | h, fs [Hz] | `(freqs [Hz] en [0, fs/2], H complejo)` |
| `filtros` | `filtrar(x, h)` | señal y coeficientes | señal filtrada del mismo largo, sin retardo |
