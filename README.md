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
