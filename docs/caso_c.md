# Caso C: imágenes, Fourier y aliasing

## Ejecutar los experimentos

Desde la raíz del repositorio, después de instalar `pip install -e ".[dev]"`:

```powershell
# Windows
.venv/Scripts/python.exe -m psi_lab.lab1.image_experiments
```

```bash
# Linux, con el entorno virtual activado
python -m psi_lab.lab1.image_experiments

# Alternativa con Docker, solo los experimentos del caso C
docker compose run --rm imagenes
```

Se generan las figuras, las tablas y una discusión en `out/caso_c/`.
No modifica ni regenera los resultados de la parte común.
Los datos de entrada y los resultados generados no se versionan.

| Archivo | Contenido |
|---|---|
| `c1_magnitud_fase.png` | Dos fotografías con su magnitud logarítmica y fase |
| `c1_intercambio_fases.png` | Originales e intercambio de fases |
| `c23_*.png` | Imágenes limpias, contaminadas y filtradas; espectro con candidatos |
| `c2_picos.json` | Interferencias inyectadas y picos detectados, por condición |
| `c3_mascaras_notch.png` | Máscaras ideal y gaussiana |
| `c3_metricas.csv` | PSNR propio y SSIM en 120 combinaciones de imagen, ruido y método |
| `c5_aliasing_zone_plate.png` | Reducciones por 2 y 4 con/sin FIR previo |
| `c5_aliasing_foto.png` | El mismo experimento con una fotografía |
| `c5_respuesta_fir.png` | Respuesta de los filtros respecto del nuevo Nyquist |
| `c5_parametros_fir.csv` | Comparación de longitudes y frecuencias de corte |
| `*_arrays.npz` | Resultados numéricos sin recortar, para inspección/reutilización |
| `manifest.json` | Parámetros y estado del caso real de esta ejecución |
| `resultados.md` | Explicaciones, resultados medidos y limitaciones |

La evaluación controlada incluye `camera` y `astronaut` de `skimage.data`,
amplitudes por tono de 0.08 y 0.2, uno/dos tonos y un tono fuera de los bins.
Compara notch ideal/gaussiano con radios de 1, 2 y 4 bins y pasabajos espacial
con sigmas de 0.7, 1.4 y 2.2 píxeles. La referencia limpia se utiliza solamente
para evaluar: el detector no recibe las frecuencias inyectadas.

## Convenciones para entender el código

- Una imagen es una matriz real `(filas, columnas)`; las fotografías cargadas
  se convierten a gris y valores `[0,1]` según el rango del tipo de dato.
- `dft2d` devuelve el espectro **sin centrar**. `fase(X)` y `reconstruir` usan
  ese mismo orden por defecto. `fase(X, centrada=True)` es para visualizar.
- `magnitud_log(X)` centra y aplica `log(1+abs(X))`. No sirve como magnitud
  de entrada de `reconstruir`: la reconstrucción requiere magnitud lineal.
- Si se centran magnitud y fase, pasar `centrada=True` a `reconstruir`.
- Un ruido se describe como `(A,u,v)`: amplitud, frecuencia horizontal y
  frecuencia vertical. `u,v` están en **ciclos/píxel**.
- Los picos del detector y los centros de las máscaras son desplazamientos
  `(fila,columna)` desde DC, en **bins**: para una imagen de `R×C`, el coseno
  de frecuencias `(u,v)` tiene centros en `±(v*R,u*C)` si coincide con la grilla.
- Cada notch incluye su pareja conjugada y protege DC. El espectro filtrado
  se devuelve al orden original mediante `ifftshift` antes de invertirlo.
- Las métricas usan los valores numéricos sin recortar y rango fijo 1. Las
  figuras muestran una escala común `[0,1]`; valores fuera del rango se saturan
  solo al mostrarlos. Los NPZ conservan los valores originales.

## Interfaz de `psi_lab.lab1.datos`

| Función | Entrada | Salida |
|---|---|---|
| `load_image(source="camera", size=None)` | `camera`, `astronaut` o ruta local; tamaño entero o `(R,C)` opcional | Imagen gris `float64` en `[0,1]` |
| `synthetic_image(size=128)` | Entero o `(R,C)` | Imagen determinista sin descarga |
| `periodic_pattern(shape, amplitude, u, v, phase=0)` | `(R,C)`, amplitud, ciclos/píxel y fase en radianes | Coseno 2D de forma `(R,C)` |
| `zone_plate(size=256)` | Lado `M` | Patrón cuadrado de frecuencias locales crecientes |

`load_image` usa `skimage` para lectura, conversión de color y remuestreo
auxiliar. El remuestreo del cargador no se utiliza para el experimento de
aliasing, que aplica explícitamente nuestro FIR y después toma muestras.

## Interfaz de `psi_lab.lab1.caso_imagen`

| Función | Entrada | Salida |
|---|---|---|
| `dft2d(x)` / `idft2d(X)` | Matriz 2D; espectros sin centrar | Matriz compleja de la misma forma |
| `magnitud_log(X)` | Espectro sin centrar | Magnitud logarítmica centrada |
| `fase(X, centrada=False)` | Espectro sin centrar | Fase en radianes, orden configurable |
| `reconstruir(magnitud, fase, centrada=False)` | Magnitud lineal y fase con igual forma/orden | Imagen real |
| `swap_phases(first, second)` | Dos imágenes de igual forma | Dos reconstrucciones con fases intercambiadas |
| `add_periodic_noise(image, interferences)` | Imagen y lista `(A,u,v)` | `(imagen_contaminada, ruido_total)` |
| `detect_periodic_peaks(spectrum, dc_radius=8, threshold_ratio=12, min_distance=5, max_pairs=4)` | Espectro sin centrar; distancias en bins | Array entero `(P,2)` de picos conjugados |
| `notch_mask(shape, peaks, radius=2, kind="ideal")` | Forma, centros, radio en bins; `ideal` o `gaussian` | Máscara centrada `[0,1]` |
| `apply_notch(image, peaks, radius=2, kind="ideal")` | Imagen y parámetros de máscara | Imagen filtrada, sin recorte |
| `separable_filter(image, kernel)` | Imagen y núcleo 1D simétrico de largo impar | Convolución centrada, borde reflejado |
| `gaussian_lowpass(image, sigma=1)` | Sigma espacial en píxeles | Imagen filtrada por convolución gaussiana |
| `psnr(reference, estimate, data_range=1)` | Dos imágenes de igual forma | PSNR en dB; infinito si son iguales |
| `image_metrics(reference, estimate)` | Referencia limpia y estimación | Diccionario `psnr_db`, `ssim` |
| `anti_alias_filter(image, factor, n_coef=65, cutoff_ratio=0.8)` | Imagen, reducción `k`, coeficientes impares y margen | Imagen prefiltrada con FIR separable propio |
| `downsample(image, factor, anti_alias=True, n_coef=65, cutoff_ratio=0.8)` | Imagen y opciones del FIR | Una muestra cada `k` en ambos ejes |

Los parámetros opcionales después de `*` en las firmas del código deben
pasarse por nombre. Los ejes no se transponen: la primera coordenada siempre
es vertical y la segunda horizontal.

## Cómo se detectan los picos y se eligen los parámetros

El detector calcula la amplitud centrada y resta su mediana logarítmica por
anillos para compensar la envolvente habitual del espectro de una imagen.
Busca máximos locales 3×3 que superen esa envolvente por un factor 12, tengan
al menos 1% del mayor coeficiente fuera de DC y estén fuera del disco central
de radio 8. Separa candidatos por al menos 5 bins y conserva hasta 4 parejas.

Estos valores son un punto de partida documentado, no una garantía universal.
El radio DC protege contenido grueso de la imagen, pero impide detectar ruido
muy cercano al origen. Las texturas y los bordes pueden generar falsos
positivos; las interferencias débiles o extendidas pueden pasar inadvertidas.
Los picos fraccionarios reparten energía entre bins y un notch demasiado
estrecho deja ruido residual. Ampliarlo retira más información legítima.

El notch ideal anula un disco por pico. El gaussiano multiplica factores
`1-exp(-D²/(2r²))` alrededor de los picos y sus conjugados. Se conserva DC.
Los radios se miden en bins, por lo que cambiar el tamaño de la imagen exige
revisarlos. El pasabajos espacial afecta una banda mucho más amplia y puede
desenfocar la foto. Las tablas comparan estos efectos con una referencia.

Para reducir por `k`, el límite del nuevo Nyquist es `1/(2k)` ciclos por píxel
de la imagen original en cada eje. Se usa `fc=0.8/(2k)` y Hamming con 65
coeficientes para dejar margen de transición. Se aplican convoluciones por
filas y columnas con los coeficientes de `fir_pasabajos` de la parte común.
La tabla compara también 33/129 coeficientes y márgenes 0.7/0.9, e informa
el peor rechazo desde el nuevo Nyquist. La atenuación es finita y el filtrado
intercambia detalle por menor aliasing.

## Caso con interferencia periódica real: falta la entrada

Se necesita una foto de pantalla, un escaneo de una impresión o una imagen
de sensor con franjas reales. Agregar un coseno a `camera` **no completa C4**.
Guardar la imagen en `data/` y ejecutar, por ejemplo:

```powershell
.venv/Scripts/python.exe -m psi_lab.lab1.image_experiments --real-image data/foto.jpg

# Si la interferencia está en una región particular:
.venv/Scripts/python.exe -m psi_lab.lab1.image_experiments --real-image data/foto.jpg --real-crop 0 0 512 512
```

```bash
docker compose run --rm imagenes --real-image /data/foto.jpg
```

El recorte predeterminado toma hasta 512×512 píxeles del centro sin remuestrear,
para no ocultar el patrón mediante un filtro previo. Se guardan la región
exacta y los picos detectados. Se pueden ajustar `--threshold-ratio`,
`--dc-radius`, `--max-pairs` y `--notch-radius` mirando espectros y residuos.
Conviene usar un directorio de salida distinto por conjunto de parámetros.

El programa genera `c4_caso_real.png`, los resultados numéricos, metadatos y
diagnósticos. Sin referencia limpia, no informa PSNR/SSIM como calidad de
restauración. La inspección del residuo debe comprobar si se retiraron franjas
o detalles auténticos; los cambios de gradiente y RMS no prueban mejora.
La desaparición de los picos seleccionados tampoco es evidencia independiente:
es una consecuencia del filtro. Registrar la procedencia y condiciones de
captura de la foto permite fundamentar la discusión.

## Verificación

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

Las pruebas de C usan imágenes y archivos sintéticos locales. Comprueban el
orden del espectro también con tamaños impares, las propiedades de los
cosenos 2D, la simetría conjugada, la conservación de DC, el PSNR conocido y
la supresión de aliasing. La integración exige PSNR >35 dB y mejora >10 dB:
35 dB equivale a un error RMS menor al 1.78% del rango completo. También exige
mejora de SSIM; no se seleccionan parámetros usando la imagen limpia.
