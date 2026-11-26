"""Reproducible Case C experiments: python -m psi_lab.lab1.image_experiments.

The output includes figures, raw arrays, CSV measurements and a Spanish
discussion. A missing real-interference photograph is recorded explicitly;
synthetic contamination of a photograph is never presented as the real case.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from psi_lab.lab1.caso_imagen import (
    add_periodic_noise,
    apply_notch,
    detect_periodic_peaks,
    dft2d,
    downsample,
    fase,
    gaussian_lowpass,
    image_metrics,
    magnitud_log,
    notch_mask,
    swap_phases,
)
from psi_lab.lab1.datos import load_image, zone_plate
from psi_lab.lab1.filtros import fir_pasabajos, respuesta_frecuencia


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _show(axis, image, title, *, spectrum=False):
    options = {} if spectrum else {"vmin": 0, "vmax": 1}
    axis.imshow(image, cmap="gray", interpolation="nearest", **options)
    axis.set_title(title, fontsize=10)
    axis.set_axis_off()


def _save(fig, directory, name):
    fig.savefig(directory / name, dpi=150, bbox_inches="tight")
    fig.clear()


def _mark_peaks(axis, peaks, shape):
    if len(peaks):
        axis.scatter(peaks[:, 1] + shape[1] // 2,
                     peaks[:, 0] + shape[0] // 2,
                     s=45, facecolors="none", edgecolors="red", linewidths=0.8)


def _fourier_experiment(images, directory):
    fig = Figure(figsize=(12, 8), layout="constrained")
    axes = fig.subplots(2, 3)
    for row, (name, image) in enumerate(images.items()):
        spectrum = dft2d(image)
        _show(axes[row, 0], image, name)
        _show(axes[row, 1], magnitud_log(spectrum), "log(1 + magnitud), centrada",
              spectrum=True)
        phase_plot = axes[row, 2].imshow(fase(spectrum, centrada=True),
                                       cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[row, 2].set_title("Fase centrada [rad]")
        axes[row, 2].set_axis_off()
        fig.colorbar(phase_plot, ax=axes[row, 2], shrink=0.7)
    _save(fig, directory, "c1_magnitud_fase.png")
    a, b = images.values()
    ab, ba = swap_phases(a, b)
    np.savez_compressed(directory / "c1_intercambio_arrays.npz",
                        camera=a, astronaut=b, mag_camera_phase_astronaut=ab,
                        mag_astronaut_phase_camera=ba)
    fig = Figure(figsize=(9, 9), layout="constrained")
    axes = fig.subplots(2, 2)
    for axis, image, title in zip(axes.flat, (a, b, ab, ba), (
            "Camera original", "Astronaut original",
            "Magnitud camera + fase astronaut", "Magnitud astronaut + fase camera")):
        _show(axis, image, title)
    fig.suptitle("Intercambio de fases: observar posiciones, bordes y formas")
    _save(fig, directory, "c1_intercambio_fases.png")


def _noise_conditions(size):
    """Integer bins isolate exact peaks; fractional bins deliberately leak."""
    uy, ux = round(0.11 * size), round(0.16 * size)
    vy, vx = round(0.19 * size), -round(0.27 * size)
    return {
        "un_tono": [(ux / size, uy / size)],
        "dos_tonos": [(ux / size, uy / size), (vx / size, vy / size)],
        "fuera_de_bin": [((ux + 0.35) / size, (uy + 0.4) / size)],
    }


def _denoising_experiments(images, directory, threshold_ratio, dc_radius, max_pairs):
    records, peak_records = [], []
    for name, image in images.items():
        for condition, frequencies in _noise_conditions(image.shape[0]).items():
            for amplitude in (0.08, 0.2):
                components = [(amplitude, u, v) for u, v in frequencies]
                noisy, noise = add_periodic_noise(image, components)
                spectrum = dft2d(noisy)
                peaks = detect_periodic_peaks(
                    spectrum, threshold_ratio=threshold_ratio,
                    dc_radius=dc_radius, max_pairs=max_pairs)
                case = f"{name}_{condition}_A{amplitude:.2f}"
                peak_records.append({"case": case, "components_A_u_v": components,
                                     "detected_row_col_bins": peaks.tolist()})
                variants = [("sin_filtrar", 0.0, noisy)]
                for kind in ("ideal", "gaussian"):
                    for radius in (1.0, 2.0, 4.0):
                        variants.append((f"notch_{kind}", radius,
                                         apply_notch(noisy, peaks, radius=radius,
                                                     kind=kind)))
                for sigma in (0.7, 1.4, 2.2):
                    variants.append(("pasabajos_gaussiano", sigma,
                                     gaussian_lowpass(noisy, sigma)))
                for method, parameter, restored in variants:
                    records.append({"imagen": name, "condicion": condition,
                                    "amplitud_por_tono": amplitude,
                                    "numero_tonos": len(components), "metodo": method,
                                    "radio_bins_o_sigma_px": parameter,
                                    "picos_detectados": len(peaks),
                                    **image_metrics(image, restored)})
                # Fixed display parameters across all conditions, not an oracle
                # selecting the best result against the unavailable clean image.
                ideal = apply_notch(noisy, peaks, radius=2, kind="ideal")
                gaussian = apply_notch(noisy, peaks, radius=2, kind="gaussian")
                lowpass = gaussian_lowpass(noisy, 1.4)
                fig = Figure(figsize=(12, 8), layout="constrained")
                axes = fig.subplots(2, 3)
                panels = [(image, "Original"), (noisy, "Con interferencia"),
                          (magnitud_log(spectrum), "Espectro y picos automáticos"),
                          (ideal, "Notch ideal, radio 2 bins"),
                          (gaussian, "Notch gaussiano, sigma 2 bins"),
                          (lowpass, "Pasabajos espacial, sigma 1.4 px")]
                for index, (axis, (array, title)) in enumerate(zip(axes.flat, panels)):
                    _show(axis, array, title, spectrum=index == 2)
                _mark_peaks(axes[0, 2], peaks, image.shape)
                fig.suptitle(f"{name} | {condition} | A={amplitude:.2f} por tono")
                _save(fig, directory, f"c23_{case}.png")
                if name == "camera" and condition == "dos_tonos" and amplitude == 0.2:
                    np.savez_compressed(directory / "c23_ejemplo_arrays.npz",
                                        clean=image, noisy=noisy, noise=noise,
                                        ideal=ideal, gaussian=gaussian,
                                        lowpass=lowpass, peaks=peaks,
                                        mask_ideal=notch_mask(image.shape, peaks),
                                        mask_gaussian=notch_mask(image.shape, peaks,
                                                                 kind="gaussian"))
                    masks = Figure(figsize=(10, 4), layout="constrained")
                    for axis, kind in zip(masks.subplots(1, 2), ("ideal", "gaussian")):
                        _show(axis, notch_mask(image.shape, peaks, kind=kind),
                              f"Máscara notch {kind}")
                    _save(masks, directory, "c3_mascaras_notch.png")
    _csv(directory / "c3_metricas.csv", records)
    (directory / "c2_picos.json").write_text(
        json.dumps(peak_records, indent=2), encoding="utf-8")
    return records


def _aliasing_experiment(photo, directory):
    for name, image in (("zone_plate", zone_plate(photo.shape[0])), ("foto", photo)):
        fig = Figure(figsize=(12, 8), layout="constrained")
        axes = fig.subplots(2, 3)
        arrays = {"original": image}
        for row, factor in enumerate((2, 4)):
            raw = downsample(image, factor, anti_alias=False)
            filtered = downsample(image, factor, anti_alias=True)
            arrays[f"k{factor}_sin_filtro"] = raw
            arrays[f"k{factor}_con_filtro"] = filtered
            _show(axes[row, 0], image, "Original")
            _show(axes[row, 1], raw, f"Reducción x{factor}, sin filtro")
            _show(axes[row, 2], filtered, f"Reducción x{factor}, FIR previo")
        fig.suptitle(f"Aliasing espacial: {name}; misma escala de intensidades")
        _save(fig, directory, f"c5_aliasing_{name}.png")
        np.savez_compressed(directory / f"c5_{name}_arrays.npz", **arrays)

    records = []
    fig = Figure(figsize=(11, 4), layout="constrained")
    axes = fig.subplots(1, 2)
    for axis, factor in zip(axes, (2, 4)):
        new_nyquist = 1 / (2 * factor)
        for n_coef in (33, 65, 129):
            for ratio in (0.7, 0.8, 0.9):
                cutoff = ratio * new_nyquist
                h = fir_pasabajos(cutoff, 1.0, n_coef, "hamming")
                frequencies, response = respuesta_frecuencia(h, 1.0, n=2048)
                db = 20 * np.log10(np.maximum(np.abs(response), 1e-15))
                records.append({"factor": factor, "n_coef": n_coef,
                                "cutoff_ratio": ratio, "fc_ciclos_pixel": cutoff,
                                "nyquist_nuevo": new_nyquist,
                                "max_rechazo_db": float(db[frequencies >= new_nyquist].max())})
                if ratio == 0.8:
                    axis.plot(frequencies, db, label=f"{n_coef} coef.")
        axis.axvline(new_nyquist, color="black", linestyle="--", label="Nuevo Nyquist")
        axis.set(xlabel="Frecuencia [ciclos/píxel original]", ylabel="Ganancia [dB]",
                 title=f"Reducción x{factor}, fc=0.8/(2k)", ylim=(-100, 5))
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    _save(fig, directory, "c5_respuesta_fir.png")
    _csv(directory / "c5_parametros_fir.csv", records)
    return records


def _real_experiment(path, directory, threshold_ratio, dc_radius, max_pairs,
                     notch_radius, crop):
    image = load_image(path)
    original_shape = image.shape
    if crop is not None:
        x, y, width, height = crop
        if (min(x, y) < 0 or min(width, height) < 16
                or x + width > image.shape[1] or y + height > image.shape[0]):
            raise ValueError("--real-crop must be an in-bounds x y width height ROI >=16")
    else:
        height, width = min(512, image.shape[0]), min(512, image.shape[1])
        y, x = (image.shape[0] - height) // 2, (image.shape[1] - width) // 2
    # Crop rather than resize: prefiltering/resizing might hide the interference.
    image = image[y:y + height, x:x + width]
    spectrum = dft2d(image)
    peaks = detect_periodic_peaks(spectrum, threshold_ratio=threshold_ratio,
                                  dc_radius=dc_radius, max_pairs=max_pairs)
    fig = Figure(figsize=(12, 8), layout="constrained")
    axes = fig.subplots(2, 3)
    _show(axes[0, 0], image, "Entrada real, sin referencia limpia")
    _show(axes[0, 1], magnitud_log(spectrum), "Espectro antes + candidatos", spectrum=True)
    _mark_peaks(axes[0, 1], peaks, image.shape)
    records = []
    arrays = {"input": image, "peaks": peaks}
    for index, kind in enumerate(("ideal", "gaussian")):
        restored = apply_notch(image, peaks, radius=notch_radius, kind=kind)
        arrays[kind] = restored
        before_energy = sum(float(np.mean(g * g)) for g in np.gradient(image))
        after_energy = sum(float(np.mean(g * g)) for g in np.gradient(restored))
        records.append({"metodo": kind, "picos_candidatos": len(peaks),
                        "cambio_rms": float(np.sqrt(np.mean((restored - image) ** 2))),
                        "cambio_media": float(restored.mean() - image.mean()),
                        "razon_energia_gradiente": after_energy / max(before_energy, 1e-30),
                        "fraccion_fuera_0_1": float(np.mean((restored < 0) | (restored > 1)))})
        _show(axes[1, index], restored, f"Notch {kind}")
    _show(axes[0, 2], magnitud_log(dft2d(arrays["gaussian"])),
          "Espectro después: notch gaussiano", spectrum=True)
    residual = image - arrays["gaussian"]
    limit = max(float(np.max(np.abs(residual))), 1e-12)
    plot = axes[1, 2].imshow(residual, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1, 2].set(title="Contenido retirado: buscar detalles perdidos")
    axes[1, 2].set_axis_off()
    fig.colorbar(plot, ax=axes[1, 2], shrink=0.7)
    _save(fig, directory, "c4_caso_real.png")
    _csv(directory / "c4_diagnosticos_sin_referencia.csv", records)
    np.savez_compressed(directory / "c4_arrays.npz", **arrays)
    result = {"source": str(Path(path).resolve()), "original_shape": original_shape,
              "crop_x_y_width_height": [x, y, width, height],
              "detected_row_col_bins": peaks.tolist(), "diagnostics": records}
    (directory / "c4_metadatos.json").write_text(json.dumps(result, indent=2),
                                                 encoding="utf-8")
    return result


def _discussion(directory, metrics, fir_records, real_result):
    selected = [r for r in metrics if r["metodo"] == "notch_gaussian"
                and r["radio_bins_o_sigma_px"] == 2]
    lines = [
        "# Caso C: resultados y discusión",
        "",
        "## C1: magnitud y fase",
        ("La magnitud describe cuánto aporta cada frecuencia; la fase sitúa esas "
        "componentes en el espacio. En c1_intercambio_fases.png se deben comparar "
        "bordes, posiciones y formas de cada reconstrucción con ambas originales. "
        "La organización reconocible suele seguir principalmente a la imagen que "
        "aportó la fase, aunque la magnitud sigue afectando contraste y textura. "
        "No implica que la magnitud sea prescindible para reconstruir fielmente."),
        "",
        ("Los cálculos usan espectros sin centrar y magnitudes lineales. fftshift "
        "y log(1+|X|) son herramientas de visualización. Las figuras usan la misma "
        "escala [0,1]; las métricas y los archivos NPZ retienen valores fuera de "
        "ese rango, sin recorte ni normalización por imagen."),
        "",
        "## C2-C3: interferencias y filtros",
        ("Se ensayan dos fotografías, dos amplitudes por componente (0.08 y 0.2), "
        "uno/dos tonos y un tono fuera de los bins. No se entrega la frecuencia "
        "inyectada al detector. Un coseno que coincide con la grilla produce "
        "dos picos conjugados; fuera de la grilla aparece fuga espectral."),
        "",
        ("Los candidatos son máximos locales de amplitud que sobresalen sobre "
        "una mediana radial, fuera del disco DC. La simetría conjugada conserva "
        "la realidad de la reconstrucción. El detector puede confundir texturas "
        "o discontinuidades de borde con ruido y perder componentes débiles."),
        "",
        ("Se comparan radios notch 1, 2 y 4 bins y sigmas espaciales 0.7, 1.4 y "
        "2.2 píxeles en c3_metricas.csv. Las figuras muestran siempre radio 2 y "
        "sigma espacial 1.4: no se elige el mejor usando la referencia limpia. "
        "Un notch ancho captura más fuga, pero retira más información legítima. "
        "El ideal puede producir oscilaciones por su corte abrupto; el gaussiano "
        "hace una transición suave. El pasabajos atenúa muchas frecuencias y "
        "también puede desenfocar bordes."),
        "",
        ("PSNR = 10 log10(1/MSE), implementado en el paquete. SSIM se calcula con "
        "skimage.metrics.structural_similarity y data_range=1. No se miden "
        "contra una versión recortada o normalizada de la salida."),
        "",
        "### Resultados con notch gaussiano de radio 2 bins",
        "| Imagen | Condición | Amplitud/tono | PSNR [dB] | SSIM |",
        "|---|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(f"| {row['imagen']} | {row['condicion']} | "
                     f"{row['amplitud_por_tono']:.2f} | {row['psnr_db']:.2f} | "
                     f"{row['ssim']:.4f} |")
    lines += [
        "",
        "## C4: interferencia real",
        ("PENDIENTE: no se suministró una fotografía o escaneo con interferencia "
         "real. Las fotografías camera y astronaut con ruido agregado pertenecen "
         "al experimento controlado y NO completan C4. Ejecutar nuevamente con "
         "--real-image data/foto.jpg cuando se disponga de esa imagen."
         if real_result is None else
         f"Imagen procesada: {real_result['source']}. Recorte sin remuestreo "
         f"[x,y,ancho,alto]: {real_result['crop_x_y_width_height']}. Se detectaron "
         f"{len(real_result['detected_row_col_bins'])} candidatos. Ver c4_caso_real.png."),
        "",
        ("Sin imagen limpia no se reportan PSNR/SSIM como calidad de restauración. "
        "Se inspeccionan original, espectros, restauración y residuo; se busca "
        "menos franjas conservando letras, bordes y textura. El cambio RMS, "
        "la energía de gradiente y la conservación de la media son diagnósticos, "
        "no una prueba de mejora. Quitar los picos seleccionados es consecuencia "
        "del filtro y tampoco demuestra que fueran ruido. Comparar regiones y "
        "varios radios ayuda a detectar pérdida de detalle y sobreprocesamiento."),
        "",
        "## C5: aliasing espacial",
        ("La zone plate usa I(x,y)=(1+cos(pi*(x²+y²)/M))/2 y coordenadas centradas. "
        "Sus frecuencias locales son fx=x/M y fy=y/M ciclos/píxel. Al diezmar "
        "por k, el nuevo Nyquist en la grilla original es 1/(2k) en cada eje. "
        "Las frecuencias que lo exceden se pliegan y originan anillos o patrones "
        "fantasma; el detalle observado no representa nuevas estructuras."),
        "",
        ("Se usa fir_pasabajos de la parte común con fs=1, Hamming, 65 coeficientes "
        "y fc=0.8/(2k), aplicándolo por convolución a filas y columnas. El 20% "
        "de margen deja espacio para la transición del FIR. La convolución "
        "centrada con borde reflejado conserva el tamaño y evita bordes negros. "
        "c5_parametros_fir.csv compara 33/65/129 coeficientes y márgenes 0.7/0.8/0.9; "
        "c5_respuesta_fir.png permite verificar la atenuación desde nuevo Nyquist. "
        "Un FIR finito no elimina todo el aliasing; más rechazo y margen también "
        "reducen resolución. Un filtro aplicado después de diezmar ya no puede "
        "separar las frecuencias que se superpusieron."),
        "",
        "### Rechazo medido del FIR utilizado",
        "| Factor k | fc [ciclos/píxel] | Máxima ganancia desde nuevo Nyquist [dB] |",
        "|---:|---:|---:|",
    ]
    for row in fir_records:
        if row["n_coef"] == 65 and row["cutoff_ratio"] == 0.8:
            lines.append(f"| {row['factor']} | {row['fc_ciclos_pixel']:.3f} | "
                         f"{row['max_rechazo_db']:.2f} |")
    lines += [
        "",
        ("La fotografía camera repite el experimento: revisar bordes y texturas "
        "en c5_aliasing_foto.png. Un aspecto más nítido sin filtro puede contener "
        "aliasing; una salida más suave no es por sí sola un fallo del filtro."),
        "",
        "## Operaciones auxiliares y reproducibilidad",
        ("skimage.data aporta camera/astronaut; skimage.io lee archivos, "
        "skimage.color convierte a gris y skimage.transform.resize iguala "
        "tamaños antes de los experimentos controlados. El caso real se recorta "
        "sin remuestrear. La detección, los notch, la convolución separable, PSNR "
        "y el FIR previo a diezmar se implementan en el paquete. NumPy FFT "
        "se utiliza en el caso aplicado conforme al enunciado. Las pruebas "
        "de integración usan exclusivamente imágenes sintéticas, sin descargas."),
    ]
    (directory / "resultados.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiments(output_dir="out/caso_c", *, size=256, real_image=None,
                    threshold_ratio=12.0, dc_radius=8, max_pairs=4,
                    notch_radius=2.0, real_crop=None):
    """Generate all Case C experiments; return paths and honest completion status."""
    if not isinstance(size, int) or size < 64:
        raise ValueError("size must be an integer >= 64")
    if real_image is not None and not Path(real_image).is_file():
        raise FileNotFoundError(real_image)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    images = {name: load_image(name, size) for name in ("camera", "astronaut")}
    _fourier_experiment(images, directory)
    metrics = _denoising_experiments(images, directory, threshold_ratio, dc_radius,
                                    max_pairs)
    fir_records = _aliasing_experiment(images["camera"], directory)
    real_result = None
    if real_image is not None:
        real_result = _real_experiment(real_image, directory, threshold_ratio,
                                       dc_radius, max_pairs, notch_radius, real_crop)
    _discussion(directory, metrics, fir_records, real_result)
    manifest = {"size": size, "threshold_ratio": threshold_ratio,
                "dc_radius_bins": dc_radius, "max_pairs": max_pairs,
                "real_notch_radius_bins": notch_radius,
                "real_case": "processed" if real_result is not None else "pending_input",
                "real_source": str(real_image) if real_image is not None else None,
                "numpy_version": np.__version__, "metric_rows": len(metrics)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                             encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimentos de imágenes: caso C")
    parser.add_argument("--output-dir", default="out/caso_c")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--real-image", help="Foto/escaneo con interferencia real")
    parser.add_argument("--real-crop", nargs=4, type=int,
                        metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--threshold-ratio", type=float, default=12.0)
    parser.add_argument("--dc-radius", type=float, default=8)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--notch-radius", type=float, default=2.0)
    args = parser.parse_args(argv)
    try:
        manifest = run_experiments(
            args.output_dir, size=args.size, real_image=args.real_image,
            threshold_ratio=args.threshold_ratio, dc_radius=args.dc_radius,
            max_pairs=args.max_pairs, notch_radius=args.notch_radius,
            real_crop=args.real_crop)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Resultados en {Path(args.output_dir).resolve()}")
    if manifest["real_case"] == "pending_input":
        print("C4 pendiente: falta una imagen con interferencia periódica real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
