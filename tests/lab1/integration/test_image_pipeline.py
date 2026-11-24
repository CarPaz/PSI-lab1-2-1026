"""Synthetic end-to-end checks: no photograph downloads, network or oracle peaks."""

import numpy as np
import pytest

from psi_lab.lab1.caso_imagen import (
    add_periodic_noise,
    apply_notch,
    detect_periodic_peaks,
    dft2d,
    image_metrics,
)
from psi_lab.lab1.datos import synthetic_image


@pytest.mark.parametrize("kind", ["ideal", "gaussian"])
@pytest.mark.parametrize("amplitude", [0.08, 0.2])
def test_generated_image_noise_detection_notch_metrics(kind, amplitude):
    clean = synthetic_image((128, 160))
    noisy, _ = add_periodic_noise(
        clean, [(amplitude, 23 / 160, 17 / 128), (amplitude / 2, -37 / 160, 29 / 128)])
    peaks = detect_periodic_peaks(dft2d(noisy))
    expected = {(17, 23), (-17, -23), (29, -37), (-29, 37)}
    assert expected <= set(map(tuple, peaks))
    restored = apply_notch(noisy, peaks, kind=kind, radius=2)
    before, after = image_metrics(clean, noisy), image_metrics(clean, restored)
    # 35 dB means RMS error < 1.78% of full range; the additional gain check
    # prevents a weak-noise input from satisfying the threshold without cleaning.
    assert after["psnr_db"] > 35
    assert after["psnr_db"] > before["psnr_db"] + 10
    assert after["ssim"] > before["ssim"]
    np.testing.assert_allclose(restored.mean(), noisy.mean(), atol=1e-12)


def test_local_image_branch_writes_only_no_reference_diagnostics(tmp_path):
    """Exercise C4's file interface with a fixture, not evidence of a real case."""
    import json

    from skimage import io

    from psi_lab.lab1.image_experiments import _real_experiment

    clean = synthetic_image(64)
    noisy, _ = add_periodic_noise(clean, [(0.1, 11 / 64, 9 / 64)])
    source = tmp_path / "synthetic_fixture.png"
    io.imsave(source, (noisy * 255).astype(np.uint8), check_contrast=False)
    result = _real_experiment(source, tmp_path, 12, 8, 4, 2, None)
    assert result["crop_x_y_width_height"] == [0, 0, 64, 64]
    assert (tmp_path / "c4_caso_real.png").is_file()
    assert (tmp_path / "c4_diagnosticos_sin_referencia.csv").is_file()
    assert "psnr" not in json.dumps(result).lower()
    assert "ssim" not in json.dumps(result).lower()


def test_manifest_marks_missing_real_case_and_uses_only_generated_inputs(tmp_path,
                                                                       monkeypatch):
    import json

    from psi_lab.lab1 import image_experiments

    def local_fixture(source, size):
        image = synthetic_image(size)
        return image if source == "camera" else np.rot90(image)

    monkeypatch.setattr(image_experiments, "load_image", local_fixture)
    result = image_experiments.run_experiments(tmp_path, size=64)
    assert result["real_case"] == "pending_input"
    assert result["metric_rows"] == 120
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["real_source"] is None
    assert not (tmp_path / "c4_caso_real.png").exists()
    assert "PENDIENTE" in (tmp_path / "resultados.md").read_text(encoding="utf-8")
    for name in ("c1_intercambio_fases.png", "c5_aliasing_zone_plate.png",
                 "c5_aliasing_foto.png", "c3_metricas.csv"):
        assert (tmp_path / name).stat().st_size > 0
