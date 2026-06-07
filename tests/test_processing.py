from pathlib import Path

import cv2
import numpy as np
import tifffile

from mira220.cli import PROCESSED_SENTINELS, _capture_output_name, _is_processed_output
from mira220.correction import apply_model, apply_scene_correction, load_yaml
from mira220.imaging import (
    compute_ndvi,
    crop_valid_region,
    ndvi_false_color,
    save_reflectance_products,
    smooth_ndvi_for_display,
)


def test_default_model_loads() -> None:
    model = load_yaml(Path("config/models/flat_patch_v1.yaml"))
    assert model["name"] == "flat_patch_v1"
    assert model["preprocessing"]["irc_cut"] == 0.0


def test_correction_equations_and_clipping() -> None:
    model = load_yaml(Path("config/models/flat_patch_v1.yaml"))
    rgb = np.array([[[0.2, 0.3, 0.4]]], dtype=np.float32)
    ir = np.array([[0.5]], dtype=np.float32)
    red, green, corrected_ir = apply_model(rgb, ir, model)
    assert 0 <= red[0, 0] <= 1
    assert 0 <= green[0, 0] <= 1
    assert 0 <= corrected_ir[0, 0] <= 1


def test_ndvi_and_false_color() -> None:
    ndvi = compute_ndvi(np.array([0.2, 0.8]), np.array([0.8, 0.2]))
    np.testing.assert_allclose(ndvi, [0.6, -0.6], atol=1e-5)
    colors = ndvi_false_color(np.array([[-0.2, 0.7]], np.float32), -0.2, 0.7)
    np.testing.assert_allclose(colors[0, 0], [120, 40, 30], atol=1)
    np.testing.assert_allclose(colors[0, 1], [0, 100, 50], atol=1)


def test_ndvi_display_smoothing_happens_on_float_image() -> None:
    ndvi = np.zeros((21, 21), np.float32)
    ndvi[10, 10] = 1.0
    smoothed = smooth_ndvi_for_display(ndvi, sigma=1.5)
    assert smoothed[10, 10] < 1.0
    assert smoothed[10, 10] > smoothed[10, 9] > 0


def test_crop_valid_region_default_size() -> None:
    cropped = crop_valid_region(np.zeros((1400, 1600), np.float32))
    assert cropped.shape == (1150, 1300)


def test_reflectance_products_include_ndvi_and_previews(tmp_path: Path) -> None:
    red = np.full((4, 4), 0.2, np.float32)
    green = np.full((4, 4), 0.3, np.float32)
    ir = np.full((4, 4), 0.8, np.float32)
    rgb = np.dstack([red, green, ir])
    save_reflectance_products(tmp_path, red, green, ir, -0.2, 0.7, rgb=rgb)
    for name in (
        "red_reflectance.tiff",
        "green_reflectance.tiff",
        "ir_reflectance.tiff",
        "rgn_reflectance.tiff",
        "ndvi.tiff",
        "ndvi_gray.png",
        "ndvi_false_color.png",
        "ndvi_false_color_crop.png",
        "rgn_preview.png",
        "rgb_preview.png",
    ):
        assert (tmp_path / name).is_file()
    assert tifffile.imread(tmp_path / "ndvi.tiff").dtype == np.float32
    assert cv2.imread(str(tmp_path / "ndvi_false_color_crop.png")).shape[:2] == (4, 4)


def test_raw_capture_output_name_uses_trimmed_timestamp() -> None:
    raw = Path("1600x1400_Bit12_RGB-IR_RGB-IRMod(GRIG)_2026-05-02_19_36_16.raw")
    assert _capture_output_name(raw) == "2026-02-05_19_36_16"


def test_raw_capture_output_name_falls_back_to_stem() -> None:
    assert _capture_output_name(Path("capture.raw")) == "capture"


def test_processed_output_requires_expected_artifacts(tmp_path: Path) -> None:
    assert not _is_processed_output(tmp_path)
    for name in PROCESSED_SENTINELS:
        (tmp_path / name).write_bytes(b"x")
    assert _is_processed_output(tmp_path)


def test_scene_correction_uses_channel_level_features() -> None:
    config = {
        "red_coefficients": [1, 0, 0, 0, 0, 0],
        "ir_coefficients": [0, 1, 0, 0, 0, 0],
    }
    red = np.array([[0.2, 0.4]], np.float32)
    ir = np.array([[0.6, 0.8]], np.float32)
    corrected_red, corrected_ir = apply_scene_correction(red, ir, config)
    np.testing.assert_allclose(corrected_red, red)
    np.testing.assert_allclose(corrected_ir, ir)


def test_scene_reference_model_applies_without_clipping_sample() -> None:
    model = load_yaml(Path("config/models/scene_reference_v1.yaml"))
    rgb = np.array([[[0.3, 0.4, 0.5]]], dtype=np.float32)
    ir = np.array([[0.5]], dtype=np.float32)
    red, _, corrected_ir = apply_model(rgb, ir, model)
    assert 0 < red[0, 0] < 1
    assert 0 < corrected_ir[0, 0] < 1

