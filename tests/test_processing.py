from pathlib import Path

import numpy as np

from mira220.correction import apply_model, apply_scene_correction, load_yaml
from mira220.imaging import compute_ndvi, ndvi_false_color


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
    assert colors[0, 0, 0] > colors[0, 0, 1]
    assert colors[0, 1, 1] > colors[0, 1, 0]


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

