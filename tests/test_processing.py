from pathlib import Path

import cv2
import numpy as np
import tifffile

from mira220.cli import (
    CANDIDATE_SENTINELS,
    NDVI_PRODUCT_SENTINELS,
    PROCESSED_SENTINELS,
    _capture_output_name,
    _is_processed_output,
    _process_run,
    _write_candidates_for_output,
)
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


def test_ndvi_product_set_skips_bulky_reflectance_tiffs(tmp_path: Path) -> None:
    red = np.full((4, 4), 0.2, np.float32)
    green = np.full((4, 4), 0.3, np.float32)
    ir = np.full((4, 4), 0.8, np.float32)
    rgb = np.dstack([red, green, ir])
    save_reflectance_products(tmp_path, red, green, ir, -0.2, 0.7, rgb=rgb, product_set="ndvi")
    for name in NDVI_PRODUCT_SENTINELS:
        assert (tmp_path / name).is_file()
    for name in (
        "red_reflectance.tiff",
        "green_reflectance.tiff",
        "ir_reflectance.tiff",
        "rgn_reflectance.tiff",
    ):
        assert not (tmp_path / name).exists()


def test_raw_capture_output_name_uses_trimmed_timestamp() -> None:
    raw = Path("1600x1400_Bit12_RGB-IR_RGB-IRMod(GRIG)_2026-05-02_19_36_16.raw")
    assert _capture_output_name(raw) == "2026-05-02_19_36_16"


def test_raw_capture_output_name_falls_back_to_stem() -> None:
    assert _capture_output_name(Path("capture.raw")) == "capture"


def test_processed_output_requires_expected_artifacts(tmp_path: Path) -> None:
    assert not _is_processed_output(tmp_path)
    for name in PROCESSED_SENTINELS:
        (tmp_path / name).write_bytes(b"x")
    assert _is_processed_output(tmp_path)


def test_processed_output_uses_product_set_specific_sentinels(tmp_path: Path) -> None:
    for name in NDVI_PRODUCT_SENTINELS:
        (tmp_path / name).write_bytes(b"x")
    assert _is_processed_output(tmp_path, product_set="ndvi")
    assert not _is_processed_output(tmp_path, product_set="full")


def test_write_candidates_for_output_generates_and_skips(tmp_path: Path) -> None:
    ndvi = np.full((220, 360), 0.24, np.float32)
    ndvi[:, :110] = 0.65
    ndvi[:, 250:] = 0.02
    tifffile.imwrite(tmp_path / "ndvi.tiff", ndvi)

    generated = _write_candidates_for_output(tmp_path)
    assert generated["status"] == "generated"
    assert generated["candidate_count"] >= 2
    for name in CANDIDATE_SENTINELS:
        assert (tmp_path / name).is_file()

    skipped = _write_candidates_for_output(tmp_path)
    assert skipped["status"] == "skipped"
    forced = _write_candidates_for_output(tmp_path, force=True)
    assert forced["status"] == "generated"


def test_process_run_ndvi_product_set_writes_candidates_next_to_ndvi(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "1600x1400_Bit12_RGB-IR_RGB-IRMod(GRIG)_2026-06-06_20_17_03.raw"
    raw_path.write_bytes(b"raw")
    model_path = tmp_path / "model.yaml"
    model_path.write_text("model", encoding="utf-8")
    run_dir = tmp_path / "runs" / "survey"

    monkeypatch.setattr("mira220.cli.is_compatible_raw", lambda path: True)
    monkeypatch.setattr(
        "mira220.cli.load_yaml",
        lambda path: {
            "preprocessing": {"sensor_bit_depth": 12},
            "display": {"ndvi_min": -0.2, "ndvi_max": 0.7},
        },
    )
    monkeypatch.setattr(
        "mira220.cli.run_openrgbir",
        lambda raw_path, output_dir, repo, config, keep_intermediates=False: (
            np.dstack(
                [
                    np.full((180, 240), 820, np.uint16),
                    np.full((180, 240), 1220, np.uint16),
                    np.full((180, 240), 900, np.uint16),
                ]
            ),
            np.full((180, 240), 2800, np.uint16),
        ),
    )
    monkeypatch.setattr(
        "mira220.cli.apply_model",
        lambda rgb, ir, model: (
            np.where(np.indices(ir.shape)[1] < 80, 0.12, np.where(np.indices(ir.shape)[1] > 160, 0.85, 0.3)).astype(
                np.float32
            ),
            np.full(ir.shape, 0.3, np.float32),
            np.where(np.indices(ir.shape)[1] < 80, 0.82, np.where(np.indices(ir.shape)[1] > 160, 0.15, 0.4)).astype(
                np.float32
            ),
        ),
    )

    summary = _process_run(
        raw_dir,
        model_path,
        run_dir,
        False,
        tmp_path,
        tmp_path,
        False,
        product_set="ndvi",
        detect_candidates_flag=True,
    )
    output_dir = run_dir / "2026-06-06_20_17_03"
    for name in NDVI_PRODUCT_SENTINELS + CANDIDATE_SENTINELS:
        assert (output_dir / name).is_file()
    assert not (output_dir / "red_reflectance.tiff").exists()
    assert summary["product_set"] == "ndvi"
    assert len(summary["processed"]) == 1
    assert len(summary["candidate_generated"]) == 1


def test_process_run_generates_missing_candidates_for_already_processed_capture(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "1600x1400_Bit12_RGB-IR_RGB-IRMod(GRIG)_2026-06-06_20_05_23.raw"
    raw_path.write_bytes(b"raw")
    run_dir = tmp_path / "runs" / "survey"
    output_dir = run_dir / "2026-06-06_20_05_23"
    output_dir.mkdir(parents=True)
    ndvi = np.full((180, 240), 0.24, np.float32)
    ndvi[:, :80] = 0.65
    ndvi[:, 160:] = 0.02
    tifffile.imwrite(output_dir / "ndvi.tiff", ndvi)
    for name in NDVI_PRODUCT_SENTINELS:
        if name != "ndvi.tiff":
            (output_dir / name).write_bytes(b"x")

    monkeypatch.setattr("mira220.cli.is_compatible_raw", lambda path: True)
    monkeypatch.setattr(
        "mira220.cli.load_yaml",
        lambda path: {
            "preprocessing": {"sensor_bit_depth": 12},
            "display": {"ndvi_min": -0.2, "ndvi_max": 0.7},
        },
    )
    monkeypatch.setattr(
        "mira220.cli.run_openrgbir",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("already processed RAW should be skipped")),
    )

    summary = _process_run(
        raw_dir,
        tmp_path / "model.yaml",
        run_dir,
        False,
        tmp_path,
        tmp_path,
        False,
        product_set="ndvi",
        detect_candidates_flag=True,
    )
    for name in CANDIDATE_SENTINELS:
        assert (output_dir / name).is_file()
    assert len(summary["processed"]) == 0
    assert len(summary["skipped"]) == 1
    assert len(summary["candidate_generated"]) == 1
    assert summary["skipped"][0]["candidate_result"]["status"] == "generated"


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

