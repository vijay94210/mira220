from pathlib import Path

# Tests for Mira-to-gold comparison and pair ranking. These checks cover marker
# alignment, regression metrics, output discovery, and partial batch failures.

import cv2
import numpy as np
import pytest
import tifffile

from mira220.comparison import (
    align_pair,
    compare_all,
    block_regression_metrics,
    block_regression_values,
    discover_mira_outputs,
    load_mira_output,
    load_rgn_tiff,
    regression_metrics,
)
from mira220.correction import load_yaml
from mira220.imaging import compute_ndvi


TARGET = load_yaml(Path("config/targets/aruco830_flat_patches.yaml"))


def _scene() -> tuple[np.ndarray, np.ndarray]:
    height, width = 420, 520
    yy, xx = np.mgrid[:height, :width]
    red = 0.15 + 0.45 * xx / width
    green = 0.2 + 0.35 * yy / height
    nir = 0.55 + 0.25 * yy / height
    rgn = np.dstack([red, green, nir]).astype(np.float32)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    marker = cv2.aruco.generateImageMarker(dictionary, 830, 120)
    marker_rgb = np.repeat((marker.astype(np.float32) / 255.0)[..., None], 3, axis=2)
    rgn[130:250, 190:310] = marker_rgb
    return rgn, compute_ndvi(rgn[..., 0], rgn[..., 2])


def _write_mira_output(path: Path, rgn: np.ndarray, ndvi: np.ndarray) -> None:
    path.mkdir(parents=True)
    tifffile.imwrite(path / "rgn_reflectance.tiff", rgn)
    tifffile.imwrite(path / "ndvi.tiff", ndvi.astype(np.float32))


def test_tiff_discovery_and_validation(tmp_path: Path) -> None:
    rgn, ndvi = _scene()
    output = tmp_path / "nested" / "capture"
    _write_mira_output(output, rgn, ndvi)
    assert discover_mira_outputs(tmp_path) == [output]
    loaded_ndvi, loaded_rgn = load_mira_output(output)
    assert loaded_ndvi.shape == ndvi.shape
    assert loaded_rgn.shape == rgn.shape
    invalid = tmp_path / "invalid.tiff"
    tifffile.imwrite(invalid, np.zeros((10, 10), np.float32))
    with pytest.raises(ValueError, match="three-channel"):
        load_rgn_tiff(invalid)


def test_regression_metrics_known_relationship() -> None:
    gold = np.linspace(-0.5, 0.8, 100, dtype=np.float32)
    mira = 0.9 * gold + 0.05
    metrics = regression_metrics(gold, mira)
    assert metrics["slope"] == pytest.approx(0.9, abs=1e-6)
    assert metrics["intercept"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["r_squared"] == pytest.approx(1.0)


def test_block_regression_metrics() -> None:
    gold = np.arange(64, dtype=np.float32).reshape(8, 8) / 64
    mira = gold * 0.9 + 0.05
    metrics = block_regression_metrics(gold, mira, np.ones_like(gold, bool), block_size=4)
    assert metrics["block_size"] == 4
    assert metrics["pearson_correlation"] == pytest.approx(1.0)


def test_block_regression_values_returns_block_averages() -> None:
    gold = np.arange(64, dtype=np.float32).reshape(8, 8) / 64
    mira = gold * 0.9 + 0.05
    gold_values, mira_values = block_regression_values(
        gold, mira, np.ones_like(gold, bool), block_size=4
    )
    assert gold_values.shape == (4,)
    np.testing.assert_allclose(mira_values, gold_values * 0.9 + 0.05)


def test_block_regression_metrics_ignores_invalid_nan_pixels() -> None:
    gold = np.arange(64, dtype=np.float32).reshape(8, 8) / 64
    mira = gold * 0.9 + 0.05
    valid = np.ones_like(gold, bool)
    valid[0, 0] = False
    gold[0, 0] = np.nan
    mira[0, 0] = np.nan
    metrics = block_regression_metrics(gold, mira, valid, block_size=4)
    assert metrics["pearson_correlation"] == pytest.approx(1.0)


def test_aruco_homography_alignment() -> None:
    gold_rgn, gold_ndvi = _scene()
    source = np.array([[0, 0], [519, 0], [519, 419], [0, 419]], np.float32)
    destination = np.array([[25, 18], [500, 5], [510, 405], [12, 415]], np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    mira_rgn = cv2.warpPerspective(gold_rgn, transform, (520, 420))
    mira_ndvi = cv2.warpPerspective(gold_ndvi, transform, (520, 420))
    result = align_pair(mira_ndvi, mira_rgn, gold_rgn, 830, "DICT_4X4_1000")
    valid = result["valid_mask"]
    error = np.mean(np.abs(result["aligned_mira_ndvi"][valid] - result["gold_ndvi"][valid]))
    assert error < 0.015
    assert result["scene_similarity"] > 0.95


def test_compare_all_records_partial_failure(tmp_path: Path) -> None:
    rgn, ndvi = _scene()
    mira_root = tmp_path / "mira"
    gold_root = tmp_path / "gold"
    _write_mira_output(mira_root / "capture", rgn, ndvi)
    gold_root.mkdir()
    tifffile.imwrite(gold_root / "valid.tiff", np.round(rgn * 65535).astype(np.uint16))
    tifffile.imwrite(gold_root / "invalid.tiff", np.zeros((20, 20), np.uint16))
    summary = compare_all(mira_root, gold_root, tmp_path / "comparisons", TARGET)
    assert summary["successful_pair_count"] == 1
    assert summary["failed_pair_count"] == 1
    assert (tmp_path / "comparisons" / "pair_rankings.csv").is_file()
    assert (tmp_path / "comparisons" / "comparison_summary.json").is_file()
    report = summary["pairs"][1]
    assert report["block_size"] == 16
    assert report["aggregation"] == "valid-pixel-weighted block mean"
    assert report["valid_block_count"] > 0
    assert "pixel_pearson_correlation" in report
