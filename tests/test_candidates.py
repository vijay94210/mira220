from __future__ import annotations

# Tests for NDVI candidate detection and candidate output files. These checks
# cover classification, artifact rejection, target exclusion, CSVs, masks, and
# overlays.

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

import mira220.candidates as candidate_module
from mira220.candidates import (
    CSV_FIELDS,
    CandidateConfig,
    build_class_masks,
    detect_candidates,
    load_ndvi_source,
    robust_normalize,
    summarize_candidates,
    target_exclusion_from_source,
    write_candidate_mask,
    write_candidate_overlay,
    write_candidates_csv,
)
from mira220.cli import build_parser
from mira220.imaging import NDVI_CROP_MARGINS, crop_valid_region, ndvi_false_color


def _scene(shape: tuple[int, int] = (240, 300), value: float = 0.25) -> np.ndarray:
    return np.full(shape, value, np.float32)


def test_processed_directory_loading_uses_cropped_ndvi_and_crop_background(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    output.mkdir()
    ndvi = np.full((1400, 1600), 0.25, np.float32)
    ndvi[300:900, 350:800] = 0.65
    tifffile.imwrite(output / "ndvi.tiff", ndvi)
    cropped = crop_valid_region(ndvi)
    crop_background = cv2.cvtColor(ndvi_false_color(cropped, -0.2, 0.7), cv2.COLOR_RGB2BGR)
    full_background = cv2.cvtColor(ndvi_false_color(ndvi, -0.2, 0.7), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output / "ndvi_false_color.png"), full_background)
    cv2.imwrite(str(output / "ndvi_false_color_crop.png"), crop_background)
    source = load_ndvi_source(output)
    assert source.crop_margins == NDVI_CROP_MARGINS
    assert source.ndvi.shape == cropped.shape
    assert source.background.shape[:2] == cropped.shape


def test_direct_tiff_and_png_inputs_load(tmp_path: Path) -> None:
    ndvi = _scene((80, 100), 0.3)
    ndvi[:, :40] = 0.7
    tif_path = tmp_path / "ndvi.tiff"
    png_dir = tmp_path / "png-only"
    png_dir.mkdir()
    png_path = png_dir / "ndvi.png"
    tifffile.imwrite(tif_path, ndvi)
    cv2.imwrite(str(png_path), np.round(robust_normalize(ndvi) * 255).astype(np.uint8))
    assert load_ndvi_source(tif_path, crop=False).physical_ndvi
    assert not load_ndvi_source(png_path, crop=False).physical_ndvi


def test_threshold_modes_produce_expected_masks() -> None:
    ndvi = _scene((120, 160), 0.1)
    ndvi[:, 50:110] = 0.65
    manual = build_class_masks(ndvi, CandidateConfig(crop=False, threshold_mode="manual", manual_threshold=0.5, min_area=50))
    otsu = build_class_masks(ndvi, CandidateConfig(crop=False, threshold_mode="otsu", min_area=50))
    assert manual["high_vegetation"].any()
    assert otsu["high_vegetation"].any()


def test_speckles_removed_and_small_holes_filled() -> None:
    ndvi = _scene((180, 220), 0.05)
    ndvi[30:150, 40:180] = 0.65
    ndvi[80:92, 100:112] = 0.05
    ndvi[10:14, 10:14] = 0.65
    masks = build_class_masks(
        ndvi,
        CandidateConfig(crop=False, min_area=800, morph_open=3, morph_close=9, hole_fill_area=500),
    )
    high = masks["high_vegetation"]
    assert high[86, 106] > 0
    assert high[12, 12] == 0


def test_broad_left_middle_right_regions_are_detected() -> None:
    ndvi = _scene((220, 360), 0.24)
    ndvi[:, :110] = 0.68
    ndvi[:, 235:] = 0.62
    candidates, _ = detect_candidates(
        ndvi,
        CandidateConfig(crop=False, min_area=3000, morph_open=3, morph_close=15),
    )
    broad = sorted(
        [candidate for candidate in candidates if candidate["candidate_type"] == "broad_ndvi_region"],
        key=lambda candidate: candidate["bbox_x"],
    )
    assert len(broad) >= 3
    assert broad[0]["bbox_x"] < 25
    assert 95 <= broad[1]["bbox_x"] <= 125
    assert broad[-1]["bbox_x"] >= 220


def test_irregular_region_produces_simplified_contour() -> None:
    ndvi = _scene((220, 260), 0.05)
    polygon = np.array([[30, 40], [115, 20], [210, 80], [170, 190], [60, 180]], np.int32)
    cv2.fillPoly(ndvi, [polygon], 0.7)
    candidates, _ = detect_candidates(
        ndvi,
        CandidateConfig(crop=False, min_area=1000, geometry="contour", simplify_epsilon=8.0),
    )
    contour = next(candidate for candidate in candidates if candidate["candidate_type"] == "broad_ndvi_region")
    assert contour["geometry_type"] == "contour"
    assert len(contour["points"]) < 15
    assert contour["area_pixels"] > 10000


def test_texture_region_detected_separately() -> None:
    ndvi = _scene((180, 220), 0.08)
    patch = np.indices((110, 130)).sum(axis=0) % 12
    ndvi[35:145, 45:175] = np.where(patch < 6, 0.16, 0.33).astype(np.float32)
    candidates, _ = detect_candidates(
        ndvi,
        CandidateConfig(
            crop=False,
            min_area=800,
            min_contrast=0.5,
            texture_sensitivity=0.12,
            morph_open=3,
            morph_close=11,
        ),
    )
    assert any(candidate["candidate_type"] == "texture_region" for candidate in candidates)


def test_exclusion_mask_removes_target_candidate_but_keeps_vegetation() -> None:
    ndvi = _scene((180, 240), 0.04)
    ndvi[30:120, 30:105] = 0.65
    ndvi[40:105, 150:215] = 0.65
    exclusion = np.zeros(ndvi.shape, np.uint8)
    exclusion[36:112, 142:222] = 255
    candidates, _ = detect_candidates(
        ndvi,
        CandidateConfig(crop=False, min_area=1000, morph_open=3, morph_close=9),
        exclusion_mask=exclusion,
    )
    assert candidates
    assert all(candidate["centroid_x"] < 130 for candidate in candidates)


def test_linear_artifacts_are_not_emitted() -> None:
    ndvi = _scene((180, 240), 0.25)
    cv2.line(ndvi, (20, 150), (220, 30), 0.75, 3)
    candidates, _ = detect_candidates(
        ndvi,
        CandidateConfig(crop=False, min_area=1000),
    )
    assert all(candidate["candidate_type"] != "linear_artifact" for candidate in candidates)


def test_geometric_box_artifacts_are_not_emitted() -> None:
    ndvi = _scene((180, 240), 0.35)
    ndvi[55:125, 70:150] = 0.02
    candidates, _ = detect_candidates(
        ndvi,
        CandidateConfig(crop=False, min_area=1500, geometry_min_solidity=0.75),
    )
    assert all(candidate["candidate_type"] != "geometric_object" for candidate in candidates)


def test_uniform_scene_does_not_produce_full_frame_candidate() -> None:
    candidates, _ = detect_candidates(
        _scene((120, 160), 0.65),
        CandidateConfig(crop=False, min_area=1000),
    )
    assert candidates == []


def test_csv_and_candidate_mask_ids_match(tmp_path: Path) -> None:
    candidates = [
        {
            "id": 1,
            "candidate_type": "broad_ndvi_region",
            "ndvi_class": "high_vegetation",
            "geometry_type": "box",
            "bbox_x": 10,
            "bbox_y": 12,
            "bbox_width": 20,
            "bbox_height": 18,
            "centroid_x": 20,
            "centroid_y": 21,
            "area_pixels": 360,
            "perimeter_pixels": 76,
            "mean_ndvi": 0.6,
            "median_ndvi": 0.6,
            "min_ndvi": 0.5,
            "max_ndvi": 0.7,
            "std_ndvi": 0.1,
            "contrast_score": 0.2,
            "aspect_ratio": 1.1,
            "solidity": 0.9,
            "circularity": 0.7,
            "extent": 0.8,
            "texture_score": 0.1,
            "line_length": 0,
            "line_angle_degrees": 0,
            "total_score": 400,
            "points": [[10, 12], [29, 12], [29, 29], [10, 29]],
        }
    ]
    csv_path = tmp_path / "candidates.csv"
    mask_path = tmp_path / "candidate_mask.png"
    write_candidates_csv(csv_path, candidates)
    write_candidate_mask(mask_path, (50, 60), candidates)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CSV_FIELDS
        row = next(reader)
        assert row["candidate_type"] == "broad_ndvi_region"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    assert int(mask[20, 20]) == 1


def test_candidate_overlay_adds_id_labels_and_legends(tmp_path: Path) -> None:
    background = np.zeros((80, 100, 3), np.uint8)
    candidates = [
        {
            "id": 1,
            "candidate_type": "broad_ndvi_region",
            "ndvi_class": "high_vegetation",
            "geometry_type": "box",
            "bbox_x": 10,
            "bbox_y": 12,
            "bbox_width": 30,
            "bbox_height": 30,
            "centroid_x": 25,
            "centroid_y": 27,
            "points": [[10, 12], [39, 12], [39, 41], [10, 41]],
        }
    ]
    path = tmp_path / "candidate_overlay.png"
    write_candidate_overlay(path, background, candidates, CandidateConfig(box_thickness=3))
    overlay = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert overlay.shape[0] == background.shape[0]
    assert overlay.shape[1] > background.shape[1]
    assert np.any(overlay[:, background.shape[1] :, :] != 255)
    assert np.any(overlay[15:40, 15:40, :] > 0)


def test_target_exclusion_metadata_reports_detected_marker(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "capture"
    output.mkdir()
    ndvi = np.full((80, 100), 0.25, np.float32)
    tifffile.imwrite(output / "ndvi.tiff", ndvi)
    tifffile.imwrite(output / "rgn_reflectance.tiff", np.dstack([ndvi, ndvi, ndvi]))
    target = {"marker_id": 830, "dictionary": "DICT_4X4_1000", "patches": [{"name": "P1"}]}
    marker = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], np.float32)
    patch = np.array([[50, 20], [70, 20], [70, 40], [50, 40]], np.float32)
    monkeypatch.setattr(candidate_module, "detect_marker", lambda image, marker_id, dictionary: marker)
    monkeypatch.setattr(candidate_module, "patch_polygons", lambda corners, target_config: [patch])

    source = load_ndvi_source(output, crop=False)
    exclusion = target_exclusion_from_source(source, target, padding=0)
    summary = summarize_candidates(source, output, CandidateConfig(crop=False), [], exclusion.metadata)

    assert exclusion.mask is not None
    assert exclusion.metadata["detected"] is True
    assert exclusion.metadata["marker_id"] == 830
    assert summary["target_exclusion"]["excluded_pixels"] > 0


def test_target_exclusion_metadata_reports_marker_failure(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "capture"
    output.mkdir()
    ndvi = np.full((80, 100), 0.25, np.float32)
    tifffile.imwrite(output / "ndvi.tiff", ndvi)
    tifffile.imwrite(output / "rgn_reflectance.tiff", np.dstack([ndvi, ndvi, ndvi]))
    target = {"marker_id": 830, "dictionary": "DICT_4X4_1000", "patches": []}
    monkeypatch.setattr(
        candidate_module,
        "detect_marker",
        lambda image, marker_id, dictionary: (_ for _ in ()).throw(RuntimeError("marker missing")),
    )

    source = load_ndvi_source(output, crop=False)
    exclusion = target_exclusion_from_source(source, target)

    assert exclusion.mask is None
    assert exclusion.metadata["detected"] is False
    assert exclusion.metadata["reason"] == "marker missing"


def test_cli_detect_candidates_writes_all_outputs(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    output.mkdir()
    ndvi = _scene((180, 240), 0.24)
    ndvi[:, :80] = 0.65
    ndvi[:, 160:] = 0.02
    tifffile.imwrite(output / "ndvi.tiff", ndvi)
    args = build_parser().parse_args(
        [
            "detect-candidates",
            str(output),
            "--output-dir",
            str(tmp_path / "candidate-output"),
            "--no-crop",
            "--min-area",
            "1000",
        ]
    )
    assert args.handler(args) == 0
    assert (tmp_path / "candidate-output" / "candidate_overlay.png").is_file()
    assert (tmp_path / "candidate-output" / "candidate_mask.png").is_file()
    assert (tmp_path / "candidate-output" / "candidates.csv").is_file()
    summary = json.loads((tmp_path / "candidate-output" / "candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_count"] >= 2
    assert "counts_by_type" in summary
    assert "candidate_colors" in summary
