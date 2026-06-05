from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile

from .correction import apply_model
from .imaging import (
    compute_ndvi,
    detect_marker,
    draw_patch_overlay,
    normalize_sensor,
    patch_polygons,
    polygon_median,
    write_rgb_preview,
)
from .openrgbir import run_openrgbir


def fit_flat_patch_model(
    mira_raw: Path,
    gold_tiff: Path,
    output_dir: Path,
    openrgbir_repo: Path,
    isp_config: Path,
    target: dict,
    keep_intermediates: bool = False,
) -> dict:
    raw_rgb, raw_ir = run_openrgbir(
        mira_raw, output_dir, openrgbir_repo, isp_config, keep_intermediates=keep_intermediates
    )
    mira_rgb = normalize_sensor(raw_rgb)
    mira_ir = normalize_sensor(raw_ir)
    gold = tifffile.imread(gold_tiff).astype(np.float32) / 65535.0
    marker_id = int(target["marker_id"])
    dictionary = target["dictionary"]
    mira_marker = detect_marker(mira_rgb, marker_id, dictionary)
    gold_marker = detect_marker(gold, marker_id, dictionary)
    mira_polygons = patch_polygons(mira_marker, target)
    gold_polygons = patch_polygons(gold_marker, target)
    mira_values = np.array(
        [
            [
                polygon_median(mira_rgb[..., 0], polygon),
                polygon_median(mira_rgb[..., 1], polygon),
                polygon_median(mira_ir, polygon),
            ]
            for polygon in mira_polygons
        ],
        dtype=np.float32,
    )
    gold_values = np.array([polygon_median(gold, polygon) for polygon in gold_polygons], dtype=np.float32)
    shared_targets = gold_values.mean(axis=1)
    red_design = np.column_stack(
        [mira_values[:, 0], mira_values[:, 2], np.square(mira_values[:, 2]), np.ones(4)]
    )
    green_design = np.column_stack(
        [mira_values[:, 1], mira_values[:, 2], np.square(mira_values[:, 2]), np.ones(4)]
    )
    ir_design = np.column_stack([mira_values[:, 2], np.square(mira_values[:, 2]), np.ones(4)])
    model = {
        "name": "flat_patch_v1",
        "description": "Post-demosaic quadratic R/G/IR correction anchored to flat-spectrum patches.",
        "preprocessing": {
            "irc_cut": 0.0,
            "awb": False,
            "ccm": False,
            "sensor_bit_depth": 12,
        },
        "provenance": {
            "mira_raw": str(mira_raw.resolve()),
            "gold_tiff": str(gold_tiff.resolve()),
            "gold_channels": {"red": 0, "green": 1, "ir": 2},
        },
        "display": {"ndvi_min": -0.2, "ndvi_max": 0.7},
        "patch_targets": [
            {"name": patch["name"], "reflectance": float(value)}
            for patch, value in zip(target["patches"], shared_targets)
        ],
        "correction": {
            "red": {
                "features": ["raw_red", "raw_ir", "raw_ir_squared", "constant"],
                "coefficients": np.linalg.lstsq(red_design, shared_targets, rcond=None)[0].tolist(),
            },
            "green": {
                "features": ["raw_green", "raw_ir", "raw_ir_squared", "constant"],
                "coefficients": np.linalg.lstsq(green_design, shared_targets, rcond=None)[0].tolist(),
            },
            "ir": {
                "features": ["raw_ir", "raw_ir_squared", "constant"],
                "coefficients": np.linalg.lstsq(ir_design, shared_targets, rcond=None)[0].tolist(),
            },
        },
    }
    red, green, ir = apply_model(mira_rgb, mira_ir, model)
    patch_results = []
    for patch, polygon in zip(target["patches"], mira_polygons):
        rv = float(polygon_median(red, polygon))
        gv = float(polygon_median(green, polygon))
        iv = float(polygon_median(ir, polygon))
        patch_results.append(
            {
                "name": patch["name"],
                "red": rv,
                "green": gv,
                "ir": iv,
                "ndvi": float((iv - rv) / (iv + rv + 1e-6)),
            }
        )
    model["validation"] = {"patch_results": patch_results, "max_abs_patch_ndvi": max(abs(x["ndvi"]) for x in patch_results)}
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "patches_annotated.png"), draw_patch_overlay(gold, gold_marker, gold_polygons, target))
    write_rgb_preview(output_dir / "gold_rgn_preview.png", gold)
    write_rgb_preview(output_dir / "mira_rgn_preview.png", np.dstack([red, green, ir]))
    ndvi = compute_ndvi(red, ir)
    tifffile.imwrite(output_dir / "ndvi_preview_source.tiff", ndvi.astype(np.float32))
    return model


def inspect_products(run_dir: Path, target: dict, model: dict) -> dict:
    red = tifffile.imread(run_dir / "red_reflectance.tiff")
    green = tifffile.imread(run_dir / "green_reflectance.tiff")
    ir = tifffile.imread(run_dir / "ir_reflectance.tiff")
    rgn = np.dstack([red, green, ir])
    marker = detect_marker(rgn, int(target["marker_id"]), target["dictionary"])
    polygons = patch_polygons(marker, target)
    patches = []
    for patch, polygon in zip(target["patches"], polygons):
        rv = float(polygon_median(red, polygon))
        gv = float(polygon_median(green, polygon))
        iv = float(polygon_median(ir, polygon))
        patches.append(
            {
                "name": patch["name"],
                "red": rv,
                "green": gv,
                "ir": iv,
                "ndvi": float((iv - rv) / (iv + rv + 1e-6)),
            }
        )
    return {
        "run_directory": str(run_dir.resolve()),
        "model": model["name"],
        "patches": patches,
        "clipping_fraction": {
            "red": float(np.mean((red <= 0) | (red >= 1))),
            "green": float(np.mean((green <= 0) | (green >= 1))),
            "ir": float(np.mean((ir <= 0) | (ir >= 1))),
        },
    }

