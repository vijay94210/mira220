from __future__ import annotations

"""Session-level adjustment helpers.

A session is a group of captures taken under similar field conditions. This
module measures the calibration target across the session, fits a simple channel
adjustment, and writes adjusted outputs when enough target data is available.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

from .correction import write_yaml
from .imaging import detect_marker, patch_polygons, polygon_median, save_reflectance_products


CHANNELS = ("red", "green", "ir")


def discover_reflectance_outputs(root: Path) -> list[Path]:
    # A usable processed capture must have the combined R/G/N TIFF and each
    # separate reflectance channel.
    return sorted(
        path.parent
        for path in root.rglob("rgn_reflectance.tiff")
        if all((path.parent / f"{channel}_reflectance.tiff").is_file() for channel in CHANNELS)
    )


def measure_output_patches(output_dir: Path, target: dict) -> list[dict]:
    # Measure Red, Green, and IR medians inside each target patch for one capture.
    rgn = tifffile.imread(output_dir / "rgn_reflectance.tiff").astype(np.float32)
    marker = detect_marker(rgn, int(target["marker_id"]), target["dictionary"])
    polygons = patch_polygons(marker, target)
    values = []
    for patch, polygon in zip(target["patches"], polygons):
        # Median is used because it is less affected by a few noisy pixels.
        median = polygon_median(rgn, polygon)
        values.append(
            {
                "name": patch["name"],
                "red": float(median[0]),
                "green": float(median[1]),
                "ir": float(median[2]),
            }
        )
    return values


def aggregate_session_patches(
    run_root: Path,
    target: dict,
    require_measurements: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    # Collect target patch measurements from every processed capture in a run.
    # Captures without a detectable marker are recorded as failures, but they do
    # not stop the whole session unless measurements are required.
    measurements = []
    failures = []
    for output_dir in discover_reflectance_outputs(run_root):
        try:
            measurements.append(
                {"output_directory": str(output_dir.resolve()), "patches": measure_output_patches(output_dir, target)}
            )
        except Exception as exc:
            failures.append({"output_directory": str(output_dir.resolve()), "error": str(exc)})
    if not measurements and require_measurements:
        raise ValueError("No session capture produced usable ArUco patch measurements.")
    if not measurements:
        return [], measurements, failures
    patch_names = [patch["name"] for patch in target["patches"]]
    aggregated = []
    for patch_name in patch_names:
        # Use the session median for each patch/channel. That gives one stable
        # observed value per patch for fitting the adjustment.
        row = {"name": patch_name}
        for channel in CHANNELS:
            row[channel] = float(
                np.median(
                    [
                        next(patch for patch in measurement["patches"] if patch["name"] == patch_name)[channel]
                        for measurement in measurements
                    ]
                )
            )
        aggregated.append(row)
    return aggregated, measurements, failures


def fit_session_adjustment(
    aggregated: list[dict],
    reference: dict,
) -> dict:
    # Fit a straight-line correction for each channel:
    #
    #   corrected = gain * observed + offset
    #
    # This is simple, explainable, and works with a small number of patches.
    reference_by_name = {patch["name"]: patch for patch in reference["patches"]}
    channels = {}
    for channel in CHANNELS:
        observed = np.asarray([patch[channel] for patch in aggregated], np.float64)
        targets = np.asarray([reference_by_name[patch["name"]][channel] for patch in aggregated], np.float64)
        if observed.size < 3:
            # A line fit is too fragile with fewer than three patch values.
            raise ValueError(
                f"Cannot fit {channel} session adjustment with fewer than 3 usable patch measurements."
            )
        if np.ptp(observed) < 1e-6:
            raise ValueError(f"Cannot fit {channel} session adjustment from degenerate patch measurements.")
        gain, offset = np.polyfit(observed, targets, 1)
        predicted = gain * observed + offset
        channels[channel] = {
            "gain": float(gain),
            "offset": float(offset),
            "rmse": float(np.sqrt(np.mean(np.square(predicted - targets)))),
            "max_abs_residual": float(np.max(np.abs(predicted - targets))),
            "usable_patch_count": int(observed.size),
            "usable_patches": [patch["name"] for patch in aggregated],
        }
    return {"method": "per-channel affine", "channels": channels}


def apply_affine(image: np.ndarray, adjustment: dict) -> np.ndarray:
    # Apply the gain/offset correction to every pixel and keep values in range.
    return np.clip(
        adjustment["gain"] * np.asarray(image, np.float32) + adjustment["offset"],
        0.0,
        1.0,
    )


def summarize_clipping(root: Path) -> dict:
    # Clipping tells us how many pixels hit exactly 0 or 1 after correction.
    captures = []
    for output_dir in discover_reflectance_outputs(root):
        clipping = {}
        for channel in CHANNELS:
            image = tifffile.imread(output_dir / f"{channel}_reflectance.tiff")
            clipping[channel] = float(np.mean((image <= 0) | (image >= 1)))
        captures.append({"output_directory": str(output_dir.resolve()), "clipping_fraction": clipping})
    if not captures:
        return {"capture_count": 0, "mean_clipping_fraction": {}, "captures": []}
    return {
        "capture_count": len(captures),
        "mean_clipping_fraction": {
            # Average the clipping rate across all captures for each channel.
            channel: float(np.mean([capture["clipping_fraction"][channel] for capture in captures]))
            for channel in CHANNELS
        },
        "captures": captures,
    }


def adjust_session(
    input_root: Path,
    output_root: Path,
    target: dict,
    reference: dict,
    display: dict,
    model_path: Path,
) -> dict:
    # Main session adjustment workflow. It tries to measure target patches,
    # compute an adjustment, apply that adjustment to every capture, and write a
    # report of what happened.
    aggregated, measurements, failures = aggregate_session_patches(input_root, target, require_measurements=False)
    report = {
        "model": str(model_path.resolve()),
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "target_reference": reference["name"],
        "mapir_used_for_fitting": False,
        "session_patch_medians": aggregated,
        "capture_measurements": measurements,
        "measurement_failures": failures,
        "adjusted": False,
        "adjustment": None,
        "processed": [],
    }
    if not measurements:
        # If the marker was never found, leave the session unadjusted and explain
        # why in the report.
        report["skip_reason"] = "No capture produced usable ArUco patch measurements."
        return report

    adjustment = fit_session_adjustment(aggregated, reference)
    processed = []
    for input_dir in discover_reflectance_outputs(input_root):
        # Keep the same relative folder structure in the adjusted output tree.
        relative = input_dir.relative_to(input_root)
        output_dir = output_root / relative
        channels = {
            channel: tifffile.imread(input_dir / f"{channel}_reflectance.tiff").astype(np.float32)
            for channel in CHANNELS
        }
        adjusted = {
            # Apply the per-channel adjustment learned from the target patches.
            channel: apply_affine(channels[channel], adjustment["channels"][channel])
            for channel in CHANNELS
        }
        save_reflectance_products(
            output_dir,
            adjusted["red"],
            adjusted["green"],
            adjusted["ir"],
            display["ndvi_min"],
            display["ndvi_max"],
        )
        rgb_preview = input_dir / "rgb_preview.png"
        if rgb_preview.is_file():
            # The visible RGB preview is not recalculated here, so copy it from
            # the original processed output when available.
            shutil.copy2(rgb_preview, output_dir / "rgb_preview.png")
        processed.append(
            {
                "input_directory": str(input_dir.resolve()),
                "output_directory": str(output_dir.resolve()),
                "clipping_fraction": {
                    channel: float(np.mean((adjusted[channel] <= 0) | (adjusted[channel] >= 1)))
                    for channel in CHANNELS
                },
            }
        )
    report["adjusted"] = True
    report["adjustment"] = adjustment
    report["processed"] = processed
    output_root.mkdir(parents=True, exist_ok=True)
    # Save both YAML and JSON because YAML is pleasant for people and JSON is
    # easy for programs to read.
    write_yaml(output_root / "session_adjustment.yaml", report)
    (output_root / "session_adjustment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
