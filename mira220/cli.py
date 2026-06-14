from __future__ import annotations

"""Command line entry points for the Mira220 pipeline.

Plain-English overview:

This file is the "control desk" for the project. A user types a command like
`python -m mira220 process`, and this file decides which function should run.

Most of the hard science and image math lives in other files. This file wires
those pieces together in the right order:

1. Read input files and settings.
2. Skip files that are already done or not usable.
3. Run image processing.
4. Save output images, tables, and summary JSON files.
5. Print short progress messages for the user.

The comments below are intentionally more detailed than normal production code
comments because this file may be read in a student report.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile

from . import (
    DEFAULT_DATA_ROOT,
    DEFAULT_ISP_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OPENRGBIR_REPO,
    DEFAULT_TARGET_PATH,
    DEFAULT_TARGET_REFERENCE_PATH,
    ROOT,
)
from .calibration import fit_flat_patch_model, inspect_products
from .candidates import (
    CandidateConfig,
    detect_candidates,
    load_ndvi_source,
    summarize_candidates,
    target_exclusion_from_source,
    write_candidate_overlay,
    write_candidate_mask,
    write_candidates_csv,
)
from .comparison import compare_all, compare_pair
from .correction import apply_model, apply_scene_correction, load_yaml, write_yaml
from .imaging import ndvi_false_color, normalize_sensor, save_reflectance_products
from .openrgbir import is_compatible_raw, run_openrgbir
from .session import adjust_session, summarize_clipping


# These filenames act like a checklist. If every file in the checklist exists,
# the program treats a capture as already processed. That prevents slow and
# expensive RAW conversion from running again by accident.
PROCESSED_SENTINELS = (
    "red_reflectance.tiff",
    "green_reflectance.tiff",
    "ir_reflectance.tiff",
    "ndvi.tiff",
    "ndvi_gray.png",
    "ndvi_false_color.png",
    "ndvi_false_color_crop.png",
    "rgb_preview.png",
    "rgn_preview.png",
    "rgn_reflectance.tiff",
)

# This shorter checklist is used when the user only wants NDVI review products.
# It leaves out the large per-channel reflectance TIFF files to save disk space.
NDVI_PRODUCT_SENTINELS = (
    "ndvi.tiff",
    "ndvi_gray.png",
    "ndvi_false_color.png",
    "ndvi_false_color_crop.png",
    "rgb_preview.png",
    "rgn_preview.png",
)

# Candidate detection creates these four review files. Together they show where
# the program found interesting NDVI regions and record those regions in formats
# that people and other programs can read.
CANDIDATE_SENTINELS = (
    "candidate_overlay.png",
    "candidate_mask.png",
    "candidates.csv",
    "candidate_summary.json",
)

# Mira RAW filenames usually contain a timestamp like:
#
#   2026-06-06_20_17_03
#
# This regular expression pulls out the date and time pieces so output folders
# and collages can be named consistently.
RAW_TIMESTAMP_RE = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})")

# If too many RAW pixels are already at the sensor's maximum value, the image is
# probably washed out. The pipeline skips those files instead of pretending they
# are reliable science data.
RAW_SATURATION_THRESHOLD_FRACTION = 0.05

# Simple colors used when building collage images. OpenCV stores images in BGR
# order, but these values are gray enough that the channel order does not matter.
COLLAGE_BACKGROUND = (245, 245, 245)
COLLAGE_TEXT = (30, 30, 30)


def _add_shared_paths(parser: argparse.ArgumentParser) -> None:
    # Several commands need the same paths: where openRGB-IR lives, which ISP
    # config to use, and where the calibration target config is stored. This
    # helper adds those options in one place so the commands stay consistent.
    parser.add_argument("--openrgbir-repo", type=Path, default=DEFAULT_OPENRGBIR_REPO)
    parser.add_argument("--isp-config", type=Path, default=DEFAULT_ISP_CONFIG_PATH)
    parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)


def _capture_output_name(raw_path: Path) -> str:
    # Turn a long RAW filename into a clean folder name. If the file has a
    # timestamp, keep only that timestamp. If it does not, use the file stem.
    match = RAW_TIMESTAMP_RE.search(raw_path.stem)
    if match is None:
        return raw_path.stem
    parts = match.groupdict()
    return (
        f"{parts['year']}-{parts['month']}-{parts['day']}_"
        f"{parts['hour']}_{parts['minute']}_{parts['second']}"
    )


def _capture_date(path: Path) -> str | None:
    # Collages are grouped by day. This searches a path and its parent folders
    # for a RAW-style timestamp, then returns just the date part.
    for part in (path.stem, *reversed(path.parts)):
        match = RAW_TIMESTAMP_RE.search(part)
        if match is not None:
            parts = match.groupdict()
            return f"{parts['year']}-{parts['month']}-{parts['day']}"
    return None


def _processed_sentinels(product_set: str) -> tuple[str, ...]:
    # Pick the right "already done" checklist for the requested output type.
    if product_set == "full":
        return PROCESSED_SENTINELS
    if product_set == "ndvi":
        return NDVI_PRODUCT_SENTINELS
    raise ValueError(f"Unsupported product set: {product_set}")


def _is_processed_output(output_dir: Path, product_set: str = "full") -> bool:
    # A capture is considered processed only when every expected output file is
    # present. This is stricter than checking for just one file, because partial
    # outputs can happen if a previous run was interrupted.
    return all((output_dir / name).is_file() for name in _processed_sentinels(product_set))


def _has_candidate_outputs(output_dir: Path) -> bool:
    # Same idea as _is_processed_output, but for the candidate-region products.
    return all((output_dir / name).is_file() for name in CANDIDATE_SENTINELS)


def _raw_saturation_summary(
    raw_path: Path,
    sensor_bit_depth: int,
    threshold_fraction: float = RAW_SATURATION_THRESHOLD_FRACTION,
) -> dict:
    # A 12-bit sensor has values from 0 to 4095. More generally, the maximum
    # value is 2^bit_depth - 1. Pixels at that maximum are clipped: the sensor
    # cannot tell how bright they really were.
    sensor_max = (1 << int(sensor_bit_depth)) - 1

    # RAW files are stored as uint16 numbers. Loading with np.fromfile is enough
    # here because we only need to count saturated values, not build an image.
    raw = np.fromfile(raw_path, dtype=np.uint16)
    saturated_fraction = float(np.mean(raw == sensor_max)) if raw.size else 0.0
    return {
        "saturated_fraction": saturated_fraction,
        "threshold_fraction": float(threshold_fraction),
        "sensor_max": int(sensor_max),
        "excessive": saturated_fraction > threshold_fraction,
    }


def _write_candidates_for_output(output_dir: Path, force: bool = False, target: dict | None = None) -> dict:
    # Candidate outputs can be reused. If they already exist and the user did not
    # ask to force regeneration, skip this capture and report why.
    if _has_candidate_outputs(output_dir) and not force:
        return {"output_directory": str(output_dir.resolve()), "status": "skipped", "reason": "already exists"}

    # Load the processed NDVI image and preview background from this capture.
    source = load_ndvi_source(output_dir)

    # If the calibration board is visible, build a mask around it. Candidate
    # detection should focus on real scene regions, not the target board.
    exclusion = target_exclusion_from_source(source, target)
    candidates, _ = detect_candidates(source, exclusion_mask=exclusion.mask)

    # Write the same candidate information in several useful forms:
    # an image overlay, an ID mask, a CSV table, and a JSON summary.
    write_candidate_overlay(output_dir / "candidate_overlay.png", source.background, candidates, CandidateConfig())
    write_candidate_mask(output_dir / "candidate_mask.png", source.ndvi.shape, candidates)
    write_candidates_csv(output_dir / "candidates.csv", candidates)
    summary = summarize_candidates(source, output_dir, CandidateConfig(), candidates, exclusion.metadata)
    (output_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "output_directory": str(output_dir.resolve()),
        "status": "generated",
        "candidate_count": len(candidates),
        "target_exclusion": exclusion.metadata,
    }


def _fit(args: argparse.Namespace) -> int:
    # Fit means "learn the correction model." The program compares one Mira RAW
    # capture with a trusted gold-camera image of the same target, then writes a
    # model file that future processing runs can use.
    target = load_yaml(args.target_config)
    model = fit_flat_patch_model(
        args.mira_raw,
        args.gold_tiff,
        args.evidence_dir,
        args.openrgbir_repo,
        args.isp_config,
        target,
        keep_intermediates=args.keep_intermediates,
    )
    write_yaml(args.model_out, model)
    write_yaml(args.evidence_dir / "fit_report.yaml", model)

    # The fitter leaves behind a temporary NDVI TIFF for review. Convert it into
    # a normal PNG preview, then delete the temporary TIFF so the evidence folder
    # stays compact.
    ndvi = tifffile.imread(args.evidence_dir / "ndvi_preview_source.tiff")
    display = model["display"]
    cv2.imwrite(
        str(args.evidence_dir / "ndvi_preview.png"),
        cv2.cvtColor(
            ndvi_false_color(ndvi, display["ndvi_min"], display["ndvi_max"]),
            cv2.COLOR_RGB2BGR,
        ),
    )
    (args.evidence_dir / "ndvi_preview_source.tiff").unlink()
    print(f"Model written to {args.model_out}")
    print(f"Evidence written to {args.evidence_dir}")
    return 0


def _process_run(
    raw_dir: Path,
    model_path: Path,
    run_dir: Path,
    recursive: bool,
    openrgbir_repo: Path,
    isp_config: Path,
    keep_intermediates: bool,
    product_set: str = "full",
    detect_candidates_flag: bool = False,
    force_candidates: bool = False,
    target_config: Path = DEFAULT_TARGET_PATH,
) -> dict:
    # This is the main batch-processing loop. It takes a folder of RAW files and
    # turns each good RAW file into output images and summary records.
    model = load_yaml(model_path)
    target = load_yaml(target_config)

    # With --recursive, search all subfolders. Without it, only use RAW files
    # directly inside raw_dir.
    pattern = "**/*.raw" if recursive else "*.raw"
    raw_paths = sorted(raw_dir.glob(pattern))
    if not raw_paths:
        raise SystemExit(f"No RAW files found in {raw_dir.resolve()}")
    processed = []
    skipped = []
    candidate_generated = []
    candidate_skipped = []
    candidate_failed = []
    for raw_path in raw_paths:
        # First safety check: only process files that match the expected Mira220
        # RAW size. This prevents random or incomplete files from entering the
        # pipeline.
        if not is_compatible_raw(raw_path):
            skipped.append({"path": str(raw_path.resolve()), "reason": "incompatible RAW size"})
            print(f"Skipping incompatible RAW: {raw_path.name}")
            continue
        relative_parent = raw_path.parent.relative_to(raw_dir)
        output_dir = run_dir / relative_parent / _capture_output_name(raw_path)

        # Second safety check: if all expected outputs already exist, skip the
        # expensive processing steps. If candidate detection was requested, the
        # code can still add or refresh candidate files for that capture.
        if _is_processed_output(output_dir, product_set):
            candidate_result = None
            if detect_candidates_flag:
                try:
                    candidate_result = _write_candidates_for_output(output_dir, force=force_candidates, target=target)
                    if candidate_result["status"] == "generated":
                        candidate_generated.append(candidate_result)
                    else:
                        candidate_skipped.append(candidate_result)
                except Exception as exc:  # pragma: no cover - defensive summary path
                    candidate_result = {
                        "output_directory": str(output_dir.resolve()),
                        "status": "failed",
                        "error": str(exc),
                    }
                    candidate_failed.append(candidate_result)
            skipped.append(
                {
                    "path": str(raw_path.resolve()),
                    "output_directory": str(output_dir.resolve()),
                    "reason": "already processed",
                    "candidate_result": candidate_result,
                }
            )
            print(f"Skipping already processed RAW: {raw_path.name} -> {output_dir}")
            continue

        # Third safety check: skip images that are too saturated. A saturated
        # image has many pixels stuck at the maximum sensor value, which means
        # detail was lost before the software ever saw the image.
        saturation = _raw_saturation_summary(raw_path, model["preprocessing"]["sensor_bit_depth"])
        if saturation["excessive"]:
            skipped.append(
                {
                    "path": str(raw_path.resolve()),
                    "output_directory": str(output_dir.resolve()),
                    "reason": "excessive RAW saturation",
                    "saturated_fraction": saturation["saturated_fraction"],
                    "threshold_fraction": saturation["threshold_fraction"],
                    "sensor_max": saturation["sensor_max"],
                }
            )
            print(
                "Skipping saturated RAW: "
                f"{raw_path.name} ({saturation['saturated_fraction']:.3%} >= "
                f"{saturation['threshold_fraction']:.3%})"
            )
            continue

        # openRGB-IR handles the camera-specific RAW conversion. It returns a
        # visible RGB image and a separate infrared image.
        raw_rgb, raw_ir = run_openrgbir(
            raw_path,
            output_dir,
            openrgbir_repo,
            isp_config,
            keep_intermediates=keep_intermediates,
        )

        # Convert integer sensor values, such as 0..4095, into floating-point
        # values from 0.0..1.0. The correction math expects that normalized form.
        rgb = normalize_sensor(raw_rgb, model["preprocessing"]["sensor_bit_depth"])
        ir = normalize_sensor(raw_ir, model["preprocessing"]["sensor_bit_depth"])

        # Apply the saved calibration model. This estimates corrected Red,
        # Green, and IR reflectance values from the raw camera channels.
        red, green, calibrated_ir = apply_model(rgb, ir, model)
        display = model["display"]

        # Save the scientific products and the review images. The full product
        # set includes reflectance TIFFs; the NDVI product set is smaller.
        save_reflectance_products(
            output_dir,
            red,
            green,
            calibrated_ir,
            display["ndvi_min"],
            display["ndvi_max"],
            rgb=rgb,
            product_set=product_set,
        )

        # Candidate detection is optional because it adds extra work and files.
        # When enabled during fresh processing, always force new candidate files
        # so they match the just-created NDVI image.
        candidate_result = None
        if detect_candidates_flag:
            try:
                candidate_result = _write_candidates_for_output(output_dir, force=True, target=target)
                candidate_generated.append(candidate_result)
            except Exception as exc:  # pragma: no cover - defensive summary path
                candidate_result = {
                    "output_directory": str(output_dir.resolve()),
                    "status": "failed",
                    "error": str(exc),
                }
                candidate_failed.append(candidate_result)
        processed.append(
            {
                "path": str(raw_path.resolve()),
                "output_directory": str(output_dir.resolve()),
                "candidate_result": candidate_result,
            }
        )
        print(f"Processed {raw_path.name} -> {output_dir}")

    # The summary is a machine-readable record of what happened during this run.
    # It is useful later when checking which files were processed, skipped, or
    # failed candidate detection.
    summary = {
        "model": str(model_path.resolve()),
        "input_directory": str(raw_dir.resolve()),
        "product_set": product_set,
        "processed": processed,
        "skipped": skipped,
        "candidate_generated": candidate_generated,
        "candidate_skipped": candidate_skipped,
        "candidate_failed": candidate_failed,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed: {len(processed)} processed, {len(skipped)} skipped.")
    return summary


def _process(args: argparse.Namespace) -> int:
    # CLI wrapper for _process_run. If the user does not choose a run id, use the
    # current date and time so each run gets a unique output folder.
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    _process_run(
        args.raw_dir,
        args.model,
        args.results_dir / run_id,
        args.recursive,
        args.openrgbir_repo,
        args.isp_config,
        args.keep_intermediates,
        product_set=args.product_set,
        detect_candidates_flag=args.detect_candidates,
        force_candidates=args.force_candidates,
        target_config=args.target_config,
    )
    return 0


def _process_session(args: argparse.Namespace) -> int:
    # A session run creates two possible output sets:
    # 1. validated/ is the normal processed output.
    # 2. affine-adjusted/ is an extra version adjusted from target measurements,
    #    but only when the target can be measured well enough.
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.results_dir / run_id
    validated_root = output_root / "validated"
    model = load_yaml(args.model)
    processing = _process_run(
        args.raw_dir,
        args.model,
        validated_root,
        args.recursive,
        args.openrgbir_repo,
        args.isp_config,
        args.keep_intermediates,
    )

    # Try to learn a session-level adjustment from the calibration target. This
    # helps handle lighting changes within a field session.
    affine = adjust_session(
        validated_root,
        output_root / "affine-adjusted",
        load_yaml(args.target_config),
        load_yaml(args.target_reference),
        model["display"],
        args.model,
    )
    summary = {
        "model": str(args.model.resolve()),
        "input_directory": str(args.raw_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "validated_root": str(validated_root.resolve()),
        "affine_adjusted_root": str((output_root / "affine-adjusted").resolve()) if affine["adjusted"] else None,
        "affine_adjusted": affine["adjusted"],
        "affine_skip_reason": affine.get("skip_reason"),
        "target_measurement_count": len(affine["capture_measurements"]),
        "target_measurement_failures": affine["measurement_failures"],
        "processing": processing,
        "affine": affine,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if affine["adjusted"]:
        print(f"Validated outputs written to {validated_root}")
        print(f"Affine-adjusted outputs written to {output_root / 'affine-adjusted'}")
    else:
        print(f"Validated outputs written to {validated_root}")
        print(f"Affine adjustment skipped: {affine.get('skip_reason')}")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    # Inspect reads one processed capture and measures the calibration target.
    # It is a diagnostic command: it helps answer "Do these outputs look sane?"
    model = load_yaml(args.model)
    target = load_yaml(args.target_config)
    path = args.image_or_run
    if path.is_file():
        # inspect normally expects a processed folder. This branch lets a user
        # inspect a standalone three-channel R/G/N TIFF by briefly creating the
        # files that inspect_products expects, then cleaning them up.
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise SystemExit("Inspect accepts a processed run directory or a three-channel RGN TIFF.")
        rgn = tifffile.imread(path)
        temporary = args.report.parent / "_inspect_temporary"
        temporary.mkdir(parents=True, exist_ok=True)
        for name, channel in zip(("red", "green", "ir"), range(3)):
            tifffile.imwrite(temporary / f"{name}_reflectance.tiff", rgn[..., channel])
        report = inspect_products(temporary, target, model)
        for item in temporary.iterdir():
            item.unlink()
        temporary.rmdir()
    else:
        report = inspect_products(path, target, model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _compare(args: argparse.Namespace) -> int:
    # Compare one Mira output against one gold-camera TIFF. The printed metrics
    # are numbers, but the overlay image still needs human review.
    report = compare_pair(args.mira_output, args.gold_tiff, args.output_dir, load_yaml(args.target_config))
    print(json.dumps(report["metrics"], indent=2))
    print("Best-pair selection requires visual confirmation of alignment_overlay.png.")
    return 0


def _compare_all(args: argparse.Namespace) -> int:
    # Compare many Mira outputs against many gold TIFFs and rank likely matches.
    # This is useful when the exact pairing is not known ahead of time.
    summary = compare_all(args.mira_root, args.gold_dir, args.output_dir, load_yaml(args.target_config))
    print(json.dumps({key: value for key, value in summary.items() if key != "pairs"}, indent=2))
    print("Best-pair selections require visual confirmation of alignment_overlay.png.")
    return 0


def _candidate_output_dir(input_path: Path) -> Path:
    # Direct candidate detection writes to a shared candidates folder by default.
    # Use the input name so each capture gets a separate candidate output folder.
    name = input_path.name if input_path.name else input_path.resolve().name
    if input_path.is_file():
        name = input_path.stem
    return DEFAULT_DATA_ROOT / "outputs" / "candidates" / name


def _detect_candidates(args: argparse.Namespace) -> int:
    # Build a CandidateConfig from command-line options. This lets users tune how
    # sensitive the detector is without editing Python code.
    config = CandidateConfig(
        top_n=args.top_n,
        threshold_mode=args.threshold_mode,
        manual_threshold=args.manual_threshold,
        invert_mask=args.invert_mask,
        min_area=args.min_area,
        min_contrast=args.min_contrast,
        large_region_area=args.large_region_area,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        hole_fill_area=args.hole_fill_area,
        simplify_epsilon=args.simplify_epsilon,
        texture_window=args.texture_window,
        texture_sensitivity=args.texture_sensitivity,
        geometry_min_solidity=args.geometry_min_solidity,
        geometry=args.geometry,
        box_thickness=args.box_thickness,
        crop=not args.no_crop,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
        low_threshold=args.low_threshold,
        show_class_colors=args.show_class_colors,
    )

    # Load the NDVI source, exclude the calibration target if possible, then run
    # the detector and write all review files.
    source = load_ndvi_source(args.image_or_output_dir, crop=config.crop)
    target = load_yaml(args.target_config)
    exclusion = target_exclusion_from_source(source, target)
    candidates, _ = detect_candidates(source, config, exclusion_mask=exclusion.mask)
    output_dir = args.output_dir or _candidate_output_dir(args.image_or_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_overlay(output_dir / "candidate_overlay.png", source.background, candidates, config)
    write_candidate_mask(output_dir / "candidate_mask.png", source.ndvi.shape, candidates)
    write_candidates_csv(output_dir / "candidates.csv", candidates)
    summary = summarize_candidates(source, output_dir, config, candidates, exclusion.metadata)
    (output_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "top_candidates"}, indent=2))
    return 0


def _collage_entries(input_root: Path, image_name: str) -> dict[str, list[Path]]:
    # Find every requested image, such as ndvi_false_color_crop.png or
    # candidate_overlay.png, and group the paths by capture date.
    grouped: dict[str, list[Path]] = {}
    for image_path in sorted(input_root.rglob(image_name)):
        if not image_path.is_file():
            continue
        date = _capture_date(image_path.parent)
        if date is None:
            continue
        grouped.setdefault(date, []).append(image_path)
    return grouped


def _fit_tile(image: np.ndarray, tile_width: int, tile_height: int) -> np.ndarray:
    # Resize one image so it fits inside a fixed-size tile without stretching.
    # Empty space is filled with a light gray background.
    height, width = image.shape[:2]
    scale = min(tile_width / width, tile_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    tile = np.full((tile_height, tile_width, 3), COLLAGE_BACKGROUND, np.uint8)
    y = (tile_height - resized_height) // 2
    x = (tile_width - resized_width) // 2
    tile[y : y + resized_height, x : x + resized_width] = resized
    return tile


def _write_collage(
    output_path: Path,
    image_paths: list[Path],
    tile_width: int,
    tile_height: int,
    columns: int,
    label_height: int = 34,
    gutter: int = 12,
) -> None:
    # A collage is a big canvas made from many smaller tiles. Each tile shows one
    # processed image, and the label below it is the capture folder name.
    if not image_paths:
        raise ValueError("Cannot write a collage with no images.")
    columns = max(1, int(columns))
    rows = int(np.ceil(len(image_paths) / columns))
    canvas_width = columns * tile_width + (columns + 1) * gutter
    canvas_height = rows * (tile_height + label_height) + (rows + 1) * gutter
    canvas = np.full((canvas_height, canvas_width, 3), COLLAGE_BACKGROUND, np.uint8)
    for index, image_path in enumerate(image_paths):
        # cv2.imread returns BGR image data. That is fine here because cv2.imwrite
        # also expects BGR data.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        row, col = divmod(index, columns)
        x = gutter + col * (tile_width + gutter)
        y = gutter + row * (tile_height + label_height + gutter)
        canvas[y : y + tile_height, x : x + tile_width] = _fit_tile(image, tile_width, tile_height)
        label = image_path.parent.name
        cv2.putText(
            canvas,
            label,
            (x, y + tile_height + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLLAGE_TEXT,
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _collage(args: argparse.Namespace) -> int:
    # CLI wrapper for collage creation. It writes one PNG per date plus a JSON
    # summary listing how many images went into each collage.
    grouped = _collage_entries(args.input_root, args.image_name)
    if not grouped:
        raise SystemExit(f"No {args.image_name} images with RAW-style dates found under {args.input_root.resolve()}")
    summary = {"input_root": str(args.input_root.resolve()), "image_name": args.image_name, "collages": []}
    for date, image_paths in grouped.items():
        output_path = args.output_dir / f"{date}_{Path(args.image_name).stem}_collage.png"
        _write_collage(output_path, image_paths, args.tile_width, args.tile_height, args.columns)
        summary["collages"].append(
            {
                "date": date,
                "image_count": len(image_paths),
                "output_path": str(output_path.resolve()),
            }
        )
        print(f"Wrote {len(image_paths)} images for {date} -> {output_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "collage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def _recalibrate(args: argparse.Namespace) -> int:
    # Recalibrate starts from existing reflectance outputs. It does not rerun RAW
    # conversion. Instead, it applies a scene correction model and writes a new
    # output tree.
    model = load_yaml(args.model)
    if "scene_correction" not in model:
        raise SystemExit(f"{args.model} does not define scene_correction.")
    input_dirs = sorted(
        path.parent
        for path in args.input_root.rglob("red_reflectance.tiff")
        if (path.parent / "green_reflectance.tiff").is_file()
        and (path.parent / "ir_reflectance.tiff").is_file()
    )
    if not input_dirs:
        raise SystemExit(f"No reflectance output directories found under {args.input_root.resolve()}")
    processed = []
    for input_dir in input_dirs:
        # Preserve the same folder layout under the new output root.
        relative = input_dir.relative_to(args.input_root)
        output_dir = args.output_root / relative
        red = tifffile.imread(input_dir / "red_reflectance.tiff").astype(np.float32)
        green = tifffile.imread(input_dir / "green_reflectance.tiff").astype(np.float32)
        ir = tifffile.imread(input_dir / "ir_reflectance.tiff").astype(np.float32)
        red, ir = apply_scene_correction(red, ir, model["scene_correction"])
        display = model["display"]
        save_reflectance_products(output_dir, red, green, ir, display["ndvi_min"], display["ndvi_max"])
        processed.append(
            {
                "input_directory": str(input_dir.resolve()),
                "output_directory": str(output_dir.resolve()),
                "clipping_fraction": {
                    "red": float(np.mean((red <= 0) | (red >= 1))),
                    "green": float(np.mean((green <= 0) | (green >= 1))),
                    "ir": float(np.mean((ir <= 0) | (ir >= 1))),
                },
            }
        )
        print(f"Recalibrated {input_dir} -> {output_dir}")
    summary = {
        "model": str(args.model.resolve()),
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "processed": processed,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "recalibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


def _adjust_session(args: argparse.Namespace) -> int:
    # Apply the session-level target adjustment to an already processed run.
    model = load_yaml(args.model)
    report = adjust_session(
        args.input_root,
        args.output_root,
        load_yaml(args.target_config),
        load_yaml(args.target_reference),
        model["display"],
        args.model,
    )
    print(json.dumps(report["adjustment"], indent=2))
    return 0


def _variant_summary(name: str, comparison: dict, variant_root: Path, processing: dict) -> dict:
    # Pull out the best comparison rows and summarize their metrics. This makes
    # validate-session easier to read than a giant list of every possible pair.
    best = [row for row in comparison["pairs"] if row.get("best_candidate")]
    if not best:
        return {"variant": name, "successful_capture_count": 0}
    metric_names = ("pearson_correlation", "r_squared", "rmse", "mae", "mean_bias")
    summary = {
        "variant": name,
        "successful_capture_count": len(best),
        "mean_best_metrics": {
            metric: float(np.mean([row[metric] for row in best])) for metric in metric_names
        },
        "best_pairs": best,
        "clipping": summarize_clipping(variant_root),
    }
    if "adjustment" in processing:
        if processing["adjustment"] is None:
            return summary
        summary["target_fit_residuals"] = {
            channel: {
                "rmse": values["rmse"],
                "max_abs_residual": values["max_abs_residual"],
            }
            for channel, values in processing["adjustment"]["channels"].items()
        }
    return summary


def _validate_session(args: argparse.Namespace) -> int:
    # validate-session is an experiment runner. It processes the same session in
    # several ways, compares each result against Mapir images, then ranks the
    # variants by their comparison metrics.
    raw_dir = args.session_dir / "mira" / "raw"
    mapir_dir = args.session_dir / "mapir" / "tiff"
    output_root = args.output_root or DEFAULT_DATA_ROOT / "outputs" / "sessions" / args.session_dir.name
    target = load_yaml(args.target_config)
    reference = load_yaml(args.target_reference)
    variants = {
        "flat-patch": args.flat_model,
        "scene-reference": args.scene_model,
    }
    processing = {}
    for variant, model_path in variants.items():
        # First process RAW files with the selected model.
        variant_root = output_root / variant
        processing[variant] = _process_run(
            raw_dir,
            model_path,
            variant_root,
            True,
            args.openrgbir_repo,
            args.isp_config,
            args.keep_intermediates,
        )
        model = load_yaml(model_path)
        adjusted_name = f"{variant}-adjusted"

        # Then try a target-based lighting adjustment for that same variant.
        processing[adjusted_name] = adjust_session(
            variant_root,
            output_root / adjusted_name,
            target,
            reference,
            model["display"],
            model_path,
        )
    comparison_summaries = {}
    variant_summaries = []
    for variant in ("flat-patch", "flat-patch-adjusted", "scene-reference", "scene-reference-adjusted"):
        # Compare each variant's outputs against the Mapir reference images.
        comparison = compare_all(
            output_root / variant,
            mapir_dir,
            output_root / "comparisons" / variant,
            target,
        )
        comparison_summaries[variant] = comparison
        variant_summaries.append(
            _variant_summary(variant, comparison, output_root / variant, processing[variant])
        )
    variant_summaries.sort(
        # Higher correlation is better. Lower RMSE is better. This sort puts the
        # strongest variant first.
        key=lambda item: (
            item.get("mean_best_metrics", {}).get("pearson_correlation", -1),
            -item.get("mean_best_metrics", {}).get("rmse", float("inf")),
        ),
        reverse=True,
    )
    summary = {
        "session_directory": str(args.session_dir.resolve()),
        "mapir_used_for_adjustment_fitting": False,
        "mapir_directory": str(mapir_dir.resolve()),
        "processing": processing,
        "variant_ranking": variant_summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "session_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(variant_summaries, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    # argparse turns command-line text into structured Python values. Each
    # subparser below is one user-facing command, such as "process" or "collage".
    parser = argparse.ArgumentParser(description="Mira220 RGB-IR reflectance processing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Fit a calibration model from a Mira RAW and a trusted gold-camera image.
    fit = subparsers.add_parser("fit", help="Fit the flat-patch correction model.")
    fit.add_argument("--mira-raw", type=Path, required=True)
    fit.add_argument("--gold-tiff", type=Path, required=True)
    fit.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    fit.add_argument("--evidence-dir", type=Path, default=ROOT / "calibration" / "evidence")
    fit.add_argument("--keep-intermediates", action="store_true")
    _add_shared_paths(fit)
    fit.set_defaults(handler=_fit)

    # Process a folder of RAW files into reflectance and NDVI products.
    process = subparsers.add_parser("process", help="Batch-process Mira220 RAW files.")
    process.add_argument("raw_dir", type=Path, nargs="?", default=DEFAULT_DATA_ROOT / "raw")
    process.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    process.add_argument("--results-dir", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "runs")
    process.add_argument("--run-id")
    process.add_argument("--recursive", action="store_true")
    process.add_argument("--keep-intermediates", action="store_true")
    process.add_argument("--product-set", choices=("full", "ndvi"), default="full")
    process.add_argument("--detect-candidates", action="store_true")
    process.add_argument("--force-candidates", action="store_true")
    _add_shared_paths(process)
    process.set_defaults(handler=_process)

    # Process a session and optionally create a target-adjusted variant.
    process_session = subparsers.add_parser(
        "process-session", help="Process RAWs and optionally write a target affine-adjusted variant."
    )
    process_session.add_argument("raw_dir", type=Path, nargs="?", default=DEFAULT_DATA_ROOT / "raw")
    process_session.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    process_session.add_argument("--results-dir", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "runs")
    process_session.add_argument("--run-id")
    process_session.add_argument("--recursive", action="store_true")
    process_session.add_argument("--keep-intermediates", action="store_true")
    process_session.add_argument("--target-reference", type=Path, default=DEFAULT_TARGET_REFERENCE_PATH)
    _add_shared_paths(process_session)
    process_session.set_defaults(handler=_process_session)

    # Inspect one processed output or R/G/N TIFF.
    inspect = subparsers.add_parser("inspect", help="Inspect patch values and clipping.")
    inspect.add_argument("image_or_run", type=Path)
    inspect.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    inspect.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)
    inspect.add_argument("--report", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "inspection.json")
    inspect.set_defaults(handler=_inspect)

    # Compare one Mira output against one gold image.
    compare = subparsers.add_parser("compare", help="Compare one Mira output with one gold TIFF.")
    compare.add_argument("mira_output", type=Path)
    compare.add_argument("gold_tiff", type=Path)
    compare.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "comparisons" / "single")
    compare.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)
    compare.set_defaults(handler=_compare)

    # Compare many Mira outputs against many gold images.
    compare_batch = subparsers.add_parser("compare-all", help="Compare every Mira output with every gold TIFF.")
    compare_batch.add_argument("--mira-root", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "runs")
    compare_batch.add_argument("--gold-dir", type=Path, default=DEFAULT_DATA_ROOT / "gold")
    compare_batch.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "comparisons")
    compare_batch.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)
    compare_batch.set_defaults(handler=_compare_all)

    # Find interesting NDVI regions in one processed output.
    candidates = subparsers.add_parser("detect-candidates", help="Detect cropped NDVI candidate regions.")
    candidates.add_argument("image_or_output_dir", type=Path)
    candidates.add_argument("--output-dir", type=Path)
    candidates.add_argument("--top-n", type=int, default=25)
    candidates.add_argument("--threshold-mode", choices=("manual", "otsu", "adaptive", "percentile"), default="percentile")
    candidates.add_argument("--manual-threshold", type=float)
    candidates.add_argument("--invert-mask", action="store_true")
    candidates.add_argument("--min-area", type=int, default=15000)
    candidates.add_argument("--min-contrast", type=float, default=0.03)
    candidates.add_argument("--large-region-area", type=int, default=50000)
    candidates.add_argument("--morph-open", type=int, default=5)
    candidates.add_argument("--morph-close", type=int, default=31)
    candidates.add_argument("--hole-fill-area", type=int, default=4000)
    candidates.add_argument("--simplify-epsilon", type=float, default=8.0)
    candidates.add_argument("--texture-window", type=int, default=21)
    candidates.add_argument("--texture-sensitivity", type=float, default=0.18)
    candidates.add_argument("--geometry-min-solidity", type=float, default=0.82)
    candidates.add_argument("--geometry", choices=("auto", "box", "contour"), default="contour")
    candidates.add_argument("--box-thickness", type=int, default=8)
    candidates.add_argument("--percentile-low", type=float, default=1.0)
    candidates.add_argument("--percentile-high", type=float, default=99.0)
    candidates.add_argument("--no-crop", action="store_true")
    candidates.add_argument("--show-class-colors", action="store_true")
    candidates.add_argument("--high-threshold", type=float, default=0.45)
    candidates.add_argument("--medium-threshold", type=float, default=0.18)
    candidates.add_argument("--low-threshold", type=float, default=0.05)
    candidates.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)
    candidates.set_defaults(handler=_detect_candidates)

    # Build dated contact sheets from existing processed images.
    collage = subparsers.add_parser("collage", help="Create dated collages from processed image products.")
    collage.add_argument("input_root", type=Path, nargs="?", default=DEFAULT_DATA_ROOT / "outputs" / "runs")
    collage.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "collages")
    collage.add_argument("--image-name", default="ndvi_false_color_crop.png")
    collage.add_argument("--tile-width", type=int, default=320)
    collage.add_argument("--tile-height", type=int, default=260)
    collage.add_argument("--columns", type=int, default=4)
    collage.set_defaults(handler=_collage)

    # Apply a scene correction to existing reflectance outputs.
    recalibrate = subparsers.add_parser(
        "recalibrate", help="Apply a scene correction to existing reflectance outputs."
    )
    recalibrate.add_argument("input_root", type=Path)
    recalibrate.add_argument("--model", type=Path, default=ROOT / "config" / "models" / "scene_reference_v1.yaml")
    recalibrate.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "runs" / "scene-reference-v1")
    recalibrate.set_defaults(handler=_recalibrate)

    # Apply session target adjustment to an already processed output tree.
    adjust = subparsers.add_parser(
        "adjust-session", help="Fit and apply a session-level target lighting adjustment."
    )
    adjust.add_argument("input_root", type=Path)
    adjust.add_argument("--output-root", type=Path, required=True)
    adjust.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    adjust.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)
    adjust.add_argument("--target-reference", type=Path, default=DEFAULT_TARGET_REFERENCE_PATH)
    adjust.set_defaults(handler=_adjust_session)

    # Run the larger validation experiment for a full session.
    validate = subparsers.add_parser(
        "validate-session", help="Process and validate four model/lighting-adjustment variants."
    )
    validate.add_argument("session_dir", type=Path)
    validate.add_argument("--output-root", type=Path)
    validate.add_argument("--flat-model", type=Path, default=DEFAULT_MODEL_PATH)
    validate.add_argument(
        "--scene-model", type=Path, default=ROOT / "config" / "models" / "scene_reference_v1.yaml"
    )
    validate.add_argument("--target-reference", type=Path, default=DEFAULT_TARGET_REFERENCE_PATH)
    validate.add_argument("--keep-intermediates", action="store_true")
    _add_shared_paths(validate)
    validate.set_defaults(handler=_validate_session)
    return parser


def main() -> None:
    # Program entry point. Parse the command, then call the handler function that
    # was attached with set_defaults(handler=...).
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))

