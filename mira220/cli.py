from __future__ import annotations

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
    write_candidate_overlay,
    write_candidate_mask,
    write_candidates_csv,
)
from .comparison import compare_all, compare_pair
from .correction import apply_model, apply_scene_correction, load_yaml, write_yaml
from .imaging import ndvi_false_color, normalize_sensor, save_reflectance_products
from .openrgbir import is_compatible_raw, run_openrgbir
from .session import adjust_session, summarize_clipping


PROCESSED_SENTINELS = (
    "ndvi.tiff",
    "ndvi_gray.png",
    "ndvi_false_color.png",
    "ndvi_false_color_crop.png",
    "rgb_preview.png",
    "rgn_preview.png",
    "rgn_reflectance.tiff",
)
RAW_TIMESTAMP_RE = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})")


def _add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--openrgbir-repo", type=Path, default=DEFAULT_OPENRGBIR_REPO)
    parser.add_argument("--isp-config", type=Path, default=DEFAULT_ISP_CONFIG_PATH)
    parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)


def _capture_output_name(raw_path: Path) -> str:
    match = RAW_TIMESTAMP_RE.search(raw_path.stem)
    if match is None:
        return raw_path.stem
    parts = match.groupdict()
    return (
        f"{parts['year']}-{parts['month']}-{parts['day']}_"
        f"{parts['hour']}_{parts['minute']}_{parts['second']}"
    )


def _is_processed_output(output_dir: Path) -> bool:
    return all((output_dir / name).is_file() for name in PROCESSED_SENTINELS)


def _fit(args: argparse.Namespace) -> int:
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
) -> dict:
    model = load_yaml(model_path)
    pattern = "**/*.raw" if recursive else "*.raw"
    raw_paths = sorted(raw_dir.glob(pattern))
    if not raw_paths:
        raise SystemExit(f"No RAW files found in {raw_dir.resolve()}")
    processed = []
    skipped = []
    for raw_path in raw_paths:
        if not is_compatible_raw(raw_path):
            skipped.append({"path": str(raw_path.resolve()), "reason": "incompatible RAW size"})
            print(f"Skipping incompatible RAW: {raw_path.name}")
            continue
        relative_parent = raw_path.parent.relative_to(raw_dir)
        output_dir = run_dir / relative_parent / _capture_output_name(raw_path)
        if _is_processed_output(output_dir):
            skipped.append(
                {
                    "path": str(raw_path.resolve()),
                    "output_directory": str(output_dir.resolve()),
                    "reason": "already processed",
                }
            )
            print(f"Skipping already processed RAW: {raw_path.name} -> {output_dir}")
            continue
        raw_rgb, raw_ir = run_openrgbir(
            raw_path,
            output_dir,
            openrgbir_repo,
            isp_config,
            keep_intermediates=keep_intermediates,
        )
        rgb = normalize_sensor(raw_rgb, model["preprocessing"]["sensor_bit_depth"])
        ir = normalize_sensor(raw_ir, model["preprocessing"]["sensor_bit_depth"])
        red, green, calibrated_ir = apply_model(rgb, ir, model)
        display = model["display"]
        save_reflectance_products(
            output_dir,
            red,
            green,
            calibrated_ir,
            display["ndvi_min"],
            display["ndvi_max"],
            rgb=rgb,
        )
        processed.append(str(raw_path.resolve()))
        print(f"Processed {raw_path.name} -> {output_dir}")
    summary = {
        "model": str(model_path.resolve()),
        "input_directory": str(raw_dir.resolve()),
        "processed": processed,
        "skipped": skipped,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed: {len(processed)} processed, {len(skipped)} skipped.")
    return summary


def _process(args: argparse.Namespace) -> int:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    _process_run(
        args.raw_dir,
        args.model,
        args.results_dir / run_id,
        args.recursive,
        args.openrgbir_repo,
        args.isp_config,
        args.keep_intermediates,
    )
    return 0


def _process_session(args: argparse.Namespace) -> int:
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
    model = load_yaml(args.model)
    target = load_yaml(args.target_config)
    path = args.image_or_run
    if path.is_file():
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
    report = compare_pair(args.mira_output, args.gold_tiff, args.output_dir, load_yaml(args.target_config))
    print(json.dumps(report["metrics"], indent=2))
    print("Best-pair selection requires visual confirmation of alignment_overlay.png.")
    return 0


def _compare_all(args: argparse.Namespace) -> int:
    summary = compare_all(args.mira_root, args.gold_dir, args.output_dir, load_yaml(args.target_config))
    print(json.dumps({key: value for key, value in summary.items() if key != "pairs"}, indent=2))
    print("Best-pair selections require visual confirmation of alignment_overlay.png.")
    return 0


def _candidate_output_dir(input_path: Path) -> Path:
    name = input_path.name if input_path.name else input_path.resolve().name
    if input_path.is_file():
        name = input_path.stem
    return DEFAULT_DATA_ROOT / "outputs" / "candidates" / name


def _detect_candidates(args: argparse.Namespace) -> int:
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
    source = load_ndvi_source(args.image_or_output_dir, crop=config.crop)
    candidates, _ = detect_candidates(source, config)
    output_dir = args.output_dir or _candidate_output_dir(args.image_or_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_overlay(output_dir / "candidate_overlay.png", source.background, candidates, config)
    write_candidate_mask(output_dir / "candidate_mask.png", source.ndvi.shape, candidates)
    write_candidates_csv(output_dir / "candidates.csv", candidates)
    summary = summarize_candidates(source, output_dir, config, candidates)
    (output_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "top_candidates"}, indent=2))
    return 0


def _recalibrate(args: argparse.Namespace) -> int:
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
    parser = argparse.ArgumentParser(description="Mira220 RGB-IR reflectance processing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="Fit the flat-patch correction model.")
    fit.add_argument("--mira-raw", type=Path, required=True)
    fit.add_argument("--gold-tiff", type=Path, required=True)
    fit.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    fit.add_argument("--evidence-dir", type=Path, default=ROOT / "calibration" / "evidence")
    fit.add_argument("--keep-intermediates", action="store_true")
    _add_shared_paths(fit)
    fit.set_defaults(handler=_fit)

    process = subparsers.add_parser("process", help="Batch-process Mira220 RAW files.")
    process.add_argument("raw_dir", type=Path, nargs="?", default=DEFAULT_DATA_ROOT / "raw")
    process.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    process.add_argument("--results-dir", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "runs")
    process.add_argument("--run-id")
    process.add_argument("--recursive", action="store_true")
    process.add_argument("--keep-intermediates", action="store_true")
    _add_shared_paths(process)
    process.set_defaults(handler=_process)

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

    inspect = subparsers.add_parser("inspect", help="Inspect patch values and clipping.")
    inspect.add_argument("image_or_run", type=Path)
    inspect.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    inspect.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)
    inspect.add_argument("--report", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "inspection.json")
    inspect.set_defaults(handler=_inspect)

    compare = subparsers.add_parser("compare", help="Compare one Mira output with one gold TIFF.")
    compare.add_argument("mira_output", type=Path)
    compare.add_argument("gold_tiff", type=Path)
    compare.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "comparisons" / "single")
    compare.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)
    compare.set_defaults(handler=_compare)

    compare_batch = subparsers.add_parser("compare-all", help="Compare every Mira output with every gold TIFF.")
    compare_batch.add_argument("--mira-root", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "runs")
    compare_batch.add_argument("--gold-dir", type=Path, default=DEFAULT_DATA_ROOT / "gold")
    compare_batch.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "comparisons")
    compare_batch.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)
    compare_batch.set_defaults(handler=_compare_all)

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
    candidates.set_defaults(handler=_detect_candidates)

    recalibrate = subparsers.add_parser(
        "recalibrate", help="Apply a scene correction to existing reflectance outputs."
    )
    recalibrate.add_argument("input_root", type=Path)
    recalibrate.add_argument("--model", type=Path, default=ROOT / "config" / "models" / "scene_reference_v1.yaml")
    recalibrate.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT / "outputs" / "runs" / "scene-reference-v1")
    recalibrate.set_defaults(handler=_recalibrate)

    adjust = subparsers.add_parser(
        "adjust-session", help="Fit and apply a session-level target lighting adjustment."
    )
    adjust.add_argument("input_root", type=Path)
    adjust.add_argument("--output-root", type=Path, required=True)
    adjust.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    adjust.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)
    adjust.add_argument("--target-reference", type=Path, default=DEFAULT_TARGET_REFERENCE_PATH)
    adjust.set_defaults(handler=_adjust_session)

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
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))

