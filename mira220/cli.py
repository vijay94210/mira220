from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile

from . import (
    DEFAULT_ISP_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OPENRGBIR_REPO,
    DEFAULT_TARGET_PATH,
    ROOT,
)
from .calibration import fit_flat_patch_model, inspect_products
from .correction import apply_model, load_yaml, write_yaml
from .imaging import ndvi_false_color, normalize_sensor, save_reflectance_products
from .openrgbir import is_compatible_raw, run_openrgbir


def _add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--openrgbir-repo", type=Path, default=DEFAULT_OPENRGBIR_REPO)
    parser.add_argument("--isp-config", type=Path, default=DEFAULT_ISP_CONFIG_PATH)
    parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)


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


def _process(args: argparse.Namespace) -> int:
    model = load_yaml(args.model)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / run_id
    pattern = "**/*.raw" if args.recursive else "*.raw"
    raw_paths = sorted(args.raw_dir.glob(pattern))
    if not raw_paths:
        raise SystemExit(f"No RAW files found in {args.raw_dir.resolve()}")
    processed = []
    skipped = []
    for raw_path in raw_paths:
        if not is_compatible_raw(raw_path):
            skipped.append({"path": str(raw_path.resolve()), "reason": "incompatible RAW size"})
            print(f"Skipping incompatible RAW: {raw_path.name}")
            continue
        relative_parent = raw_path.parent.relative_to(args.raw_dir)
        output_dir = run_dir / relative_parent / raw_path.stem
        raw_rgb, raw_ir = run_openrgbir(
            raw_path,
            output_dir,
            args.openrgbir_repo,
            args.isp_config,
            keep_intermediates=args.keep_intermediates,
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
        )
        processed.append(str(raw_path.resolve()))
        print(f"Processed {raw_path.name} -> {output_dir}")
    summary = {
        "run_id": run_id,
        "model": str(args.model.resolve()),
        "input_directory": str(args.raw_dir.resolve()),
        "processed": processed,
        "skipped": skipped,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed: {len(processed)} processed, {len(skipped)} skipped.")
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
    process.add_argument("raw_dir", type=Path, nargs="?", default=ROOT / "data" / "raw")
    process.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    process.add_argument("--results-dir", type=Path, default=ROOT / "results")
    process.add_argument("--run-id")
    process.add_argument("--recursive", action="store_true")
    process.add_argument("--keep-intermediates", action="store_true")
    _add_shared_paths(process)
    process.set_defaults(handler=_process)

    inspect = subparsers.add_parser("inspect", help="Inspect patch values and clipping.")
    inspect.add_argument("image_or_run", type=Path)
    inspect.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    inspect.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_PATH)
    inspect.add_argument("--report", type=Path, default=ROOT / "results" / "inspection.json")
    inspect.set_defaults(handler=_inspect)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))

