from __future__ import annotations

"""Compare Mira NDVI outputs against gold-camera reference TIFFs.

This module aligns a Mira output with a trusted gold-camera image using the
ArUco marker, computes NDVI differences, writes plots, and ranks possible image
pairs when many captures are compared.
"""

import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tifffile

from .imaging import compute_ndvi, detect_marker, preview_u8


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


MAX_PLOT_POINTS = 100_000
# Pixel blocks reduce noise when comparing two aligned NDVI images.
BLOCK_SIZE = 16


def load_rgn_tiff(path: Path) -> np.ndarray:
    # RGN means Red, Green, Near-infrared. Gold-camera TIFFs must have these
    # three channels.
    image = tifffile.imread(path)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{path} must be a three-channel Red/Green/NIR TIFF.")
    if np.issubdtype(image.dtype, np.integer):
        # Integer TIFFs are normalized by their maximum possible value.
        image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    else:
        image = image.astype(np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{path} contains non-finite RGN values.")
    return np.clip(image, 0.0, 1.0)


def load_mira_output(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    # A Mira output must include NDVI plus the R/G/N reflectance stack.
    ndvi_path = output_dir / "ndvi.tiff"
    rgn_path = output_dir / "rgn_reflectance.tiff"
    if not ndvi_path.is_file() or not rgn_path.is_file():
        raise FileNotFoundError(
            f"{output_dir} must contain ndvi.tiff and rgn_reflectance.tiff."
        )
    ndvi = tifffile.imread(ndvi_path)
    if ndvi.ndim != 2 or not np.issubdtype(ndvi.dtype, np.floating):
        raise ValueError(f"{ndvi_path} must be a two-dimensional floating-point TIFF.")
    ndvi = ndvi.astype(np.float32)
    rgn = load_rgn_tiff(rgn_path)
    if ndvi.shape != rgn.shape[:2]:
        raise ValueError(f"{ndvi_path} and {rgn_path} must have matching dimensions.")
    return ndvi, rgn


def discover_mira_outputs(root: Path) -> list[Path]:
    # Any folder with both ndvi.tiff and rgn_reflectance.tiff is a comparison
    # candidate.
    return sorted(
        path.parent
        for path in root.rglob("ndvi.tiff")
        if (path.parent / "rgn_reflectance.tiff").is_file()
    )


def discover_gold_tiffs(root: Path) -> list[Path]:
    # Gold references are discovered by TIFF extension.
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in {".tif", ".tiff"})


def regression_metrics(gold: np.ndarray, mira: np.ndarray) -> dict[str, float | int]:
    # Compute common statistics that describe how closely Mira matches gold.
    if gold.size < 2:
        raise ValueError("At least two valid overlapping pixels are required.")
    slope, intercept = np.polyfit(gold, mira, 1)
    predicted = slope * gold + intercept
    residual_sum = float(np.sum(np.square(mira - predicted)))
    total_sum = float(np.sum(np.square(mira - np.mean(mira))))
    correlation = float(np.corrcoef(gold, mira)[0, 1])
    difference = mira - gold
    return {
        "valid_pixel_count": int(gold.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - residual_sum / total_sum) if total_sum > 0 else 0.0,
        "pearson_correlation": correlation if np.isfinite(correlation) else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
        "mean_bias": float(np.mean(difference)),
    }


def block_regression_values(
    gold: np.ndarray,
    mira: np.ndarray,
    valid: np.ndarray,
    block_size: int = BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    # Trim to a whole number of blocks so reshape can split the image cleanly.
    height = gold.shape[0] // block_size * block_size
    width = gold.shape[1] // block_size * block_size

    def block_sum(image: np.ndarray) -> np.ndarray:
        # Reshape turns the image into rows of blocks and columns of blocks, then
        # sum each block.
        return image[:height, :width].reshape(
            height // block_size, block_size, width // block_size, block_size
        ).sum(axis=(1, 3))

    block_counts = block_sum(valid.astype(np.float32))
    # Only keep blocks that are almost completely valid overlap.
    block_valid = block_counts >= block_size * block_size * 0.95
    gold_means = block_sum(np.where(valid, gold, 0.0)) / np.maximum(block_counts, 1)
    mira_means = block_sum(np.where(valid, mira, 0.0)) / np.maximum(block_counts, 1)
    return gold_means[block_valid], mira_means[block_valid]


def block_regression_metrics(
    gold: np.ndarray,
    mira: np.ndarray,
    valid: np.ndarray,
    block_size: int = BLOCK_SIZE,
) -> dict[str, float | int]:
    # Same metrics as pixel regression, but on block averages.
    gold_values, mira_values = block_regression_values(gold, mira, valid, block_size)
    metrics = regression_metrics(gold_values, mira_values)
    metrics["block_size"] = block_size
    return metrics


def _gray(image: np.ndarray) -> np.ndarray:
    # Convert RGB-like data to grayscale so scene similarity can use one channel.
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)


def _scene_similarity(gold_rgn: np.ndarray, aligned_mira_rgn: np.ndarray, mask: np.ndarray) -> float:
    # Correlation of grayscale scene texture helps rank likely image pairs.
    gold = _gray(gold_rgn)[mask]
    mira = _gray(aligned_mira_rgn)[mask]
    if gold.size < 2 or np.std(gold) == 0 or np.std(mira) == 0:
        return 0.0
    value = float(np.corrcoef(gold, mira)[0, 1])
    return value if np.isfinite(value) else 0.0


def align_pair(
    mira_ndvi: np.ndarray,
    mira_rgn: np.ndarray,
    gold_rgn: np.ndarray,
    marker_id: int,
    dictionary: str,
) -> dict[str, np.ndarray | float]:
    # Find the marker in both images and compute a perspective transform from
    # Mira coordinates into gold-camera coordinates.
    mira_marker = detect_marker(mira_rgn, marker_id, dictionary)
    gold_marker = detect_marker(gold_rgn, marker_id, dictionary)
    homography = cv2.getPerspectiveTransform(mira_marker, gold_marker)
    height, width = gold_rgn.shape[:2]
    # warpPerspective uses the homography to place Mira NDVI onto the gold image
    # grid. Pixels outside the source become NaN.
    aligned_ndvi = cv2.warpPerspective(
        mira_ndvi,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    source_valid = np.isfinite(mira_ndvi).astype(np.uint8)
    valid = cv2.warpPerspective(
        source_valid,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    aligned_rgn = cv2.warpPerspective(
        mira_rgn,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    gold_ndvi = compute_ndvi(gold_rgn[..., 0], gold_rgn[..., 2])
    valid &= np.isfinite(aligned_ndvi) & np.isfinite(gold_ndvi)
    # Do not compare the ArUco marker itself; it is a printed target, not the
    # field scene.
    marker_mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(marker_mask, np.round(gold_marker).astype(np.int32), 1)
    valid &= marker_mask == 0
    if np.count_nonzero(valid) < 2:
        raise ValueError("Alignment produced insufficient valid overlapping pixels.")
    return {
        "mira_marker": mira_marker,
        "gold_marker": gold_marker,
        "homography": homography,
        "aligned_mira_ndvi": aligned_ndvi.astype(np.float32),
        "aligned_mira_rgn": aligned_rgn.astype(np.float32),
        "gold_ndvi": gold_ndvi.astype(np.float32),
        "valid_mask": valid,
        "scene_similarity": _scene_similarity(gold_rgn, aligned_rgn, valid),
    }


def _sample(gold: np.ndarray, mira: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Plotting every point can be slow. Evenly sample when there are too many.
    if gold.size <= MAX_PLOT_POINTS:
        return gold, mira
    indices = np.linspace(0, gold.size - 1, MAX_PLOT_POINTS, dtype=np.int64)
    return gold[indices], mira[indices]


def _write_regression_plot(path: Path, gold: np.ndarray, mira: np.ndarray, metrics: dict) -> None:
    # Write a visual plot showing gold NDVI versus Mira NDVI.
    plot_gold, plot_mira = _sample(gold, mira)
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.hexbin(plot_gold, plot_mira, gridsize=80, mincnt=1, cmap="viridis")
    limits = [
        float(min(plot_gold.min(), plot_mira.min())),
        float(max(plot_gold.max(), plot_mira.max())),
    ]
    axis.plot(limits, limits, "k--", label="1:1")
    axis.plot(
        limits,
        np.asarray(limits) * metrics["slope"] + metrics["intercept"],
        "r-",
        label="Fit",
    )
    block_size = metrics.get("block_size")
    label_suffix = f" ({block_size}x{block_size} block mean)" if block_size else ""
    axis.set(
        xlabel=f"Gold NDVI{label_suffix}",
        ylabel=f"Mira NDVI{label_suffix}",
        xlim=limits,
        ylim=limits,
    )
    axis.set_title(f"R2={metrics['r_squared']:.4f}  RMSE={metrics['rmse']:.4f}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_difference_heatmap(path: Path, difference: np.ndarray, valid: np.ndarray) -> None:
    # Write a map of where Mira is higher or lower than gold.
    shown = np.where(valid, difference, np.nan)
    limit = float(np.nanpercentile(np.abs(shown), 99))
    limit = max(limit, 1e-6)
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(shown, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_title("Mira NDVI - Gold NDVI")
    axis.axis("off")
    figure.colorbar(image, ax=axis, label="NDVI difference")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_alignment_overlay(path: Path, gold_rgn: np.ndarray, aligned_mira_rgn: np.ndarray) -> None:
    # Blend the aligned images so a human can verify the alignment visually.
    gold = preview_u8(gold_rgn, 2.2)
    mira = preview_u8(aligned_mira_rgn, 2.2)
    overlay = cv2.addWeighted(gold, 0.5, mira, 0.5, 0)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def compare_pair(
    mira_output: Path,
    gold_tiff: Path,
    output_dir: Path,
    target: dict,
) -> dict:
    # Full comparison workflow for one Mira output and one gold TIFF.
    mira_ndvi, mira_rgn = load_mira_output(mira_output)
    gold_rgn = load_rgn_tiff(gold_tiff)
    aligned = align_pair(
        mira_ndvi,
        mira_rgn,
        gold_rgn,
        int(target["marker_id"]),
        target["dictionary"],
    )
    valid = aligned["valid_mask"]
    # Keep both pixel-level metrics and block-mean metrics. The block metrics are
    # the primary summary because they reduce pixel noise.
    pixel_metrics = regression_metrics(
        aligned["gold_ndvi"][valid], aligned["aligned_mira_ndvi"][valid]
    )
    gold_values, mira_values = block_regression_values(
        aligned["gold_ndvi"], aligned["aligned_mira_ndvi"], valid
    )
    metrics = regression_metrics(gold_values, mira_values)
    metrics["valid_block_count"] = metrics.pop("valid_pixel_count")
    metrics["block_size"] = BLOCK_SIZE
    metrics["aggregation"] = "valid-pixel-weighted block mean"
    metrics["valid_overlap_fraction"] = float(np.mean(valid))
    metrics["scene_similarity"] = float(aligned["scene_similarity"])
    difference = aligned["aligned_mira_ndvi"] - aligned["gold_ndvi"]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save raw aligned arrays and visual evidence for review.
    tifffile.imwrite(output_dir / "mira_ndvi_aligned.tiff", aligned["aligned_mira_ndvi"])
    tifffile.imwrite(output_dir / "gold_ndvi.tiff", aligned["gold_ndvi"])
    tifffile.imwrite(output_dir / "ndvi_difference.tiff", difference.astype(np.float32))
    tifffile.imwrite(output_dir / "valid_overlap_mask.tiff", valid.astype(np.uint8))
    _write_regression_plot(output_dir / "regression.png", gold_values, mira_values, metrics)
    _write_difference_heatmap(output_dir / "difference_heatmap.png", difference, valid)
    _write_alignment_overlay(output_dir / "alignment_overlay.png", gold_rgn, aligned["aligned_mira_rgn"])
    report = {
        "status": "success",
        "mira_output": str(mira_output.resolve()),
        "gold_tiff": str(gold_tiff.resolve()),
        "visual_confirmation_required": True,
        "mira_marker_corners": aligned["mira_marker"].tolist(),
        "gold_marker_corners": aligned["gold_marker"].tolist(),
        "homography": aligned["homography"].tolist(),
        "metrics": metrics,
        "pixel_metrics": pixel_metrics,
    }
    (output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare_all(mira_root: Path, gold_root: Path, output_dir: Path, target: dict) -> dict:
    # Try every Mira output against every gold TIFF. This is useful when capture
    # ordering or naming is uncertain.
    mira_outputs = discover_mira_outputs(mira_root)
    gold_tiffs = discover_gold_tiffs(gold_root)
    if not mira_outputs:
        raise ValueError(f"No Mira output directories found under {mira_root}.")
    if not gold_tiffs:
        raise ValueError(f"No gold TIFFs found under {gold_root}.")
    rows = []
    for mira_output in mira_outputs:
        for gold_tiff in gold_tiffs:
            # Put each pair's evidence in its own folder.
            mira_relative = mira_output.relative_to(mira_root)
            gold_relative = gold_tiff.relative_to(gold_root).with_suffix("")
            pair_dir = output_dir / mira_relative / gold_relative
            try:
                report = compare_pair(mira_output, gold_tiff, pair_dir, target)
                rows.append(
                    {
                        "mira_output": str(mira_output.resolve()),
                        "gold_tiff": str(gold_tiff.resolve()),
                        "status": "success",
                        "error": "",
                        **report["metrics"],
                        **{f"pixel_{key}": value for key, value in report["pixel_metrics"].items()},
                    }
                )
            except Exception as exc:
                # Pair failures are recorded instead of stopping the whole batch.
                rows.append(
                    {
                        "mira_output": str(mira_output.resolve()),
                        "gold_tiff": str(gold_tiff.resolve()),
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    for mira_output in mira_outputs:
        # Rank successful gold matches for each Mira output by scene similarity
        # and valid overlap.
        successful = [
            row
            for row in rows
            if row["mira_output"] == str(mira_output.resolve()) and row["status"] == "success"
        ]
        successful.sort(
            key=lambda row: (row["scene_similarity"], row["valid_overlap_fraction"]),
            reverse=True,
        )
        for rank, row in enumerate(successful, start=1):
            row["rank"] = rank
            row["best_candidate"] = rank == 1
    output_dir.mkdir(parents=True, exist_ok=True)
    # CSV is convenient for spreadsheets; JSON keeps the full structured summary.
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "pair_rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "visual_confirmation_required": True,
        "mira_output_count": len(mira_outputs),
        "gold_tiff_count": len(gold_tiffs),
        "successful_pair_count": sum(row["status"] == "success" for row in rows),
        "failed_pair_count": sum(row["status"] == "failed" for row in rows),
        "pairs": rows,
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
