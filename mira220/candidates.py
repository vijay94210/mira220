from __future__ import annotations

"""Find and describe interesting NDVI candidate regions.

Candidate detection is a review aid. It does not decide final plant health by
itself. Instead, it finds regions in processed NDVI images that look worth a
human checking: broad vegetation regions, texture-heavy regions, and lower-NDVI
areas. It also tries to ignore calibration targets and obvious image artifacts.
"""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import tifffile

from .imaging import (
    NDVI_CROP_MARGINS,
    crop_valid_region,
    crop_valid_region_or_full,
    detect_marker,
    ndvi_false_color,
    patch_polygons,
)


ThresholdMode = Literal["manual", "otsu", "adaptive", "percentile"]
GeometryMode = Literal["auto", "box", "contour"]

# Overlay colors use OpenCV's BGR order because OpenCV writes images that way.
CANDIDATE_COLORS_BGR = {
    "broad_ndvi_region": (0, 0, 255),
    "texture_region": (0, 165, 255),
}

# RGB hex colors are stored in summaries for people or browser-based tools.
CANDIDATE_COLORS_RGB = {
    "broad_ndvi_region": "#FF0000",
    "texture_region": "#FFA500",
}

# These class names explain what kind of NDVI range a candidate came from.
NDVI_CLASS_LABELS = {
    "high_vegetation": "High vegetation",
    "medium_vegetation": "Medium vegetation",
    "low_vegetation": "Low vegetation / exposed ground",
    "non_vegetation_artifact": "Non-vegetation / artifact",
}
NDVI_CLASS_COLORS_BGR = {
    "high_vegetation": (35, 145, 35),
    "medium_vegetation": (90, 185, 120),
    "low_vegetation": (60, 145, 215),
    "non_vegetation_artifact": (180, 80, 180),
}

# The ordered class list matters because masks are made exclusive in this order:
# high vegetation gets first claim, then medium, then low, then non-vegetation.
NDVI_CLASSES = (
    "high_vegetation",
    "medium_vegetation",
    "low_vegetation",
    "non_vegetation_artifact",
)

# CSV field order is fixed so spreadsheets open candidate tables consistently.
CSV_FIELDS = (
    "id",
    "candidate_type",
    "ndvi_class",
    "geometry_type",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "centroid_x",
    "centroid_y",
    "area_pixels",
    "perimeter_pixels",
    "mean_ndvi",
    "median_ndvi",
    "min_ndvi",
    "max_ndvi",
    "std_ndvi",
    "contrast_score",
    "aspect_ratio",
    "solidity",
    "circularity",
    "extent",
    "texture_score",
    "line_length",
    "line_angle_degrees",
    "total_score",
    "points",
)


@dataclass(frozen=True)
class CandidateConfig:
    # Tunable settings for candidate detection. These defaults are chosen for
    # Mira220 NDVI review images but can be overridden from the CLI.
    top_n: int = 25
    threshold_mode: ThresholdMode = "percentile"
    manual_threshold: float | None = None
    invert_mask: bool = False
    min_area: int = 15000
    min_contrast: float = 0.03
    large_region_area: int = 50000
    morph_open: int = 5
    morph_close: int = 31
    hole_fill_area: int = 4000
    simplify_epsilon: float = 8.0
    texture_window: int = 21
    texture_sensitivity: float = 0.18
    geometry_min_solidity: float = 0.82
    geometry: GeometryMode = "contour"
    box_thickness: int = 8
    crop: bool = True
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    high_threshold: float = 0.45
    medium_threshold: float = 0.18
    low_threshold: float = 0.05
    show_class_colors: bool = False


@dataclass(frozen=True)
class NdviSource:
    # Bundle all input data needed by candidate detection: the NDVI values, the
    # preview image used for overlays, and crop/source metadata.
    input_path: Path
    ndvi_path: Path | None
    ndvi: np.ndarray
    background: np.ndarray
    physical_ndvi: bool
    crop_margins: tuple[int, int, int, int] | None
    original_shape: tuple[int, int]
    source_bgr: np.ndarray | None = None


@dataclass(frozen=True)
class TargetExclusion:
    # Mask is the pixels to ignore, usually the calibration target. Metadata is
    # written to JSON so a reviewer knows whether exclusion succeeded.
    mask: np.ndarray | None
    metadata: dict[str, Any]


def _read_ndvi_tiff(path: Path) -> np.ndarray:
    # Candidate detection expects a single-channel floating NDVI TIFF.
    ndvi = tifffile.imread(path)
    if ndvi.ndim != 2:
        raise ValueError(f"{path} must be a two-dimensional NDVI TIFF.")
    return ndvi.astype(np.float32)


def _filled_ndvi(ndvi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Many image operations do not like NaN values. This returns a filled copy
    # plus a mask showing which pixels were originally valid.
    values = np.asarray(ndvi, np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("NDVI input contains no finite pixels.")
    filled = values.copy()
    filled[~finite] = float(np.nanmedian(values[finite]))
    return filled, finite


def robust_normalize(
    ndvi: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    # Robust normalization ignores extreme low/high percentiles so one odd pixel
    # does not control the whole display or threshold scale.
    filled, finite = _filled_ndvi(ndvi)
    low, high = np.percentile(filled[finite], [low_percentile, high_percentile])
    if high <= low:
        high = low + 1e-6
    normalized = np.clip((filled - low) / (high - low), 0.0, 1.0)
    normalized[~finite] = 0.0
    return normalized.astype(np.float32)


def _default_background(ndvi: np.ndarray) -> np.ndarray:
    # If no preview PNG exists, create a false-color background from NDVI itself.
    filled, finite = _filled_ndvi(ndvi)
    filled[~finite] = float(np.nanmedian(filled[finite]))
    return cv2.cvtColor(ndvi_false_color(filled, -0.2, 0.7), cv2.COLOR_RGB2BGR)


def _read_background(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    # Only use a background image if it exists and matches the expected shape.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != shape:
        return None
    return image


def _crop_or_full(image: np.ndarray, crop: bool) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    # Crop to the central valid region when possible. If the image is too small,
    # fall back to the full image.
    if not crop:
        return np.asarray(image).copy(), None
    try:
        return crop_valid_region(image), NDVI_CROP_MARGINS
    except ValueError:
        return np.asarray(image).copy(), None


def _load_png_source(path: Path, crop: bool) -> NdviSource:
    # A PNG can be used directly for visual review. If an adjacent ndvi.tiff is
    # available, use that real NDVI data instead of guessing from image brightness.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image {path}.")
    adjacent_ndvi = path.parent / "ndvi.tiff"
    if adjacent_ndvi.is_file():
        physical = _read_ndvi_tiff(adjacent_ndvi)
        expected = crop_valid_region_or_full(physical).shape if crop else physical.shape
        if expected == image.shape[:2]:
            ndvi, margins = _crop_or_full(physical, crop)
            return NdviSource(path, adjacent_ndvi, ndvi, image, True, margins, physical.shape, image)
        if physical.shape == image.shape[:2]:
            return NdviSource(path, adjacent_ndvi, physical, image, True, None, physical.shape, image)
    source_bgr, margins = _crop_or_full(image, crop)
    # Without a physical NDVI TIFF, use grayscale as a fallback signal.
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return NdviSource(path, None, gray, source_bgr, False, margins, image.shape[:2], source_bgr)


def load_ndvi_source(path: Path, crop: bool = True) -> NdviSource:
    # Accept either a processed output folder, an NDVI TIFF, or a preview image.
    path = Path(path)
    if path.is_dir():
        ndvi_path = path / "ndvi.tiff"
        if not ndvi_path.is_file():
            raise ValueError(f"{path} must contain ndvi.tiff.")
        full_ndvi = _read_ndvi_tiff(ndvi_path)
        ndvi, margins = _crop_or_full(full_ndvi, crop)
        background = None
        if crop:
            # Prefer the already-cropped false-color preview if it matches.
            background = _read_background(path / "ndvi_false_color_crop.png", ndvi.shape)
        if background is None:
            # Otherwise crop the full false-color preview.
            full_background = _read_background(path / "ndvi_false_color.png", full_ndvi.shape)
            if full_background is not None:
                background, _ = _crop_or_full(full_background, crop)
        if background is None:
            background = _default_background(ndvi)
        return NdviSource(path, ndvi_path, ndvi, background, True, margins, full_ndvi.shape, background)

    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        full_ndvi = _read_ndvi_tiff(path)
        ndvi, margins = _crop_or_full(full_ndvi, crop)
        return NdviSource(path, path, ndvi, _default_background(ndvi), True, margins, full_ndvi.shape)
    if suffix in {".png", ".jpg", ".jpeg"}:
        return _load_png_source(path, crop)
    raise ValueError(f"Unsupported candidate input type: {path}")


def _odd_size(value: int, minimum: int = 3) -> int:
    # OpenCV filters often require odd kernel sizes.
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _kernel(size: int) -> np.ndarray:
    # Elliptical kernels smooth shapes without making corners too square.
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_size(size), _odd_size(size)))


def _threshold_mask(normalized: np.ndarray, config: CandidateConfig) -> np.ndarray:
    # Build a basic binary mask from normalized image values. Several threshold
    # modes are supported for experimentation.
    image_u8 = np.round(np.clip(normalized, 0, 1) * 255).astype(np.uint8)
    if config.threshold_mode == "manual":
        threshold = 0.5 if config.manual_threshold is None else float(config.manual_threshold)
        mask = normalized >= threshold
    elif config.threshold_mode == "otsu":
        _, mask_u8 = cv2.threshold(image_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask_u8 > 0
    elif config.threshold_mode == "adaptive":
        size = _odd_size(config.texture_window, 15)
        mask_u8 = cv2.adaptiveThreshold(image_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, -2)
        mask = mask_u8 > 0
    else:
        threshold = np.percentile(normalized, 45)
        mask = normalized >= threshold
    return ~mask if config.invert_mask else mask


def _false_color_class_masks(image_bgr: np.ndarray, config: CandidateConfig) -> dict[str, np.ndarray]:
    # If only a false-color PNG is available, estimate NDVI classes from color.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    green = (hue >= 35) & (hue <= 95) & (saturation >= 0.18)
    orange = ((hue <= 25) | (hue >= 165)) & (saturation >= 0.12)
    high = green & (value < 0.55)
    medium = green & ~high
    low = orange
    non = ~(high | medium | low)
    return {
        "high_vegetation": high.astype(np.uint8) * 255,
        "medium_vegetation": medium.astype(np.uint8) * 255,
        "low_vegetation": low.astype(np.uint8) * 255,
        "non_vegetation_artifact": non.astype(np.uint8) * 255,
    }


def _fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    # Fill small empty islands inside a region, but do not fill holes connected
    # to the image edge.
    if max_hole_area <= 0 or not np.any(mask):
        return mask
    inverse = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    output = mask.copy()
    height, width = mask.shape
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_edge = x == 0 or y == 0 or x + w >= width or y + h >= height
        if not touches_edge and area <= max_hole_area:
            output[labels == label] = 255
    return output


def _clean_and_sieve(mask: np.ndarray, config: CandidateConfig) -> np.ndarray:
    # Morphological opening removes tiny specks. Closing bridges small gaps.
    # The connected-component step keeps only regions large enough to review.
    clean = (mask > 0).astype(np.uint8) * 255
    if config.morph_open > 0:
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, _kernel(config.morph_open))
    if config.morph_close > 0:
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, _kernel(config.morph_close))
    clean = _fill_small_holes(clean, config.hole_fill_area)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((clean > 0).astype(np.uint8), 8)
    sieved = np.zeros_like(clean)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= config.min_area:
            sieved[labels == label] = 255
    return sieved


def _clean_class_mask(class_name: str, mask: np.ndarray, config: CandidateConfig) -> np.ndarray:
    # Vegetation classes use a smaller close kernel so nearby plant regions do
    # not get merged too aggressively.
    if class_name not in {"high_vegetation", "medium_vegetation"}:
        return _clean_and_sieve(mask, config)
    class_config = CandidateConfig(
        **{
            **asdict(config),
            "morph_close": min(config.morph_close, 9),
        }
    )
    return _clean_and_sieve(mask, class_config)


def build_class_masks(
    source_or_ndvi: NdviSource | np.ndarray,
    config: CandidateConfig,
) -> dict[str, np.ndarray]:
    # Create one cleaned mask per NDVI class. Masks are then made exclusive so a
    # pixel belongs to only one class.
    if isinstance(source_or_ndvi, NdviSource) and not source_or_ndvi.physical_ndvi and source_or_ndvi.source_bgr is not None:
        raw_masks = _false_color_class_masks(source_or_ndvi.source_bgr, config)
    else:
        ndvi = source_or_ndvi.ndvi if isinstance(source_or_ndvi, NdviSource) else np.asarray(source_or_ndvi, np.float32)
        filled, finite = _filled_ndvi(ndvi)
        smooth = cv2.GaussianBlur(filled, (0, 0), 2.0)
        if isinstance(source_or_ndvi, NdviSource) and source_or_ndvi.physical_ndvi:
            # Physical NDVI values can use meaningful thresholds.
            high = finite & (smooth >= config.high_threshold)
            medium = finite & (smooth >= config.medium_threshold) & (smooth < config.high_threshold)
            low = finite & (smooth >= config.low_threshold) & (smooth < config.medium_threshold)
            non = finite & (smooth < config.low_threshold)
        else:
            # Non-physical image sources use normalized brightness thresholds.
            normalized = robust_normalize(filled, config.percentile_low, config.percentile_high)
            base = _threshold_mask(normalized, config)
            high = finite & base & (normalized >= 0.68)
            medium = finite & base & (normalized >= 0.38) & (normalized < 0.68)
            low = finite & (normalized >= 0.18) & (normalized < 0.38)
            non = finite & (normalized < 0.18)
        raw_masks = {
            "high_vegetation": high.astype(np.uint8) * 255,
            "medium_vegetation": medium.astype(np.uint8) * 255,
            "low_vegetation": low.astype(np.uint8) * 255,
            "non_vegetation_artifact": non.astype(np.uint8) * 255,
        }
    cleaned = {name: _clean_class_mask(name, mask, config) for name, mask in raw_masks.items()}
    occupied = np.zeros(next(iter(cleaned.values())).shape, bool)
    exclusive: dict[str, np.ndarray] = {}
    for name in NDVI_CLASSES:
        mask = (cleaned[name] > 0) & ~occupied
        exclusive[name] = mask.astype(np.uint8) * 255
        occupied |= mask
    return exclusive


def _local_texture(normalized: np.ndarray, config: CandidateConfig) -> np.ndarray:
    # Texture score combines local standard deviation and edge strength. It helps
    # find regions that may not be extreme in average NDVI but still look unusual.
    size = _odd_size(config.texture_window)
    mean = cv2.blur(normalized, (size, size))
    mean_sq = cv2.blur(normalized * normalized, (size, size))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    lap = np.abs(cv2.Laplacian(normalized, cv2.CV_32F, ksize=3))
    lap_norm = robust_normalize(lap)
    return np.clip(0.7 * robust_normalize(std) + 0.3 * lap_norm, 0.0, 1.0)


def _contrast_score(normalized: np.ndarray, component: np.ndarray) -> float:
    # Compare the inside of a candidate to a ring around it. Strong contrast
    # means the region stands out from its neighbors.
    if not np.any(component):
        return 0.0
    kernel = _kernel(17)
    dilated = cv2.dilate(component.astype(np.uint8), kernel) > 0
    ring = dilated & ~component
    inside = normalized[component]
    outside = normalized[ring]
    if inside.size == 0 or outside.size == 0:
        return 0.0
    return float(abs(np.mean(inside) - np.mean(outside)))


def _candidate_geometry(contour: np.ndarray, geometry: GeometryMode, config: CandidateConfig) -> tuple[str, list[list[int]]]:
    # Convert a contour into either a simple box or a simplified polygon.
    x, y, w, h = cv2.boundingRect(contour)
    area = max(1.0, cv2.contourArea(contour))
    extent = area / max(1, w * h)
    hull = cv2.convexHull(contour)
    hull_area = max(1.0, cv2.contourArea(hull))
    solidity = area / hull_area
    panel_like = h > w * 1.3 and extent > 0.35 and solidity >= config.geometry_min_solidity * 0.75
    if geometry == "box" or (geometry == "auto" and panel_like):
        return "box", [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]
    simplified = cv2.approxPolyDP(contour, max(0.5, config.simplify_epsilon), True)
    return "contour", simplified.reshape(-1, 2).astype(int).tolist()


def _component_stats(
    ndvi: np.ndarray,
    normalized: np.ndarray,
    texture: np.ndarray,
    component: np.ndarray,
    contour: np.ndarray,
) -> dict[str, float]:
    # Calculate measurements that are written to CSV/JSON and used for filtering
    # and scoring.
    x, y, w, h = cv2.boundingRect(contour)
    values = ndvi[component]
    if values.size == 0:
        values = np.array([np.nan], np.float32)
    area = float(np.count_nonzero(component))
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = max(1.0, cv2.contourArea(cv2.convexHull(contour)))
    contour_area = max(1.0, cv2.contourArea(contour))
    return {
        "bbox_x": float(x),
        "bbox_y": float(y),
        "bbox_width": float(w),
        "bbox_height": float(h),
        "centroid_x": float(x + w / 2.0),
        "centroid_y": float(y + h / 2.0),
        "area_pixels": area,
        "perimeter_pixels": perimeter,
        "mean_ndvi": float(np.nanmean(values)),
        "median_ndvi": float(np.nanmedian(values)),
        "min_ndvi": float(np.nanmin(values)),
        "max_ndvi": float(np.nanmax(values)),
        "std_ndvi": float(np.nanstd(values)),
        "contrast_score": _contrast_score(normalized, component),
        "aspect_ratio": float(w / max(1, h)),
        "solidity": float(contour_area / hull_area),
        "circularity": float((4.0 * np.pi * contour_area) / max(1.0, perimeter * perimeter)),
        "extent": float(contour_area / max(1, w * h)),
        "texture_score": float(np.mean(texture[component])) if np.any(component) else 0.0,
    }


def _region_candidates(
    ndvi: np.ndarray,
    normalized: np.ndarray,
    texture: np.ndarray,
    masks: dict[str, np.ndarray],
    config: CandidateConfig,
) -> list[dict[str, Any]]:
    # Broad region candidates come directly from each cleaned NDVI class mask.
    features: list[dict[str, Any]] = []
    for ndvi_class, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < config.min_area:
                continue
            component_mask = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, -1)
            component = component_mask > 0
            stats = _component_stats(ndvi, normalized, texture, component, contour)
            height, width = mask.shape
            if stats["area_pixels"] > 0.90 * height * width:
                # A nearly full-frame region is usually a thresholding failure.
                continue
            if _is_artifact_candidate(stats, mask.shape, ndvi_class):
                continue
            if stats["contrast_score"] < config.min_contrast and stats["area_pixels"] < config.large_region_area:
                # Small low-contrast regions are not useful enough to report.
                continue
            geometry_type, points = _candidate_geometry(contour, config.geometry, config)
            score = stats["area_pixels"] * (1.0 + stats["contrast_score"] + 0.2 * stats["texture_score"])
            candidate_type = (
                "texture_region"
                if stats["texture_score"] >= max(0.45, config.texture_sensitivity * 3.0)
                else "broad_ndvi_region"
            )
            features.append(
                {
                    "candidate_type": candidate_type,
                    "ndvi_class": ndvi_class,
                    "geometry_type": geometry_type,
                    **_integer_bbox_stats(stats),
                    "line_length": 0.0,
                    "line_angle_degrees": 0.0,
                    "total_score": float(score),
                    "points": points,
                }
            )
    return features


def _rasterize_candidates(shape: tuple[int, int], candidates: list[dict[str, Any]]) -> np.ndarray:
    # Turn candidate polygons back into a mask. This helps avoid duplicate
    # texture candidates on top of broad candidates.
    mask = np.zeros(shape, np.uint8)
    for candidate in candidates:
        points = np.array(candidate.get("points", []), np.int32)
        if points.size == 0 or candidate.get("geometry_type") == "line":
            continue
        cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    return mask


def _texture_candidates(
    ndvi: np.ndarray,
    normalized: np.ndarray,
    texture: np.ndarray,
    masks: dict[str, np.ndarray],
    broad_candidates: list[dict[str, Any]],
    config: CandidateConfig,
) -> list[dict[str, Any]]:
    # Texture candidates look for rough or detailed areas inside medium/low
    # vegetation zones that were not already covered by broad candidates.
    candidate_zone = (masks["medium_vegetation"] > 0) | (masks["low_vegetation"] > 0)
    occupied = _rasterize_candidates(normalized.shape, broad_candidates) > 0
    texture_mask = (texture >= config.texture_sensitivity) & candidate_zone
    texture_config = CandidateConfig(**{**asdict(config), "morph_close": min(config.morph_close, 9)})
    clean = _clean_and_sieve(texture_mask.astype(np.uint8) * 255, texture_config)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features: list[dict[str, Any]] = []
    for contour in contours:
        if cv2.contourArea(contour) < config.min_area:
            continue
        component_mask = np.zeros(clean.shape, np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        component = component_mask > 0
        overlap = np.count_nonzero(component & occupied) / max(1, np.count_nonzero(component))
        if overlap > 0.35:
            # Skip texture regions that mostly duplicate an existing candidate.
            continue
        stats = _component_stats(ndvi, normalized, texture, component, contour)
        if _is_artifact_candidate(stats, clean.shape, "medium_vegetation"):
            continue
        if stats["texture_score"] < config.texture_sensitivity:
            continue
        geometry_type, points = _candidate_geometry(contour, "contour", config)
        score = stats["area_pixels"] * (1.0 + stats["texture_score"] + stats["contrast_score"])
        features.append(
            {
                "candidate_type": "texture_region",
                "ndvi_class": "medium_vegetation",
                "geometry_type": geometry_type,
                **_integer_bbox_stats(stats),
                "line_length": 0.0,
                "line_angle_degrees": 0.0,
                "total_score": float(score),
                "points": points,
            }
        )
    return features


def _integer_bbox_stats(stats: dict[str, float]) -> dict[str, float | int]:
    # Bounding box values are easier to read as integers; scores stay floats.
    output: dict[str, float | int] = {}
    for key, value in stats.items():
        if key in {"bbox_x", "bbox_y", "bbox_width", "bbox_height", "area_pixels"}:
            output[key] = int(round(value))
        else:
            output[key] = float(value)
    return output


def _crop_mask_to_source(mask: np.ndarray, source: NdviSource) -> np.ndarray:
    # Target masks are built in full-image coordinates. Candidate detection may
    # be running on a cropped NDVI image, so crop or resize the mask to match.
    if source.crop_margins is None:
        if mask.shape != source.ndvi.shape:
            return cv2.resize(mask, (source.ndvi.shape[1], source.ndvi.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask
    left, top, right, bottom = source.crop_margins
    height, width = mask.shape
    cropped = mask[top : height - bottom, left : width - right]
    if cropped.shape != source.ndvi.shape:
        return cv2.resize(cropped, (source.ndvi.shape[1], source.ndvi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cropped


def _candidate_overlap_fraction(candidate: dict[str, Any], mask: np.ndarray) -> float:
    # Measure how much of a candidate polygon overlaps an exclusion mask.
    candidate_mask = _rasterize_candidates(mask.shape, [candidate]) > 0
    area = int(np.count_nonzero(candidate_mask))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(candidate_mask & (mask > 0)) / area)


def _candidate_exclusion_bbox_fraction(candidate: dict[str, Any], mask: np.ndarray) -> float:
    # Also check the bounding box, because some candidate geometries may be
    # simplified and miss part of the excluded region.
    x = int(candidate.get("bbox_x", 0))
    y = int(candidate.get("bbox_y", 0))
    w = int(candidate.get("bbox_width", 0))
    h = int(candidate.get("bbox_height", 0))
    if w <= 0 or h <= 0:
        return 0.0
    x0 = int(np.clip(x, 0, mask.shape[1]))
    y0 = int(np.clip(y, 0, mask.shape[0]))
    x1 = int(np.clip(x + w, 0, mask.shape[1]))
    y1 = int(np.clip(y + h, 0, mask.shape[0]))
    area = max(1, int(candidate.get("area_pixels", 0)))
    return float(np.count_nonzero(mask[y0:y1, x0:x1] > 0) / area)


def _apply_exclusion_mask(
    masks: dict[str, np.ndarray],
    exclusion_mask: np.ndarray | None,
) -> dict[str, np.ndarray]:
    # Remove the calibration target area from class masks before candidates are
    # found. A slightly wider vertical band is removed for some non-high classes
    # because target boards can create nearby artifacts.
    if exclusion_mask is None:
        return masks
    keep = exclusion_mask == 0
    non_high_keep = keep.copy()
    rows, cols = np.where(exclusion_mask > 0)
    if rows.size and cols.size:
        pad = 8
        x0 = max(0, int(cols.min()) - pad)
        x1 = min(exclusion_mask.shape[1], int(cols.max()) + pad + 1)
        non_high_keep[:, x0:x1] = False
    return {
        name: (
            (mask > 0)
            & (non_high_keep if name in {"medium_vegetation", "non_vegetation_artifact"} else keep)
        ).astype(np.uint8)
        * 255
        for name, mask in masks.items()
    }


def _filter_excluded_candidates(
    candidates: list[dict[str, Any]],
    exclusion_mask: np.ndarray | None,
    overlap_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    # Final safety pass: drop candidates that still overlap the excluded target
    # area too much.
    if exclusion_mask is None:
        return candidates
    return [
        candidate
        for candidate in candidates
        if _candidate_overlap_fraction(candidate, exclusion_mask) <= overlap_threshold
        and _candidate_exclusion_bbox_fraction(candidate, exclusion_mask) <= 0.50
    ]


def _is_artifact_candidate(stats: dict[str, float], shape: tuple[int, int], ndvi_class: str) -> bool:
    # Heuristic filters for shapes that are likely borders, strips, or boards
    # rather than useful plant regions.
    height, width = shape
    x = int(round(stats["bbox_x"]))
    y = int(round(stats["bbox_y"]))
    w = int(round(stats["bbox_width"]))
    h = int(round(stats["bbox_height"]))
    aspect_ratio = float(stats["aspect_ratio"])
    touches_frame = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
    edge_margin = max(12, int(0.05 * min(height, width)))
    near_frame = x <= edge_margin or y <= edge_margin or x + w >= width - edge_margin or y + h >= height - edge_margin
    skinny = aspect_ratio >= 10.0 or aspect_ratio <= 0.10
    frame_spanning = w >= 0.95 * width and h >= 0.95 * height
    edge_strip = near_frame and ndvi_class != "high_vegetation" and (aspect_ratio >= 4.0 or aspect_ratio <= 0.33)
    if ndvi_class != "high_vegetation" and frame_spanning:
        return True
    if edge_strip and stats["area_pixels"] < 0.20 * height * width:
        return True
    if skinny and stats["area_pixels"] < 0.20 * height * width:
        return True
    if touches_frame and skinny and stats["extent"] > 0.25:
        return True
    board_like = (
        ndvi_class in {"low_vegetation", "non_vegetation_artifact"}
        and stats["extent"] >= 0.78
        and stats["solidity"] >= 0.92
        and 0.35 <= aspect_ratio <= 3.0
        and stats["area_pixels"] < 0.20 * height * width
    )
    return board_like


def target_exclusion_from_source(
    source: NdviSource,
    target: dict[str, Any] | None,
    padding: int = 24,
) -> TargetExclusion:
    # Try to build a mask around the calibration marker and patch board. If
    # anything is missing or detection fails, return metadata explaining why and
    # let candidate detection continue without exclusion.
    metadata: dict[str, Any] = {
        "enabled": target is not None,
        "detected": False,
        "marker_id": int(target["marker_id"]) if target is not None and "marker_id" in target else None,
        "reason": None,
    }
    if target is None:
        metadata["reason"] = "target config not provided"
        return TargetExclusion(None, metadata)
    if not source.input_path.is_dir():
        metadata["reason"] = "input is not a processed output directory"
        return TargetExclusion(None, metadata)
    rgn_path = source.input_path / "rgn_reflectance.tiff"
    if not rgn_path.is_file():
        metadata["reason"] = "rgn_reflectance.tiff not found"
        return TargetExclusion(None, metadata)
    try:
        rgn = tifffile.imread(rgn_path).astype(np.float32)
        marker = detect_marker(rgn, int(target["marker_id"]), target["dictionary"])
        polygons = patch_polygons(marker, target)
    except Exception as exc:
        metadata["reason"] = str(exc)
        return TargetExclusion(None, metadata)

    full_mask = np.zeros(rgn.shape[:2], np.uint8)
    # Mark the ArUco marker and every target patch.
    cv2.fillConvexPoly(full_mask, np.round(marker).astype(np.int32), 255)
    for polygon in polygons:
        cv2.fillConvexPoly(full_mask, np.round(polygon).astype(np.int32), 255)
    all_points = np.vstack([marker, *polygons])
    hull = cv2.convexHull(np.round(all_points).astype(np.int32))
    # Fill the convex hull around all target pieces so gaps between patches are
    # also excluded.
    cv2.fillConvexPoly(full_mask, hull, 255)
    if padding > 0:
        # Expand the mask a little to catch shadows and border artifacts.
        full_mask = cv2.dilate(full_mask, _kernel(padding))
    mask = _crop_mask_to_source(full_mask, source)
    metadata.update(
        {
            "detected": True,
            "reason": None,
            "excluded_pixels": int(np.count_nonzero(mask)),
            "padding_pixels": int(padding),
        }
    )
    return TargetExclusion(mask, metadata)


def detect_candidates(
    source_or_ndvi: NdviSource | np.ndarray,
    config: CandidateConfig | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    # High-level candidate detection pipeline:
    # 1. Normalize NDVI for thresholding and texture.
    # 2. Build class masks.
    # 3. Remove target-board pixels if an exclusion mask exists.
    # 4. Find broad and texture candidates.
    # 5. Sort by score and keep the top N.
    config = config or CandidateConfig()
    source = source_or_ndvi if isinstance(source_or_ndvi, NdviSource) else None
    ndvi = source.ndvi if source is not None else np.asarray(source_or_ndvi, np.float32)
    normalized = robust_normalize(ndvi, config.percentile_low, config.percentile_high)
    texture = _local_texture(normalized, config)
    masks = build_class_masks(source if source is not None else ndvi, config)
    masks = _apply_exclusion_mask(masks, exclusion_mask)
    features = _region_candidates(ndvi, normalized, texture, masks, config)
    features.extend(_texture_candidates(ndvi, normalized, texture, masks, features, config))
    features = _filter_excluded_candidates(features, exclusion_mask)
    features.sort(key=lambda item: item["total_score"], reverse=True)
    for index, feature in enumerate(features[: config.top_n], start=1):
        # Candidate IDs are assigned after sorting so ID 1 is the strongest
        # candidate according to the score.
        feature["id"] = index
    return features[: config.top_n], masks


def write_candidates_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    # Write a spreadsheet-friendly table of all candidate measurements.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            row = {field: candidate.get(field, "") for field in CSV_FIELDS}
            row["points"] = json.dumps(candidate.get("points", []))
            writer.writerow(row)


def write_candidate_mask(path: Path, shape: tuple[int, int], candidates: list[dict[str, Any]]) -> None:
    # Write an ID mask: pixel value 1 belongs to candidate 1, value 2 belongs to
    # candidate 2, and so on. Zero means no candidate.
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, np.uint16)
    for candidate in candidates:
        points = np.array(candidate.get("points", []), np.int32)
        if points.size == 0:
            continue
        value = int(candidate["id"])
        if candidate["geometry_type"] == "line":
            cv2.line(mask, tuple(points[0]), tuple(points[-1]), value, 5)
        elif candidate["geometry_type"] == "box":
            cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], value)
        else:
            cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], value)
    cv2.imwrite(str(path), mask)


def _draw_points(overlay: np.ndarray, candidate: dict[str, Any], thickness: int) -> None:
    # Draw a candidate outline with a black halo so it is visible on bright and
    # dark backgrounds.
    points = candidate.get("points", [])
    if len(points) < 2:
        return
    color = CANDIDATE_COLORS_BGR[str(candidate["candidate_type"])]
    pts = np.array(points, np.int32).reshape(-1, 1, 2)
    if candidate["geometry_type"] == "line":
        cv2.polylines(overlay, [pts], False, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.polylines(overlay, [pts], False, color, thickness, cv2.LINE_AA)
    else:
        cv2.polylines(overlay, [pts], True, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.polylines(overlay, [pts], True, color, thickness, cv2.LINE_AA)


def _draw_candidate_label(overlay: np.ndarray, candidate: dict[str, Any]) -> None:
    # Draw the candidate ID near its centroid.
    label = str(candidate.get("id", ""))
    if not label:
        return
    x = int(round(float(candidate.get("centroid_x", candidate.get("bbox_x", 0)))))
    y = int(round(float(candidate.get("centroid_y", candidate.get("bbox_y", 0)))))
    x = int(np.clip(x, 8, overlay.shape[1] - 8))
    y = int(np.clip(y, 20, overlay.shape[0] - 8))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 5
    top_left = (x - pad, y - text_height - pad)
    bottom_right = (x + text_width + pad, y + baseline + pad)
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(overlay, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _put_panel_text(
    panel: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (30, 30, 30),
    thickness: int = 1,
) -> None:
    # Small wrapper so legend text uses one consistent font style.
    cv2.putText(panel, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_wrapped_text(panel: np.ndarray, text: str, x: int, y: int, max_width: int) -> int:
    # OpenCV does not wrap text automatically, so split labels into lines that
    # fit inside the legend panel.
    words = text.split()
    line = ""
    for word in words:
        trial = word if not line else f"{line} {word}"
        width = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        if width <= max_width:
            line = trial
            continue
        if line:
            _put_panel_text(panel, line, (x, y), 0.48)
            y += 18
        line = word
    if line:
        _put_panel_text(panel, line, (x, y), 0.48)
        y += 18
    return y


def _draw_ndvi_legend(panel: np.ndarray, x: int, y: int, height: int = 210) -> int:
    # Draw the false-color scale that explains how NDVI values map to colors.
    _put_panel_text(panel, "NDVI false color", (x, y), 0.62, thickness=2)
    y += 18
    height = max(30, min(height, panel.shape[0] - y - 12))
    values = np.linspace(0.7, -0.2, height, dtype=np.float32).reshape(height, 1)
    bar = cv2.cvtColor(ndvi_false_color(values, -0.2, 0.7), cv2.COLOR_RGB2BGR)
    bar = np.repeat(bar, 24, axis=1)
    panel[y : y + height, x : x + 24] = bar
    cv2.rectangle(panel, (x, y), (x + 24, y + height), (40, 40, 40), 1)
    for value, label in ((0.7, "0.70 high"), (0.25, "0.25 mid"), (-0.2, "-0.20 low")):
        tick_y = int(y + (0.7 - value) / 0.9 * height)
        cv2.line(panel, (x + 26, tick_y), (x + 36, tick_y), (40, 40, 40), 1)
        _put_panel_text(panel, label, (x + 42, tick_y + 5), 0.48)
    return y + height + 34


def _draw_class_legend(panel: np.ndarray, candidates: list[dict[str, Any]], x: int, y: int, width: int) -> int:
    # Draw class meanings and the list of candidate IDs included in the overlay.
    _put_panel_text(panel, "Candidate classes", (x, y), 0.62, thickness=2)
    y += 24
    for class_name in NDVI_CLASSES:
        color = NDVI_CLASS_COLORS_BGR[class_name]
        cv2.rectangle(panel, (x, y - 12), (x + 18, y + 6), color, -1, cv2.LINE_AA)
        cv2.rectangle(panel, (x, y - 12), (x + 18, y + 6), (40, 40, 40), 1, cv2.LINE_AA)
        y = _draw_wrapped_text(panel, NDVI_CLASS_LABELS[class_name], x + 28, y + 2, width - 36)
        y += 8
    y += 8
    _put_panel_text(panel, "Candidate IDs", (x, y), 0.62, thickness=2)
    y += 24
    for candidate in candidates[:18]:
        class_name = str(candidate.get("ndvi_class", ""))
        label = NDVI_CLASS_LABELS.get(class_name, class_name)
        text = f"{candidate.get('id')}: {label}"
        y = _draw_wrapped_text(panel, text, x, y, width)
        y += 4
        if y > panel.shape[0] - 24:
            _put_panel_text(panel, "...", (x, y), 0.55)
            break
    return y


def _append_legend_panel(overlay: np.ndarray, candidates: list[dict[str, Any]]) -> np.ndarray:
    # Add a right-side explanation panel to the overlay image.
    panel_width = 320
    gap = 12
    height = overlay.shape[0]
    panel = np.full((height, panel_width, 3), 245, np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, height - 1), (190, 190, 190), 1)
    y = _draw_ndvi_legend(panel, 18, 28)
    _draw_class_legend(panel, candidates, 18, y, panel_width - 36)
    spacer = np.full((height, gap, 3), 255, np.uint8)
    return np.hstack([overlay, spacer, panel])


def write_candidate_overlay(
    path: Path,
    background: np.ndarray,
    candidates: list[dict[str, Any]],
    config: CandidateConfig,
) -> None:
    # Draw all candidate outlines and labels on top of the NDVI preview.
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = np.asarray(background).copy()
    thickness = max(1, int(config.box_thickness))
    for candidate in candidates:
        _draw_points(overlay, candidate, thickness)
    for candidate in candidates:
        _draw_candidate_label(overlay, candidate)
    overlay = _append_legend_panel(overlay, candidates)
    cv2.imwrite(str(path), overlay)


def summarize_candidates(
    source: NdviSource,
    output_dir: Path,
    config: CandidateConfig,
    candidates: list[dict[str, Any]],
    target_exclusion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Build the JSON summary that records inputs, parameters, counts, and the
    # top candidate rows.
    counts_by_type: dict[str, int] = {}
    counts_by_class: dict[str, int] = {}
    for candidate in candidates:
        counts_by_type[str(candidate["candidate_type"])] = counts_by_type.get(str(candidate["candidate_type"]), 0) + 1
        counts_by_class[str(candidate["ndvi_class"])] = counts_by_class.get(str(candidate["ndvi_class"]), 0) + 1
    return {
        "input_path": str(source.input_path.resolve()),
        "ndvi_path": str(source.ndvi_path.resolve()) if source.ndvi_path is not None else None,
        "output_dir": str(output_dir.resolve()),
        "physical_ndvi": source.physical_ndvi,
        "crop_margins": source.crop_margins,
        "original_shape": source.original_shape,
        "cropped_shape": source.ndvi.shape,
        "candidate_colors": CANDIDATE_COLORS_RGB,
        "parameters": asdict(config),
        "candidate_count": len(candidates),
        "counts_by_type": counts_by_type,
        "counts_by_class": counts_by_class,
        "target_exclusion": target_exclusion,
        "top_candidates": candidates,
    }
