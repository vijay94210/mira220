from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile


EPSILON = 1e-6
NDVI_SMOOTHING_SIGMA = 1.0
NDVI_CROP_MARGINS = (150, 125, 150, 125)
NDVI_COLORMAP = [
    (0.00, (120, 40, 30)),
    (0.30, (200, 110, 60)),
    (0.50, (220, 210, 130)),
    (0.70, (120, 185, 90)),
    (1.00, (0, 100, 50)),
]


def normalize_sensor(image: np.ndarray, bit_depth: int = 12) -> np.ndarray:
    return np.clip(np.asarray(image, np.float32) / float((1 << bit_depth) - 1), 0.0, 1.0)


def compute_ndvi(red: np.ndarray, ir: np.ndarray) -> np.ndarray:
    red = np.asarray(red, np.float32)
    ir = np.asarray(ir, np.float32)
    return np.clip((ir - red) / (ir + red + EPSILON), -1.0, 1.0)


def preview_u8(image: np.ndarray, gamma: float | None = None) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    if gamma is not None:
        clipped = np.power(clipped, 1.0 / gamma)
    return np.round(clipped * 255.0).astype(np.uint8)


def smooth_ndvi_for_display(ndvi: np.ndarray, sigma: float = NDVI_SMOOTHING_SIGMA) -> np.ndarray:
    values = np.asarray(ndvi, np.float32)
    if sigma <= 0:
        return values
    finite = np.isfinite(values)
    if finite.all():
        return cv2.GaussianBlur(values, (0, 0), sigma)
    filled = values.copy()
    filled[~finite] = float(np.nanmedian(values[finite])) if np.any(finite) else 0.0
    smoothed = cv2.GaussianBlur(filled, (0, 0), sigma)
    smoothed[~finite] = np.nan
    return smoothed


def ndvi_false_color(ndvi: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    normalized = np.clip((ndvi - minimum) / (maximum - minimum), 0.0, 1.0)
    positions = np.array([point for point, _ in NDVI_COLORMAP], np.float32)
    colors_rgb = np.array([color for _, color in NDVI_COLORMAP], np.float32)
    channels = [np.interp(normalized, positions, colors_rgb[:, channel]) for channel in range(3)]
    return np.round(np.stack(channels, axis=-1)).astype(np.uint8)


def crop_valid_region(
    image: np.ndarray,
    margins: tuple[int, int, int, int] = NDVI_CROP_MARGINS,
) -> np.ndarray:
    left, top, right, bottom = margins
    height, width = image.shape[:2]
    if left < 0 or top < 0 or right < 0 or bottom < 0:
        raise ValueError("Crop margins must be non-negative.")
    if left + right >= width or top + bottom >= height:
        raise ValueError(f"Crop margins {margins} remove the entire {width}x{height} image.")
    return image[top : height - bottom, left : width - right].copy()


def crop_valid_region_or_full(image: np.ndarray) -> np.ndarray:
    try:
        return crop_valid_region(image)
    except ValueError:
        return np.asarray(image).copy()


def write_rgb_preview(path: Path, image: np.ndarray, gamma: float = 2.2) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(preview_u8(image, gamma), cv2.COLOR_RGB2BGR))


def save_reflectance_products(
    output_dir: Path,
    red: np.ndarray,
    green: np.ndarray,
    ir: np.ndarray,
    ndvi_min: float,
    ndvi_max: float,
    rgb: np.ndarray | None = None,
    product_set: str = "full",
) -> None:
    if product_set not in {"full", "ndvi"}:
        raise ValueError(f"Unsupported product set: {product_set}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rgn = np.dstack([red, green, ir]).astype(np.float32)
    ndvi = compute_ndvi(red, ir)
    ndvi_display = smooth_ndvi_for_display(ndvi)
    if product_set == "full":
        tifffile.imwrite(output_dir / "red_reflectance.tiff", red.astype(np.float32))
        tifffile.imwrite(output_dir / "green_reflectance.tiff", green.astype(np.float32))
        tifffile.imwrite(output_dir / "ir_reflectance.tiff", ir.astype(np.float32))
        tifffile.imwrite(output_dir / "rgn_reflectance.tiff", rgn)
    tifffile.imwrite(output_dir / "ndvi.tiff", ndvi.astype(np.float32))
    if rgb is not None:
        write_rgb_preview(output_dir / "rgb_preview.png", np.asarray(rgb, np.float32))
    write_rgb_preview(output_dir / "rgn_preview.png", rgn)
    cv2.imwrite(
        str(output_dir / "ndvi_false_color.png"),
        cv2.cvtColor(ndvi_false_color(ndvi_display, ndvi_min, ndvi_max), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(output_dir / "ndvi_false_color_crop.png"),
        cv2.cvtColor(ndvi_false_color(crop_valid_region_or_full(ndvi_display), ndvi_min, ndvi_max), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(output_dir / "ndvi_gray.png"),
        preview_u8((ndvi_display - ndvi_min) / (ndvi_max - ndvi_min)),
    )


def detect_marker(image_rgb: np.ndarray, marker_id: int, dictionary_name: str) -> np.ndarray:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("ArUco support requires opencv-contrib-python.")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    gray = cv2.cvtColor(preview_u8(image_rgb, 2.2), cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    if ids is None or marker_id not in ids.reshape(-1):
        raise RuntimeError(f"ArUco marker {marker_id} was not detected.")
    index = int(np.where(ids.reshape(-1) == marker_id)[0][0])
    return corners[index].reshape(4, 2).astype(np.float32)


def patch_polygons(marker_corners: np.ndarray, target: dict) -> list[np.ndarray]:
    marker_plane = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    transform = cv2.getPerspectiveTransform(marker_plane, marker_corners.astype(np.float32))
    inset = float(target.get("inset_fraction", 0.15))
    polygons = []
    for patch in target["patches"]:
        x0, y0, x1, y1 = map(float, patch["rect"])
        width, height = x1 - x0, y1 - y0
        quad = np.array(
            [
                [x0 + inset * width, y0 + inset * height],
                [x1 - inset * width, y0 + inset * height],
                [x1 - inset * width, y1 - inset * height],
                [x0 + inset * width, y1 - inset * height],
            ],
            np.float32,
        )
        polygons.append(cv2.perspectiveTransform(quad[None, ...], transform)[0])
    return polygons


def polygon_median(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
    pixels = image[mask > 0]
    if pixels.size == 0:
        raise RuntimeError("A reflectance patch polygon falls outside the image.")
    return np.median(pixels, axis=0)


def draw_patch_overlay(image: np.ndarray, marker: np.ndarray, polygons: list[np.ndarray], target: dict) -> np.ndarray:
    overlay = cv2.cvtColor(preview_u8(image, 2.2), cv2.COLOR_RGB2BGR)
    cv2.polylines(overlay, [np.round(marker).astype(np.int32)], True, (0, 255, 0), 4)
    for patch, polygon in zip(target["patches"], polygons):
        points = np.round(polygon).astype(np.int32)
        cv2.polylines(overlay, [points], True, (0, 255, 255), 3)
        center = tuple(np.round(points.mean(axis=0)).astype(int))
        cv2.putText(overlay, patch["name"], center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return overlay

