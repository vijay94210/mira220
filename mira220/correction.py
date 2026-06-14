from __future__ import annotations

"""Calibration math for turning raw camera channels into reflectance channels.

The Mira220 camera gives us raw Red, Green, and IR-like measurements. These are
not directly the final scientific reflectance values. This module applies saved
model coefficients to correct those channels.
"""

from pathlib import Path

import numpy as np
import yaml


def load_yaml(path: Path) -> dict:
    # YAML files are used for human-readable model and target configuration.
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, value: dict) -> None:
    # Make the parent folder first so callers can write reports to new folders.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def apply_scene_correction(
    red: np.ndarray,
    ir: np.ndarray,
    scene_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    # Scene correction is a second-stage adjustment. It uses Red, IR, squared
    # terms, their interaction, and a constant term as features.
    features = np.stack(
        [
            red,
            ir,
            np.square(red),
            np.square(ir),
            red * ir,
            np.ones_like(red),
        ],
        axis=-1,
    )
    red_coefficients = np.asarray(scene_config["red_coefficients"], dtype=np.float32)
    ir_coefficients = np.asarray(scene_config["ir_coefficients"], dtype=np.float32)
    if red_coefficients.shape != (6,) or ir_coefficients.shape != (6,):
        raise ValueError("Scene correction requires six red and six IR coefficients.")

    # Matrix multiplication applies the same six-feature equation to every pixel.
    # Values are clipped to the valid reflectance range.
    return (
        np.clip(features @ red_coefficients, 0.0, 1.0),
        np.clip(features @ ir_coefficients, 0.0, 1.0),
    )


def apply_model(rgb: np.ndarray, ir: np.ndarray, model_config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The main flat-patch model has separate coefficient lists for corrected Red,
    # corrected Green, and corrected IR.
    model = model_config["correction"]
    red_coefficients = np.asarray(model["red"]["coefficients"], dtype=np.float32)
    green_coefficients = np.asarray(model["green"]["coefficients"], dtype=np.float32)
    ir_coefficients = np.asarray(model["ir"]["coefficients"], dtype=np.float32)
    red = np.clip(
        # Red is estimated from raw red, raw IR, raw IR squared, and a constant.
        red_coefficients[0] * rgb[..., 0]
        + red_coefficients[1] * ir
        + red_coefficients[2] * np.square(ir)
        + red_coefficients[3],
        0.0,
        1.0,
    )
    green = np.clip(
        # Green uses raw green plus IR information, because the channels can leak
        # into each other on an RGB-IR sensor.
        green_coefficients[0] * rgb[..., 1]
        + green_coefficients[1] * ir
        + green_coefficients[2] * np.square(ir)
        + green_coefficients[3],
        0.0,
        1.0,
    )
    calibrated_ir = np.clip(
        # IR is corrected from raw IR and raw IR squared.
        ir_coefficients[0] * ir + ir_coefficients[1] * np.square(ir) + ir_coefficients[2],
        0.0,
        1.0,
    )
    if "scene_correction" in model_config:
        # Some models include an optional second-stage scene correction.
        red, calibrated_ir = apply_scene_correction(red, calibrated_ir, model_config["scene_correction"])
    return red, green, calibrated_ir

