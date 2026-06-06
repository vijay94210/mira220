from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def apply_scene_correction(
    red: np.ndarray,
    ir: np.ndarray,
    scene_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
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
    return (
        np.clip(features @ red_coefficients, 0.0, 1.0),
        np.clip(features @ ir_coefficients, 0.0, 1.0),
    )


def apply_model(rgb: np.ndarray, ir: np.ndarray, model_config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model_config["correction"]
    red_coefficients = np.asarray(model["red"]["coefficients"], dtype=np.float32)
    green_coefficients = np.asarray(model["green"]["coefficients"], dtype=np.float32)
    ir_coefficients = np.asarray(model["ir"]["coefficients"], dtype=np.float32)
    red = np.clip(
        red_coefficients[0] * rgb[..., 0]
        + red_coefficients[1] * ir
        + red_coefficients[2] * np.square(ir)
        + red_coefficients[3],
        0.0,
        1.0,
    )
    green = np.clip(
        green_coefficients[0] * rgb[..., 1]
        + green_coefficients[1] * ir
        + green_coefficients[2] * np.square(ir)
        + green_coefficients[3],
        0.0,
        1.0,
    )
    calibrated_ir = np.clip(
        ir_coefficients[0] * ir + ir_coefficients[1] * np.square(ir) + ir_coefficients[2],
        0.0,
        1.0,
    )
    if "scene_correction" in model_config:
        red, calibrated_ir = apply_scene_correction(red, calibrated_ir, model_config["scene_correction"])
    return red, green, calibrated_ir

