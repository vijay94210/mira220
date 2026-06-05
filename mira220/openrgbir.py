from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from . import MIRA_RAW_SIZE


def is_compatible_raw(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == MIRA_RAW_SIZE


def run_openrgbir(
    raw_path: Path,
    work_dir: Path,
    repo: Path,
    config_path: Path,
    keep_intermediates: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    isp = repo / "isp.py"
    if not isp.exists():
        raise FileNotFoundError(f"openRGB-IR ISP not found: {isp}")
    if not is_compatible_raw(raw_path):
        raise ValueError(f"{raw_path.name} is not a compatible {MIRA_RAW_SIZE}-byte Mira220 RAW.")
    intermediate = work_dir / "_work" / "openrgbir"
    intermediate.mkdir(parents=True, exist_ok=True)
    effective_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    effective_config_path = intermediate / "effective_config.yaml"
    effective_config_path.write_text(yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8")
    env = os.environ.copy()
    env["OPENRGBIR_OUTPUT_DIR"] = str(intermediate.resolve())
    subprocess.run(
        [
            sys.executable,
            str(isp),
            "-s",
            str(raw_path.resolve()),
            "-c",
            str(effective_config_path.resolve()),
            "-o",
            "linear",
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    rgb = np.load(intermediate / "linear_rgb.npy")
    ir = np.load(intermediate / "linear_ir.npy")
    if not keep_intermediates:
        shutil.rmtree(work_dir / "_work")
    return rgb, ir

