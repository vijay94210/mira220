from __future__ import annotations

"""Bridge between this project and the external openRGB-IR RAW converter.

The Mira220 RAW files are not normal PNG or JPEG images. This module calls the
openRGB-IR `isp.py` script, waits for it to produce NumPy arrays, and then loads
those arrays back into this pipeline.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from . import MIRA_RAW_SIZE


def is_compatible_raw(path: Path) -> bool:
    # A quick file-size check catches many wrong inputs before expensive work.
    return path.is_file() and path.stat().st_size == MIRA_RAW_SIZE


def run_openrgbir(
    raw_path: Path,
    work_dir: Path,
    repo: Path,
    config_path: Path,
    keep_intermediates: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    # openRGB-IR is a separate repository. This project expects its main script
    # to be named isp.py.
    isp = repo / "isp.py"
    if not isp.exists():
        raise FileNotFoundError(f"openRGB-IR ISP not found: {isp}")
    if not is_compatible_raw(raw_path):
        raise ValueError(f"{raw_path.name} is not a compatible {MIRA_RAW_SIZE}-byte Mira220 RAW.")

    # Keep external-tool outputs in a temporary work folder inside the capture
    # output directory. That makes cleanup easy after processing succeeds.
    intermediate = work_dir / "_work" / "openrgbir"
    intermediate.mkdir(parents=True, exist_ok=True)

    # Copy the config into the work folder so the exact settings used for this
    # run are visible to the external tool.
    effective_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    effective_config_path = intermediate / "effective_config.yaml"
    effective_config_path.write_text(yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8")

    # The external tool uses this environment variable to choose where to write
    # its .npy output arrays.
    env = os.environ.copy()
    env["OPENRGBIR_OUTPUT_DIR"] = str(intermediate.resolve())

    # Run the external Python script. check=True means Python raises an error if
    # openRGB-IR exits unsuccessfully.
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

    # openRGB-IR writes linear visible RGB and infrared arrays. The rest of this
    # project uses those arrays as the starting point for calibration.
    rgb = np.load(intermediate / "linear_rgb.npy")
    ir = np.load(intermediate / "linear_ir.npy")
    if not keep_intermediates:
        # Remove temporary files unless the user asked to keep them for debugging.
        shutil.rmtree(work_dir / "_work")
    return rgb, ir
