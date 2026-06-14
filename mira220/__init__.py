"""Shared constants for the Mira220 RGB-IR reflectance project.

This package processes Mira220 camera RAW files into plant-health review images.
The constants below are the default paths used by the command-line tools.
Keeping them in one file means the rest of the codebase can agree on where
models, target configs, and large data folders are located.
"""

import os
from pathlib import Path


# ROOT is the repository folder. It is used to build paths to checked-in config
# files no matter where the command is launched from.
ROOT = Path(__file__).resolve().parent.parent

# Large data files are intentionally outside Git. The environment variable lets
# a user point the same code at a different drive or data folder.
DEFAULT_DATA_ROOT = Path(os.environ.get("MIRA220_DATA_ROOT", r"C:\Users\jayal\droid-data"))

# Default checked-in configuration files.
DEFAULT_MODEL_PATH = ROOT / "config" / "models" / "flat_patch_v1.yaml"
DEFAULT_ISP_CONFIG_PATH = ROOT / "config" / "isp" / "zero_cut.yaml"
DEFAULT_TARGET_PATH = ROOT / "config" / "targets" / "aruco830_flat_patches.yaml"
DEFAULT_TARGET_REFERENCE_PATH = ROOT / "config" / "targets" / "aruco830_patch_reference.yaml"

# The project expects a local checkout of openRGB-IR for RAW conversion.
DEFAULT_OPENRGBIR_REPO = Path(r"C:\Users\jayal\git\openRGB-IR")

# Mira220 RAW files are 1600 x 1400 pixels with 2 bytes per pixel.
MIRA_RAW_SIZE = 1600 * 1400 * 2

