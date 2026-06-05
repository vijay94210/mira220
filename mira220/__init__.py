"""Mira220 RGB-IR reflectance processing."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "config" / "models" / "flat_patch_v1.yaml"
DEFAULT_ISP_CONFIG_PATH = ROOT / "config" / "isp" / "zero_cut.yaml"
DEFAULT_TARGET_PATH = ROOT / "config" / "targets" / "aruco830_flat_patches.yaml"
DEFAULT_OPENRGBIR_REPO = Path(r"C:\Users\jayal\git\openRGB-IR")
MIRA_RAW_SIZE = 1600 * 1400 * 2

