from pathlib import Path

# Tests for calibration target geometry. Patch order matters because calibration
# assumes P1/P2 are on top and P3/P4 are below them.

import numpy as np

from mira220.correction import load_yaml
from mira220.imaging import patch_polygons


def test_patch_order_relative_to_marker() -> None:
    # Use a simple square marker and verify the computed patch centers land in
    # the expected left/right and top/bottom order.
    target = load_yaml(Path("config/targets/aruco830_flat_patches.yaml"))
    marker = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.float32)
    centers = np.array([polygon.mean(axis=0) for polygon in patch_polygons(marker, target)])
    assert centers[0, 0] < centers[1, 0]
    assert centers[2, 0] < centers[3, 0]
    assert centers[0, 1] < centers[2, 1]
    assert centers[1, 1] < centers[3, 1]
