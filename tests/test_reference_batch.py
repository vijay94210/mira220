from pathlib import Path

# Tests for basic reference assumptions: expected RAW file size and the saved
# model's flat-patch NDVI acceptance threshold.

from mira220.correction import load_yaml
from mira220.openrgbir import is_compatible_raw


def test_raw_size_filtering(tmp_path: Path) -> None:
    # A compatible Mira220 RAW has the exact expected byte count.
    compatible = tmp_path / "compatible.raw"
    incompatible = tmp_path / "incompatible.raw"
    compatible.write_bytes(b"\0" * (1600 * 1400 * 2))
    incompatible.write_bytes(b"\0" * 100)
    assert is_compatible_raw(compatible)
    assert not is_compatible_raw(incompatible)


def test_patch_validation_meets_acceptance() -> None:
    # The fitted model should make the neutral target patches nearly zero NDVI.
    model = load_yaml(Path("config/models/flat_patch_v1.yaml"))
    results = model["validation"]["patch_results"]
    assert max(abs(row["ndvi"]) for row in results) < 0.022
