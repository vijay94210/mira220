from pathlib import Path

from mira220.correction import load_yaml
from mira220.openrgbir import is_compatible_raw


def test_raw_size_filtering(tmp_path: Path) -> None:
    compatible = tmp_path / "compatible.raw"
    incompatible = tmp_path / "incompatible.raw"
    compatible.write_bytes(b"\0" * (1600 * 1400 * 2))
    incompatible.write_bytes(b"\0" * 100)
    assert is_compatible_raw(compatible)
    assert not is_compatible_raw(incompatible)


def test_patch_validation_meets_acceptance() -> None:
    model = load_yaml(Path("config/models/flat_patch_v1.yaml"))
    results = model["validation"]["patch_results"]
    assert max(abs(row["ndvi"]) for row in results) < 0.022

