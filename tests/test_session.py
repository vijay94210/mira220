from pathlib import Path

# Tests for session-level target adjustment. These checks cover patch
# aggregation, affine fitting, output writing, clipping summaries, and skip cases.

import numpy as np
import pytest
import tifffile

import mira220.session as session
from mira220.session import (
    adjust_session,
    aggregate_session_patches,
    apply_affine,
    fit_session_adjustment,
    summarize_clipping,
)


TARGET = {
    "marker_id": 830,
    "dictionary": "DICT_4X4_1000",
    "patches": [{"name": name} for name in ("P1", "P2", "P3", "P4")],
}


def _patches(offset: float = 0.0) -> list[dict]:
    return [
        {"name": f"P{index + 1}", "red": value + offset, "green": value + offset, "ir": value + offset}
        for index, value in enumerate((0.2, 0.4, 0.6, 0.8))
    ]


def _write_output(path: Path) -> None:
    path.mkdir(parents=True)
    image = np.full((8, 8), 0.5, np.float32)
    for channel in session.CHANNELS:
        tifffile.imwrite(path / f"{channel}_reflectance.tiff", image)
    tifffile.imwrite(path / "rgn_reflectance.tiff", np.dstack([image, image, image]))


def test_fit_session_adjustment_recovers_known_affine_mapping() -> None:
    observed = _patches()
    reference = {
        "patches": [
            {
                "name": patch["name"],
                **{channel: 1.2 * patch[channel] + 0.03 for channel in session.CHANNELS},
            }
            for patch in observed
        ]
    }
    adjustment = fit_session_adjustment(observed, reference)
    for channel in session.CHANNELS:
        assert adjustment["channels"][channel]["gain"] == pytest.approx(1.2)
        assert adjustment["channels"][channel]["offset"] == pytest.approx(0.03)


def test_fit_session_adjustment_requires_three_usable_patches_per_channel() -> None:
    observed = _patches()[:2]
    reference = {"patches": _patches()[:2]}
    with pytest.raises(ValueError, match="fewer than 3 usable patch"):
        fit_session_adjustment(observed, reference)


def test_session_patch_aggregation_uses_capture_median(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("a", "b", "outlier"):
        _write_output(tmp_path / name)
    values = {"a": _patches(-0.01), "b": _patches(0.01), "outlier": _patches(0.4)}
    monkeypatch.setattr(session, "measure_output_patches", lambda path, target: values[path.name])
    aggregated, measurements, failures = aggregate_session_patches(tmp_path, TARGET)
    assert aggregated[0]["red"] == pytest.approx(0.21)
    assert len(measurements) == 3
    assert failures == []


def test_session_patch_aggregation_reports_marker_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_output(tmp_path / "good")
    _write_output(tmp_path / "bad")

    def measure(path: Path, target: dict) -> list[dict]:
        if path.name == "bad":
            raise RuntimeError("marker missing")
        return _patches()

    monkeypatch.setattr(session, "measure_output_patches", measure)
    _, measurements, failures = aggregate_session_patches(tmp_path, TARGET)
    assert len(measurements) == 1
    assert failures[0]["error"] == "marker missing"


def test_session_patch_aggregation_can_skip_without_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_output(tmp_path / "bad")
    monkeypatch.setattr(session, "measure_output_patches", lambda path, target: (_ for _ in ()).throw(RuntimeError("marker missing")))
    aggregated, measurements, failures = aggregate_session_patches(tmp_path, TARGET, require_measurements=False)
    assert aggregated == []
    assert measurements == []
    assert failures[0]["error"] == "marker missing"


def test_apply_affine_clips() -> None:
    adjusted = apply_affine(np.array([-1.0, 0.5, 2.0], np.float32), {"gain": 2.0, "offset": 0.1})
    np.testing.assert_allclose(adjusted, [0.0, 1.0, 1.0])


def test_summarize_clipping(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    _write_output(output)
    tifffile.imwrite(output / "red_reflectance.tiff", np.array([[0.0, 0.5], [1.0, 0.5]], np.float32))
    summary = summarize_clipping(tmp_path)
    assert summary["capture_count"] == 1
    assert summary["mean_clipping_fraction"]["red"] == pytest.approx(0.5)


def test_adjust_session_does_not_accept_mapir_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "input" / "capture"
    _write_output(input_dir)
    monkeypatch.setattr(
        session,
        "aggregate_session_patches",
        lambda run_root, target, require_measurements=True: (
            _patches(),
            [{"output_directory": "capture", "patches": _patches()}],
            [],
        ),
    )
    report = adjust_session(
        tmp_path / "input",
        tmp_path / "output",
        TARGET,
        {"name": "reference", "patches": _patches()},
        {"ndvi_min": -0.2, "ndvi_max": 0.7},
        Path("model.yaml"),
    )
    assert report["mapir_used_for_fitting"] is False
    assert report["adjusted"] is True
    assert (tmp_path / "output" / "session_adjustment.yaml").is_file()


def test_adjust_session_skips_cleanly_without_target_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input" / "capture"
    _write_output(input_dir)
    monkeypatch.setattr(
        session,
        "aggregate_session_patches",
        lambda run_root, target, require_measurements=True: ([], [], [{"output_directory": "capture", "error": "marker missing"}]),
    )
    report = adjust_session(
        tmp_path / "input",
        tmp_path / "output",
        TARGET,
        {"name": "reference", "patches": _patches()},
        {"ndvi_min": -0.2, "ndvi_max": 0.7},
        Path("model.yaml"),
    )
    assert report["adjusted"] is False
    assert report["processed"] == []
    assert "No capture produced usable ArUco" in report["skip_reason"]


def test_adjust_session_applies_target_fit_to_markerless_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_capture = tmp_path / "input" / "target"
    markerless_capture = tmp_path / "input" / "markerless"
    _write_output(target_capture)
    _write_output(markerless_capture)
    monkeypatch.setattr(
        session,
        "aggregate_session_patches",
        lambda run_root, target, require_measurements=True: (
            _patches(),
            [{"output_directory": str(target_capture), "patches": _patches()}],
            [{"output_directory": str(markerless_capture), "error": "marker missing"}],
        ),
    )
    report = adjust_session(
        tmp_path / "input",
        tmp_path / "output",
        TARGET,
        {"name": "reference", "patches": _patches()},
        {"ndvi_min": -0.2, "ndvi_max": 0.7},
        Path("model.yaml"),
    )
    assert report["adjusted"] is True
    assert len(report["processed"]) == 2
    assert (tmp_path / "output" / "target" / "ndvi.tiff").is_file()
    assert (tmp_path / "output" / "markerless" / "ndvi.tiff").is_file()
