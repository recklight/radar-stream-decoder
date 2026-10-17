"""Every example script runs end to end on synthetic data (with small sizes)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_synthetic_capture(tmp_path):
    out = tmp_path / "demo"
    assert load("decode_synthetic_capture").main(["--out", str(out), "--frames", "40"]) == 0
    for name in (
        "capture.bin",
        "radar.cfg",
        "points.csv",
        "arrays.npz",
        "frames.jsonl",
        "report.json",
    ):
        assert (out / name).is_file(), name


def test_stream_from_synthetic_source():
    assert load("stream_from_synthetic_source").main(["--frames", "30"]) == 0


def test_live_serial_capture_dry_run(tmp_path):
    points, record = tmp_path / "points.csv", tmp_path / "session.bin"
    args = ["--dry-run", "--duration", "0.6", "--points", str(points), "--record", str(record)]
    assert load("live_serial_capture").main(args) == 0
    assert points.is_file() and record.stat().st_size > 0


def test_live_serial_capture_needs_a_port():
    with pytest.raises(SystemExit):
        load("live_serial_capture").main([])


def test_vital_signs_forensics():
    assert load("vital_signs_forensics").main(["--frames", "200"]) == 0


@pytest.mark.parametrize("schema", ["people-counting-3d", "people-counting-2d", "people-tracking"])
def test_people_counting_capture(tmp_path, schema):
    out = tmp_path / schema
    assert (
        load("people_counting_capture").main(
            ["--schema", schema, "--out", str(out), "--frames", "60"]
        )
        == 0
    )
    for name in ("capture.bin", "radar.cfg", "points.csv", "targets.csv"):
        assert (out / name).is_file(), name
    header = (out / "points.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.endswith("snr_linear,target_id,snr_method")
