"""Command line, end to end on synthetic captures."""

from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from radar_stream_decoder.cli import app

RUNNER = CliRunner()
SRC = Path(__file__).resolve().parents[1] / "src"


def run(*args: object):
    return RUNNER.invoke(app, [str(arg) for arg in args])


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    path = tmp_path / "demo.bin"
    result = run("synth", path, "--frames", 40, "--faults", "light", "--seed", 3)
    assert result.exit_code == 0, result.output
    return path


def test_synth_writes_capture_and_configuration(capture):
    assert capture.stat().st_size > 0
    config_text = capture.with_suffix(".cfg").read_text(encoding="utf-8")
    assert "profileCfg" in config_text and "channelCfg 15 3 0" in config_text


def test_inspect_prints_and_saves_the_report(capture, tmp_path):
    report_path = tmp_path / "report.json"
    result = run(
        "inspect", capture, "--sensor-config", capture.with_suffix(".cfg"), "--json", report_path
    )
    assert result.exit_code == 0, result.output
    assert "unique frames" in result.output and "drop ratio" in result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schemas"] == {"oob-sdk2": report["unique_frames"]}


def test_decode_writes_every_format(capture, tmp_path):
    out = tmp_path / "decoded"
    result = run("decode", capture, "--out", out, "--sensor-config", capture.with_suffix(".cfg"))
    assert result.exit_code == 0, result.output
    for name in ("points.csv", "frames.jsonl", "stats.jsonl", "arrays.npz", "report.json"):
        assert (out / name).is_file(), name
    header = (out / "points.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("frame,subframe,timestamp_s,point,x_m")
    only_csv = tmp_path / "csv_only"
    assert run("decode", capture, "-o", only_csv, "--format", "csv").exit_code == 0
    assert sorted(p.name for p in only_csv.iterdir()) == ["points.csv", "report.json"]


def test_vital_signs_round_trip_through_the_cli(tmp_path):
    path = tmp_path / "vital.bin"
    assert (
        run(
            "synth", path, "--schema", "vital-signs", "--frames", 30, "--faults", "block-dump"
        ).exit_code
        == 0
    )
    out = tmp_path / "vital_out"
    result = run("decode", path, "-o", out, "--strict", "--schema", "vital-signs")
    assert result.exit_code == 0, result.output
    rows = (out / "vital_signs.jsonl").read_text(encoding="utf-8").splitlines()
    assert rows and "heart_rate_fft_bpm" in json.loads(rows[0])
    assert "repeats dropped" in result.output


def test_sensor_config_command(tmp_path):
    result = run("sensor-config", "--preset", "iwr1642_oob_2tx4rx")
    assert result.exit_code == 0 and "range_bin_m" in result.output and "0.0732422" in result.output
    as_json = run("sensor-config", "--preset", "iwr1443_vital_signs", "--json")
    assert json.loads(as_json.output)["num_tx"] == 1
    config_file = tmp_path / "radar.cfg"
    config_file.write_text(
        "channelCfg 15 3 0\nprofileCfg 0 77 7 6 60 0 0 40 1 256 5000 0 0 30\n", encoding="utf-8"
    )
    from_file = run("sensor-config", config_file, "--platform", "16xx")
    assert from_file.exit_code == 0 and "num_rx" in from_file.output
    assert run("sensor-config").exit_code == 2
    assert run("sensor-config", config_file, "--preset", "iwr1642_oob_2tx4rx").exit_code == 2


def test_sensor_config_conventions_are_selectable_and_stated():
    default = json.loads(run("sensor-config", "--preset", "iwr6843_3d_3tx4rx", "--json").output)
    assert default["convention_wavelength_reference"] == "start"
    assert default["max_range_m"] == pytest.approx(0.8 * default["if_range_limit_m"])
    firmware = run(
        "sensor-config",
        "--preset",
        "iwr6843_3d_3tx4rx",
        "--wavelength-at",
        "center",
        "--max-range-factor",
        "1",
        "--no-doppler-padding",
        "--json",
    )
    assert firmware.exit_code == 0, firmware.output
    values = json.loads(firmware.output)
    assert values["convention_wavelength_reference"] == "center"
    assert values["convention_doppler_zero_padding"] is False
    assert values["max_range_m"] == pytest.approx(values["if_range_limit_m"])
    assert values["max_velocity_mps"] < default["max_velocity_mps"]
    # 64 chirps per transmitter: a power of two, so there is no zero padding to undo.
    assert values["doppler_resolution_mps"] == pytest.approx(values["doppler_bin_mps"])
    table = run("sensor-config", "--preset", "iwr6843_3d_3tx4rx")
    assert "convention_wavelength_reference" in table.output
    preset = ("sensor-config", "--preset", "iwr6843_3d_3tx4rx")
    assert run(*preset, "--wavelength-at", "mid").exit_code == 2
    assert run(*preset, "--max-range-factor", "2").exit_code == 2


def test_evaluate_command(tmp_path):
    out = tmp_path / "evaluation.json"
    result = run(
        "evaluate",
        "--frames",
        20,
        "--trials",
        2,
        "--faults",
        "light",
        "--faults",
        "heavy",
        "--estimator-frames",
        10,
        "--out",
        out,
    )
    assert result.exit_code == 0, result.output
    assert "sensitivity" in result.output and "azimuth RMSE" in result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [entry["faults"] for entry in data["recovery"]] == ["light", "heavy"]
    assert "estimators_log_bias_corrected" in data
    vital = run(
        "evaluate", "--schema", "vital-signs", "--frames", 20, "--trials", 1, "--faults", "none"
    )
    assert vital.exit_code == 0 and "estimator accuracy" not in vital.output


def test_stream_demo_records_and_exports(tmp_path):
    record, points = tmp_path / "live.bin", tmp_path / "points.csv"
    result = run("stream", "--demo", "--max-frames", 8, "--record", record, "--points-csv", points)
    assert result.exit_code == 0, result.output
    assert "8 frames" in result.output
    assert record.stat().st_size > 0 and points.is_file()
    rows = list(csv.DictReader(points.open(encoding="utf-8")))
    times = sorted({float(row["timestamp_s"]) for row in rows})
    assert len(times) == 8 and times[0] == 0.0
    assert all(abs(b - a - 0.05) < 1e-3 for a, b in itertools.pairwise(times))
    replay = tmp_path / "replay"
    assert run("decode", record, "-o", replay, "--format", "jsonl").exit_code == 0


def test_stream_stops_cleanly_on_ctrl_c(tmp_path, monkeypatch):
    from radar_stream_decoder.capture import FrameTimestamper

    original = FrameTimestamper.__call__
    seen: list[int] = []

    def interrupt_after_five_frames(self, frame):
        seen.append(frame.frame_number)
        if len(seen) > 5:
            raise KeyboardInterrupt
        return original(self, frame)

    monkeypatch.setattr(FrameTimestamper, "__call__", interrupt_after_five_frames)
    points = tmp_path / "points.csv"
    result = run("stream", "--demo", "--max-frames", 50, "--points-csv", points)
    assert result.exit_code == 0, result.output
    assert "5 frames" in result.output and "Ctrl+C" in result.output
    frames = {row["frame"] for row in csv.DictReader(points.open(encoding="utf-8"))}
    assert frames == {"1", "2", "3", "4", "5"}


@pytest.mark.parametrize(
    "args",
    [
        ["inspect", "missing.bin"],
        ["decode", "missing.bin", "-o", "out", "--format", "xml"],
        ["synth", "x.bin", "--faults", "meteor"],
        ["synth", "x.bin", "--schema", "sdk9"],
        ["evaluate", "--faults", "meteor"],
        ["stream"],
        ["viz", "--source", "tv"],
        ["viz", "--source", "file"],
        ["viz", "--source", "serial"],
    ],
)
def test_expected_failures_exit_with_code_2(args, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run(*args)
    assert result.exit_code == 2, result.output
    assert "error:" in result.output or "Usage" in result.output


def test_noise_window_that_does_not_fit_does_not_stop_the_decode(capture, tmp_path):
    settings = tmp_path / "guard7.json"
    settings.write_text('{"snr_guard_cells": 7}', encoding="utf-8")
    result = run(
        "inspect", capture, "--sensor-config", capture.with_suffix(".cfg"), "--config", settings
    )
    assert result.exit_code == 0, result.output
    assert "invalid-value" in result.output


def test_viz_sends_the_cfg_only_through_a_cli_port(monkeypatch):
    # A stand-in for the viewer module, so the test needs neither PySide6 nor a screen.
    calls: list[dict] = []
    viewer = types.ModuleType("radar_stream_decoder.viz")
    viewer.launch_viewer = lambda **kwargs: calls.append(kwargs) or 0
    monkeypatch.setitem(sys.modules, "radar_stream_decoder.viz", viewer)
    cfg = Path(__file__).resolve().parents[1] / "configs" / "iwr1642_oob_2tx4rx.cfg"
    serial = ("viz", "--source", "serial", "--data-port", "COM4", "--sensor-config", cfg)
    assert run(*serial).exit_code == 0
    assert run(*serial, "--cli-port", "COM3").exit_code == 0
    assert calls[0]["cli_port"] is None and calls[0]["cli_text"] is None
    assert calls[0]["sensor"].num_tx == 2
    assert "sensorStart" in calls[1]["cli_text"]


def test_bad_decoder_config_file(tmp_path, capture):
    bad = tmp_path / "decoder.json"
    bad.write_text('{"strictness": true}', encoding="utf-8")
    result = run("inspect", capture, "--config", bad)
    assert result.exit_code == 2 and "unknown decoder setting" in result.output


def test_bad_log_level(capture):
    assert run("--log-level", "chatty", "inspect", capture).exit_code == 2


def test_module_entry_point():
    # Pin the pipe encoding on both ends: the help text is not ASCII, and the
    # locale default (cp950, cp1252, ...) would otherwise depend on the machine
    # and on whether PYTHONIOENCODING happens to be set in the parent.
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "radar_stream_decoder", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )
    assert completed.returncode == 0
    for command in ("inspect", "decode", "sensor-config", "synth", "evaluate", "stream", "viz"):
        assert command in completed.stdout
