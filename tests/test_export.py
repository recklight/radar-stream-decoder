"""CSV, JSON Lines, NPZ and DataFrame output."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from radar_stream_decoder.export import (
    POINT_COLUMNS,
    frame_summary_rows,
    points_to_rows,
    save_npz,
    stats_rows,
    to_dataframe,
    vital_signs_rows,
    write_jsonl,
    write_points_csv,
)
from radar_stream_decoder.frame import decode_packet
from radar_stream_decoder.synthetic import SyntheticScene, generate_packets


@pytest.fixture
def frames_16xx(scene_16xx):
    return [decode_packet(p, sensor=scene_16xx.radar()) for p in generate_packets(scene_16xx, 6)]


def test_point_rows(frames_16xx):
    times = np.arange(6) * 0.05
    rows = points_to_rows(frames_16xx, times)
    assert len(rows) == sum(f.num_points for f in frames_16xx)
    first = rows[0]
    assert set(first) == set(POINT_COLUMNS)
    assert first["frame"] == 1 and first["point"] == 0 and first["timestamp_s"] == 0.0
    assert isinstance(first["range_idx"], int) and isinstance(first["doppler_idx"], int)
    assert first["snr_method"] == "detection_matrix"
    with pytest.raises(ValueError):
        points_to_rows(frames_16xx, times[:3])


def test_unknown_values_become_empty(scene_sdk3):
    frame = decode_packet(next(generate_packets(scene_sdk3, 1)))
    row = points_to_rows([frame])[0]
    assert row["range_idx"] is None and row["peak"] is None and row["timestamp_s"] is None


def test_points_csv(tmp_path, frames_16xx):
    path = tmp_path / "nested" / "points.csv"
    count = write_points_csv(path, frames_16xx)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == count and list(rows[0]) == list(POINT_COLUMNS)
    assert rows[0]["timestamp_s"] == ""
    assert float(rows[0]["range_m"]) > 0


def test_jsonl_records(tmp_path, frames_16xx, packets_vital):
    path = tmp_path / "frames.jsonl"
    assert write_jsonl(path, frame_summary_rows(frames_16xx)) == 6
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["tlv_types"] == [1, 2, 3, 4, 5, 6]
    stats = stats_rows(frames_16xx)
    assert len(stats) == 6 and "active_frame_cpu_load_pct" in stats[0]
    vital = vital_signs_rows([decode_packet(p) for p in packets_vital[:3]])
    assert len(vital) == 3 and "heart_rate_fft_bpm" in vital[0]
    special = tmp_path / "special.jsonl"
    write_jsonl(
        special, [{"a": float("nan"), "b": np.float32(1.5), "c": np.arange(2), "d": (1, 2)}]
    )
    assert json.loads(special.read_text(encoding="utf-8")) == {
        "a": None,
        "b": 1.5,
        "c": [0, 1],
        "d": [1, 2],
    }


def test_npz_stacks_arrays(tmp_path, frames_16xx):
    path = tmp_path / "arrays.npz"
    keys = save_npz(path, frames_16xx, np.arange(6) * 0.05)
    with np.load(path) as data:
        assert data["range_profile"].shape == (6, 256)
        assert data["range_doppler"].shape == (6, 256, 16)
        np.testing.assert_array_equal(data["range_profile__frame"], np.arange(1, 7))
        assert data["timestamp_s"][1] == pytest.approx(0.05)
        assert float(data["db_per_lsb"]) > 0
    assert "noise_profile" in keys


def test_npz_splits_mixed_shapes(tmp_path):
    small = SyntheticScene(seed=1, sensor=None)
    frames = [decode_packet(p) for p in generate_packets(small, 2)]
    other_scene = SyntheticScene(platform="14xx", seed=2)
    frames += [decode_packet(p) for p in generate_packets(other_scene, 2)]
    keys = save_npz(tmp_path / "mixed.npz", frames)
    # Both presets have 256 range bins but 16 and 32 Doppler bins.
    assert "range_profile" in keys
    assert {"range_doppler__256x16", "range_doppler__256x32"} <= set(keys)
    assert "range_doppler" not in keys
    with np.load(tmp_path / "mixed.npz") as data:
        assert not any(array.dtype == object for array in data.values())


def test_dataframe(frames_16xx):
    pd = pytest.importorskip("pandas")
    table = to_dataframe(frames_16xx)
    assert isinstance(table, pd.DataFrame) and list(table.columns) == list(POINT_COLUMNS)
    assert len(to_dataframe(frames_16xx, "frames")) == 6
    with pytest.raises(ValueError):
        to_dataframe(frames_16xx, "nonsense")
