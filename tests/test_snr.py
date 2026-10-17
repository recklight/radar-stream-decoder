"""Log-domain noise estimation and per-point SNR."""

from __future__ import annotations

import numpy as np
import pytest

from radar_stream_decoder.payloads import MagnitudeProfile, PointCloud, RangeDopplerMap, SnrMethod
from radar_stream_decoder.snr import (
    LOG_DETECTOR_BIAS_DB,
    ca_noise_floor,
    estimate_point_snr,
    peak_level_db,
)


def test_cell_averaging_and_smallest_of():
    row = np.full(64, 10.0)
    row[40:52] = 20.0  # clutter edge on the right of cell 36
    ca = ca_noise_floor(row, 36, guard=1, train=8, mode="ca")[0]
    so = ca_noise_floor(row, 36, guard=1, train=8, mode="so")[0]
    assert so == pytest.approx(10.0)
    assert ca > so


def test_doppler_axis_wraps_around():
    row = np.zeros(16)
    row[14] = 30.0  # two cells left of cell 0, through the wrap
    assert ca_noise_floor(row, 0, guard=1, train=2, cyclic=True)[0] == pytest.approx(30.0 / 4)
    # Without wrapping only the right side exists.
    assert ca_noise_floor(row, 0, guard=1, train=2, cyclic=False)[0] == pytest.approx(0.0)


def test_training_window_shrinks_to_fit_short_rows():
    row = np.arange(8, dtype=float)
    value = ca_noise_floor(row, 0, guard=1, train=8)[0]  # window shrinks to 2 cells per side
    assert value == pytest.approx(np.mean([row[2], row[3], row[6], row[5]]))
    with pytest.raises(ValueError):
        ca_noise_floor(np.zeros(3), 0, guard=2, train=1)


def test_many_rows_at_once():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(5, 32))
    idx = np.array([0, 5, 16, 30, 31])
    together = ca_noise_floor(rows, idx, guard=2, train=4)
    one_by_one = [ca_noise_floor(rows[i], idx[i], guard=2, train=4)[0] for i in range(5)]
    np.testing.assert_allclose(together, one_by_one)
    with pytest.raises(ValueError):
        ca_noise_floor(rows, idx[:3])


def _points(range_idx, doppler_idx):
    n = len(range_idx)
    return PointCloud(
        np.zeros(n), np.ones(n), np.zeros(n), range_idx=range_idx, doppler_idx=doppler_idx
    )


def test_snr_from_the_detection_matrix_is_exact_on_a_flat_floor():
    matrix = np.full((32, 16), 1000, dtype=np.uint16)
    matrix[10, 3] = 1000 + 2000
    matrix[20, 14] = 1000 + 500  # Doppler index -2
    points = _points([10, 20], [3, -2])
    estimate = estimate_point_snr(points, rd_map=RangeDopplerMap(matrix), db_per_lsb=0.01)
    assert estimate.method is SnrMethod.DETECTION_MATRIX
    np.testing.assert_allclose(estimate.snr_db, [20.0, 5.0])
    corrected = estimate_point_snr(
        points, rd_map=RangeDopplerMap(matrix), db_per_lsb=0.01, log_bias_correction=True
    )
    np.testing.assert_allclose(
        corrected.snr_db, [20.0 - LOG_DETECTOR_BIAS_DB, 5.0 - LOG_DETECTOR_BIAS_DB]
    )


def test_noise_profile_fallback_and_unavailable():
    signal = MagnitudeProfile("range", np.array([0, 3000, 0], dtype=np.uint16))
    noise = MagnitudeProfile("noise", np.array([0, 1000, 0], dtype=np.uint16))
    points = _points([1], [0])
    estimate = estimate_point_snr(
        points, range_profile=signal, noise_profile=noise, db_per_lsb=0.01
    )
    assert estimate.method is SnrMethod.NOISE_PROFILE
    assert estimate.snr_db[0] == pytest.approx(20.0)
    nothing = estimate_point_snr(points, db_per_lsb=0.01)
    assert nothing.method is SnrMethod.UNAVAILABLE and np.isnan(nothing.snr_db[0])
    no_indices = PointCloud([0.0], [1.0], [0.0])
    assert estimate_point_snr(no_indices, db_per_lsb=0.01).method is SnrMethod.UNAVAILABLE
    out_of_range = _points([7], [0])
    assert (
        estimate_point_snr(
            out_of_range, range_profile=signal, noise_profile=noise, db_per_lsb=0.01
        ).method
        is SnrMethod.UNAVAILABLE
    )


def test_log_detector_bias_matches_rayleigh_noise():
    rng = np.random.default_rng(1)
    power = rng.exponential(1.0, 400_000)
    bias = 10 * np.log10(power.mean()) - np.mean(10 * np.log10(power))
    assert bias == pytest.approx(LOG_DETECTOR_BIAS_DB, abs=0.02)
    assert pytest.approx(2.507, abs=1e-3) == LOG_DETECTOR_BIAS_DB


def test_peak_level_conventions():
    np.testing.assert_allclose(peak_level_db([10, 100], 0xA1642), [20.0, 40.0])
    assert np.isnan(peak_level_db([0], 0xA1642)[0])
    np.testing.assert_allclose(peak_level_db([9, 99], 0xA1443), [10.0, 20.0])
