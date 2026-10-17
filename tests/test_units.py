"""Index, velocity, decibel and geometry conversions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar_stream_decoder.constants import DB_PER_LOG2_UNIT, Platform, cpu_clock_hz, parse_platform
from radar_stream_decoder.units import (
    Log2Combining,
    antenna_shift_bits,
    cartesian_to_spherical,
    default_log2_combining,
    default_log2_fraction_bits,
    doppler_to_velocity,
    fft_scale_compensation_db,
    log2_magnitude_scale,
    log2_to_db,
    next_power_of_two,
    range_index_to_m,
    signed_doppler,
    spherical_to_cartesian,
    xyz_q_format,
)


def test_next_power_of_two():
    assert [next_power_of_two(n) for n in (1, 2, 3, 200, 256, 257)] == [1, 2, 4, 256, 256, 512]
    with pytest.raises(ValueError):
        next_power_of_two(0)


def test_signed_doppler_is_a_circular_shift():
    bins = np.arange(32)
    signed = signed_doppler(bins, 32)
    assert signed.min() == -16 and signed.max() == 15
    np.testing.assert_array_equal(signed % 32, bins)
    np.testing.assert_array_equal(signed_doppler([16, 17], 32), [-16, -15])
    with pytest.raises(ValueError):
        signed_doppler([1], 0)


def test_velocity_and_range_from_indices():
    np.testing.assert_allclose(doppler_to_velocity([-4, 3], 0.5), [-2.0, 1.5])
    np.testing.assert_allclose(range_index_to_m([0, 5, 100], 0.04, 0.1), [0.0, 0.1, 3.9])


def test_log2_scale_conventions():
    assert default_log2_fraction_bits(Platform.XWR16XX, "oob-sdk2") == 8
    assert default_log2_fraction_bits(Platform.XWR14XX, "oob-sdk2") == 9
    assert default_log2_fraction_bits(Platform.XWR16XX, "oob-sdk3") == 8
    assert default_log2_combining(Platform.XWR16XX, "oob-sdk2") is Log2Combining.SUM
    assert default_log2_combining(Platform.XWR14XX, "oob-sdk2") is Log2Combining.AVERAGE
    assert default_log2_combining(Platform.XWR68XX, "oob-sdk3") is Log2Combining.SHIFTED_SUM
    assert log2_magnitude_scale(Platform.XWR16XX, "oob-sdk2", 4, 2) == pytest.approx(1 / 2048)
    assert log2_magnitude_scale(Platform.XWR14XX, "oob-sdk2") == pytest.approx(1 / 512)
    assert log2_magnitude_scale(Platform.XWR14XX, "oob-sdk2", elevation_tx=True) == pytest.approx(
        4 / 3 / 512
    )
    # SDK 3.x: each antenna shifted right by ceil(log2 12) = 4 before the sum.
    assert log2_magnitude_scale(Platform.XWR16XX, "oob-sdk3", 4, 3) == pytest.approx(
        16 / (256 * 12)
    )
    assert log2_magnitude_scale(
        Platform.XWR16XX, "oob-sdk2", fraction_bits=9, combining="average"
    ) == pytest.approx(1 / 512)
    with pytest.raises(ValueError, match="antenna counts"):
        log2_magnitude_scale(Platform.XWR16XX, "oob-sdk2")
    with pytest.raises(ValueError):
        log2_magnitude_scale(Platform.XWR16XX, "oob-sdk2", 4, 2, combining="median")


def test_antenna_shift_is_the_ceiling_of_log2():
    assert [antenna_shift_bits(n) for n in (1, 2, 3, 4, 5, 8, 9, 12, 16)] == [
        0,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        4,
    ]
    with pytest.raises(ValueError):
        antenna_shift_bits(0)


def test_log2_to_db_matches_amplitude_ratios():
    # 512 raw units at Q9 averaged = one doubling of amplitude = 6.02 dB.
    assert log2_to_db([512], 1 / 512)[0] == pytest.approx(20 * math.log10(2))
    assert pytest.approx(6.0206, abs=1e-4) == DB_PER_LOG2_UNIT
    assert log2_to_db([0], 1.0, offset_db=-3.0)[0] == -3.0


def test_fft_scale_compensation():
    assert fft_scale_compensation_db(Platform.XWR14XX, 256, 16) == pytest.approx(
        20 * math.log10(32 / 256)
    )
    # xWR16xx: s(N) = ceil(log4 N - 1)^2 / N, so s(256)/s(64) = (9/256)/(4/64) = 9/16.
    expected = 20 * math.log10(16 / 32) + 20 * math.log10(9 / 16)
    assert fft_scale_compensation_db(Platform.XWR16XX, 256, 32) == pytest.approx(expected)
    with pytest.raises(ValueError):
        fft_scale_compensation_db(Platform.XWR16XX, 0, 16)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spherical_round_trip(seed):
    rng = np.random.default_rng(seed)
    x, y, z = rng.normal(size=50), rng.uniform(0.1, 9, 50), rng.normal(size=50)
    distance, azimuth, elevation = cartesian_to_spherical(x, y, z)
    back = spherical_to_cartesian(distance, azimuth, elevation)
    np.testing.assert_allclose(np.vstack(back), np.vstack([x, y, z]), atol=1e-12)
    assert np.all(np.abs(azimuth) < 90)


def test_boresight_conventions():
    _distance, azimuth, elevation = cartesian_to_spherical([1.0, 0.0], [0.0, 2.0], [0.0, 2.0])
    np.testing.assert_allclose(azimuth, [90.0, 0.0])
    np.testing.assert_allclose(elevation, [0.0, 45.0])


def test_q_format():
    assert xyz_q_format(0.0732) == 8
    assert xyz_q_format(0.04) == 9
    with pytest.raises(ValueError):
        xyz_q_format(0.0)


def test_platform_parsing_and_clocks():
    assert parse_platform("AWR1642") == Platform.XWR16XX
    assert parse_platform("0xA1843") == Platform.XWR18XX
    assert parse_platform(0xA6843) == Platform.XWR68XX
    assert parse_platform("") is None
    with pytest.raises(ValueError):
        parse_platform("0x12345")
    with pytest.raises(ValueError):
        parse_platform("bananas")
    assert cpu_clock_hz(Platform.XWR16XX) == 600e6
    assert cpu_clock_hz(Platform.XWR18XX) == 200e6
