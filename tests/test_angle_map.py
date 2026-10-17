"""Complex decoding, the angle FFT with a synthetic steering vector, and Cartesian resampling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar_stream_decoder.angle_map import (
    AzimuthHeatmap,
    angle_axis_deg,
    angle_axis_rad,
    angle_bins,
    angle_spectrum,
    decode_complex_iq,
    encode_complex_iq,
    peak_angle_deg,
    polar_to_cartesian,
)
from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.constants import Platform
from radar_stream_decoder.encoder import (
    HeaderTemplate,
    encode_azimuth_heatmap,
    encode_frame,
    encode_magnitude_profile,
)
from radar_stream_decoder.errors import IssueCode
from radar_stream_decoder.frame import decode_packet


def steering(theta_deg: float, antennas: int = 8) -> np.ndarray:
    return np.exp(1j * math.pi * math.sin(math.radians(theta_deg)) * np.arange(antennas))


def test_complex_round_trip_and_part_order(rng):
    values = rng.integers(-30000, 30000, 20) + 1j * rng.integers(-30000, 30000, 20)
    for order in ("imag_real", "real_imag"):
        np.testing.assert_array_equal(
            decode_complex_iq(encode_complex_iq(values, order), order), values
        )
    swapped = decode_complex_iq(encode_complex_iq(values, "imag_real"), "real_imag")
    np.testing.assert_array_equal(swapped, 1j * np.conj(values))
    with pytest.raises(ValueError):
        decode_complex_iq(b"\x00" * 6)
    with pytest.raises(ValueError):
        decode_complex_iq(b"\x00" * 8, order="other")


def test_angle_axis():
    assert angle_bins(64)[0] == -31 and angle_bins(64)[-1] == 31
    axis = angle_axis_deg(64)
    assert axis.size == 63 and np.all(np.diff(axis) > 0)
    np.testing.assert_allclose(axis, -axis[::-1], atol=1e-12)
    np.testing.assert_allclose(np.sin(angle_axis_rad(64)), 2 * angle_bins(64) / 64)
    with pytest.raises(ValueError):
        angle_bins(63)


@pytest.mark.parametrize("theta", [-55.0, -20.0, -3.0, 0.0, 7.5, 33.0, 60.0])
def test_steering_vector_peaks_at_its_angle(theta):
    spectrum = angle_spectrum(steering(theta), 64)
    # The strongest bin is the FFT bin nearest to k = N sin(theta) / 2.
    peak_bin = int(np.argmax(spectrum[0])) - 31
    assert abs(peak_bin - 64 * math.sin(math.radians(theta)) / 2) <= 0.5 + 1e-9
    refined = peak_angle_deg(spectrum, 64)[0]
    assert refined == pytest.approx(theta, abs=0.6 if abs(theta) < 50 else 1.5)


def test_wrong_part_order_mirrors_the_angle():
    cube = 3000 * steering(25.0)[None, :]
    raw = encode_complex_iq(cube.reshape(-1), "imag_real")
    right = peak_angle_deg(angle_spectrum(decode_complex_iq(raw, "imag_real")[None, :]))[0]
    wrong = peak_angle_deg(angle_spectrum(decode_complex_iq(raw, "real_imag")[None, :]))[0]
    assert right == pytest.approx(25.0, abs=0.6)
    assert wrong == pytest.approx(-25.0, abs=0.6)


def test_spectrum_validation():
    with pytest.raises(ValueError):
        angle_spectrum(np.ones((2, 70)), 64)
    with pytest.raises(ValueError):
        peak_angle_deg(np.ones((2, 10)), 64)


def test_polar_to_cartesian_places_a_peak_where_it_belongs():
    ranges = np.linspace(0.0, 10.0, 101)
    theta = angle_axis_rad(64)
    spectrum = np.zeros((ranges.size, theta.size))
    r_index, t_index = 60, 40  # 6 m at asin(2 * 9 / 64)
    spectrum[r_index, t_index] = 1.0
    heat = polar_to_cartesian(spectrum, ranges, theta, grid_points=401)
    y_index, x_index = np.unravel_index(np.nanargmax(heat.image), heat.image.shape)
    expected_x = 6.0 * math.sin(theta[t_index])
    expected_y = 6.0 * math.cos(theta[t_index])
    assert heat.x_m[x_index] == pytest.approx(expected_x, abs=0.1)
    assert heat.y_m[y_index] == pytest.approx(expected_y, abs=0.1)
    assert np.isnan(heat.image[0, 0])  # (-10, 0) lies outside the angle interval
    assert heat.extent == (-10.0, 10.0, 0.0, 10.0)


def test_polar_to_cartesian_validation():
    with pytest.raises(ValueError):
        polar_to_cartesian(np.zeros((3, 4)), [0, 1, 2], [0.0, 0.1, 0.2])
    with pytest.raises(ValueError):
        polar_to_cartesian(np.zeros((3, 2)), [0, 2, 1], [0.0, 0.1])
    with pytest.raises(ValueError):
        polar_to_cartesian(np.zeros((2, 2)), [0, 1], [0.0, 0.1], grid_points=1)


def test_heatmap_object(sensor_16xx):
    rows = sensor_16xx.num_range_bins
    cube = np.zeros((rows, 8), dtype=complex)
    cube[40] = 2000 * steering(-12.0)
    heatmap = AzimuthHeatmap(cube.astype(np.complex64))
    assert heatmap.resolved
    assert heatmap.peak_azimuth_deg()[40] == pytest.approx(-12.0, abs=0.6)
    cartesian = heatmap.to_cartesian(np.arange(rows) * sensor_16xx.range_bin_m, grid_points=50)
    assert cartesian.image.shape == (50, 50)
    with pytest.raises(ValueError):
        AzimuthHeatmap(np.zeros(6, dtype=np.complex64)).spectrum()


def test_decoder_infers_the_antenna_count():
    cube = np.zeros((64, 8), dtype=complex)
    cube[10] = 1500 * steering(30.0)
    template = HeaderTemplate(platform=Platform.XWR16XX)
    heatmap_tlv = (4, encode_azimuth_heatmap(cube))
    profile_tlv = (2, encode_magnitude_profile(np.ones(64)))
    with_profile = decode_packet(encode_frame(template, [profile_tlv, heatmap_tlv]))
    assert with_profile.azimuth_heatmap.cube.shape == (64, 8)
    configured = decode_packet(
        encode_frame(template, [heatmap_tlv]), DecoderConfig(num_virtual_antennas=8)
    )
    assert configured.azimuth_heatmap.cube.shape == (64, 8)
    unknown = decode_packet(encode_frame(template, [heatmap_tlv]))
    assert not unknown.azimuth_heatmap.resolved
    assert IssueCode.UNRESOLVED_SHAPE in [issue.code for issue in unknown.issues]
