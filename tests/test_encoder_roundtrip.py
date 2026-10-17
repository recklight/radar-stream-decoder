"""Encode -> decode round trips, field by field, over several seeds."""

from __future__ import annotations

import numpy as np
import pytest

from radar_stream_decoder.angle_map import decode_complex_iq
from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.constants import MAGIC_WORD, PACKET_ALIGNMENT, Platform
from radar_stream_decoder.encoder import (
    HeaderTemplate,
    encode_azimuth_heatmap,
    encode_frame,
    encode_magnitude_profile,
    encode_points_float,
    encode_points_qformat,
    encode_range_doppler,
    encode_range_segment,
    encode_side_info,
    encode_vital_summary,
)
from radar_stream_decoder.frame import decode_packet
from radar_stream_decoder.header import SdkVersion
from radar_stream_decoder.payloads import PointCloud
from radar_stream_decoder.synthetic import SyntheticScene, generate_packets
from radar_stream_decoder.vital_signs import VitalSignsSummary

SEEDS = [0, 1, 2, 3, 4]
SDK2 = HeaderTemplate(SdkVersion(2, 0, 0, 4), Platform.XWR16XX, frame_number=3)


@pytest.mark.parametrize("seed", SEEDS)
def test_qformat_points_round_trip(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    q = int(rng.integers(6, 11))
    limit = 30000 / (1 << q)
    cloud = PointCloud(
        rng.uniform(-limit, limit, n),
        rng.uniform(0, limit, n),
        rng.uniform(-limit / 4, limit / 4, n),
        range_idx=rng.integers(0, 512, n),
        doppler_idx=rng.integers(-32, 32, n),
        peak=rng.integers(0, 65535, n),
    )
    decoded = decode_packet(encode_frame(SDK2, [(1, encode_points_qformat(cloud, q))])).points
    step = 1.0 / (1 << q)
    for name in ("x_m", "y_m", "z_m"):
        np.testing.assert_allclose(
            getattr(decoded, name), getattr(cloud, name), atol=step / 2 + 1e-12
        )
    np.testing.assert_array_equal(decoded.range_idx, cloud.range_idx)
    np.testing.assert_array_equal(decoded.doppler_idx, cloud.doppler_idx)
    np.testing.assert_array_equal(decoded.peak, cloud.peak)


@pytest.mark.parametrize("seed", SEEDS)
def test_float_points_and_side_info_round_trip(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 30))
    cloud = PointCloud(
        rng.normal(size=n),
        rng.uniform(0, 9, n),
        rng.normal(size=n),
        velocity_mps=rng.normal(size=n),
    )
    snr, noise = rng.uniform(0, 50, n), rng.uniform(30, 60, n)
    template = HeaderTemplate(SdkVersion(3, 5), Platform.XWR16XX, declared_point_count=n)
    packet = encode_frame(
        template, [(1, encode_points_float(cloud)), (7, encode_side_info(snr, noise))]
    )
    decoded = decode_packet(packet).points
    for name in ("x_m", "y_m", "z_m", "velocity_mps"):
        np.testing.assert_array_equal(
            getattr(decoded, name), getattr(cloud, name).astype(np.float32).astype(np.float64)
        )
    np.testing.assert_allclose(decoded.snr_db, snr, atol=0.05 + 1e-9)
    np.testing.assert_allclose(decoded.noise_db, noise, atol=0.05 + 1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_arrays_round_trip(seed, sensor_16xx):
    rng = np.random.default_rng(seed)
    rows, cols, antennas = sensor_16xx.num_range_bins, sensor_16xx.num_doppler_bins, 8
    profile = rng.integers(0, 65536, rows).astype(np.uint16)
    matrix = rng.integers(0, 65536, (rows, cols)).astype(np.uint16)
    cube = rng.integers(-3000, 3000, (rows, antennas)) + 1j * rng.integers(
        -3000, 3000, (rows, antennas)
    )
    packet = encode_frame(
        SDK2,
        [
            (2, encode_magnitude_profile(profile)),
            (3, encode_magnitude_profile(profile[::-1])),
            (4, encode_azimuth_heatmap(cube)),
            (5, encode_range_doppler(matrix)),
        ],
    )
    frame = decode_packet(packet, sensor=sensor_16xx)
    np.testing.assert_array_equal(frame.range_profile.raw, profile)
    np.testing.assert_array_equal(frame.noise_profile.raw, profile[::-1])
    np.testing.assert_array_equal(frame.range_doppler.raw, matrix)
    np.testing.assert_array_equal(frame.azimuth_heatmap.cube, cube.astype(np.complex64))
    assert not frame.issues


@pytest.mark.parametrize("seed", SEEDS)
def test_vital_signs_round_trip(seed):
    rng = np.random.default_rng(seed)
    values = {name: float(rng.normal()) for name in VitalSignsSummary.__dataclass_fields__}
    values.update(
        peak_range_bin=18,
        phase_range_bin=17,
        processing_cycles=int(rng.integers(0, 2**32)),
        first_range_bin=8,
        last_range_bin=29,
        motion_detected=bool(seed % 2),
        reserved=tuple(float(v) for v in rng.normal(size=8)),
    )
    summary = VitalSignsSummary(**values)
    segment = rng.integers(-2000, 2000, 22) + 1j * rng.integers(-2000, 2000, 22)
    template = HeaderTemplate.for_schema("vital-signs", Platform.XWR14XX, declared_point_count=99)
    packet = encode_frame(
        template, [(0, encode_vital_summary(summary)), (1, encode_range_segment(segment))]
    )
    frame = decode_packet(packet)
    decoded = frame.vital_signs
    for name, value in values.items():
        expected = np.float32(value) if isinstance(value, float) else value
        if name == "reserved":
            expected = tuple(float(np.float32(v)) for v in value)
        assert getattr(decoded, name) == pytest.approx(expected), name
    np.testing.assert_array_equal(frame.range_segment.values, segment.astype(np.complex64))
    assert frame.range_segment.start_bin == 8


def test_frame_layout():
    one = encode_frame(SDK2, [(6, bytes(24))], padding_byte=0x0F)
    assert one.startswith(MAGIC_WORD) and len(one) % PACKET_ALIGNMENT == 0
    assert one[72:] == b"\x0f" * (len(one) - 72)
    short = encode_frame(HeaderTemplate(subframe_number=None), [(6, bytes(24))])
    assert len(short) == 96  # 36 + 32 bytes of TLV, padded to 96
    empty = encode_frame(SDK2, [])
    assert len(empty) == 64 and decode_packet(empty).tlv_types == ()
    with pytest.raises(ValueError):
        encode_frame(SDK2, [], padding_byte=300)


def test_complex_encoding_clips_and_orders():
    raw = encode_azimuth_heatmap(np.array([[40000 + 1j, -1 - 50000j]]))
    np.testing.assert_array_equal(decode_complex_iq(raw), [32767 + 1j, -1 - 32768j])
    with pytest.raises(ValueError):
        encode_azimuth_heatmap(np.zeros(4))
    with pytest.raises(ValueError):
        encode_range_doppler(np.zeros(4))


@pytest.mark.parametrize("schema", ["oob-sdk2", "oob-sdk3", "vital-signs"])
def test_generation_is_byte_reproducible(schema):
    first = list(generate_packets(SyntheticScene(schema=schema, seed=9), 5))
    again = list(generate_packets(SyntheticScene(schema=schema, seed=9), 5))
    other = list(generate_packets(SyntheticScene(schema=schema, seed=10), 5))
    assert first == again
    assert first != other


def test_strict_decoding_accepts_every_synthetic_family():
    strict = DecoderConfig(strict=True)
    for schema, platform in [
        ("oob-sdk2", "16xx"),
        ("oob-sdk2", "14xx"),
        ("oob-sdk3", "16xx"),
        ("vital-signs", "14xx"),
    ]:
        scene = SyntheticScene(schema=schema, platform=platform, seed=2, padding_byte=0x0F)
        for packet in generate_packets(scene, 3):
            assert not decode_packet(packet, strict, scene.radar()).has_errors
