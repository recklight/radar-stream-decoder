"""Payload decoders: exact values, signed Doppler, length checks, cross-TLV enrichment."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.constants import DemoTlvType, Platform
from radar_stream_decoder.encoder import (
    HeaderTemplate,
    encode_frame,
    encode_magnitude_profile,
    encode_points_float,
    encode_points_qformat,
    encode_range_doppler,
    encode_side_info,
    encode_stats,
    encode_temperature,
)
from radar_stream_decoder.errors import IssueCode, Severity
from radar_stream_decoder.frame import decode_packet
from radar_stream_decoder.header import SdkVersion
from radar_stream_decoder.payloads import (
    MagnitudeProfile,
    PointCloud,
    ProcessingStats,
    RangeDopplerMap,
    SnrMethod,
    TemperatureReport,
)
from radar_stream_decoder.sensor_config import parse_cli_config

SDK2_16XX = HeaderTemplate(SdkVersion(2, 0, 0, 4), Platform.XWR16XX, frame_number=10)


def _cloud(**extra):
    base = {
        "x_m": [0.5, -1.25, 2.0],
        "y_m": [3.0, 4.5, 1.0],
        "z_m": [0.0, 0.25, -0.5],
        "range_idx": [41, 64, 31],
        "doppler_idx": [-3, 0, 5],
        "peak": [1200, 800, 4000],
    }
    base.update(extra)
    return PointCloud(**base)


def _codes(frame):
    return [issue.code for issue in frame.issues]


class TestQFormatPoints:
    def test_exact_values_and_signed_doppler(self):
        packet = encode_frame(SDK2_16XX, [(1, encode_points_qformat(_cloud(), q_format=9))])
        points = decode_packet(packet).points
        assert points.q_format == 9
        np.testing.assert_array_equal(points.x_m, [0.5, -1.25, 2.0])
        np.testing.assert_array_equal(points.doppler_idx, [-3, 0, 5])
        np.testing.assert_array_equal(points.range_idx, [41, 64, 31])
        np.testing.assert_array_equal(points.peak, [1200, 800, 4000])
        assert np.all(np.isnan(points.velocity_mps))  # no radar configuration

    def test_velocity_needs_the_radar_configuration(self, sensor_16xx):
        packet = encode_frame(SDK2_16XX, [(1, encode_points_qformat(_cloud(), q_format=9))])
        points = decode_packet(packet, sensor=sensor_16xx).points
        np.testing.assert_allclose(
            points.velocity_mps, np.array([-3, 0, 5]) * sensor_16xx.doppler_resolution_mps
        )

    def test_sdk10_unsigned_doppler_is_made_signed(self, sensor_16xx):
        payload = encode_points_qformat(_cloud(), 9, unsigned_doppler_bins=16)
        template = HeaderTemplate(SdkVersion(1, 0), Platform.XWR16XX, subframe_number=None)
        packet = encode_frame(template, [(1, payload)])
        raw = decode_packet(packet).points
        assert IssueCode.MISSING_DOPPLER_BINS in _codes(decode_packet(packet))
        np.testing.assert_array_equal(raw.doppler_idx, [13, 0, 5])  # still unsigned
        fixed = decode_packet(packet, sensor=sensor_16xx).points
        np.testing.assert_array_equal(fixed.doppler_idx, [-3, 0, 5])
        forced = decode_packet(packet, DecoderConfig(num_doppler_bins=16)).points
        np.testing.assert_array_equal(forced.doppler_idx, [-3, 0, 5])

    def test_declared_count_beyond_the_payload_is_not_trusted(self):
        payload = bytearray(encode_points_qformat(_cloud(), 9))
        struct.pack_into("<H", payload, 0, 50)  # descriptor claims 50 points
        frame = decode_packet(encode_frame(SDK2_16XX, [(1, bytes(payload))]))
        assert len(frame.points) == 3
        assert IssueCode.LENGTH_MISMATCH in _codes(frame)
        assert frame.has_errors

    def test_ragged_payload_is_reported(self):
        payload = encode_points_qformat(_cloud(), 9) + b"\x01\x02\x03"
        frame = decode_packet(encode_frame(SDK2_16XX, [(1, payload)]))
        assert len(frame.points) == 3 and frame.has_errors

    def test_descriptor_only_and_absurd_q_format(self):
        frame = decode_packet(encode_frame(SDK2_16XX, [(1, b"\x00\x00")]))
        assert frame.points is not None and len(frame.points) == 0 and frame.has_errors
        payload = bytearray(encode_points_qformat(_cloud(), 9))
        struct.pack_into("<H", payload, 2, 40)
        frame = decode_packet(encode_frame(SDK2_16XX, [(1, bytes(payload))]))
        assert IssueCode.INVALID_VALUE in _codes(frame) and len(frame.points) == 0

    def test_geometry_and_records(self):
        cloud = _cloud()
        np.testing.assert_allclose(cloud.range_m, np.sqrt([9.25, 21.875, 5.25]))
        assert cloud.azimuth_deg[0] == pytest.approx(np.degrees(np.arctan2(0.5, 3.0)))
        records = cloud.to_records()
        assert records.dtype.names[:3] == ("x_m", "y_m", "z_m")
        assert len(cloud.select(cloud.peak > 1000)) == 2
        np.testing.assert_allclose(cloud.range_from_index_m(0.1, 0.05), [4.05, 6.35, 3.05])

    def test_column_lengths_are_checked(self):
        with pytest.raises(ValueError, match="y_m"):
            PointCloud([0.0, 1.0], [1.0], [0.0, 0.0])


class TestSdk3Points:
    def test_float_points_merge_side_info(self):
        cloud = PointCloud([1.0, -2.0], [3.0, 4.0], [0.5, 0.0], velocity_mps=[0.25, -1.5])
        template = HeaderTemplate(SdkVersion(3, 5), Platform.XWR16XX, declared_point_count=2)
        packet = encode_frame(
            template,
            [(1, encode_points_float(cloud)), (7, encode_side_info([21.4, 9.0], [41.0, 40.2]))],
        )
        frame = decode_packet(packet)
        points = frame.points
        np.testing.assert_array_equal(points.velocity_mps, [0.25, -1.5])
        np.testing.assert_allclose(points.snr_db, [21.4, 9.0])
        np.testing.assert_allclose(points.noise_db, [41.0, 40.2])
        assert points.snr_method is SnrMethod.SIDE_INFO
        assert points.range_idx is None and points.peak is None
        assert not frame.issues

    def test_side_info_for_another_point_count_is_not_merged(self):
        cloud = PointCloud([1.0], [3.0], [0.0], velocity_mps=[0.0])
        template = HeaderTemplate(SdkVersion(3, 5), Platform.XWR16XX, declared_point_count=1)
        packet = encode_frame(
            template,
            [(1, encode_points_float(cloud)), (7, encode_side_info([1.0, 2.0], [3.0, 4.0]))],
        )
        frame = decode_packet(packet)
        assert IssueCode.SIDE_INFO_MISMATCH in _codes(frame)
        assert np.isnan(frame.points.snr_db[0])

    def test_header_count_mismatch_is_a_warning(self):
        cloud = PointCloud([1.0], [3.0], [0.0])
        template = HeaderTemplate(SdkVersion(3, 5), Platform.XWR16XX, declared_point_count=4)
        frame = decode_packet(encode_frame(template, [(1, encode_points_float(cloud))]))
        assert _codes(frame) == [IssueCode.POINT_COUNT_MISMATCH]
        assert not frame.has_errors


class TestMagnitudeArrays:
    def test_profiles_and_decibels(self, sensor_16xx):
        raw = np.arange(256, dtype=np.uint16) * 100
        packet = encode_frame(
            SDK2_16XX,
            [(2, encode_magnitude_profile(raw)), (3, encode_magnitude_profile(raw[::-1]))],
        )
        frame = decode_packet(packet, sensor=sensor_16xx)
        profile = frame.range_profile
        np.testing.assert_array_equal(profile.raw, raw)
        assert frame.noise_profile.kind == "noise"
        expected_scale = 20 * np.log10(2) / (256 * 4 * 2)  # Q8, summed over 8 virtual antennas
        assert profile.db_per_lsb == pytest.approx(expected_scale)
        np.testing.assert_allclose(profile.to_db(), raw * expected_scale)
        axis = profile.range_axis(sensor=sensor_16xx)
        assert axis[10] == pytest.approx(10 * sensor_16xx.range_bin_m)

    def test_scale_is_unknown_without_antenna_counts(self):
        packet = encode_frame(SDK2_16XX, [(2, encode_magnitude_profile(np.ones(8)))])
        profile = decode_packet(packet).range_profile
        assert profile.db_per_lsb is None
        with pytest.raises(ValueError, match="decibel scale"):
            profile.to_db()
        assert decode_packet(packet, DecoderConfig(num_virtual_antennas=8)).range_profile.db_per_lsb
        with pytest.raises(ValueError):
            profile.range_axis()

    def test_14xx_scale_needs_no_antenna_counts(self):
        template = HeaderTemplate(SdkVersion(1, 2), Platform.XWR14XX, subframe_number=None)
        packet = encode_frame(template, [(2, encode_magnitude_profile(np.ones(8)))])
        profile = decode_packet(packet).range_profile
        assert profile.db_per_lsb == pytest.approx(20 * np.log10(2) / 512)

    def test_cfg_without_platform_takes_the_header_platform(self):
        # Three transmitters on an xWR14xx: TX1 is the elevation antenna, which
        # puts the 4/3 factor on the averaged values whether or not
        # parse_cli_config was given the platform.
        text = (
            "channelCfg 15 7 0\n"
            "profileCfg 0 77 10 6 52 0 0 30 1 256 6000 0 0 30\n"
            "chirpCfg 0 0 0 0 0 0 0 1\n"
            "chirpCfg 1 1 0 0 0 0 0 4\n"
            "chirpCfg 2 2 0 0 0 0 0 2\n"
            "frameCfg 0 2 32 0 50 1 0\n"
        )
        template = HeaderTemplate(SdkVersion(1, 2), Platform.XWR14XX, subframe_number=None)
        packet = encode_frame(template, [(2, encode_magnitude_profile(np.ones(256)))])
        bare = decode_packet(packet, sensor=parse_cli_config(text)).range_profile
        named = decode_packet(packet, sensor=parse_cli_config(text, "14xx")).range_profile
        assert named.db_per_lsb == pytest.approx(20 * np.log10(2) / 512 * 4 / 3)
        assert bare.db_per_lsb == pytest.approx(named.db_per_lsb)

    def test_odd_profile_length(self):
        packet = encode_frame(SDK2_16XX, [(2, encode_magnitude_profile(np.ones(4)) + b"\x07")])
        frame = decode_packet(packet)
        assert len(frame.range_profile) == 4 and frame.has_errors

    def test_range_doppler_shape_sources(self, sensor_16xx):
        matrix = np.arange(256 * 16, dtype=np.uint16).reshape(256, 16)
        rd = encode_range_doppler(matrix)
        alone = decode_packet(encode_frame(SDK2_16XX, [(5, rd)]))
        assert not alone.range_doppler.resolved
        assert IssueCode.UNRESOLVED_SHAPE in _codes(alone)
        with_profile = decode_packet(
            encode_frame(SDK2_16XX, [(2, encode_magnitude_profile(matrix[:, 0])), (5, rd)])
        )
        np.testing.assert_array_equal(with_profile.range_doppler.raw, matrix)
        with_sensor = decode_packet(encode_frame(SDK2_16XX, [(5, rd)]), sensor=sensor_16xx)
        np.testing.assert_array_equal(with_sensor.range_doppler.raw, matrix)
        with_setting = decode_packet(
            encode_frame(SDK2_16XX, [(5, rd)]), DecoderConfig(num_range_bins=256)
        )
        assert with_setting.range_doppler.raw.shape == (256, 16)

    def test_range_doppler_centring_and_axes(self, sensor_16xx):
        matrix = np.zeros((4, 8), dtype=np.uint16)
        matrix[1, 0] = 9  # zero Doppler
        matrix[2, 7] = 5  # Doppler bin -1
        rd = RangeDopplerMap(matrix, db_per_lsb=0.5)
        centred = rd.centered()
        assert centred[1, 4] == 9 and centred[2, 3] == 5
        np.testing.assert_allclose(rd.values_at([2], [-1]), [5.0])
        ranges, velocities = rd.axes(0.1, 0.25)
        np.testing.assert_allclose(velocities, (np.arange(8) - 4) * 0.25)
        np.testing.assert_allclose(ranges, np.arange(4) * 0.1)
        assert rd.to_db(centered=True)[1, 4] == pytest.approx(4.5)
        assert rd.num_range_bins == 4 and rd.num_doppler_bins == 8
        with pytest.raises(ValueError):
            RangeDopplerMap(np.zeros(6, dtype=np.uint16)).centered()

    def test_magnitude_profile_helpers(self):
        profile = MagnitudeProfile("range", np.array([0, 512], dtype=np.uint16))
        np.testing.assert_allclose(profile.to_db(db_per_lsb=1 / 512, offset_db=1.0), [1.0, 2.0])


class TestStatsTemperatureUnknown:
    def test_stats(self):
        stats = ProcessingStats(1500, 400, 45000, 25, 37, 12)
        frame = decode_packet(encode_frame(SDK2_16XX, [(6, encode_stats(stats))]))
        assert frame.stats == stats
        assert frame.stats.as_dict()["active_frame_cpu_load_pct"] == 37

    def test_short_stats_are_dropped_with_an_error(self):
        frame = decode_packet(encode_frame(SDK2_16XX, [(6, bytes(20))]))
        assert frame.stats is None and frame.has_errors

    def test_temperature(self):
        report = TemperatureReport(0, 1234, (40, 41, 42, 43), (45, 46, 47), 39, (44, 45))
        template = HeaderTemplate(SdkVersion(3, 5), Platform.XWR16XX)
        frame = decode_packet(encode_frame(template, [(9, encode_temperature(report))]))
        assert frame.temperature == report and frame.temperature.valid

    def test_unknown_and_duplicate_tlvs_are_kept_as_warnings(self):
        stats = encode_stats(ProcessingStats(1, 2, 3, 4, 5, 6))
        frame = decode_packet(encode_frame(SDK2_16XX, [(42, b"abcd"), (6, stats), (6, stats)]))
        assert frame.unknown_tlvs[0].tlv_type == 42 and frame.unknown_tlvs[0].raw == b"abcd"
        assert IssueCode.UNKNOWN_TLV in _codes(frame)
        assert IssueCode.DUPLICATE_TLV in _codes(frame)
        assert all(issue.severity is Severity.WARNING for issue in frame.issues)
        assert frame.tlv_types == (42, 6, 6)


class TestSnrEnrichment:
    def test_points_get_snr_from_the_detection_matrix(self, scene_16xx):
        from radar_stream_decoder.synthetic import generate_packets

        frame = decode_packet(next(generate_packets(scene_16xx, 1)), sensor=scene_16xx.radar())
        assert frame.points.snr_method is SnrMethod.DETECTION_MATRIX
        assert np.all(np.isfinite(frame.points.snr_db))

    def test_noise_profile_fallback(self, scene_16xx):
        from radar_stream_decoder.synthetic import generate_packets

        packet = next(
            generate_packets(scene_16xx, 1, outputs=("points", "range_profile", "noise_profile"))
        )
        frame = decode_packet(packet, sensor=scene_16xx.radar())
        assert frame.points.snr_method is SnrMethod.NOISE_PROFILE

    def test_no_arrays_no_snr(self, scene_16xx):
        from radar_stream_decoder.synthetic import generate_packets

        packet = next(generate_packets(scene_16xx, 1, outputs=("points",)))
        frame = decode_packet(packet, sensor=scene_16xx.radar())
        assert frame.points.snr_method is SnrMethod.UNAVAILABLE
        assert np.all(np.isnan(frame.points.snr_db))

    def test_noise_window_wider_than_the_doppler_axis(self, scene_16xx):
        from radar_stream_decoder.synthetic import generate_packets

        # 16 Doppler bins leave no training cell beside 7 guard cells per side.
        packet = next(generate_packets(scene_16xx, 1))
        config = DecoderConfig(snr_guard_cells=7)
        frame = decode_packet(packet, config, sensor=scene_16xx.radar())
        assert frame.num_points and frame.points.snr_method is SnrMethod.UNAVAILABLE
        assert np.all(np.isnan(frame.points.snr_db))
        assert IssueCode.INVALID_VALUE in _codes(frame) and not frame.has_errors


def test_frame_accessors_and_summary(scene_16xx):
    from radar_stream_decoder.synthetic import generate_packets

    frame = decode_packet(next(generate_packets(scene_16xx, 1)))
    summary = frame.summary()
    assert summary["schema"] == "oob-sdk2" and summary["num_points"] == frame.num_points
    assert summary["tlv_types"] == [1, 2, 3, 4, 5, 6]
    assert frame.key == (1, 0) and frame.platform == Platform.XWR16XX
    assert frame.azimuth_heatmap is not None and frame.side_info is None
    assert frame.vital_signs is None and frame.range_segment is None
    assert int(DemoTlvType.STATS) in frame.tlv_types
