"""Synthetic scenes against their ground truth, and the fault injector's bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from radar_stream_decoder.constants import MAGIC_WORD
from radar_stream_decoder.frame import decode_packet
from radar_stream_decoder.payloads import PayloadKind
from radar_stream_decoder.synthetic import (
    FAULT_PRESETS,
    FaultPlan,
    PacketFate,
    SyntheticScene,
    Target,
    content_end,
    generate_packets,
    inject_faults,
    packet_crc,
    scene_truth,
    split_into_chunks,
    synthetic_capture,
)


def test_decoded_points_match_the_truth(scene_16xx):
    sensor = scene_16xx.radar()
    for index, packet in enumerate(generate_packets(scene_16xx, 40)):
        truth = scene_truth(scene_16xx, index)
        points = decode_packet(packet, sensor=sensor).points
        detected = np.flatnonzero(truth.detected)
        if not detected.size:
            assert points is None
            continue
        step = 1.0 / (1 << sensor.xyz_q_format)
        np.testing.assert_allclose(points.x_m, truth.position_m[detected, 0], atol=step)
        np.testing.assert_allclose(points.range_m, truth.range_m[detected], atol=2 * step)
        np.testing.assert_array_equal(points.range_idx, truth.range_idx[detected])
        np.testing.assert_array_equal(points.doppler_idx, truth.doppler_idx[detected])
        np.testing.assert_allclose(
            points.velocity_mps,
            truth.radial_velocity_mps[detected],
            atol=sensor.doppler_resolution_mps / 2 + 1e-9,
        )


def test_profiles_are_cuts_of_the_detection_matrix(scene_16xx):
    frame = decode_packet(next(generate_packets(scene_16xx, 1)), sensor=scene_16xx.radar())
    matrix = frame.range_doppler.raw
    np.testing.assert_array_equal(frame.range_profile.raw, matrix[:, 0])
    np.testing.assert_array_equal(frame.noise_profile.raw, matrix[:, matrix.shape[1] // 2 - 1])


def test_targets_move_and_swing():
    walker = Target(0.0, 2.0, vy_mps=1.0)
    assert walker.state_at(3.0)[0][1] == pytest.approx(5.0)
    swinger = Target(0.0, 2.0, vy_mps=1.0, swing_period_s=4.0)
    position, velocity = swinger.state_at(2.0)
    assert position[1] == pytest.approx(2.0, abs=1e-12) and velocity[1] == pytest.approx(-1.0)


def test_output_selection_and_declared_count(scene_16xx):
    only = scene_16xx.replace(outputs=("points", "stats"), declared_point_count=99)
    frame = decode_packet(next(generate_packets(only, 1)))
    assert frame.tlv_types == (1, 6) and frame.header.declared_point_count == 99
    empty = SyntheticScene(targets=(), seed=1)
    frame = decode_packet(next(generate_packets(empty, 1)))
    assert frame.points is None and 1 not in frame.tlv_types  # no points, no point TLV


def test_sdk3_extras(scene_sdk3):
    frame = decode_packet(next(generate_packets(scene_sdk3, 1)))
    assert PayloadKind.POINT_SIDE_INFO in frame.payloads and frame.temperature.valid
    assert frame.header.version.major == 3 and frame.header.header_len == 40


def test_scene_validation():
    assert SyntheticScene(schema="vital-signs", platform="16xx").platform == 0xA1443
    with pytest.raises(ValueError):
        SyntheticScene(padding_byte=256)
    with pytest.raises(ValueError):
        scene_truth(SyntheticScene(schema="vital-signs"), 0)
    with pytest.raises(ValueError, match="outside"):
        list(
            generate_packets(
                SyntheticScene(
                    schema="vital-signs", chest=type(SyntheticScene().chest)(range_m=2.0)
                ),
                1,
            )
        )


def test_fault_bookkeeping(packets_vital):
    plan = FaultPlan(
        leading_junk_bytes=50, drop_every=5, truncate_probability=0.2, duplicate_probability=0.3
    )
    faulted = inject_faults(packets_vital, plan, 4)
    assert len(faulted.fates) == len(packets_vital)
    assert [faulted.fates[i] for i in (4, 9, 14)] == [PacketFate.DROPPED] * 3
    assert faulted.count(PacketFate.TRUNCATED) > 0
    assert faulted.data.find(MAGIC_WORD) == 50
    delivered = [
        p for p, f in zip(packets_vital, faulted.fates, strict=True) if f is PacketFate.DELIVERED
    ]
    for packet, copies in zip(packets_vital, faulted.copies, strict=True):
        assert faulted.data.count(packet) == copies
    assert all(faulted.data.count(p) >= 1 for p in delivered)


def test_truncation_cuts_inside_the_content(packets_vital):
    faulted = inject_faults(packets_vital[:5], FaultPlan(truncate_probability=1.0), 1)
    assert faulted.count(PacketFate.TRUNCATED) == 5
    assert len(faulted.data) < sum(content_end(p) for p in packets_vital[:5])


def test_block_dump_layout(packets_vital):
    faulted = inject_faults(packets_vital[:6], FaultPlan(block_dump_size=2880), 1)
    assert len(faulted.data) == 6 * 2880
    blocks = [faulted.data[i : i + 2880] for i in range(0, len(faulted.data), 2880)]
    assert blocks[0].startswith(packets_vital[0]) and set(blocks[0][288:]) == {0}
    assert blocks[3].startswith(packets_vital[2] + packets_vital[3])
    assert faulted.copies == (2, 2, 2, 2, 2, 1)


def test_block_grows_for_large_packets(packets_16xx):
    faulted = inject_faults(packets_16xx[:3], FaultPlan(block_dump_size=2880), 1)
    block = len(faulted.data) // 3
    assert block % 2880 == 0 and block >= 2 * len(packets_16xx[0])


def test_bit_flips_stay_inside_the_content(packets_vital):
    faulted = inject_faults(packets_vital[:10], FaultPlan(bitflip_probability=1.0), 2)
    assert faulted.count(PacketFate.CORRUPTED) == 10
    for original, start in zip(packets_vital[:10], range(0, 2880, 288), strict=True):
        damaged = faulted.data[start : start + 288]
        diff = [i for i in range(288) if damaged[i] != original[i]]
        assert len(diff) == 1 and len(MAGIC_WORD) <= diff[0] < content_end(original)


def test_fault_plan_validation():
    for bad in (
        {"junk_probability": 1.5},
        {"drop_every": 0},
        {"block_dump_size": 10},
        {"junk_max_bytes": 0},
    ):
        with pytest.raises(ValueError):
            FaultPlan(**bad)
    assert set(FAULT_PRESETS) == {"none", "light", "heavy", "block-dump", "bitflip"}


def test_chunks_and_helpers():
    data = bytes(range(256)) * 10
    chunks = split_into_chunks(data, 1, max_chunk=50, min_chunk=10)
    assert b"".join(chunks) == data and all(len(c) <= 50 for c in chunks)
    with pytest.raises(ValueError):
        split_into_chunks(data, 1, max_chunk=5, min_chunk=6)
    stream = synthetic_capture(SyntheticScene(schema="vital-signs"), 5)
    assert stream.count(PacketFate.DELIVERED) == 5 and stream.junk_bytes == 0
    assert packet_crc(b"abc") == 891568578
