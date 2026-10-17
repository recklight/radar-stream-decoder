"""Capture reading, de-duplication, gap statistics and timestamp unwrapping."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from radar_stream_decoder.assembler import AssemblerStats
from radar_stream_decoder.capture import (
    FrameDeduplicator,
    FrameTimestamper,
    FrameVerdict,
    analyze_capture,
    deduplicate,
    estimate_frame_period_s,
    frame_timestamps,
    iter_capture,
    unwrap_cpu_cycles,
)
from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.constants import CYCLE_COUNTER_MODULUS
from radar_stream_decoder.errors import IssueCode
from radar_stream_decoder.frame import decode_packet
from radar_stream_decoder.synthetic import (
    FAULT_PRESETS,
    FaultPlan,
    PacketFate,
    SyntheticScene,
    generate_packets,
    synthetic_capture,
)


def write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_iter_capture_is_independent_of_chunk_size(capture_file):
    small = list(iter_capture(capture_file, chunk_size=17))
    large = list(iter_capture(capture_file, chunk_size=1 << 20))
    assert [f.crc32 for f in small] == [f.crc32 for f in large]
    assert len(small) == 12


def test_iter_capture_reports_assembler_stats(capture_file):
    stats = AssemblerStats()
    frames = list(iter_capture(capture_file, stats=stats))
    assert stats.frames_decoded == len(frames) == 12
    assert stats.bytes_skipped == len(b"\x00\xff junk ")


def test_missing_capture_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_capture(tmp_path / "absent.bin"))


def test_block_dump_recording(tmp_path):
    scene = SyntheticScene(schema="vital-signs", seed=12)
    stream = synthetic_capture(scene, 200, FAULT_PRESETS["block-dump"], 5)
    unique = stream.count(PacketFate.DELIVERED)
    report = analyze_capture(write(tmp_path / "dump.bin", stream.data)).report
    assert report.unique_frames == unique
    assert report.packets == 2 * unique - 1
    assert report.duplicates == unique - 1
    assert report.conflicts == 0 and report.frame_resets == 0
    expected_span = report.last_frame - report.first_frame + 1
    assert report.missing_frames == expected_span - unique
    assert report.drop_ratio == pytest.approx(report.missing_frames / expected_span)
    assert sum(report.gap_histogram.values()) == unique - 1
    assert report.estimated_frame_period_s == pytest.approx(0.05, rel=1e-3)
    assert report.clock_hz == 200e6
    assert report.schemas == {"vital-signs": unique}
    assert report.mean_packet_bytes == 288


def test_drop_pattern_statistics(tmp_path):
    scene = SyntheticScene(schema="vital-signs", seed=2)
    stream = synthetic_capture(scene, 30, FaultPlan(drop_every=3), 1)
    report = analyze_capture(write(tmp_path / "drops.bin", stream.data)).report
    # Frames 3, 6, ..., 30 are dropped: 20 remain, with nine 2-steps and ten 1-steps.
    assert report.unique_frames == 20
    assert report.gap_histogram == {1: 10, 2: 9}
    assert report.missing_frames == 9
    assert report.drop_ratio == pytest.approx(9 / 29)


def test_conflict_and_restart_detection(packets_vital, scene_vital):
    frames = [decode_packet(p) for p in packets_vital[:5]]
    restarted = [
        decode_packet(p) for p in generate_packets(scene_vital.replace(seed=99), 3)
    ]  # frame numbers 1..3 again, different content
    altered = decode_packet(packets_vital[4][:-1] + b"\x01")  # frame 5, other bytes
    sequence = [*frames, frames[4], altered, *restarted]
    unique, stats = deduplicate(sequence)
    assert stats.duplicates == 1 and stats.conflicts == 1 and stats.resets == 1
    assert [f.frame_number for f in unique] == [1, 2, 3, 4, 5, 1, 2, 3]
    codes = [issue.code for issue in stats.issues]
    assert IssueCode.FRAME_CONFLICT in codes and IssueCode.FRAME_RESET in codes


def test_altered_copy_of_an_older_frame_is_a_conflict(packets_vital):
    # A buffer dump writes frames 3..5 again with frame 3 cut short. Taken for a
    # restart, the cut copy would also turn the repeats of 4 and 5 into new frames.
    frames = [decode_packet(p) for p in packets_vital[:8]]
    cut = decode_packet(packets_vital[2][:200])
    unique, stats = deduplicate([*frames[:5], cut, frames[3], frames[4], *frames[5:]])
    assert [f.frame_number for f in unique] == list(range(1, 9))
    assert stats.conflicts == 1 and stats.duplicates == 2 and stats.resets == 0
    # An intact copy with other bytes but the same header timestamp: one bit flipped.
    flipped = bytearray(packets_vital[2])
    flipped[60] ^= 0x01
    altered = decode_packet(bytes(flipped))
    assert altered.header.cpu_cycles == frames[2].header.cpu_cycles
    unique, stats = deduplicate([*frames[:5], altered, *frames[5:]])
    assert [f.frame_number for f in unique] == list(range(1, 9))
    assert stats.conflicts == 1 and stats.resets == 0


def test_empty_capture_has_no_timing(tmp_path):
    report = analyze_capture(write(tmp_path / "empty.bin", b"")).report
    assert report.timestamp_source == "none" and report.unique_frames == 0
    assert "frame period          unknown" in report.summary_lines()


def test_intact_copy_replaces_a_copy_cut_short(tmp_path, packets_vital):
    # A buffer dump that ends inside packet 2, followed by a dump that repeats
    # packets 1 and 2 whole: the intact copy of frame 2 must win.
    p = packets_vital
    data = p[0] + p[1][:200] + bytes(64) + p[0] + p[1] + p[2]
    analysis = analyze_capture(write(tmp_path / "cut_dump.bin", data))
    assert [f.crc32 for f in analysis.frames] == [zlib.crc32(x) for x in p[:3]]
    report = analysis.report
    assert report.replaced_copies == 1 and report.conflicts == 0 and report.duplicates == 1
    assert report.issue_counts.get("frame-replaced") == 1
    streamed = FrameDeduplicator()
    verdicts = [streamed.judge(decode_packet(x)) for x in (p[0], p[1][:200], p[0], p[1])]
    assert verdicts == [
        FrameVerdict.NEW,
        FrameVerdict.NEW,
        FrameVerdict.REPEAT,
        FrameVerdict.REPLACEMENT,
    ]
    # A damaged copy never displaces an intact one.
    assert not streamed.accept(decode_packet(p[1][:200]))


def test_deduplicator_window_is_bounded(packets_vital):
    dedupe = FrameDeduplicator(window=4)
    frames = [decode_packet(p) for p in packets_vital[:10]]
    assert all(dedupe.accept(frame) for frame in frames)
    assert len(dedupe._recent) == 4
    with pytest.raises(ValueError):
        FrameDeduplicator(window=0)


class TestTimestamps:
    def test_wrap_is_resolved(self):
        cycles = [CYCLE_COUNTER_MODULUS - 4_000_000, 6_000_000, 16_000_000]
        seconds = unwrap_cpu_cycles(cycles, [1, 2, 3], 200e6)
        np.testing.assert_allclose(seconds, [0.0, 0.05, 0.10])

    def test_gap_longer_than_a_wrap_needs_the_period(self):
        # 600 frames of 50 ms = 30 s, more than one 21.47 s wrap of a 200 MHz counter.
        step = 600 * 10_000_000
        cycles = [0, 10_000_000, (10_000_000 + step) % CYCLE_COUNTER_MODULUS]
        frames = [1, 2, 602]
        seconds = unwrap_cpu_cycles(cycles, frames, 200e6, nominal_period_s=0.05)
        assert seconds[-1] == pytest.approx(30.05)
        estimated = unwrap_cpu_cycles(cycles, frames, 200e6)  # period from the 1-frame step
        assert estimated[-1] == pytest.approx(30.05)
        modular = unwrap_cpu_cycles(cycles[1:], frames[1:], 200e6)  # no small step to learn from
        assert modular[-1] == pytest.approx((cycles[2] - cycles[1]) % CYCLE_COUNTER_MODULUS / 200e6)

    def test_period_estimate_uses_small_steps_only(self):
        cycles = np.arange(0, 20) * 12_000_000
        frames = np.arange(20)
        assert estimate_frame_period_s(cycles, frames, 600e6) == pytest.approx(0.02)
        assert estimate_frame_period_s([5], [1], 200e6) is None

    def test_input_validation(self):
        with pytest.raises(ValueError):
            unwrap_cpu_cycles([1, 2], [1], 200e6)
        with pytest.raises(ValueError):
            unwrap_cpu_cycles([1], [1], 0.0)
        assert unwrap_cpu_cycles([], [], 200e6).size == 0

    def test_synthetic_stream_across_counter_wraps(self):
        scene = SyntheticScene(
            schema="vital-signs", seed=4, first_cpu_cycles=CYCLE_COUNTER_MODULUS - 30_000_000
        )
        frames = [decode_packet(p) for p in generate_packets(scene, 900)]  # 45 s, two wraps
        seconds = frame_timestamps(frames)
        assert np.all(np.diff(seconds) > 0.04)
        assert seconds[-1] == pytest.approx(899 * 0.05, abs=1e-3)
        overridden = frame_timestamps(frames, DecoderConfig(clock_hz=100e6))
        assert overridden[-1] == pytest.approx(2 * seconds[-1], rel=1e-6)

    @pytest.mark.parametrize("nominal", [None, 0.05])
    def test_streaming_timestamps_match_the_batch_ones(self, tmp_path, nominal):
        # Frame gaps of a buffer-dump recording plus two counter wraps.
        scene = SyntheticScene(
            schema="vital-signs", seed=6, first_cpu_cycles=CYCLE_COUNTER_MODULUS - 30_000_000
        )
        stream = synthetic_capture(scene, 900, FAULT_PRESETS["block-dump"], 3)
        config = DecoderConfig(nominal_frame_period_s=nominal)
        analysis = analyze_capture(write(tmp_path / "gaps.bin", stream.data), config)
        stamp = FrameTimestamper(config)
        streamed = [stamp(frame) for frame in analysis.frames]
        np.testing.assert_allclose(streamed, analysis.timestamps_s, atol=1e-9)
        assert stamp.clock_hz == 200e6
        with pytest.raises(ValueError):
            FrameTimestamper(history=0)


def test_report_serialises(tmp_path, capture_file):
    analysis = analyze_capture(capture_file, DecoderConfig(), None)
    data = json.loads(json.dumps(analysis.report.to_dict()))
    assert data["unique_frames"] == 12 and data["tlv_types"]["1"] == 12
    text = "\n".join(analysis.report.summary_lines())
    assert "unique frames" in text and "drop ratio" in text and "link load" in text
    assert analysis.timestamps_s.shape == (12,)
    assert analysis.report.data_port_load > 1.0  # full heatmaps at 20 fps overrun a UART


def test_deduplication_can_be_switched_off(tmp_path):
    scene = SyntheticScene(schema="vital-signs", seed=8)
    stream = synthetic_capture(scene, 20, FaultPlan(duplicate_probability=1.0), 1)
    path = write(tmp_path / "doubled.bin", stream.data)
    assert analyze_capture(path).report.unique_frames == 20
    kept = analyze_capture(path, DecoderConfig(deduplicate=False))
    assert kept.report.unique_frames == 40 and kept.report.duplicates == 0
