"""Byte sources and the frame stream that the CLI and the viewer share."""

from __future__ import annotations

import zlib

import pytest

from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.sources import (
    ByteSource,
    FileByteSource,
    FrameStream,
    RecordingByteSource,
    SerialByteSource,
    SyntheticByteSource,
    iter_frames,
)
from radar_stream_decoder.synthetic import FAULT_PRESETS, FaultPlan, SyntheticScene


def test_file_source(capture_file):
    source = FileByteSource(capture_file, chunk_size=1000)
    assert b"".join(source.chunks()) == capture_file.read_bytes()
    assert source.size_bytes == capture_file.stat().st_size
    assert "capture.bin" in source.description
    with pytest.raises(FileNotFoundError):
        FileByteSource(capture_file.with_name("missing.bin"))
    with pytest.raises(ValueError):
        FileByteSource(capture_file, chunk_size=0)


def test_sources_satisfy_the_protocol(capture_file):
    assert isinstance(FileByteSource(capture_file), ByteSource)
    assert isinstance(SyntheticByteSource(n_frames=1), ByteSource)
    assert isinstance(SerialByteSource("COM9"), ByteSource)


def test_synthetic_source_is_deterministic():
    scene = SyntheticScene(schema="vital-signs", seed=3)
    first = b"".join(SyntheticByteSource(scene, 20, faults=FAULT_PRESETS["light"]).chunks())
    again = b"".join(SyntheticByteSource(scene, 20, faults=FAULT_PRESETS["light"]).chunks())
    assert first == again
    clean = b"".join(SyntheticByteSource(scene, 20).chunks())
    assert len(clean) == 20 * 288


def test_synthetic_source_stops_when_closed():
    source = SyntheticByteSource(SyntheticScene(schema="vital-signs"), None, batch_frames=4)
    taken = 0
    for chunk in source.chunks():
        taken += len(chunk)
        if taken > 10_000:
            source.close()
    assert taken < 20_000


def test_recording_source(tmp_path, capture_file):
    recording = tmp_path / "copy.bin"
    source = RecordingByteSource(FileByteSource(capture_file, 333), recording)
    frames = list(FrameStream(source))
    assert len(frames) == 12
    assert recording.read_bytes() == capture_file.read_bytes()
    assert "copy.bin" in source.description


def test_frame_stream_counts_and_deduplicates():
    scene = SyntheticScene(schema="vital-signs", seed=5)
    doubled = SyntheticByteSource(scene, 15, faults=FaultPlan(duplicate_probability=1.0))
    stream = FrameStream(doubled)
    frames = list(stream)
    assert len(frames) == 15 and stream.frames_seen == 30
    assert stream.dedupe_stats.duplicates == 15
    assert stream.assembler_stats.frames_decoded == 30
    raw = list(
        iter_frames(
            SyntheticByteSource(scene, 15, faults=FaultPlan(duplicate_probability=1.0)),
            DecoderConfig(deduplicate=False),
        )
    )
    assert len(raw) == 30


def test_frame_stream_matches_the_packets(scene_16xx, packets_16xx):
    frames = list(
        iter_frames(SyntheticByteSource(scene_16xx, 12, max_chunk=777), sensor=scene_16xx.radar())
    )
    assert [f.crc32 for f in frames] == [zlib.crc32(p) for p in packets_16xx]
    assert all(f.points is not None for f in frames)
