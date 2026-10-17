"""Incremental assembly: chunking invariance, resynchronisation, damaged packets, end of stream."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from radar_stream_decoder.assembler import FrameAssembler
from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.constants import MAGIC_WORD
from radar_stream_decoder.encoder import HeaderTemplate, encode_frame, encode_magnitude_profile
from radar_stream_decoder.errors import IssueCode
from radar_stream_decoder.synthetic import (
    FAULT_PRESETS,
    FaultPlan,
    PacketFate,
    SyntheticScene,
    content_end,
    generate_packets,
    inject_faults,
    split_into_chunks,
)


def feed_all(data: bytes, chunks: list[bytes] | None = None, config: DecoderConfig | None = None):
    assembler = FrameAssembler(config)
    frames = []
    for chunk in chunks if chunks is not None else [data]:
        frames.extend(assembler.feed(chunk))
    frames.extend(assembler.finish())
    return frames, assembler.stats


def signature(frames):
    return [(f.frame_number, f.crc32, f.truncated, len(f.issues)) for f in frames]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_any_chunking_gives_the_same_frames(seed, packets_vital):
    faulted = inject_faults(packets_vital, FAULT_PRESETS["heavy"], seed)
    reference, _ = feed_all(faulted.data)
    random_chunks, _ = feed_all(faulted.data, split_into_chunks(faulted.data, seed, max_chunk=97))
    single_bytes, _ = feed_all(faulted.data, [bytes([b]) for b in faulted.data])
    assert signature(reference) == signature(random_chunks) == signature(single_bytes)
    assert reference


def test_large_packets_in_random_chunks(packets_16xx):
    data = b"".join(packets_16xx)
    frames, stats = feed_all(data, split_into_chunks(data, 4, max_chunk=1500))
    assert [f.crc32 for f in frames] == [zlib.crc32(p) for p in packets_16xx]
    assert stats.bytes_skipped == 0 and stats.resyncs == 0
    assert stats.bytes_in == len(data)


def test_leading_junk_is_skipped_without_counting_a_resync(packets_16xx):
    frames, stats = feed_all(b"\x07" * 333 + b"".join(packets_16xx[:3]))
    assert [f.frame_number for f in frames] == [1, 2, 3]
    assert stats.bytes_skipped == 333 and stats.resyncs == 0


def test_junk_between_packets_counts_resyncs(packets_vital):
    data = packets_vital[0] + b"garbage" + packets_vital[1] + b"more garbage" + packets_vital[2]
    frames, stats = feed_all(data)
    assert len(frames) == 3 and stats.resyncs == 2
    assert stats.bytes_skipped == len(b"garbage") + len(b"more garbage")
    assert sum(issue.code is IssueCode.SKIPPED_BYTES for issue in stats.issues) == 2


def test_bare_sync_words_are_rejected(packets_vital):
    data = MAGIC_WORD + b"\xff" * 40 + packets_vital[0] + MAGIC_WORD + packets_vital[1]
    frames, stats = feed_all(data)
    assert [f.crc32 for f in frames] == [zlib.crc32(p) for p in packets_vital[:2]]
    assert stats.frames_rejected == 2


def test_forged_header_with_garbage_chain_is_rejected(packets_vital):
    plan = FaultPlan(junk_probability=1.0, fake_sync_probability=1.0, junk_max_bytes=200)
    faulted = inject_faults(packets_vital[:10], plan, 3)
    assert faulted.fake_sync_words == 10
    frames, stats = feed_all(faulted.data)
    assert [f.crc32 for f in frames] == [zlib.crc32(p) for p in packets_vital[:10]]
    assert stats.frames_rejected >= 10


@pytest.mark.parametrize("strict", [False, True])
def test_cut_packet_does_not_take_the_next_one_down(strict, packets_16xx):
    victim = packets_16xx[1]
    cut = content_end(victim) - 600  # inside the heatmap; the TLVs before it survive
    data = packets_16xx[0] + victim[:cut] + packets_16xx[2]
    frames, stats = feed_all(data, config=DecoderConfig(strict=strict))
    crcs = [f.crc32 for f in frames]
    assert zlib.crc32(packets_16xx[0]) in crcs and zlib.crc32(packets_16xx[2]) in crcs
    salvaged = [f for f in frames if f.truncated]
    if strict:
        assert not salvaged and stats.frames_rejected == 1
    else:
        assert len(salvaged) == 1 and stats.frames_salvaged == 1
        frame = salvaged[0]
        assert frame.frame_number == 2 and frame.has_errors
        assert IssueCode.TRUNCATED_PACKET in [issue.code for issue in frame.issues]
        assert frame.points is not None  # TLVs before the cut were decoded


def test_padding_lost_on_the_link_is_still_an_error(packets_vital):
    # Ten bytes missing: the chain still fits, but the loss could have been anywhere.
    data = packets_vital[0][:-10] + packets_vital[1]
    frames, _ = feed_all(data)
    first = frames[0]
    assert first.truncated and first.has_errors
    assert IssueCode.SHORTENED_PACKET in [issue.code for issue in first.issues]
    assert frames[1].crc32 == zlib.crc32(packets_vital[1])


@pytest.mark.parametrize("lost", range(1, 8))
def test_next_sync_word_straddling_the_declared_end(lost, packets_vital):
    # The shortened packet's declared span ends inside the next packet's sync word.
    # Fed one byte at a time, the assembler must see that sync word exactly as it
    # does with the whole buffer, instead of swallowing its first bytes as padding.
    data = packets_vital[0] + packets_vital[1][:-lost] + packets_vital[2]
    whole, whole_stats = feed_all(data)
    single, single_stats = feed_all(data, [bytes([b]) for b in data])
    assert signature(single) == signature(whole)
    assert single_stats.as_dict() == whole_stats.as_dict()
    assert [f.frame_number for f in whole] == [1, 2, 3]
    assert whole[1].truncated and whole[2].crc32 == zlib.crc32(packets_vital[2])


def test_complete_packet_ending_like_a_sync_word_waits_for_more_bytes():
    # 0x02 padding matches the first sync-word byte, so whether the next packet
    # starts inside the span is only known once more bytes (or the end) arrive.
    packet = encode_frame(HeaderTemplate(frame_number=4), [(6, bytes(24))], padding_byte=0x02)
    assembler = FrameAssembler()
    assert assembler.feed(packet) == []
    assert [f.frame_number for f in assembler.feed(packet)] == [4]
    assert [f.frame_number for f in assembler.finish()] == [4]
    assert assembler.stats.bytes_skipped == 0


def test_junk_in_the_padding_is_detected(packets_vital):
    packet = bytearray(packets_vital[0])
    packet[-5:] = b"\x13\x37\x00\x42\x99"
    frames, _ = feed_all(bytes(packet))
    assert IssueCode.CORRUPT_PADDING in [issue.code for issue in frames[0].issues]


def test_0x0f_padding_is_accepted():
    scene = SyntheticScene(schema="vital-signs", seed=1, padding_byte=0x0F)
    packets = list(generate_packets(scene, 4))
    frames, _ = feed_all(b"".join(packets), config=DecoderConfig(strict=True))
    assert len(frames) == 4 and not any(f.issues for f in frames)


def test_sync_word_inside_a_payload_does_not_cut_the_packet():
    # A range profile whose values spell the sync word, followed by junk that is no header.
    values = np.frombuffer(MAGIC_WORD + bytes(range(40)), dtype="<u2")
    packet = encode_frame(HeaderTemplate(frame_number=8), [(2, encode_magnitude_profile(values))])
    frames, stats = feed_all(packet + packet)
    assert [f.frame_number for f in frames] == [8, 8]
    assert not any(f.issues for f in frames) and stats.frames_rejected == 0


@pytest.mark.parametrize("strict", [False, True])
def test_unfinished_final_packet(strict, packets_16xx):
    tail = packets_16xx[1][: len(packets_16xx[1]) - 2000]
    frames, stats = feed_all(packets_16xx[0] + tail, config=DecoderConfig(strict=strict))
    assert stats.truncated_tail_bytes == len(tail)
    assert IssueCode.TRUNCATED_TAIL in [issue.code for issue in stats.issues]
    if strict:
        assert len(frames) == 1
    else:
        assert len(frames) == 2 and frames[1].truncated and stats.frames_salvaged == 1


def test_stream_ending_inside_a_header_is_a_truncated_tail(packets_vital):
    frames, stats = feed_all(packets_vital[0] + packets_vital[1][:20])
    assert len(frames) == 1
    assert stats.truncated_tail_bytes == 20 and stats.bytes_skipped == 0
    assert stats.issues[-1].code is IssueCode.TRUNCATED_TAIL


def test_trailing_bytes_without_sync_word_are_skipped(packets_vital):
    frames, stats = feed_all(packets_vital[0] + b"\x01\x02\x03")
    assert len(frames) == 1 and stats.bytes_skipped == 3 and stats.truncated_tail_bytes == 0


def test_oversized_length_claim_is_rejected(packets_vital):
    forged = bytearray(packets_vital[0])
    struct.pack_into("<I", forged, 12, 10_000_000)
    frames, stats = feed_all(bytes(forged) + packets_vital[1])
    assert [f.crc32 for f in frames] == [zlib.crc32(packets_vital[1])]
    assert stats.frames_rejected >= 1


def test_feed_after_finish_and_reset(packets_vital):
    assembler = FrameAssembler()
    assembler.feed(packets_vital[0])
    assert assembler.finish() == []
    with pytest.raises(RuntimeError):
        assembler.feed(packets_vital[1])
    assembler.reset()
    assert len(assembler.feed(packets_vital[1])) == 1
    stats = assembler.flush()
    assert stats.frames_decoded == 1 and assembler.buffered_bytes == 0


def test_recorded_issues_are_bounded(packets_vital):
    junky = b"".join(p + b"xyz" for p in packets_vital)
    assembler = FrameAssembler(max_recorded_issues=5)
    assembler.feed(junky)
    stats = assembler.flush()
    assert len(stats.issues) == 5 and stats.issue_count == len(packets_vital)


@pytest.mark.parametrize("preset", ["light", "heavy", "block-dump"])
def test_every_intact_packet_is_recovered_and_nothing_damaged_passes(preset, packets_vital):
    faulted = inject_faults(packets_vital, FAULT_PRESETS[preset], 11)
    frames, _ = feed_all(faulted.data, split_into_chunks(faulted.data, 2, max_chunk=300))
    intact = {
        zlib.crc32(p)
        for p, fate in zip(packets_vital, faulted.fates, strict=True)
        if fate is PacketFate.DELIVERED
    }
    clean = [f for f in frames if not f.has_errors and not f.truncated]
    assert {f.crc32 for f in clean} == intact
