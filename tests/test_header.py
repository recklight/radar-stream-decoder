"""Header decoding, plausibility checks and the 36/40-byte layout decision."""

from __future__ import annotations

import struct

import pytest

from radar_stream_decoder.constants import MAGIC_WORD, Platform
from radar_stream_decoder.encoder import HeaderTemplate, encode_frame, encode_stats
from radar_stream_decoder.errors import DecodeError, IssueCode
from radar_stream_decoder.frame import decode_packet, resolve_header_len
from radar_stream_decoder.header import (
    SdkVersion,
    decode_frame_header,
    header_len_rule,
    header_problem,
    infer_header_len,
    read_header_fields,
    walk_tlv_chain,
)
from radar_stream_decoder.payloads import ProcessingStats
from radar_stream_decoder.schema import PacketSchema, chain_matches_schema

STATS = encode_stats(ProcessingStats(1, 2, 3, 4, 5, 6))


def _header(version=0x02000004, length=96, platform=0xA1642, frame=1, cycles=0, points=0, tlvs=1):
    return MAGIC_WORD + struct.pack("<7I", version, length, platform, frame, cycles, points, tlvs)


class TestSdkVersion:
    def test_word_round_trip(self):
        for word in (0x01000005, 0x01020005, 0x02000004, 0x03050004, 0xFFFFFFFF):
            assert SdkVersion.from_word(word).to_word() == word

    def test_ordering_and_text(self):
        assert SdkVersion(1, 1) > SdkVersion(1, 0, 9, 9)
        assert SdkVersion.parse("1.2.0.5") == SdkVersion(1, 2, 0, 5)
        assert str(SdkVersion(3, 5, 0, 4)) == "3.5.0.4"
        assert SdkVersion(1, 2).major_minor == (1, 2)

    @pytest.mark.parametrize("text", ["1", "1.2.3.4.5", "a.b", ""])
    def test_parse_rejects_malformed_text(self, text):
        with pytest.raises(ValueError):
            SdkVersion.parse(text)

    def test_components_must_fit_a_byte(self):
        with pytest.raises(ValueError):
            SdkVersion(256, 0)


class TestHeaderFields:
    def test_needs_36_bytes(self):
        with pytest.raises(DecodeError, match="36 bytes"):
            read_header_fields(_header()[:30])

    def test_needs_the_sync_word(self):
        with pytest.raises(DecodeError, match="sync word"):
            read_header_fields(b"\x00" * 8 + _header()[8:])

    def test_fields_are_read_little_endian(self):
        fields = read_header_fields(_header(frame=0x01020304, cycles=7, tlvs=3, length=160))
        assert fields.frame_number == 0x01020304
        assert fields.cpu_cycles == 7
        assert fields.tlv_count == 3
        assert fields.version == SdkVersion(2, 0, 0, 4)

    def test_decode_both_layouts(self):
        raw = _header(tlvs=0, length=64) + struct.pack("<I", 9) + bytes(24)
        short = decode_frame_header(raw, header_len=36)
        long = decode_frame_header(raw, header_len=40)
        assert short.subframe_number is None and not short.has_subframe
        assert long.subframe_number == 9 and long.header_len == 40
        assert long.platform_name == "xWR16xx"
        with pytest.raises(ValueError):
            decode_frame_header(raw, header_len=38)


LIMITS = {"max_packet_bytes": 4096, "max_tlvs": 32}


class TestHeaderProblem:
    def check(self, raw, strict=False):
        return header_problem(read_header_fields(raw), strict=strict, **LIMITS)

    def test_plausible_header(self):
        assert self.check(_header()) is None

    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"version": 0x09000000}, "major version 9"),
            ({"platform": 0x12345}, "not a TI mmWave device"),
            ({"tlvs": 40}, "40 TLVs"),
            ({"length": 8192}, "exceeds"),
            ({"length": 40, "tlvs": 2}, "cannot hold"),
        ],
    )
    def test_implausible_headers(self, kwargs, fragment):
        assert fragment in (self.check(_header(**kwargs)) or "")

    def test_alignment_only_in_strict_mode(self):
        raw = _header(length=100)
        assert self.check(raw) is None
        assert "multiple of 32" in self.check(raw, strict=True)


class TestHeaderRules:
    @pytest.mark.parametrize(
        ("platform", "version", "schema", "expected"),
        [
            (Platform.XWR14XX, SdkVersion(1, 2, 0, 5), "vital-signs", 40),
            (Platform.XWR16XX, SdkVersion(3, 5), None, 40),
            (Platform.XWR14XX, SdkVersion(1, 2), "oob-sdk3", 40),
            (Platform.XWR16XX, SdkVersion(1, 1), None, 40),
            (Platform.XWR16XX, SdkVersion(2, 0), None, 40),
            (Platform.XWR16XX, SdkVersion(1, 0), None, 36),
            (Platform.XWR14XX, SdkVersion(1, 2), "oob-sdk2", 36),
            (Platform.XWR14XX, SdkVersion(1, 2), None, None),
            (Platform.XWR18XX, SdkVersion(2, 1), None, None),
        ],
    )
    def test_rule_table(self, platform, version, schema, expected):
        rule = header_len_rule(platform, version, schema)
        assert (rule.header_len if rule else None) == expected

    def test_inference_tells_the_layouts_apart(self, scene_14xx, scene_vital):
        from radar_stream_decoder.synthetic import generate_packets

        oob = next(generate_packets(scene_14xx, 1))
        vital = next(generate_packets(scene_vital, 1))

        def any_schema(locations):
            return any(chain_matches_schema(s, locations) for s in PacketSchema)

        assert infer_header_len(oob, 0, read_header_fields(oob), any_schema) == (36,)
        assert infer_header_len(vital, 0, read_header_fields(vital), any_schema) == (40,)
        assert resolve_header_len(oob, read_header_fields(oob), None)[0] == 36
        assert resolve_header_len(vital, read_header_fields(vital), None)[0] == 40

    def test_data_overrules_a_rule_that_does_not_fit(self):
        # An xWR16xx SDK 2.0 header must have the sub-frame word -- this one does not.
        packet = encode_frame(
            HeaderTemplate(platform=Platform.XWR16XX, frame_number=4, subframe_number=None),
            [(6, STATS)],
        )
        frame = decode_packet(packet)
        assert frame.header.header_len == 36
        assert frame.stats is not None and frame.stats.inter_frame_cpu_load_pct == 6
        assert [issue.code for issue in frame.issues] == [IssueCode.HEADER_LENGTH_CONFLICT]


class TestTlvChain:
    def test_complete_chain(self):
        packet = encode_frame(HeaderTemplate(frame_number=1), [(6, STATS), (2, bytes(8))])
        walk = walk_tlv_chain(packet, 0, 40, 2, len(packet))
        assert walk.complete and walk.problem is None
        assert [loc.tlv_type for loc in walk.locations] == [6, 2]
        assert walk.locations[0].payload_offset == 48 and walk.locations[0].end == 72

    def test_overflow_is_reported(self):
        packet = encode_frame(HeaderTemplate(frame_number=1), [(6, STATS)])
        broken = bytearray(packet)
        struct.pack_into("<I", broken, 44, 4000)  # TLV length far beyond the packet
        walk = walk_tlv_chain(bytes(broken), 0, 40, 1, len(packet))
        assert not walk.complete and not walk.truncated
        assert "overflows" in walk.problem

    def test_truncation_is_distinguished_from_overflow(self):
        packet = encode_frame(HeaderTemplate(frame_number=1), [(6, STATS)])
        walk = walk_tlv_chain(packet, 0, 40, 1, len(packet), available=60)
        assert walk.truncated and not walk.complete
