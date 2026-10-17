"""Schema names, registries and automatic detection."""

from __future__ import annotations

import pytest

from radar_stream_decoder.constants import Platform
from radar_stream_decoder.errors import IssueCode, SchemaError, UnsupportedFormatError
from radar_stream_decoder.frame import decode_packet
from radar_stream_decoder.header import FrameHeader, SdkVersion, TlvLocation
from radar_stream_decoder.schema import (
    PacketSchema,
    chain_matches_schema,
    detect_schema,
    parse_schema,
    schema_registry,
)
from radar_stream_decoder.synthetic import generate_packets


def _header(version=(2, 0), platform=Platform.XWR16XX, header_len=40):
    return FrameHeader(SdkVersion(*version), 288, platform, 1, 0, 0, 2, 0, header_len)


def test_parse_schema_accepts_names_and_auto():
    assert parse_schema("OOB_SDK3") is PacketSchema.OOB_SDK3
    assert parse_schema(" vital-signs ") is PacketSchema.VITAL_SIGNS
    assert parse_schema(PacketSchema.OOB_SDK2) is PacketSchema.OOB_SDK2
    assert parse_schema("auto") is None and parse_schema(None) is None


def test_parse_schema_rejects_unknown_names():
    with pytest.raises(SchemaError, match="oob-sdk2"):
        parse_schema("sdk4")


def test_registries_cover_the_documented_types():
    assert sorted(schema_registry(PacketSchema.OOB_SDK2)) == [1, 2, 3, 4, 5, 6]
    assert sorted(schema_registry(PacketSchema.OOB_SDK3)) == list(range(1, 10))
    assert sorted(schema_registry(PacketSchema.VITAL_SIGNS)) == [0, 1]
    assert sorted(schema_registry(PacketSchema.PEOPLE_COUNTING_3D)) == [6, 7, 8, 11]
    assert sorted(schema_registry(PacketSchema.PEOPLE_COUNTING_2D)) == [6, 7, 8]
    assert sorted(schema_registry(PacketSchema.PEOPLE_TRACKING)) == [1010, 1011, 1012, 1020, 1021]
    # Type 6 is the statistics block of the out-of-box demo but a point cloud here.
    assert schema_registry(PacketSchema.OOB_SDK2)[6].kind != (
        schema_registry(PacketSchema.PEOPLE_COUNTING_3D)[6].kind
    )
    # Type 1 means different things in different firmware.
    assert schema_registry(PacketSchema.OOB_SDK2)[1].description != (
        schema_registry(PacketSchema.VITAL_SIGNS)[1].description
    )


def test_chain_plausibility():
    points = TlvLocation(1, 4 + 12 * 3, 48)
    assert chain_matches_schema(PacketSchema.OOB_SDK2, [points])
    assert not chain_matches_schema(PacketSchema.OOB_SDK2, [TlvLocation(1, 41, 48)])
    assert not chain_matches_schema(PacketSchema.OOB_SDK2, [TlvLocation(0, 128, 48)])
    assert chain_matches_schema(PacketSchema.VITAL_SIGNS, [TlvLocation(0, 128, 48)])
    assert chain_matches_schema(PacketSchema.OOB_SDK3, [TlvLocation(9, 28, 48)])


@pytest.mark.parametrize(
    ("header", "tlvs", "expected"),
    [
        (_header(version=(3, 5)), [(1, 32)], PacketSchema.OOB_SDK3),
        (_header((1, 2), Platform.XWR14XX), [(0, 128), (1, 88)], PacketSchema.VITAL_SIGNS),
        (_header(), [(1, 16), (2, 512)], PacketSchema.OOB_SDK2),
        (_header(), [(1, 17)], PacketSchema.OOB_SDK2),  # known type, broken length
        (_header(), [], PacketSchema.OOB_SDK2),
        (_header(version=(3, 5)), [], PacketSchema.OOB_SDK3),
        (_header(version=(3, 5)), [(1020, 28), (1010, 112)], PacketSchema.PEOPLE_TRACKING),
        # An unknown type next to known ones does not change the family.
        (_header(version=(3, 5)), [(1, 32), (42, 4)], PacketSchema.OOB_SDK3),
    ],
)
def test_detect_schema(header, tlvs, expected):
    assert detect_schema(header, tlvs) is expected


@pytest.mark.parametrize(
    ("header", "tlvs"),
    [
        # A 36-byte header or another platform rules out the Vital Signs lab, and
        # the out-of-box demo has no type 0: nothing explains the packet.
        (_header((1, 2), Platform.XWR14XX, 36), [(0, 128)]),
        (_header((1, 2), Platform.XWR16XX), [(0, 128)]),
        # SDK 3.x with types neither the out-of-box demo nor people tracking define.
        (_header(version=(3, 5)), [(301, 64), (306, 40)]),
    ],
)
def test_unexplained_packets_are_not_forced_into_a_schema(header, tlvs):
    with pytest.raises(UnsupportedFormatError, match="unsupported format") as caught:
        detect_schema(header, tlvs)
    assert [issue.code for issue in caught.value.issues] == [IssueCode.UNSUPPORTED_FORMAT]


def test_every_synthetic_family_is_detected(scene_16xx, scene_14xx, scene_sdk3, scene_vital):
    for scene in (scene_16xx, scene_14xx, scene_sdk3, scene_vital):
        frame = decode_packet(next(generate_packets(scene, 1)))
        assert frame.schema is scene.schema
        assert not frame.issues


def test_forced_schema_is_used_even_when_detection_would_differ(scene_vital):
    packet = next(generate_packets(scene_vital, 1))
    from radar_stream_decoder.config import DecoderConfig

    frame = decode_packet(packet, DecoderConfig(schema="oob-sdk2"))
    assert frame.schema is PacketSchema.OOB_SDK2
    # Under the wrong schema type 0 is unknown and type 1 is not a point list.
    assert frame.unknown_tlvs and frame.has_errors
