"""Packet schemas: which TLV type means what, per firmware family.

TLV type numbers are not global. Type 1 is the detected-point list in the
out-of-box demo but a complex range segment in the Vital Signs lab; type 6 is
the statistics block of the out-of-box demo but the point cloud of the people
counting labs; and the point format itself changed between SDK 2.x (Q-format
integers) and SDK 3.x (floats). A packet can therefore only be decoded against
an explicit schema:

=======================  =================================================  ===============
Schema                   Firmware                                           TLV types
=======================  =================================================  ===============
``oob-sdk2``             out-of-box demo, SDK 1.x-2.x (xWR14xx/16xx)         1-6
``oob-sdk3``             out-of-box demo, SDK 3.x                            1-9
``vital-signs``          Vital Signs lab (xWR14xx)                           0-1
``people-counting-3d``   3D People Counting lab, 48-byte header (xWR68xx)    6-8, 11
``people-counting-2d``   2D people counting lab, 52-byte header (xWR16xx)    6-8
``people-tracking``      Radar Toolbox people tracking, 40-byte header       1010-1012,
                                                                             1020-1021
=======================  =================================================  ===============

:func:`detect_schema` picks one from the header layout and the TLV chain and
refuses to guess when nothing fits. When the firmware is known, the schema can
be forced instead (``--schema`` on the command line).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any

from .constants import (
    LONG_HEADER_LEN,
    DemoTlvType,
    PeopleCountingTlvType,
    PeopleTrackingTlvType,
    Platform,
    VitalSignsTlvType,
)
from .errors import DecodeIssue, IssueCode, SchemaError, Severity, UnsupportedFormatError
from .header import HeaderLayout

if TYPE_CHECKING:
    from .header import FrameHeader, TlvLocation
    from .payloads import DecodeContext, PayloadKind


class PacketSchema(str, Enum):
    """Firmware family whose TLV definitions a packet follows."""

    OOB_SDK2 = "oob-sdk2"
    OOB_SDK3 = "oob-sdk3"
    VITAL_SIGNS = "vital-signs"
    PEOPLE_COUNTING_3D = "people-counting-3d"
    PEOPLE_COUNTING_2D = "people-counting-2d"
    PEOPLE_TRACKING = "people-tracking"

    @property
    def description(self) -> str:
        """One-line description for reports."""
        return {
            PacketSchema.OOB_SDK2: "out-of-box demo, SDK 1.x/2.x (Q-format points, TLV 1-6)",
            PacketSchema.OOB_SDK3: "out-of-box demo, SDK 3.x (float points, TLV 1-9)",
            PacketSchema.VITAL_SIGNS: "Vital Signs lab (summary + complex range segment)",
            PacketSchema.PEOPLE_COUNTING_3D: (
                "3D People Counting lab (48-byte checksummed header, TLV 6-8, 11)"
            ),
            PacketSchema.PEOPLE_COUNTING_2D: (
                "2D people counting lab (52-byte checksummed header, TLV 6-8)"
            ),
            PacketSchema.PEOPLE_TRACKING: (
                "Radar Toolbox people tracking (40-byte header, TLV 1010-1012, 1020-1021)"
            ),
        }[self]

    @property
    def header_layout(self) -> HeaderLayout:
        """Header layout the firmware family writes."""
        if self is PacketSchema.PEOPLE_COUNTING_3D:
            return HeaderLayout.CHECKSUMMED_48
        if self is PacketSchema.PEOPLE_COUNTING_2D:
            return HeaderLayout.CHECKSUMMED_52
        return HeaderLayout.PADDED

    @property
    def tracks_targets(self) -> bool:
        """Whether the firmware family reports tracker targets."""
        return self in TRACKING_SCHEMAS

    @property
    def padding_is_uninitialised(self) -> bool:
        """Whether the firmware sends padding bytes it never sets (SDK 3.x based).

        TI mmWave SDK 03.05.00.04 sends the padding of the out-of-box demo from
        an uninitialised local array (``mss_main.c`` of the xwr68xx, xwr18xx and
        xwr16xx demos, ``main.c`` of xwr64xx), so its content carries no
        integrity information.
        """
        return self in (PacketSchema.OOB_SDK3, PacketSchema.PEOPLE_TRACKING)


#: Schemas whose firmware runs a tracker on the sensor.
TRACKING_SCHEMAS: frozenset[PacketSchema] = frozenset(
    {
        PacketSchema.PEOPLE_COUNTING_3D,
        PacketSchema.PEOPLE_COUNTING_2D,
        PacketSchema.PEOPLE_TRACKING,
    }
)

#: Schemas that use the padded (36/40-byte) header layout.
PADDED_SCHEMAS: tuple[PacketSchema, ...] = (
    PacketSchema.OOB_SDK2,
    PacketSchema.OOB_SDK3,
    PacketSchema.VITAL_SIGNS,
    PacketSchema.PEOPLE_TRACKING,
)


def parse_schema(value: str | PacketSchema | None) -> PacketSchema | None:
    """Turn a user-facing schema name into a :class:`PacketSchema`.

    Parameters
    ----------
    value : str, PacketSchema or None
        ``"auto"`` or ``None`` select automatic detection.

    Returns
    -------
    PacketSchema or None

    Raises
    ------
    SchemaError
        For an unknown name.

    Examples
    --------
    >>> parse_schema("vital-signs")
    <PacketSchema.VITAL_SIGNS: 'vital-signs'>
    >>> parse_schema("auto") is None
    True
    """
    if value is None or isinstance(value, PacketSchema):
        return value
    text = value.strip().lower().replace("_", "-")
    if text in {"", "auto"}:
        return None
    try:
        return PacketSchema(text)
    except ValueError:
        names = ", ".join(schema.value for schema in PacketSchema)
        raise SchemaError(f"unknown packet schema {value!r}; choose auto, {names}") from None


@dataclass(frozen=True)
class TlvSpec:
    """How one TLV type of one schema is decoded.

    Attributes
    ----------
    tlv_type : int
        Type number on the wire.
    kind : PayloadKind
        Key under which the decoded payload is stored in a frame.
    description : str
        Readable name for messages.
    decode : callable
        ``decode(payload, context) -> object``.
    length_ok : callable
        ``length_ok(length) -> bool``; a structural plausibility check used
        when inferring the header layout and detecting the schema.
    """

    tlv_type: int
    kind: PayloadKind
    description: str
    decode: Callable[[memoryview, DecodeContext], Any]
    length_ok: Callable[[int], bool]


def _even_nonzero(length: int) -> bool:
    return length > 0 and length % 2 == 0


def _complex_nonzero(length: int) -> bool:
    return length > 0 and length % 4 == 0


@cache
def _registries() -> dict[PacketSchema, dict[int, TlvSpec]]:
    # Imported here: the payload modules import this module for PacketSchema.
    from .angle_map import decode_azimuth_elevation_heatmap, decode_azimuth_heatmap
    from .payloads import (
        PayloadKind,
        decode_noise_profile,
        decode_points_float,
        decode_points_qformat,
        decode_range_doppler,
        decode_range_profile,
        decode_side_info,
        decode_stats,
        decode_temperature,
    )
    from .tracking import (
        decode_compressed_points,
        decode_polar_points,
        decode_presence,
        decode_target_association,
        decode_target_heights,
        decode_target_list_2d,
        decode_target_list_3d,
    )
    from .vital_signs import decode_range_segment, decode_vital_summary

    common = {
        DemoTlvType.RANGE_PROFILE: TlvSpec(
            DemoTlvType.RANGE_PROFILE,
            PayloadKind.RANGE_PROFILE,
            "range profile",
            decode_range_profile,
            _even_nonzero,
        ),
        DemoTlvType.NOISE_PROFILE: TlvSpec(
            DemoTlvType.NOISE_PROFILE,
            PayloadKind.NOISE_PROFILE,
            "noise profile",
            decode_noise_profile,
            _even_nonzero,
        ),
        DemoTlvType.AZIMUTH_HEATMAP: TlvSpec(
            DemoTlvType.AZIMUTH_HEATMAP,
            PayloadKind.AZIMUTH_HEATMAP,
            "azimuth static heatmap",
            decode_azimuth_heatmap,
            _complex_nonzero,
        ),
        DemoTlvType.RANGE_DOPPLER_HEATMAP: TlvSpec(
            DemoTlvType.RANGE_DOPPLER_HEATMAP,
            PayloadKind.RANGE_DOPPLER,
            "range-Doppler heatmap",
            decode_range_doppler,
            _even_nonzero,
        ),
        DemoTlvType.STATS: TlvSpec(
            DemoTlvType.STATS,
            PayloadKind.STATS,
            "processing statistics",
            decode_stats,
            lambda length: length == 24,
        ),
    }
    sdk2 = {
        DemoTlvType.POINTS: TlvSpec(
            DemoTlvType.POINTS,
            PayloadKind.POINTS,
            "detected points (Q-format)",
            decode_points_qformat,
            lambda length: length >= 4 and (length - 4) % 12 == 0,
        ),
        **common,
    }
    sdk3 = {
        DemoTlvType.POINTS: TlvSpec(
            DemoTlvType.POINTS,
            PayloadKind.POINTS,
            "detected points (float)",
            decode_points_float,
            lambda length: length % 16 == 0,
        ),
        **common,
        DemoTlvType.POINT_SIDE_INFO: TlvSpec(
            DemoTlvType.POINT_SIDE_INFO,
            PayloadKind.POINT_SIDE_INFO,
            "point side info",
            decode_side_info,
            lambda length: length % 4 == 0,
        ),
        DemoTlvType.AZIMUTH_ELEVATION_HEATMAP: TlvSpec(
            DemoTlvType.AZIMUTH_ELEVATION_HEATMAP,
            PayloadKind.AZIMUTH_ELEVATION_HEATMAP,
            "azimuth-elevation static heatmap",
            decode_azimuth_elevation_heatmap,
            _complex_nonzero,
        ),
        DemoTlvType.TEMPERATURE: TlvSpec(
            DemoTlvType.TEMPERATURE,
            PayloadKind.TEMPERATURE,
            "temperature report",
            decode_temperature,
            lambda length: length == 28,
        ),
    }
    vital = {
        VitalSignsTlvType.SUMMARY: TlvSpec(
            VitalSignsTlvType.SUMMARY,
            PayloadKind.VITAL_SIGNS,
            "vital-signs summary",
            decode_vital_summary,
            lambda length: length == 128,
        ),
        VitalSignsTlvType.RANGE_SEGMENT: TlvSpec(
            VitalSignsTlvType.RANGE_SEGMENT,
            PayloadKind.RANGE_SEGMENT,
            "complex range segment",
            decode_range_segment,
            _complex_nonzero,
        ),
    }

    def compressed_points(tlv_type: int) -> TlvSpec:
        return TlvSpec(
            tlv_type,
            PayloadKind.POINTS,
            "compressed spherical point cloud",
            decode_compressed_points,
            lambda length: length >= 20 and (length - 20) % 8 == 0,
        )

    def targets_3d(tlv_type: int) -> TlvSpec:
        return TlvSpec(
            tlv_type,
            PayloadKind.TARGETS,
            "3D target list",
            decode_target_list_3d,
            lambda length: length % 112 == 0 or length % 40 == 0,
        )

    def association(tlv_type: int) -> TlvSpec:
        return TlvSpec(
            tlv_type,
            PayloadKind.TARGET_ASSOCIATION,
            "per-point target IDs",
            decode_target_association,
            lambda length: True,
        )

    def presence(tlv_type: int) -> TlvSpec:
        return TlvSpec(
            tlv_type,
            PayloadKind.PRESENCE,
            "presence indication",
            decode_presence,
            lambda length: length == 4,
        )

    counting_3d = {
        PeopleCountingTlvType.POINTS: compressed_points(PeopleCountingTlvType.POINTS),
        PeopleCountingTlvType.TARGETS: targets_3d(PeopleCountingTlvType.TARGETS),
        PeopleCountingTlvType.TARGET_ASSOCIATION: association(
            PeopleCountingTlvType.TARGET_ASSOCIATION
        ),
        PeopleCountingTlvType.PRESENCE: presence(PeopleCountingTlvType.PRESENCE),
    }
    counting_2d = {
        PeopleCountingTlvType.POINTS: TlvSpec(
            PeopleCountingTlvType.POINTS,
            PayloadKind.POINTS,
            "polar point cloud",
            decode_polar_points,
            lambda length: length % 16 == 0,
        ),
        PeopleCountingTlvType.TARGETS: TlvSpec(
            PeopleCountingTlvType.TARGETS,
            PayloadKind.TARGETS,
            "2D target list",
            decode_target_list_2d,
            lambda length: length % 68 == 0,
        ),
        PeopleCountingTlvType.TARGET_ASSOCIATION: association(
            PeopleCountingTlvType.TARGET_ASSOCIATION
        ),
    }
    tracking = {
        PeopleTrackingTlvType.POINTS: compressed_points(PeopleTrackingTlvType.POINTS),
        PeopleTrackingTlvType.TARGETS: targets_3d(PeopleTrackingTlvType.TARGETS),
        PeopleTrackingTlvType.TARGET_ASSOCIATION: association(
            PeopleTrackingTlvType.TARGET_ASSOCIATION
        ),
        PeopleTrackingTlvType.TARGET_HEIGHTS: TlvSpec(
            PeopleTrackingTlvType.TARGET_HEIGHTS,
            PayloadKind.TARGET_HEIGHTS,
            "target heights",
            decode_target_heights,
            lambda length: length % 12 == 0,
        ),
        PeopleTrackingTlvType.PRESENCE: presence(PeopleTrackingTlvType.PRESENCE),
    }
    return {
        PacketSchema.OOB_SDK2: {int(key): spec for key, spec in sdk2.items()},
        PacketSchema.OOB_SDK3: {int(key): spec for key, spec in sdk3.items()},
        PacketSchema.VITAL_SIGNS: {int(key): spec for key, spec in vital.items()},
        PacketSchema.PEOPLE_COUNTING_3D: {int(key): spec for key, spec in counting_3d.items()},
        PacketSchema.PEOPLE_COUNTING_2D: {int(key): spec for key, spec in counting_2d.items()},
        PacketSchema.PEOPLE_TRACKING: {int(key): spec for key, spec in tracking.items()},
    }


def schema_registry(schema: PacketSchema) -> Mapping[int, TlvSpec]:
    """Return the TLV definitions of one schema, keyed by type number.

    Examples
    --------
    >>> sorted(schema_registry(PacketSchema.OOB_SDK2))
    [1, 2, 3, 4, 5, 6]
    >>> schema_registry(PacketSchema.VITAL_SIGNS)[0].description
    'vital-signs summary'
    >>> sorted(schema_registry(PacketSchema.PEOPLE_TRACKING))
    [1010, 1011, 1012, 1020, 1021]
    """
    return _registries()[PacketSchema(schema)]


def chain_matches_schema(schema: PacketSchema, tlvs: Sequence[TlvLocation]) -> bool:
    """Whether every TLV of a chain is a known type with a plausible length.

    Parameters
    ----------
    schema : PacketSchema
        Schema to test against.
    tlvs : sequence of TlvLocation
        Chain found under some header layout.
    """
    registry = schema_registry(schema)
    for location in tlvs:
        spec = registry.get(location.tlv_type)
        if spec is None or not spec.length_ok(location.length):
            return False
    return True


def explained_tlvs(schema: PacketSchema, tlvs: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """How well a schema explains a chain of ``(type, length)`` pairs.

    Returns
    -------
    tuple of int
        TLVs whose type the schema defines, and of those the ones whose
        length is plausible too.
    """
    registry = schema_registry(schema)
    known = [(tlv_type, length) for tlv_type, length in tlvs if tlv_type in registry]
    plausible = sum(1 for tlv_type, length in known if registry[tlv_type].length_ok(length))
    return len(known), plausible


def detect_schema(header: FrameHeader, tlvs: Sequence[tuple[int, int]]) -> PacketSchema:
    """Pick the schema of a packet from its header and its TLV types and lengths.

    The SDK version word alone cannot do this: the 3D People Counting lab and
    the Radar Toolbox people tracking demos write the SDK 3.x version just like
    the SDK 3.x out-of-box demo. The rules, in order:

    1. A 48-byte checksummed header is the 3D People Counting lab, a 52-byte
       one the 2D people counting lab (the header layouts are unique).
    2. A 40-byte header from an xWR14xx with a 128-byte type-0 TLV is the
       Vital Signs lab (its summary structure is exactly 128 bytes, and the
       out-of-box demo never uses type 0).
    3. Otherwise the candidates are the out-of-box demo of the header's SDK
       generation and, for SDK 3.x, the people tracking demos. The candidate
       that defines the most TLV types of the packet wins, then the one with
       the most plausible lengths; remaining ties go to the out-of-box demo.
       A packet without TLVs is the out-of-box demo.
    4. A packet with TLVs none of whose types any candidate defines is not
       guessed at: :class:`~radar_stream_decoder.errors.UnsupportedFormatError`.

    Parameters
    ----------
    header : FrameHeader
        Decoded header.
    tlvs : sequence of (int, int)
        ``(type, payload length)`` of every TLV in the packet.

    Returns
    -------
    PacketSchema

    Raises
    ------
    UnsupportedFormatError
        If no known firmware family explains the packet.
    """
    if header.layout is HeaderLayout.CHECKSUMMED_48:
        return PacketSchema.PEOPLE_COUNTING_3D
    if header.layout is HeaderLayout.CHECKSUMMED_52:
        return PacketSchema.PEOPLE_COUNTING_2D
    lengths = dict(reversed(tlvs))
    if (
        lengths.get(int(VitalSignsTlvType.SUMMARY)) == 128
        and header.header_len == LONG_HEADER_LEN
        and header.platform == Platform.XWR14XX
    ):
        return PacketSchema.VITAL_SIGNS
    if header.version.major >= 3:
        candidates = (PacketSchema.OOB_SDK3, PacketSchema.PEOPLE_TRACKING)
    else:
        candidates = (PacketSchema.OOB_SDK2,)
    if not tlvs:
        return candidates[0]
    scores = [explained_tlvs(schema, tlvs) for schema in candidates]
    best = max(range(len(candidates)), key=lambda index: (*scores[index], -index))
    if scores[best][0] == 0:
        types = ", ".join(str(tlv_type) for tlv_type, _ in tlvs)
        names = " or ".join(schema.value for schema in candidates)
        message = (
            f"unsupported format: a {header.header_len}-byte header of SDK "
            f"{header.version} with TLV types {types}, none of which {names} defines; "
            "force a schema with --schema if the firmware is known"
        )
        issue = DecodeIssue(
            IssueCode.UNSUPPORTED_FORMAT, message, Severity.ERROR, header.frame_number
        )
        raise UnsupportedFormatError(message, (issue,))
    return candidates[best]
