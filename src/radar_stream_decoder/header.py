"""Frame headers: the three layouts TI firmware writes, and how to tell them apart.

Every packet starts with the sync word ``02 01 04 03 06 05 08 07``. What follows
depends on the firmware family (all fields little-endian):

**Padded layout**, written by the out-of-box demo (SDK 1.x-3.x), the Vital
Signs lab and Radar Toolbox people tracking. TLV lengths exclude the 8-byte
type/length pair and the packet is padded to a multiple of 32 bytes::

    offset  size  field
         0     8  sync word
         8     4  version          major<<24 | minor<<16 | bugfix<<8 | build
        12     4  packet length    header + TLVs + padding, a multiple of 32
        16     4  platform         0xA1443, 0xA1642, ...
        20     4  frame number
        24     4  CPU cycles       free-running 32-bit timestamp
        28     4  point count      as declared by the firmware (not always true)
        32     4  TLV count
       (36     4  sub-frame number, only in the 40-byte variant)

**Checksummed 48-byte layout** of the 3D People Counting lab (TI mmWave
Industrial Toolbox 4.5-4.12, ``Pcount3DDemo_output_message_header``)::

         8     4  version          the SDK 3.x version word
        12     4  packet length    header + TLVs, not padded
        16     4  platform         0xA6843
        20     4  frame number
        24     4  sub-frame number
        28     4  chirp processing margin (us)
        32     4  frame processing time (us)
        36     4  tracking time (us)
        40     4  UART transmit time (us)
        44     2  TLV count        uint16
        46     2  header checksum  uint16

**Checksummed 52-byte layout** of the 2D people counting lab (TI mmWave
Industrial Toolbox 2.x, the lab's ``MmwDemo_output_message_header``)::

         8     4  version
        12     4  platform         0xA1642
        16     4  CPU cycles       R4F timestamp
        20     4  packet length    header + TLVs, not padded
        24     4  frame number
        28     4  sub-frame number
        32     4  chirp processing margin (us)
        36     4  frame processing margin (us)
        40     4  tracking time (us)
        44     4  UART transmit time (us)
        48     2  TLV count        uint16
        50     2  header checksum  uint16

In both checksummed layouts every TLV length includes its 8-byte type/length
pair and the firmware computes the checksum as
``~((sum >> 16) + (sum & 0xFFFF))`` over the header's ``uint16`` words with the
checksum field set to zero (``mss_main.c`` of the 3D lab, ``task_mbox.c`` of the
2D lab).

The SDK version word cannot tell the layouts apart, because the 3D People
Counting lab writes the SDK 3.5 version just like the SDK 3.x out-of-box demo.
:func:`examine_header` therefore goes by the header's shape. The word at
offset 12 is a platform code (``0xAxxxx``) only in the 52-byte layout, where it
is followed by a checksum that must match; otherwise the padded layout is
tried, then the 48-byte one with its checksum. A header that fits no layout is
reported, never forced into one.

Whether the padded header carries the sub-frame word depends on the platform,
the SDK release and the firmware family, and the header itself does not say.
The decoder first consults
:data:`~radar_stream_decoder.constants.HEADER_RULES`; when no rule applies, or
a rule disagrees with the data, it walks the TLV chain under both variants and
keeps the one whose chain fits the declared packet length.
"""

from __future__ import annotations

import functools
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from .constants import (
    CHECKSUMMED_HEADER_LEN_2D,
    CHECKSUMMED_HEADER_LEN_3D,
    DEFAULT_MAX_PACKET_BYTES,
    HEADER_LENGTHS,
    HEADER_RULES,
    MAGIC_WORD,
    MAX_TLVS_PER_PACKET,
    PACKET_ALIGNMENT,
    PLATFORM_FAMILY_TAG,
    SHORT_HEADER_LEN,
    SUPPORTED_MAJOR_VERSIONS,
    TLV_HEADER_LEN,
    HeaderRule,
    platform_label,
)
from .errors import DecodeError, IssueCode

_HEADER_PREFIX = struct.Struct("<8s7I")
_HEADER_3D = struct.Struct("<8s9I2H")
_HEADER_2D = struct.Struct("<8s10I2H")
_UINT32 = struct.Struct("<I")
_UINT16 = struct.Struct("<H")
_TLV_HEADER = struct.Struct("<2I")

assert _HEADER_PREFIX.size == SHORT_HEADER_LEN
assert _HEADER_3D.size == CHECKSUMMED_HEADER_LEN_3D
assert _HEADER_2D.size == CHECKSUMMED_HEADER_LEN_2D
assert _TLV_HEADER.size == TLV_HEADER_LEN


class HeaderLayout(str, Enum):
    """Byte layout of a frame header (see the module documentation)."""

    PADDED = "padded"
    CHECKSUMMED_48 = "checksummed-48"
    CHECKSUMMED_52 = "checksummed-52"

    @property
    def checksummed(self) -> bool:
        """Whether the header carries a checksum (and TLV lengths include their pair)."""
        return self is not HeaderLayout.PADDED

    @property
    def min_length(self) -> int:
        """Bytes needed to read the layout's fixed fields."""
        return _LAYOUT_LENGTHS[self]


_LAYOUT_LENGTHS: dict[HeaderLayout, int] = {
    HeaderLayout.PADDED: SHORT_HEADER_LEN,
    HeaderLayout.CHECKSUMMED_48: CHECKSUMMED_HEADER_LEN_3D,
    HeaderLayout.CHECKSUMMED_52: CHECKSUMMED_HEADER_LEN_2D,
}

#: Plausibility limits of :func:`examine_header` when no decoder settings apply.
DEFAULT_LIMITS: dict[str, int] = {
    "max_packet_bytes": DEFAULT_MAX_PACKET_BYTES,
    "max_tlvs": MAX_TLVS_PER_PACKET,
}

#: Header length of each checksummed layout.
CHECKSUMMED_LAYOUT_BY_LENGTH: dict[int, HeaderLayout] = {
    CHECKSUMMED_HEADER_LEN_3D: HeaderLayout.CHECKSUMMED_48,
    CHECKSUMMED_HEADER_LEN_2D: HeaderLayout.CHECKSUMMED_52,
}


@dataclass(frozen=True, order=True)
class SdkVersion:
    """mmWave SDK release that built the firmware, as packed in the header.

    Examples
    --------
    >>> SdkVersion.from_word(0x01020005)
    SdkVersion(major=1, minor=2, bugfix=0, build=5)
    >>> str(SdkVersion.from_word(0x03050004))
    '3.5.0.4'
    >>> SdkVersion(1, 2) >= SdkVersion(1, 1)
    True
    """

    major: int
    minor: int
    bugfix: int = 0
    build: int = 0

    def __post_init__(self) -> None:
        for name in ("major", "minor", "bugfix", "build"):
            value = getattr(self, name)
            if not 0 <= value <= 0xFF:
                raise ValueError(f"version {name} must fit in one byte, got {value}")

    @classmethod
    def from_word(cls, word: int) -> SdkVersion:
        """Unpack ``major<<24 | minor<<16 | bugfix<<8 | build``."""
        return cls((word >> 24) & 0xFF, (word >> 16) & 0xFF, (word >> 8) & 0xFF, word & 0xFF)

    @classmethod
    def parse(cls, text: str) -> SdkVersion:
        """Parse ``"1.2"``, ``"1.2.0"`` or ``"1.2.0.5"``.

        >>> SdkVersion.parse("2.0")
        SdkVersion(major=2, minor=0, bugfix=0, build=0)
        """
        parts = text.strip().split(".")
        if not 2 <= len(parts) <= 4 or not all(part.isdigit() for part in parts):
            raise ValueError(f"expected a version such as 1.2.0.5, got {text!r}")
        numbers = [int(part) for part in parts] + [0] * (4 - len(parts))
        return cls(*numbers)

    def to_word(self) -> int:
        """Pack back into the 32-bit header representation."""
        return (self.major << 24) | (self.minor << 16) | (self.bugfix << 8) | self.build

    @property
    def major_minor(self) -> tuple[int, int]:
        """``(major, minor)``, the granularity the format rules care about."""
        return (self.major, self.minor)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.bugfix}.{self.build}"


@dataclass(frozen=True)
class HeaderTiming:
    """Processing times the people counting firmware reports in its header.

    All values are in microseconds. The 3D lab (48-byte header) reports its
    frame processing *time*, the 2D lab (52-byte header) its frame processing
    *margin*; the other field is ``None``.
    """

    chirp_margin_us: int
    tracking_time_us: int
    uart_time_us: int
    frame_processing_time_us: int | None = None
    frame_margin_us: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        """Field names to values."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class HeaderFields:
    """The fixed part of a header, read before the full packet is examined.

    For the padded layout this is the 36-byte prefix (the sub-frame word, if
    any, is settled later); checksummed layouts are read completely.

    Attributes
    ----------
    cpu_cycles : int or None
        Header timestamp; ``None`` for the 48-byte layout, which has none.
    declared_point_count : int or None
        Point count claimed by the padded layout; ``None`` for the others.
    layout : HeaderLayout
    subframe_number : int or None
        Read from checksummed layouts; padded ones settle it later.
    checksum : int or None
        Checksum field of a checksummed header.
    timing : HeaderTiming or None
        Processing times of a checksummed header.
    """

    version_word: int
    packet_length: int
    platform: int
    frame_number: int
    cpu_cycles: int | None
    declared_point_count: int | None
    tlv_count: int
    layout: HeaderLayout = HeaderLayout.PADDED
    subframe_number: int | None = None
    checksum: int | None = None
    timing: HeaderTiming | None = None

    @property
    def version(self) -> SdkVersion:
        """The packed version word as an :class:`SdkVersion`."""
        return SdkVersion.from_word(self.version_word)


def read_header_fields(buf: bytes | bytearray | memoryview, offset: int = 0) -> HeaderFields:
    """Read the fixed 36-byte part of a padded-layout header.

    Parameters
    ----------
    buf : bytes-like
        Buffer holding at least 36 bytes from ``offset``.
    offset : int, optional
        Position of the sync word.

    Returns
    -------
    HeaderFields

    Raises
    ------
    DecodeError
        If fewer than 36 bytes are available or the sync word is missing.
    """
    if len(buf) - offset < SHORT_HEADER_LEN:
        raise DecodeError(
            f"a frame header needs {SHORT_HEADER_LEN} bytes, only {len(buf) - offset} available"
        )
    magic, *values = _HEADER_PREFIX.unpack_from(buf, offset)
    if magic != MAGIC_WORD:
        raise DecodeError(f"no sync word at offset {offset}")
    return HeaderFields(*values)


def header_checksum(header: bytes | bytearray | memoryview) -> int:
    """Checksum the people counting firmware writes into its header.

    The sum of the header's little-endian ``uint16`` words, with the checksum
    field (the last two bytes) taken as zero, folded once and complemented:
    ``~((sum >> 16) + (sum & 0xFFFF))`` truncated to 16 bits.

    Examples
    --------
    >>> header_checksum(bytes(46) + b"\\xff\\xff")
    65535
    """
    if len(header) % 2:
        raise ValueError(f"a header has an even length, got {len(header)} bytes")
    total = sum(_checksum_words(len(header) // 2 - 1).unpack_from(header, 0))
    return (~((total >> 16) + (total & 0xFFFF))) & 0xFFFF


def stored_checksum_matches(buf: bytes | bytearray | memoryview, offset: int, length: int) -> bool:
    """Whether the last two bytes of ``buf[offset:offset + length]`` hold its header checksum."""
    if len(buf) - offset < length:
        return False
    header = memoryview(buf)[offset : offset + length]
    (stored,) = _UINT16.unpack_from(header, length - 2)
    return header_checksum(header) == stored


@functools.lru_cache(maxsize=8)
def _checksum_words(count: int) -> struct.Struct:
    return struct.Struct(f"<{count}H")


def read_checksummed_fields(
    buf: bytes | bytearray | memoryview, offset: int, layout: HeaderLayout
) -> HeaderFields:
    """Read a complete 48- or 52-byte people counting header.

    Raises
    ------
    DecodeError
        If the buffer is too short or does not start with the sync word.
    ValueError
        If ``layout`` is not a checksummed layout.
    """
    if not layout.checksummed:
        raise ValueError(f"{layout.value} is not a checksummed layout")
    size = layout.min_length
    if len(buf) - offset < size:
        raise DecodeError(f"a {size}-byte header needs {size} bytes, only {len(buf) - offset}")
    if layout is HeaderLayout.CHECKSUMMED_48:
        (
            magic,
            version,
            length,
            platform,
            frame,
            subframe,
            chirp_margin,
            frame_time,
            tracking,
            uart,
            tlvs,
            checksum,
        ) = _HEADER_3D.unpack_from(buf, offset)
        cycles = None
        timing = HeaderTiming(chirp_margin, tracking, uart, frame_processing_time_us=frame_time)
    else:
        (
            magic,
            version,
            platform,
            cycles,
            length,
            frame,
            subframe,
            chirp_margin,
            frame_margin,
            tracking,
            uart,
            tlvs,
            checksum,
        ) = _HEADER_2D.unpack_from(buf, offset)
        timing = HeaderTiming(chirp_margin, tracking, uart, frame_margin_us=frame_margin)
    if magic != MAGIC_WORD:
        raise DecodeError(f"no sync word at offset {offset}")
    return HeaderFields(
        version_word=version,
        packet_length=length,
        platform=platform,
        frame_number=frame,
        cpu_cycles=cycles,
        declared_point_count=None,
        tlv_count=tlvs,
        layout=layout,
        subframe_number=subframe,
        checksum=checksum,
        timing=timing,
    )


def header_problem(
    fields: HeaderFields,
    *,
    max_packet_bytes: int,
    max_tlvs: int,
    strict: bool = False,
) -> str | None:
    """Return why a header cannot be genuine, or ``None`` if it is plausible.

    The structural checks shared by all layouts: a supported SDK major version,
    a TI platform code, a bounded TLV count and a packet length that can hold
    the header and one TL pair per declared TLV. Strict mode also insists on
    the 32-byte alignment of the padded layout. The checksum of a checksummed
    header is verified separately (:func:`examine_header`).

    Parameters
    ----------
    fields : HeaderFields
        The header to check.
    max_packet_bytes : int
        Longest packet considered genuine.
    max_tlvs : int
        Largest TLV count considered genuine.
    strict : bool, optional
        Require ``packet_length % 32 == 0`` (padded layout only).

    Returns
    -------
    str or None
        A one-line reason, or ``None``.
    """
    version = fields.version
    if version.major not in SUPPORTED_MAJOR_VERSIONS:
        return f"SDK major version {version.major} is not supported"
    if fields.platform >> 16 != PLATFORM_FAMILY_TAG:
        return f"platform code 0x{fields.platform:X} is not a TI mmWave device"
    if fields.tlv_count > max_tlvs:
        return f"{fields.tlv_count} TLVs declared, at most {max_tlvs} accepted"
    if fields.packet_length > max_packet_bytes:
        return f"packet length {fields.packet_length} exceeds the {max_packet_bytes}-byte limit"
    minimum = fields.layout.min_length + TLV_HEADER_LEN * fields.tlv_count
    if fields.packet_length < minimum:
        return (
            f"packet length {fields.packet_length} cannot hold a header and "
            f"{fields.tlv_count} TLVs ({minimum} bytes)"
        )
    if strict and fields.layout is HeaderLayout.PADDED and fields.packet_length % PACKET_ALIGNMENT:
        return f"packet length {fields.packet_length} is not a multiple of {PACKET_ALIGNMENT}"
    return None


@dataclass(frozen=True)
class HeaderVerdict:
    """Outcome of :func:`examine_header`.

    Exactly one of three things holds: ``fields`` is a plausible header,
    ``problem`` says why no layout fits (``code`` classifies it), or
    ``needed`` is the number of bytes from the sync word that must be present
    before a verdict is possible.
    """

    fields: HeaderFields | None = None
    problem: str | None = None
    code: IssueCode = IssueCode.INVALID_HEADER
    needed: int = 0

    @property
    def pending(self) -> bool:
        """Whether more bytes are needed."""
        return self.needed > 0


def examine_header(
    buf: bytes | bytearray | memoryview,
    offset: int = 0,
    *,
    max_packet_bytes: int,
    max_tlvs: int,
    strict: bool = False,
    layout: HeaderLayout | None = None,
) -> HeaderVerdict:
    """Decide which header layout follows a sync word, if any.

    Parameters
    ----------
    buf : bytes-like
        Buffer with the sync word at ``offset``.
    offset : int, optional
        Position of the sync word.
    max_packet_bytes, max_tlvs, strict
        Limits, as in :func:`header_problem`.
    layout : HeaderLayout, optional
        Only try this layout (a schema forced by the user implies one).

    Returns
    -------
    HeaderVerdict
        A plausible padded header is returned as soon as 36 bytes are present;
        it may still turn out to be a 48-byte header once the whole packet is
        known (both put the packet length at offset 12), which
        :func:`radar_stream_decoder.frame.decode_packet` settles.

    Examples
    --------
    >>> raw = MAGIC_WORD + struct.pack("<8I", 0x02000004, 64, 0xA1642, 1, 0, 0, 0, 0)
    >>> examine_header(raw, max_packet_bytes=4096, max_tlvs=32).fields.layout.value
    'padded'
    >>> examine_header(raw[:20], max_packet_bytes=4096, max_tlvs=32).needed
    36
    """
    have = len(buf) - offset
    if bytes(buf[offset : offset + len(MAGIC_WORD)]) != MAGIC_WORD[: max(0, min(have, 8))]:
        return HeaderVerdict(problem=f"no sync word at offset {offset}")
    if layout is None:
        if have < 16:
            return HeaderVerdict(needed=16)
        (word12,) = _UINT32.unpack_from(buf, offset + 12)
        candidates: tuple[HeaderLayout, ...] = (HeaderLayout.PADDED, HeaderLayout.CHECKSUMMED_48)
        if word12 >> 16 == PLATFORM_FAMILY_TAG:
            # A platform code where the others keep the packet length: the 52-byte
            # layout. The others stay as a fall-back for limits above 0xA0000 bytes.
            candidates = (HeaderLayout.CHECKSUMMED_52, *candidates)
    else:
        candidates = (layout,)
    reasons: list[str] = []
    code = IssueCode.INVALID_HEADER
    for candidate in candidates:
        if have < candidate.min_length:
            return HeaderVerdict(needed=candidate.min_length)
        if candidate is HeaderLayout.PADDED:
            fields = read_header_fields(buf, offset)
        else:
            fields = read_checksummed_fields(buf, offset, candidate)
        problem = header_problem(
            fields, max_packet_bytes=max_packet_bytes, max_tlvs=max_tlvs, strict=strict
        )
        if problem is None and candidate.checksummed:
            expected = header_checksum(buf[offset : offset + candidate.min_length])
            if expected != fields.checksum:
                problem = (
                    f"header checksum 0x{fields.checksum:04X} does not match the "
                    f"0x{expected:04X} computed over the header"
                )
                code = IssueCode.HEADER_CHECKSUM_MISMATCH
        if problem is None:
            return HeaderVerdict(fields=fields)
        reasons.append(f"{_layout_name(candidate)}: {problem}")
    return HeaderVerdict(problem="; ".join(reasons), code=code)


def _layout_name(layout: HeaderLayout) -> str:
    if layout is HeaderLayout.PADDED:
        return "36/40-byte header"
    return f"{layout.min_length}-byte header"


@dataclass(frozen=True)
class FrameHeader:
    """A decoded frame header.

    Attributes
    ----------
    version : SdkVersion
        SDK release of the firmware.
    packet_length : int
        Declared length of the whole packet, padding included.
    platform : int
        Device code (see :class:`~radar_stream_decoder.constants.Platform`).
    frame_number : int
        Frame counter; restarts when the sensor restarts.
    cpu_cycles : int or None
        Free-running 32-bit timestamp of the core that wrote the packet;
        ``None`` for the 48-byte layout, which carries no timestamp.
    declared_point_count : int or None
        Point count written by the firmware (padded layout only). Some
        firmware writes a constant here, so the decoder always counts points
        from the payload itself.
    tlv_count : int
        Number of TLVs that follow.
    subframe_number : int or None
        Sub-frame index, ``None`` in the 36-byte layout.
    header_len : int
        36, 40, 48 or 52.
    layout : HeaderLayout
    checksum : int or None
        Checksum field of a checksummed header (already verified).
    timing : HeaderTiming or None
        Processing times of a checksummed header.
    """

    version: SdkVersion
    packet_length: int
    platform: int
    frame_number: int
    cpu_cycles: int | None
    declared_point_count: int | None
    tlv_count: int
    subframe_number: int | None
    header_len: int
    layout: HeaderLayout = HeaderLayout.PADDED
    checksum: int | None = None
    timing: HeaderTiming | None = None

    @property
    def platform_name(self) -> str:
        """Readable platform name such as ``"xWR16xx"``."""
        return platform_label(self.platform)

    @property
    def has_subframe(self) -> bool:
        """Whether the header carries a sub-frame number."""
        return self.subframe_number is not None

    @property
    def tlv_length_includes_header(self) -> bool:
        """Whether TLV length fields count their own 8-byte type/length pair."""
        return self.layout.checksummed


def decode_frame_header(
    buf: bytes | bytearray | memoryview, offset: int = 0, header_len: int = SHORT_HEADER_LEN
) -> FrameHeader:
    """Decode a header whose layout is already known.

    Parameters
    ----------
    buf : bytes-like
        Buffer with the packet.
    offset : int, optional
        Position of the sync word.
    header_len : {36, 40, 48, 52}
        Header layout: 36 and 40 are the padded variants, 48 and 52 the
        checksummed ones.

    Returns
    -------
    FrameHeader

    Raises
    ------
    DecodeError
        If the buffer is too short or does not start with the sync word.
    ValueError
        If ``header_len`` is none of the four layouts.

    Examples
    --------
    >>> raw = MAGIC_WORD + struct.pack("<8I", 0x01020005, 64, 0xA1443, 7, 1000, 0, 0, 0)
    >>> header = decode_frame_header(raw, header_len=40)
    >>> header.frame_number, header.subframe_number, str(header.version)
    (7, 0, '1.2.0.5')
    """
    if header_len in CHECKSUMMED_LAYOUT_BY_LENGTH:
        fields = read_checksummed_fields(buf, offset, CHECKSUMMED_LAYOUT_BY_LENGTH[header_len])
        return FrameHeader(
            version=fields.version,
            packet_length=fields.packet_length,
            platform=fields.platform,
            frame_number=fields.frame_number,
            cpu_cycles=fields.cpu_cycles,
            declared_point_count=None,
            tlv_count=fields.tlv_count,
            subframe_number=fields.subframe_number,
            header_len=header_len,
            layout=fields.layout,
            checksum=fields.checksum,
            timing=fields.timing,
        )
    if header_len not in HEADER_LENGTHS:
        raise ValueError(
            f"header_len must be one of {(*HEADER_LENGTHS, *CHECKSUMMED_LAYOUT_BY_LENGTH)}, "
            f"got {header_len}"
        )
    fields = read_header_fields(buf, offset)
    subframe: int | None = None
    if header_len > SHORT_HEADER_LEN:
        if len(buf) - offset < header_len:
            raise DecodeError(f"a {header_len}-byte header needs {header_len} bytes")
        (subframe,) = _UINT32.unpack_from(buf, offset + SHORT_HEADER_LEN)
    return FrameHeader(
        version=fields.version,
        packet_length=fields.packet_length,
        platform=fields.platform,
        frame_number=fields.frame_number,
        cpu_cycles=fields.cpu_cycles,
        declared_point_count=fields.declared_point_count,
        tlv_count=fields.tlv_count,
        subframe_number=subframe,
        header_len=header_len,
    )


def header_len_rule(platform: int, version: SdkVersion, schema: str | None) -> HeaderRule | None:
    """Return the first header-length rule of the padded layout that applies, or ``None``.

    Parameters
    ----------
    platform : int
        Header platform code.
    version : SdkVersion
        Header version.
    schema : str or None
        Packet schema when it is known (forced by the user); ``None`` while it
        is still being detected.

    Examples
    --------
    >>> header_len_rule(0xA1642, SdkVersion(2, 0), None).header_len
    40
    >>> header_len_rule(0xA1443, SdkVersion(1, 2), "oob-sdk2").header_len
    36
    >>> header_len_rule(0xA1443, SdkVersion(1, 2), None) is None
    True
    """
    for rule in HEADER_RULES:
        if rule.applies_to(platform, version.major_minor, schema):
            return rule
    return None


@dataclass(frozen=True)
class TlvLocation:
    """Where one TLV's payload sits inside a packet.

    ``length`` is always the payload length, whatever the layout's length
    field counts.
    """

    tlv_type: int
    length: int
    payload_offset: int

    @property
    def end(self) -> int:
        """Offset just past the payload."""
        return self.payload_offset + self.length


@dataclass(frozen=True)
class ChainWalk:
    """Result of following the TLV chain through a packet.

    Attributes
    ----------
    locations : tuple of TlvLocation
        Every TLV whose payload lies completely inside the available bytes.
    complete : bool
        All declared TLVs were found.
    truncated : bool
        The chain ran past the bytes actually available (but not past the
        declared packet length).
    end_offset : int
        Offset just past the last complete TLV.
    problem : str or None
        Why the walk stopped early.
    """

    locations: tuple[TlvLocation, ...]
    complete: bool
    truncated: bool
    end_offset: int
    problem: str | None = None


def walk_tlv_chain(
    buf: bytes | bytearray | memoryview,
    offset: int,
    header_len: int,
    tlv_count: int,
    packet_length: int,
    available: int | None = None,
    *,
    length_includes_header: bool = False,
) -> ChainWalk:
    """Follow ``tlv_count`` type/length pairs through a packet.

    Parameters
    ----------
    buf : bytes-like
        Buffer holding the packet.
    offset : int
        Position of the packet's sync word in ``buf``.
    header_len : int
        Header length to assume.
    tlv_count : int
        Number of TLVs the header declares.
    packet_length : int
        Declared packet length; no TLV may extend past it.
    available : int, optional
        Bytes of the packet actually present (for a packet cut short on the
        link). Defaults to everything in ``buf`` up to ``packet_length``.
    length_includes_header : bool, optional
        The length fields count their own 8-byte type/length pair (checksummed
        layouts).

    Returns
    -------
    ChainWalk
    """
    have = min(packet_length, len(buf) - offset)
    if available is not None:
        have = min(have, available)
    position = header_len
    locations: list[TlvLocation] = []
    for index in range(tlv_count):
        if position + TLV_HEADER_LEN > packet_length:
            return ChainWalk(
                tuple(locations),
                False,
                False,
                position,
                f"TLV {index} header at byte {position} overflows the {packet_length}-byte packet",
            )
        if position + TLV_HEADER_LEN > have:
            return ChainWalk(
                tuple(locations), False, True, position, f"packet ends inside TLV {index} header"
            )
        tlv_type, length = _TLV_HEADER.unpack_from(buf, offset + position)
        if length_includes_header:
            if length < TLV_HEADER_LEN:
                return ChainWalk(
                    tuple(locations),
                    False,
                    False,
                    position,
                    f"TLV {index} (type {tlv_type}) declares {length} bytes, fewer than its "
                    f"own {TLV_HEADER_LEN}-byte type/length pair",
                )
            length -= TLV_HEADER_LEN
        payload_offset = position + TLV_HEADER_LEN
        if payload_offset + length > packet_length:
            return ChainWalk(
                tuple(locations),
                False,
                False,
                position,
                f"TLV {index} (type {tlv_type}, {length} bytes) overflows the "
                f"{packet_length}-byte packet",
            )
        if payload_offset + length > have:
            return ChainWalk(
                tuple(locations),
                False,
                True,
                position,
                f"packet ends inside TLV {index} (type {tlv_type})",
            )
        locations.append(TlvLocation(tlv_type, length, payload_offset))
        position = payload_offset + length
    return ChainWalk(tuple(locations), True, False, position)


def infer_header_len(
    buf: bytes | bytearray | memoryview,
    offset: int,
    fields: HeaderFields,
    accept: Callable[[Sequence[TlvLocation]], bool],
    available: int | None = None,
) -> tuple[int, ...]:
    """Return the padded header lengths under which the TLV chain is consistent.

    A layout is consistent when its chain does not overflow the declared
    packet length and ``accept`` approves the TLV types and lengths it finds.
    When more than one layout qualifies, those that also leave less than 32
    bytes of padding (the firmware's alignment guarantee) are preferred.

    Parameters
    ----------
    buf : bytes-like
        Buffer holding the packet.
    offset : int
        Position of the sync word.
    fields : HeaderFields
        The already-read fixed header part.
    accept : callable
        Receives the TLV locations of a candidate layout and returns whether
        their types and lengths make sense for some packet schema.
    available : int, optional
        Bytes of the packet actually present.

    Returns
    -------
    tuple of int
        Consistent layouts, possibly empty.
    """
    have = min(fields.packet_length, len(buf) - offset)
    if available is not None:
        have = min(have, available)
    viable: list[tuple[int, ChainWalk]] = []
    for header_len in HEADER_LENGTHS:
        if fields.packet_length < header_len or have < header_len:
            continue
        walk = walk_tlv_chain(buf, offset, header_len, fields.tlv_count, fields.packet_length, have)
        if walk.problem is not None and not walk.truncated:
            continue
        if not accept(walk.locations):
            continue
        viable.append((header_len, walk))
    aligned = [
        header_len
        for header_len, walk in viable
        if walk.complete and fields.packet_length - walk.end_offset < PACKET_ALIGNMENT
    ]
    if aligned:
        return tuple(aligned)
    return tuple(header_len for header_len, _ in viable)
