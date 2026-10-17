"""Protocol constants of the TI mmWave UART output stream.

Defined here: the sync word, the header and TLV sizes, the packet alignment,
the platform codes and their CPU clocks, the TLV type numbers of each firmware
family, and the table of rules that decides whether a frame header carries
the optional sub-frame word.

All multi-byte fields of the stream are little-endian (the R4F and C674x cores
that write it are little-endian).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

#: Sync word that opens every packet: the little-endian ``uint16`` sequence
#: ``0x0102, 0x0304, 0x0506, 0x0708`` as it appears on the wire.
MAGIC_WORD: bytes = bytes((0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07))

#: Packets are padded so that their total length is a multiple of this.
PACKET_ALIGNMENT = 32

#: Frame header without the sub-frame word (xWR14xx out-of-box demo, SDK 1.x).
SHORT_HEADER_LEN = 36

#: Frame header with the trailing ``uint32`` sub-frame word.
#:
#: TI documents disagree here: the out-of-box UART page of the mmWave Industrial
#: Toolbox 4.10 states "Length: 44 Bytes", but its own field list adds up to 40
#: bytes. The firmware is authoritative: ``MmwDemo_output_message_header`` in
#: ``mmw_output.h`` of TI mmWave SDK 03.05.00.04 (xwr68xx, xwr64xx, xwr18xx
#: demos) is 40 bytes, and ``mss_main.c`` starts every packet length from
#: ``sizeof`` of that structure.
LONG_HEADER_LEN = 40

#: The two header layouts of the padded framing, shortest first.
HEADER_LENGTHS: tuple[int, int] = (SHORT_HEADER_LEN, LONG_HEADER_LEN)

#: Header of the 3D People Counting lab (TI mmWave Industrial Toolbox 4.5-4.12,
#: ``Pcount3DDemo_output_message_header`` in ``pcount3D_output.h``): no CPU
#: timestamp, a ``uint16`` TLV count and a ``uint16`` header checksum.
CHECKSUMMED_HEADER_LEN_3D = 48

#: Header of the 2D people counting lab (TI mmWave Industrial Toolbox 2.x,
#: ``MmwDemo_output_message_header`` in the lab's ``mmw_output.h``): the platform
#: word moves to offset 12 and a CPU timestamp precedes the packet length.
CHECKSUMMED_HEADER_LEN_2D = 52

#: Size of a TLV's type and length words. The padded framing's length field
#: excludes them; the checksummed framing's length field includes them.
TLV_HEADER_LEN = 8

#: Upper bound on the TLV count a sane header may declare.
MAX_TLVS_PER_PACKET = 32

#: Upper bound on a packet's declared length; longer claims are treated as corruption.
DEFAULT_MAX_PACKET_BYTES = 262_144

#: ``platform >> 16`` of every TI mmWave device code (``0xA1443``, ``0xA1642``, ...).
PLATFORM_FAMILY_TAG = 0xA

#: SDK major versions whose output format the decoder understands.
SUPPORTED_MAJOR_VERSIONS: tuple[int, ...] = (1, 2, 3)

#: Speed of light used for every FMCW conversion. TI's tools use exactly 3e8,
#: so the decoder does too; the difference to 299 792 458 m/s is 0.07 %.
SPEED_OF_LIGHT_MPS = 3.0e8

#: Angle FFT length the out-of-box demo uses for its azimuth processing.
ANGLE_FFT_SIZE = 64

#: The header's CPU-cycle timestamp is a free-running 32-bit counter.
CYCLE_COUNTER_MODULUS = 1 << 32

#: Decibels per unit of log2 amplitude: a factor of two in amplitude is 6.02 dB.
DB_PER_LOG2_UNIT = 20.0 * math.log10(2.0)

#: Default timestamp clock when the platform does not say otherwise (R4F at 200 MHz).
DEFAULT_CPU_CLOCK_HZ = 200e6


class Platform(IntEnum):
    """Device codes written into the header's ``platform`` field."""

    XWR14XX = 0xA1443
    XWR16XX = 0xA1642
    XWR18XX = 0xA1843
    XWR68XX = 0xA6843


_PLATFORM_LABELS: dict[int, str] = {
    Platform.XWR14XX: "xWR14xx",
    Platform.XWR16XX: "xWR16xx",
    Platform.XWR18XX: "xWR18xx",
    Platform.XWR68XX: "xWR68xx",
}

#: Clock of the core that stamps ``timeCpuCycles``. On the xWR14xx the R4F
#: (200 MHz) writes the header; on the xWR16xx with SDK 1.x/2.x it is the C674x
#: DSP (600 MHz). Other platforms fall back to :data:`DEFAULT_CPU_CLOCK_HZ`.
PLATFORM_CPU_CLOCK_HZ: dict[int, float] = {
    Platform.XWR14XX: 200e6,
    Platform.XWR16XX: 600e6,
}

#: Firmware families whose header timestamp is read on the R4F (200 MHz)
#: whatever the platform. TI mmWave SDK 03.05.00.04 stamps the out-of-box demo
#: header with ``Pmu_getCount(0)`` in ``mss_main.c`` on every platform; the 2D
#: people counting lab stamps it with ``Cycleprofiler_getTimeStamp()`` on the
#: MSS (``task_mbox.c``). The Radar Toolbox people tracking firmware is built
#: on SDK 3.x and assumed to follow the out-of-box demo.
R4F_TIMESTAMP_SCHEMAS: frozenset[str] = frozenset(
    {"oob-sdk3", "people-counting-2d", "people-tracking"}
)


def cpu_clock_hz(platform: int, schema: str | None = None) -> float:
    """Clock of the header timestamp for a platform and firmware family.

    SDK 3.x firmware and the people counting labs stamp headers on the R4F at
    200 MHz; xWR16xx firmware of SDK 1.x/2.x uses the 600 MHz DSP clock.

    Examples
    --------
    >>> cpu_clock_hz(0xA1642), cpu_clock_hz(0xA1642, "oob-sdk3"), cpu_clock_hz(0xA1443)
    (600000000.0, 200000000.0, 200000000.0)
    >>> cpu_clock_hz(0xA1642, "people-counting-2d")
    200000000.0
    """
    if schema is not None and str(getattr(schema, "value", schema)) in R4F_TIMESTAMP_SCHEMAS:
        return DEFAULT_CPU_CLOCK_HZ
    return PLATFORM_CPU_CLOCK_HZ.get(int(platform), DEFAULT_CPU_CLOCK_HZ)


def platform_label(code: int) -> str:
    """Return a readable name for a platform code.

    Parameters
    ----------
    code : int
        Value of the header's ``platform`` field.

    Returns
    -------
    str
        ``"xWR14xx"`` style names for known devices, the hex code otherwise.

    Examples
    --------
    >>> platform_label(0xA1443)
    'xWR14xx'
    >>> platform_label(0xA9999)
    '0xA9999'
    """
    return _PLATFORM_LABELS.get(int(code), f"0x{int(code):X}")


_PLATFORM_ALIASES: dict[str, Platform] = {
    "14xx": Platform.XWR14XX,
    "1443": Platform.XWR14XX,
    "16xx": Platform.XWR16XX,
    "1642": Platform.XWR16XX,
    "18xx": Platform.XWR18XX,
    "1843": Platform.XWR18XX,
    "68xx": Platform.XWR68XX,
    "6843": Platform.XWR68XX,
}


def parse_platform(value: int | str | None) -> int | None:
    """Interpret a platform given as a code, a device name or a family name.

    Parameters
    ----------
    value : int, str or None
        ``0xA1642``, ``"0xA1642"``, ``"16xx"``, ``"IWR1642"``, ``"xwr16xx"`` and
        similar spellings are accepted. ``None`` and ``"auto"`` mean "take the
        platform from the packet header".

    Returns
    -------
    int or None
        The platform code, or ``None`` for automatic selection.

    Raises
    ------
    ValueError
        If the value names no known platform and is not a valid code.

    Examples
    --------
    >>> hex(parse_platform("IWR1443"))
    '0xa1443'
    >>> hex(parse_platform("16xx"))
    '0xa1642'
    >>> parse_platform("auto") is None
    True
    """
    if value is None:
        return None
    if isinstance(value, int):
        code = int(value)
    else:
        text = value.strip().lower()
        if text in {"", "auto"}:
            return None
        for alias, platform in _PLATFORM_ALIASES.items():
            if text.endswith(alias):
                return int(platform)
        try:
            code = int(text, 0)
        except ValueError:
            raise ValueError(
                f"unknown platform {value!r}; use 14xx, 16xx, 18xx, 68xx or a code such as 0xA1642"
            ) from None
    if code >> 16 != PLATFORM_FAMILY_TAG:
        raise ValueError(f"0x{code:X} is not a TI mmWave platform code (expected 0xAxxxx)")
    return code


class DemoTlvType(IntEnum):
    """TLV types of the out-of-box demo output (types 7-9 exist from SDK 3.x on)."""

    POINTS = 1
    RANGE_PROFILE = 2
    NOISE_PROFILE = 3
    AZIMUTH_HEATMAP = 4
    RANGE_DOPPLER_HEATMAP = 5
    STATS = 6
    POINT_SIDE_INFO = 7
    AZIMUTH_ELEVATION_HEATMAP = 8
    TEMPERATURE = 9


class VitalSignsTlvType(IntEnum):
    """TLV types of the Vital Signs lab output."""

    SUMMARY = 0
    RANGE_SEGMENT = 1


class PeopleCountingTlvType(IntEnum):
    """TLV types of the people counting labs (checksummed 48- and 52-byte headers).

    Numbers from ``Pcount3DDemo_output_message_type`` (``pcount3D_output.h``,
    TI mmWave Industrial Toolbox 4.6.0) and ``MmwDemo_output_message_type`` of
    the 2D lab (Industrial Toolbox 2.x). Types 6-8 mean something entirely
    different here than in the out-of-box demo (statistics, side info, heatmap).
    """

    POINTS = 6
    TARGETS = 7
    TARGET_ASSOCIATION = 8
    PRESENCE = 11


class PeopleTrackingTlvType(IntEnum):
    """TLV types of the TI Radar Toolbox people tracking demos (40-byte header)."""

    TARGETS = 1010
    TARGET_ASSOCIATION = 1011
    TARGET_HEIGHTS = 1012
    POINTS = 1020
    PRESENCE = 1021


#: Largest identifier the trackers give a target (TI documents IDs 0-249).
MAX_TARGET_ID = 249

#: Per-point target-ID values that mean "not associated with any target".
UNASSOCIATED_WEAK_SNR = 253
UNASSOCIATED_OUTSIDE_BOUNDARY = 254
UNASSOCIATED_NOISE = 255


@dataclass(frozen=True)
class HeaderRule:
    """One entry of the table that decides the frame-header length.

    A rule applies when every condition it sets matches; unset conditions
    match anything. Rules are tried in order and the first match wins.

    Attributes
    ----------
    description : str
        Why the rule holds; quoted in decode issues.
    header_len : int
        36 or 40 bytes.
    platform : int, optional
        Platform code the rule is restricted to.
    schema : str, optional
        Packet schema value (``"oob-sdk2"`` ...) the rule is restricted to.
    min_version, max_version : tuple of int, optional
        ``(major, minor)`` bounds; the minimum is inclusive, the maximum exclusive.
    """

    description: str
    header_len: int
    platform: int | None = None
    schema: str | None = None
    min_version: tuple[int, int] | None = None
    max_version: tuple[int, int] | None = None

    def applies_to(self, platform: int, version: tuple[int, int], schema: str | None) -> bool:
        """Return whether the rule covers this platform, ``(major, minor)`` and schema."""
        if self.platform is not None and platform != self.platform:
            return False
        if self.schema is not None and schema != self.schema:
            return False
        if self.min_version is not None and version < self.min_version:
            return False
        return not (self.max_version is not None and version >= self.max_version)


#: Header-length rules, most specific first. A packet that matches no rule (an
#: xWR14xx stream whose firmware family is not known yet, an unfamiliar
#: platform) has its header length inferred from which layout makes the TLV
#: chain fit the declared packet length.
HEADER_RULES: tuple[HeaderRule, ...] = (
    HeaderRule(
        "the Vital Signs lab firmware always writes the sub-frame word",
        LONG_HEADER_LEN,
        schema="vital-signs",
    ),
    HeaderRule(
        "SDK 3.x firmware always writes the sub-frame word",
        LONG_HEADER_LEN,
        schema="oob-sdk3",
    ),
    HeaderRule(
        "the Radar Toolbox people tracking firmware always writes the sub-frame word",
        LONG_HEADER_LEN,
        schema="people-tracking",
    ),
    HeaderRule(
        "SDK 3.x firmware always writes the sub-frame word",
        LONG_HEADER_LEN,
        min_version=(3, 0),
    ),
    HeaderRule(
        "xWR16xx firmware from SDK 1.1 on writes the sub-frame word",
        LONG_HEADER_LEN,
        platform=Platform.XWR16XX,
        min_version=(1, 1),
    ),
    HeaderRule(
        "xWR16xx firmware before SDK 1.1 has no sub-frame word",
        SHORT_HEADER_LEN,
        platform=Platform.XWR16XX,
        max_version=(1, 1),
    ),
    HeaderRule(
        "the xWR14xx out-of-box demo has no sub-frame word",
        SHORT_HEADER_LEN,
        platform=Platform.XWR14XX,
        schema="oob-sdk2",
    ),
)
