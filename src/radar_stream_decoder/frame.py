"""Decoding one packet into a :class:`RadarFrame`.

:func:`decode_packet` is the single entry point every reader goes through:

1. identify the header layout (padded 36/40 bytes, checksummed 48 or 52
   bytes) and verify it, checksum included;
2. settle the padded header length (rule table, then TLV-chain inference);
3. walk the TLV chain, slicing each payload by its own length field;
4. pick the schema (forced or detected; a packet no known firmware family
   explains raises :class:`~radar_stream_decoder.errors.UnsupportedFormatError`)
   and run each TLV's decoder;
5. check what follows the last TLV: padding below 32 bytes for the padded
   layout, nothing at all for the checksummed ones;
6. enrich across TLVs: point velocities and SNR, the decibel scale of the
   log2-magnitude arrays, the first bin of a Vital Signs range segment, target
   heights.

Problems that leave the rest of the packet usable become issues on the frame.
In strict mode any error-level issue, a misaligned length, excess padding or a
packet cut short raises :class:`~radar_stream_decoder.errors.DecodeError`.
"""

from __future__ import annotations

import dataclasses
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .angle_map import AzimuthHeatmap
from .config import DecoderConfig
from .constants import LONG_HEADER_LEN, PACKET_ALIGNMENT, SHORT_HEADER_LEN, Platform
from .errors import DecodeError, DecodeIssue, IssueCode, Severity
from .header import (
    FrameHeader,
    HeaderFields,
    HeaderLayout,
    TlvLocation,
    decode_frame_header,
    examine_header,
    header_len_rule,
    infer_header_len,
    stored_checksum_matches,
    walk_tlv_chain,
)
from .payloads import (
    DecodeContext,
    MagnitudeProfile,
    PayloadKind,
    PointCloud,
    PointSideInfo,
    ProcessingStats,
    RangeDopplerMap,
    SnrMethod,
    TemperatureReport,
    UnknownTlv,
)
from .schema import (
    PADDED_SCHEMAS,
    PacketSchema,
    chain_matches_schema,
    detect_schema,
    schema_registry,
)
from .sensor_config import SensorConfig
from .snr import estimate_point_snr
from .tracking import (
    PresenceIndication,
    TargetAssociation,
    TargetHeights,
    TargetList,
    merge_target_heights,
)
from .vital_signs import RangeSegment, VitalSignsSummary


@dataclass(repr=False)
class RadarFrame:
    """One decoded packet.

    Attributes
    ----------
    header : FrameHeader
    schema : PacketSchema
        Schema the payloads were decoded with.
    tlv_types : tuple of int
        Types of all TLVs found, in packet order.
    payloads : dict
        Decoded payloads keyed by :class:`~radar_stream_decoder.payloads.PayloadKind`.
    unknown_tlvs : list of UnknownTlv
        TLVs the schema does not define.
    issues : list of DecodeIssue
        Everything that went wrong while decoding this packet.
    raw : bytes
        The packet bytes (empty when ``keep_raw_packets`` is off).
    crc32 : int
        CRC-32 of the packet bytes, the identity used for de-duplication.
    truncated : bool
        The packet arrived shorter than its declared length. Only TLVs that
        lie entirely before the cut were decoded, and since the lost bytes may
        have been anywhere, even those can be shifted; such frames always carry
        an error-level issue.
    """

    header: FrameHeader
    schema: PacketSchema
    tlv_types: tuple[int, ...]
    payloads: dict[str, Any] = field(default_factory=dict)
    unknown_tlvs: list[UnknownTlv] = field(default_factory=list)
    issues: list[DecodeIssue] = field(default_factory=list)
    raw: bytes = b""
    crc32: int = 0
    truncated: bool = False

    def __repr__(self) -> str:
        return (
            f"RadarFrame(frame={self.header.frame_number}, subframe={self.header.subframe_number}, "
            f"schema={self.schema.value!r}, tlv_types={self.tlv_types}, points={self.num_points}, "
            f"issues={len(self.issues)}, truncated={self.truncated})"
        )

    # -- identity -----------------------------------------------------------------

    @property
    def frame_number(self) -> int:
        """Frame counter of the header."""
        return self.header.frame_number

    @property
    def subframe_number(self) -> int | None:
        """Sub-frame index, ``None`` in the 36-byte header layout."""
        return self.header.subframe_number

    @property
    def key(self) -> tuple[int, int]:
        """``(frame, sub-frame)`` ordering key."""
        return (self.header.frame_number, self.header.subframe_number or 0)

    @property
    def platform(self) -> int:
        """Platform code of the header."""
        return self.header.platform

    @property
    def has_errors(self) -> bool:
        """Whether any issue is error-level."""
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    # -- payload accessors -------------------------------------------------------

    def _payload(self, kind: PayloadKind) -> Any:
        return self.payloads.get(kind)

    @property
    def points(self) -> PointCloud | None:
        """Detected points."""
        return self._payload(PayloadKind.POINTS)

    @property
    def num_points(self) -> int:
        """Detected points in the payload (not the header's claim)."""
        points = self.points
        return len(points) if points is not None else 0

    @property
    def range_profile(self) -> MagnitudeProfile | None:
        """Zero-Doppler range profile."""
        return self._payload(PayloadKind.RANGE_PROFILE)

    @property
    def noise_profile(self) -> MagnitudeProfile | None:
        """Noise profile."""
        return self._payload(PayloadKind.NOISE_PROFILE)

    @property
    def range_doppler(self) -> RangeDopplerMap | None:
        """Detection matrix."""
        return self._payload(PayloadKind.RANGE_DOPPLER)

    @property
    def azimuth_heatmap(self) -> AzimuthHeatmap | None:
        """Azimuth static heatmap."""
        return self._payload(PayloadKind.AZIMUTH_HEATMAP)

    @property
    def azimuth_elevation_heatmap(self) -> AzimuthHeatmap | None:
        """SDK 3.x azimuth-elevation heatmap."""
        return self._payload(PayloadKind.AZIMUTH_ELEVATION_HEATMAP)

    @property
    def stats(self) -> ProcessingStats | None:
        """Processing statistics."""
        return self._payload(PayloadKind.STATS)

    @property
    def side_info(self) -> PointSideInfo | None:
        """SDK 3.x per-point SNR and noise."""
        return self._payload(PayloadKind.POINT_SIDE_INFO)

    @property
    def temperature(self) -> TemperatureReport | None:
        """SDK 3.x temperature report."""
        return self._payload(PayloadKind.TEMPERATURE)

    @property
    def vital_signs(self) -> VitalSignsSummary | None:
        """Vital Signs lab summary."""
        return self._payload(PayloadKind.VITAL_SIGNS)

    @property
    def range_segment(self) -> RangeSegment | None:
        """Vital Signs lab complex range segment."""
        return self._payload(PayloadKind.RANGE_SEGMENT)

    @property
    def targets(self) -> TargetList | None:
        """Tracker targets (people counting and tracking firmware)."""
        return self._payload(PayloadKind.TARGETS)

    @property
    def num_targets(self) -> int:
        """Targets in the target list."""
        targets = self.targets
        return len(targets) if targets is not None else 0

    @property
    def target_association(self) -> TargetAssociation | None:
        """Per-point target IDs of the *previous* frame's point cloud."""
        return self._payload(PayloadKind.TARGET_ASSOCIATION)

    @property
    def target_heights(self) -> TargetHeights | None:
        """Target height extents (Radar Toolbox people tracking)."""
        return self._payload(PayloadKind.TARGET_HEIGHTS)

    @property
    def presence(self) -> PresenceIndication | None:
        """Presence indication."""
        return self._payload(PayloadKind.PRESENCE)

    def summary(self) -> dict[str, Any]:
        """JSON-friendly overview of the frame (no arrays)."""
        return {
            "frame": self.header.frame_number,
            "subframe": self.header.subframe_number,
            "platform": self.header.platform_name,
            "sdk_version": str(self.header.version),
            "schema": self.schema.value,
            "packet_length": self.header.packet_length,
            "header_len": self.header.header_len,
            "cpu_cycles": self.header.cpu_cycles,
            "tlv_types": list(self.tlv_types),
            "num_points": self.num_points,
            "declared_points": self.header.declared_point_count,
            "num_targets": self.num_targets,
            "presence": None if self.presence is None else self.presence.occupied,
            "truncated": self.truncated,
            "issues": [issue.code.value for issue in self.issues],
        }


def _any_schema_accepts(locations: Sequence[TlvLocation]) -> bool:
    return any(chain_matches_schema(schema, locations) for schema in PADDED_SCHEMAS)


def resolve_header_len(
    packet: bytes | memoryview,
    fields: HeaderFields,
    schema: PacketSchema | None,
    available: int | None = None,
) -> tuple[int, list[DecodeIssue]]:
    """Decide between the 36- and 40-byte variants of the padded header layout.

    The rule table decides when a rule applies and its layout yields a
    consistent TLV chain (or neither layout does). When the rule's layout does
    not fit but the other one does, the data wins and a warning is returned.
    Without an applicable rule the layout whose chain fits is taken; if both
    fit, the out-of-box convention of the platform breaks the tie.

    Returns
    -------
    header_len : int
    issues : list of DecodeIssue
    """
    rule = header_len_rule(fields.platform, fields.version, schema)
    if schema is not None:

        def accept(locations: Sequence[TlvLocation]) -> bool:
            return chain_matches_schema(schema, locations)

    else:
        accept = _any_schema_accepts
    viable = infer_header_len(packet, 0, fields, accept, available)
    if rule is not None:
        if rule.header_len in viable or not viable:
            return rule.header_len, []
        chosen = viable[0]
        issue = DecodeIssue(
            IssueCode.HEADER_LENGTH_CONFLICT,
            f"{rule.description}, which implies a {rule.header_len}-byte header, but only "
            f"a {chosen}-byte header gives a consistent TLV chain",
            Severity.WARNING,
            fields.frame_number,
        )
        return chosen, [issue]
    if len(viable) == 1:
        return viable[0], []
    if fields.platform == Platform.XWR14XX:
        return SHORT_HEADER_LEN, []
    return LONG_HEADER_LEN, []


def identify_header(
    packet: bytes | bytearray | memoryview,
    config: DecoderConfig,
    *,
    available: int | None = None,
) -> HeaderFields:
    """Read the header of a packet in whichever layout fits it.

    A padded header and a 48-byte checksummed header both put the packet length
    at offset 12, and a 48-byte header can pass the padded checks when its
    processing-time words are small. The checksummed reading then wins when its
    checksum matches and its TLV chain ends exactly at the declared length. For
    a packet cut short (``available`` or the buffer ends early) the chain only
    has to run into the cut without overflowing the declared length.

    Raises
    ------
    DecodeError
        If no layout fits (or the forced schema's layout does not).
    """
    forced = config.schema.header_layout if config.schema is not None else None
    limits = {
        "max_packet_bytes": config.max_packet_bytes,
        "max_tlvs": config.max_tlvs,
        "strict": config.strict,
    }
    verdict = examine_header(packet, 0, layout=forced, **limits)
    if verdict.pending:
        raise DecodeError(f"only {len(packet)} bytes; the header needs at least {verdict.needed}")
    if verdict.fields is None:
        raise DecodeError(
            f"invalid header: {verdict.problem}",
            (DecodeIssue(verdict.code, verdict.problem or "", Severity.ERROR),),
        )
    fields = verdict.fields
    if (
        forced is None
        and fields.layout is HeaderLayout.PADDED
        and stored_checksum_matches(packet, 0, HeaderLayout.CHECKSUMMED_48.min_length)
    ):
        alternative = examine_header(packet, 0, layout=HeaderLayout.CHECKSUMMED_48, **limits)
        if alternative.fields is not None:
            candidate = alternative.fields
            have = min(len(packet), candidate.packet_length)
            if available is not None:
                have = min(have, available)
            walk = walk_tlv_chain(
                packet,
                0,
                HeaderLayout.CHECKSUMMED_48.min_length,
                candidate.tlv_count,
                candidate.packet_length,
                have,
                length_includes_header=True,
            )
            # An intact packet must end exactly where its TLVs do; a packet cut
            # short on the link cannot show that, but its chain must not overflow.
            if walk.truncated or (
                walk.problem is None and walk.end_offset == candidate.packet_length
            ):
                return candidate
    return fields


def decode_packet(
    packet: bytes | bytearray | memoryview,
    config: DecoderConfig | None = None,
    sensor: SensorConfig | None = None,
    *,
    available: int | None = None,
) -> RadarFrame:
    """Decode one packet.

    Parameters
    ----------
    packet : bytes-like
        Bytes starting at the sync word; normally exactly one packet. Fewer
        bytes than the header declares mean the packet was cut short.
    config : DecoderConfig, optional
        Decoder settings (schema, strictness, fall-back sizes ...).
    sensor : SensorConfig, optional
        Radar configuration for physical units.
    available : int, optional
        Treat only the first ``available`` bytes as present.

    Returns
    -------
    RadarFrame

    Raises
    ------
    DecodeError
        If the buffer does not start with a plausible header, or in strict
        mode for any error-level issue, a cut-short packet or excess padding.
    UnsupportedFormatError
        If the packet is intact but no known firmware family explains it
        (a :class:`DecodeError` too).

    Examples
    --------
    >>> from radar_stream_decoder.encoder import HeaderTemplate, encode_frame, encode_stats
    >>> from radar_stream_decoder.payloads import ProcessingStats
    >>> stats = ProcessingStats(1200, 300, 48000, 20, 35, 12)
    >>> packet = encode_frame(HeaderTemplate(frame_number=5, platform=0xA1642),
    ...                       [(6, encode_stats(stats))])
    >>> frame = decode_packet(packet)
    >>> frame.frame_number, frame.schema.value, frame.stats.active_frame_cpu_load_pct
    (5, 'oob-sdk2', 35)
    """
    config = config or DecoderConfig()
    data = bytes(packet)
    fields = identify_header(data, config, available=available)
    have = min(len(data), fields.packet_length)
    if available is not None:
        have = min(have, max(0, int(available)))
    data = data[:have]
    if have < fields.layout.min_length:
        raise DecodeError(f"only {have} bytes of a {fields.packet_length}-byte packet")
    cut_short = have < fields.packet_length

    issues: list[DecodeIssue] = []
    if fields.layout is HeaderLayout.PADDED:
        header_len, header_issues = resolve_header_len(data, fields, config.schema, have)
        issues.extend(header_issues)
    else:
        header_len = fields.layout.min_length
    if header_len > have:
        raise DecodeError(f"packet cut short inside its {header_len}-byte header")
    header = decode_frame_header(data, 0, header_len)
    walk = walk_tlv_chain(
        data,
        0,
        header_len,
        header.tlv_count,
        header.packet_length,
        have,
        length_includes_header=header.tlv_length_includes_header,
    )
    found = [(location.tlv_type, location.length) for location in walk.locations]
    if config.schema is not None:
        schema = config.schema
    elif walk.problem is not None and not walk.truncated and not found:
        # Nothing decodable: report the broken chain rather than guess a family.
        schema = _fallback_schema(header)
    else:
        schema = detect_schema(header, found)

    if walk.problem is not None:
        code = IssueCode.TRUNCATED_PACKET if walk.truncated else IssueCode.TLV_OVERFLOW
        issues.append(
            DecodeIssue(code, walk.problem, Severity.ERROR, header.frame_number, walk.end_offset)
        )
    elif cut_short:
        # Firmware never sends a short packet, so bytes were lost on the way --
        # possibly inside a payload, shifting the bytes after the loss. The chain
        # fitting proves nothing about where the loss happened.
        issues.append(
            DecodeIssue(
                IssueCode.SHORTENED_PACKET,
                f"packet holds {have} of its declared {header.packet_length} bytes; the TLV "
                "chain still fits, but the lost bytes may have been anywhere, so payload "
                "contents can be shifted",
                Severity.ERROR,
                header.frame_number,
            )
        )
    else:
        issues.extend(_trailing_bytes_issues(data, header, walk.end_offset, schema, config))

    ctx = DecodeContext(
        header=header,
        schema=schema,
        config=config,
        sensor=sensor,
        tlv_lengths=dict(reversed(found)),
        issues=issues,
    )
    if sensor is not None and sensor.platform is None:
        # A .cfg loaded without --platform counts every transmitter as azimuth,
        # but on the three-transmitter devices TX1 sits in elevation, which
        # changes the dB scale and the width of the azimuth heatmap. The packet
        # header names the device, so the platform is taken from there.
        ctx.sensor = dataclasses.replace(sensor, platform=ctx.platform)
    registry = schema_registry(schema)
    view = memoryview(data)
    payloads: dict[str, Any] = {}
    unknown: list[UnknownTlv] = []
    for location in walk.locations:
        payload = view[location.payload_offset : location.end]
        spec = registry.get(location.tlv_type)
        if spec is None:
            unknown.append(UnknownTlv(location.tlv_type, bytes(payload)))
            ctx.report(
                IssueCode.UNKNOWN_TLV,
                f"TLV type {location.tlv_type} ({location.length} bytes) is not part of "
                f"the {schema.value} schema",
                offset=location.payload_offset,
            )
            continue
        if spec.kind in payloads:
            ctx.report(
                IssueCode.DUPLICATE_TLV,
                f"second {spec.description} TLV ignored",
                offset=location.payload_offset,
            )
            continue
        decoded = spec.decode(payload, ctx)
        if decoded is not None:
            payloads[spec.kind] = decoded
    _enrich(payloads, ctx)

    frame = RadarFrame(
        header=header,
        schema=schema,
        tlv_types=tuple(tlv_type for tlv_type, _ in found),
        payloads=payloads,
        unknown_tlvs=unknown,
        issues=issues,
        raw=data if config.keep_raw_packets else b"",
        crc32=zlib.crc32(data),
        truncated=cut_short,
    )
    if config.strict and frame.has_errors:
        first = next(issue for issue in issues if issue.severity is Severity.ERROR)
        raise DecodeError(f"strict mode rejected frame {header.frame_number}: {first}", issues)
    return frame


def _fallback_schema(header: FrameHeader) -> PacketSchema:
    if header.layout is HeaderLayout.CHECKSUMMED_48:
        return PacketSchema.PEOPLE_COUNTING_3D
    if header.layout is HeaderLayout.CHECKSUMMED_52:
        return PacketSchema.PEOPLE_COUNTING_2D
    return PacketSchema.OOB_SDK3 if header.version.major >= 3 else PacketSchema.OOB_SDK2


def _trailing_bytes_issues(
    data: bytes,
    header: FrameHeader,
    end: int,
    schema: PacketSchema,
    config: DecoderConfig,
) -> list[DecodeIssue]:
    """Check what follows the last TLV of a complete packet."""
    trailing = header.packet_length - end
    issues: list[DecodeIssue] = []
    excess = Severity.ERROR if config.strict else Severity.WARNING
    if header.layout.checksummed:
        if trailing:
            issues.append(
                DecodeIssue(
                    IssueCode.EXCESS_PADDING,
                    f"{trailing} bytes follow the last TLV, but this firmware does not pad: "
                    "its packet length is the header plus the TLVs",
                    excess,
                    header.frame_number,
                    end,
                )
            )
        return issues
    filler = data[end : header.packet_length]
    if len(set(filler)) > 1:
        if schema.padding_is_uninitialised:
            issues.append(
                DecodeIssue(
                    IssueCode.UNEVEN_PADDING,
                    f"the {trailing} padding bytes differ; SDK 3.x firmware sends its padding "
                    "from an uninitialised buffer, so this is not treated as damage",
                    Severity.INFO,
                    header.frame_number,
                    end,
                )
            )
        else:
            issues.append(
                DecodeIssue(
                    IssueCode.CORRUPT_PADDING,
                    f"the {trailing} padding bytes are not one repeated value; bytes from "
                    "outside this packet (a cut on the link followed by other data) "
                    "probably fill part of it",
                    Severity.ERROR,
                    header.frame_number,
                    end,
                )
            )
    if trailing >= PACKET_ALIGNMENT:
        issues.append(
            DecodeIssue(
                IssueCode.EXCESS_PADDING,
                f"{trailing} bytes follow the last TLV; the firmware pads to a multiple "
                f"of {PACKET_ALIGNMENT} and never adds a whole block",
                excess,
                header.frame_number,
                end,
            )
        )
    return issues


def _enrich(payloads: dict[str, Any], ctx: DecodeContext) -> None:
    """Fill in what needs more than one TLV or the radar configuration."""
    scale = ctx.db_per_lsb()
    for kind in (PayloadKind.RANGE_PROFILE, PayloadKind.NOISE_PROFILE, PayloadKind.RANGE_DOPPLER):
        array = payloads.get(kind)
        if array is not None:
            array.db_per_lsb = scale

    points: PointCloud | None = payloads.get(PayloadKind.POINTS)
    side: PointSideInfo | None = payloads.get(PayloadKind.POINT_SIDE_INFO)
    if points is not None and side is not None:
        if len(side) == len(points):
            points.snr_db = side.snr_db.copy()
            points.noise_db = side.noise_db.copy()
            points.snr_method = SnrMethod.SIDE_INFO
        else:
            ctx.report(
                IssueCode.SIDE_INFO_MISMATCH,
                f"side info for {len(side)} points accompanies {len(points)} points; not merged",
                Severity.ERROR,
            )
    elif points is not None and len(points) and points.range_idx is not None and scale:
        try:
            estimate = estimate_point_snr(
                points,
                rd_map=payloads.get(PayloadKind.RANGE_DOPPLER),
                range_profile=payloads.get(PayloadKind.RANGE_PROFILE),
                noise_profile=payloads.get(PayloadKind.NOISE_PROFILE),
                db_per_lsb=scale,
                guard_cells=ctx.config.snr_guard_cells,
                train_cells=ctx.config.snr_train_cells,
                estimator=ctx.config.snr_estimator,
                log_bias_correction=ctx.config.snr_log_bias_correction,
            )
        except ValueError as exc:
            # The guard cells leave no training cells on a short Doppler axis
            # (4 bins with the default 2 guard cells per side). The points
            # stay usable; only their SNR is unknown.
            ctx.report(IssueCode.INVALID_VALUE, f"point SNR not estimated: {exc}")
        else:
            points.snr_db = estimate.snr_db
            points.snr_method = estimate.method

    targets: TargetList | None = payloads.get(PayloadKind.TARGETS)
    heights: TargetHeights | None = payloads.get(PayloadKind.TARGET_HEIGHTS)
    if targets is not None and heights is not None:
        matched = merge_target_heights(targets, heights)
        if matched < len(heights):
            ctx.report(
                IssueCode.ASSOCIATION_MISMATCH,
                f"{len(heights) - matched} target heights name targets the target list "
                "does not hold",
            )

    segment: RangeSegment | None = payloads.get(PayloadKind.RANGE_SEGMENT)
    if segment is not None:
        summary: VitalSignsSummary | None = payloads.get(PayloadKind.VITAL_SIGNS)
        if summary is None:
            ctx.report(
                IssueCode.MISSING_RANGE_START,
                "range segment without a summary: its first range bin is unknown",
            )
        else:
            segment.start_bin = summary.first_range_bin
            if len(segment) != summary.segment_bins:
                ctx.report(
                    IssueCode.LENGTH_MISMATCH,
                    f"summary announces bins {summary.first_range_bin}..{summary.last_range_bin} "
                    f"({summary.segment_bins}) but the segment holds {len(segment)}",
                    Severity.ERROR,
                )
