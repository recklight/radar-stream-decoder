"""Packet encoder: the byte-exact inverse of the decoder.

Every payload decoder has an ``encode_*`` counterpart here, and
:func:`encode_frame` wraps TLVs into a packet with the header layout of the
chosen firmware family: the padded layout with a length padded to a multiple
of 32 and a selectable padding byte (``0x00``; SDK 2.0 xWR16xx debug builds pad
with ``0x0F``), or a checksummed people counting layout whose TLV lengths count
their type/length pair and whose packet is not padded.

The encoder exists for testing and demonstration: the synthetic scenes, the
fault-injection tests, the GUI's demo mode and the round-trip tests all build
their packets with it, so no recorded data is needed anywhere. Tests that pin
the byte layout against TI's firmware build their reference packets with
``struct.pack`` instead, so encoder and decoder cannot share a mistake.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .angle_map import encode_complex_iq
from .constants import MAGIC_WORD, PACKET_ALIGNMENT, TLV_HEADER_LEN, Platform
from .header import HeaderLayout, HeaderTiming, SdkVersion, header_checksum
from .payloads import (
    FLOAT_POINT_DTYPE,
    QFORMAT_POINT_DTYPE,
    SIDE_INFO_DTYPE,
    UNSIGNED_DOPPLER_POINT_DTYPE,
    PointCloud,
    ProcessingStats,
    TemperatureReport,
)
from .tracking import (
    COMPRESSED_POINT_DTYPE,
    POLAR_POINT_DTYPE,
    TARGET_2D_DTYPE,
    TARGET_3D_COMPACT_DTYPE,
    TARGET_3D_DTYPE,
    PointUnits,
    TargetHeights,
    TargetList,
)
from .vital_signs import VitalSignsSummary

_TLV_HEADER = struct.Struct("<2I")
_HEADER_WORDS = struct.Struct("<7I")
_SUBFRAME_WORD = struct.Struct("<I")
_HEADER_3D_WORDS = struct.Struct("<9IHH")
_HEADER_2D_WORDS = struct.Struct("<10IHH")

#: SDK release written into synthetic packets of each firmware family.
DEFAULT_VERSIONS: dict[str, SdkVersion] = {
    "oob-sdk2-14xx": SdkVersion(1, 2, 0, 0),
    "oob-sdk2": SdkVersion(2, 0, 0, 4),
    "oob-sdk3": SdkVersion(3, 5, 0, 4),
    "vital-signs": SdkVersion(1, 2, 0, 5),
    "people-counting-3d": SdkVersion(3, 5, 0, 4),
    "people-counting-2d": SdkVersion(1, 2, 0, 5),
    "people-tracking": SdkVersion(3, 6, 0, 0),
}

#: Point units the 3D People Counting lab writes (``mss_main.c``, Industrial Toolbox 4.6.0).
DEFAULT_POINT_UNITS = PointUnits(
    elevation_rad=0.01, azimuth_rad=0.01, doppler_mps=0.00028, range_m=0.00025, snr=0.04
)


@dataclass(frozen=True)
class HeaderTemplate:
    """Header fields the encoder cannot derive from the TLVs.

    The packet length and TLV count are filled in by :func:`encode_frame`.
    ``subframe_number=None`` selects the 36-byte variant of the padded layout;
    ``layout`` selects a checksummed people counting header instead (which has
    no point count, and in the 48-byte layout no timestamp either).
    """

    version: SdkVersion = field(default_factory=lambda: DEFAULT_VERSIONS["oob-sdk2"])
    platform: int = Platform.XWR16XX
    frame_number: int = 0
    cpu_cycles: int = 0
    declared_point_count: int = 0
    subframe_number: int | None = 0
    layout: HeaderLayout = HeaderLayout.PADDED
    timing: HeaderTiming = field(default_factory=lambda: HeaderTiming(0, 0, 0, 0, 0))

    @classmethod
    def for_schema(
        cls,
        schema: str,
        platform: int,
        frame_number: int = 0,
        cpu_cycles: int = 0,
        declared_point_count: int = 0,
    ) -> HeaderTemplate:
        """Template with the version and header layout a firmware family uses.

        Examples
        --------
        >>> HeaderTemplate.for_schema("oob-sdk2", 0xA1443).subframe_number is None
        True
        >>> str(HeaderTemplate.for_schema("vital-signs", 0xA1443).version)
        '1.2.0.5'
        """
        name = str(getattr(schema, "value", schema))
        layout = {
            "people-counting-3d": HeaderLayout.CHECKSUMMED_48,
            "people-counting-2d": HeaderLayout.CHECKSUMMED_52,
        }.get(name, HeaderLayout.PADDED)
        if name == "oob-sdk2" and platform == Platform.XWR14XX:
            version, subframe = DEFAULT_VERSIONS["oob-sdk2-14xx"], None
        else:
            version, subframe = DEFAULT_VERSIONS[name], 0
        return cls(
            version, platform, frame_number, cpu_cycles, declared_point_count, subframe, layout
        )


def encode_frame(
    template: HeaderTemplate,
    tlvs: Sequence[tuple[int, bytes]],
    padding_byte: int = 0x00,
) -> bytes:
    """Assemble a packet: header, TLVs and padding to a multiple of 32 bytes.

    Parameters
    ----------
    template : HeaderTemplate
        Header fields; the layout follows ``template.subframe_number``.
    tlvs : sequence of (int, bytes)
        TLV type and payload, in packet order.
    padding_byte : int, optional
        Value of the padding bytes.

    Returns
    -------
    bytes

    Examples
    --------
    >>> packet = encode_frame(HeaderTemplate(frame_number=3), [(6, bytes(24))])
    >>> len(packet), packet[:8] == MAGIC_WORD
    (96, True)
    """
    if not 0 <= padding_byte <= 0xFF:
        raise ValueError(f"padding_byte must be a byte value, got {padding_byte}")
    if template.layout.checksummed:
        return _encode_checksummed_frame(template, tlvs)
    header_len = 36 if template.subframe_number is None else 40
    body = b"".join(_TLV_HEADER.pack(int(t), len(p)) + bytes(p) for t, p in tlvs)
    unpadded = header_len + len(body)
    total = -(-unpadded // PACKET_ALIGNMENT) * PACKET_ALIGNMENT
    header = MAGIC_WORD + _HEADER_WORDS.pack(
        template.version.to_word(),
        total,
        int(template.platform),
        int(template.frame_number) & 0xFFFFFFFF,
        int(template.cpu_cycles) & 0xFFFFFFFF,
        int(template.declared_point_count) & 0xFFFFFFFF,
        len(tlvs),
    )
    if template.subframe_number is not None:
        header += _SUBFRAME_WORD.pack(int(template.subframe_number))
    return header + body + bytes([padding_byte]) * (total - unpadded)


def _encode_checksummed_frame(template: HeaderTemplate, tlvs: Sequence[tuple[int, bytes]]) -> bytes:
    body = b"".join(_TLV_HEADER.pack(int(t), TLV_HEADER_LEN + len(p)) + bytes(p) for t, p in tlvs)
    timing = template.timing
    subframe = int(template.subframe_number or 0)
    if template.layout is HeaderLayout.CHECKSUMMED_48:
        total = 48 + len(body)
        words = _HEADER_3D_WORDS.pack(
            template.version.to_word(),
            total,
            int(template.platform),
            int(template.frame_number) & 0xFFFFFFFF,
            subframe,
            timing.chirp_margin_us,
            timing.frame_processing_time_us or 0,
            timing.tracking_time_us,
            timing.uart_time_us,
            len(tlvs),
            0,
        )
    else:
        total = 52 + len(body)
        words = _HEADER_2D_WORDS.pack(
            template.version.to_word(),
            int(template.platform),
            int(template.cpu_cycles) & 0xFFFFFFFF,
            total,
            int(template.frame_number) & 0xFFFFFFFF,
            subframe,
            timing.chirp_margin_us,
            timing.frame_margin_us or 0,
            timing.tracking_time_us,
            timing.uart_time_us,
            len(tlvs),
            0,
        )
    header = MAGIC_WORD + words
    header = header[:-2] + struct.pack("<H", header_checksum(header))
    return header + body


def _int16(values: ArrayLike) -> np.ndarray:
    return np.clip(np.rint(np.asarray(values, dtype=np.float64)), -32768, 32767)


def _uint16(values: ArrayLike) -> np.ndarray:
    return np.clip(np.rint(np.asarray(values, dtype=np.float64)), 0, 65535)


def encode_points_qformat(
    points: PointCloud,
    q_format: int | None = None,
    *,
    unsigned_doppler_bins: int | None = None,
) -> bytes:
    """Encode an SDK 1.x/2.x point list.

    Parameters
    ----------
    points : PointCloud
        Points; missing range/Doppler indices and peaks are written as zero.
    q_format : int, optional
        Coordinate format; defaults to the cloud's own ``q_format`` or 9.
    unsigned_doppler_bins : int, optional
        Write SDK 1.0 style unsigned Doppler bins for this FFT size.
    """
    q = (
        q_format
        if q_format is not None
        else (points.q_format if points.q_format is not None else 9)
    )
    n = len(points)
    dtype = UNSIGNED_DOPPLER_POINT_DTYPE if unsigned_doppler_bins else QFORMAT_POINT_DTYPE
    records = np.zeros(n, dtype=dtype)
    scale = float(1 << q)
    records["x"] = _int16(points.x_m * scale)
    records["y"] = _int16(points.y_m * scale)
    records["z"] = _int16(points.z_m * scale)
    if points.range_idx is not None:
        records["range_idx"] = _uint16(points.range_idx)
    if points.doppler_idx is not None:
        doppler = points.doppler_idx.astype(np.int64)
        if unsigned_doppler_bins:
            records["doppler_idx"] = doppler % unsigned_doppler_bins
        else:
            records["doppler_idx"] = _int16(doppler)
    if points.peak is not None:
        records["peak"] = _uint16(points.peak)
    return struct.pack("<2H", n, q) + records.tobytes()


def encode_points_float(points: PointCloud) -> bytes:
    """Encode an SDK 3.x point list (unknown velocities are written as zero)."""
    records = np.zeros(len(points), dtype=FLOAT_POINT_DTYPE)
    records["x"], records["y"], records["z"] = points.x_m, points.y_m, points.z_m
    records["velocity"] = np.nan_to_num(points.velocity_mps, nan=0.0)
    return records.tobytes()


def encode_side_info(snr_db: ArrayLike, noise_db: ArrayLike) -> bytes:
    """Encode SDK 3.x side info, rounding to 0.1 dB."""
    snr = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    noise = np.asarray(noise_db, dtype=np.float64).reshape(-1)
    if snr.size != noise.size:
        raise ValueError(f"{snr.size} SNR values for {noise.size} noise values")
    records = np.zeros(snr.size, dtype=SIDE_INFO_DTYPE)
    records["snr"] = _int16(snr * 10.0)
    records["noise"] = _int16(noise * 10.0)
    return records.tobytes()


def encode_magnitude_profile(raw: ArrayLike) -> bytes:
    """Encode a range or noise profile of raw log2-magnitude words."""
    return _uint16(raw).astype("<u2").tobytes()


def encode_range_doppler(raw: ArrayLike) -> bytes:
    """Encode a detection matrix of shape ``(range bins, Doppler bins)`` range-major."""
    matrix = np.asarray(raw)
    if matrix.ndim != 2:
        raise ValueError(f"expected a (range, Doppler) matrix, got shape {matrix.shape}")
    return _uint16(matrix).astype("<u2").tobytes(order="C")


def encode_azimuth_heatmap(cube: ArrayLike) -> bytes:
    """Encode ``(range bins, antennas)`` complex samples, imaginary part first."""
    samples = np.asarray(cube)
    if samples.ndim != 2:
        raise ValueError(f"expected a (range, antenna) array, got shape {samples.shape}")
    return encode_complex_iq(samples.reshape(-1), order="imag_real")


def encode_stats(stats: ProcessingStats) -> bytes:
    """Encode the six processing statistics."""
    return struct.pack("<6I", *(int(v) for v in stats.as_dict().values()))


def encode_temperature(report: TemperatureReport) -> bytes:
    """Encode an SDK 3.x temperature report."""
    return struct.pack(
        "<iI10h",
        report.status,
        report.time_ms,
        *report.rx_c,
        *report.tx_c,
        report.power_management_c,
        *report.digital_c,
    )


def encode_vital_summary(summary: VitalSignsSummary) -> bytes:
    """Encode the 128-byte Vital Signs summary."""
    return summary.to_record().tobytes()


def encode_range_segment(values: ArrayLike) -> bytes:
    """Encode a Vital Signs range segment, real part first."""
    return encode_complex_iq(values, order="real_imag")


# -- people counting and people tracking ------------------------------------------------------


def encode_compressed_points(
    range_m: ArrayLike,
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
    doppler_mps: ArrayLike,
    snr: ArrayLike,
    units: PointUnits = DEFAULT_POINT_UNITS,
) -> bytes:
    """Encode a compressed spherical point cloud: the units, then 8-byte points.

    Values are divided by their unit and rounded, as the firmware does.
    """
    rng = np.asarray(range_m, dtype=np.float64).reshape(-1)
    azimuth = np.asarray(azimuth_rad, dtype=np.float64).reshape(-1)
    elevation = np.asarray(elevation_rad, dtype=np.float64).reshape(-1)
    doppler = np.asarray(doppler_mps, dtype=np.float64).reshape(-1)
    strength = np.asarray(snr, dtype=np.float64).reshape(-1)
    records = np.zeros(rng.size, dtype=COMPRESSED_POINT_DTYPE)
    records["elevation"] = np.clip(np.rint(elevation / units.elevation_rad), -128, 127)
    records["azimuth"] = np.clip(np.rint(azimuth / units.azimuth_rad), -128, 127)
    records["doppler"] = _int16(doppler / units.doppler_mps)
    records["range"] = _uint16(rng / units.range_m)
    records["snr"] = _uint16(strength / units.snr)
    head = struct.pack(
        "<5f", units.elevation_rad, units.azimuth_rad, units.doppler_mps, units.range_m, units.snr
    )
    return head + records.tobytes()


def encode_polar_points(
    range_m: ArrayLike, azimuth_rad: ArrayLike, doppler_mps: ArrayLike, snr: ArrayLike
) -> bytes:
    """Encode the 2D people counting point cloud (``float32`` range, azimuth, Doppler, SNR)."""
    rng = np.asarray(range_m, dtype=np.float64).reshape(-1)
    records = np.zeros(rng.size, dtype=POLAR_POINT_DTYPE)
    records["range"] = rng
    records["azimuth"] = np.asarray(azimuth_rad, dtype=np.float64).reshape(-1)
    records["doppler"] = np.asarray(doppler_mps, dtype=np.float64).reshape(-1)
    records["snr"] = np.asarray(snr, dtype=np.float64).reshape(-1)
    return records.tobytes()


def encode_target_list(targets: TargetList, record: str = "3d") -> bytes:
    """Encode tracker targets as 112-byte (``"3d"``), 40-byte (``"3d-compact"``) or
    68-byte (``"2d"``) records; missing covariance, gain and confidence are zero."""
    n = len(targets)
    covariance = targets.covariance
    gain = targets.gating_gain if targets.gating_gain is not None else np.zeros(n)
    if record == "3d":
        records = np.zeros(n, dtype=TARGET_3D_DTYPE)
        records["position"] = targets.position_m
        records["velocity"] = targets.velocity_mps
        records["acceleration"] = targets.acceleration_mps2
        if covariance is not None:
            records["covariance"] = covariance.reshape(n, 16)
        records["gating_gain"] = gain
        records["confidence"] = (
            targets.confidence if targets.confidence is not None else np.zeros(n)
        )
    elif record == "3d-compact":
        records = np.zeros(n, dtype=TARGET_3D_COMPACT_DTYPE)
        for axis, name in enumerate("xyz"):
            records[f"pos_{name}"] = targets.position_m[:, axis]
            records[f"vel_{name}"] = targets.velocity_mps[:, axis]
            records[f"acc_{name}"] = targets.acceleration_mps2[:, axis]
    elif record == "2d":
        records = np.zeros(n, dtype=TARGET_2D_DTYPE)
        records["position"] = targets.position_m[:, :2]
        records["velocity"] = targets.velocity_mps[:, :2]
        records["acceleration"] = targets.acceleration_mps2[:, :2]
        if covariance is not None:
            records["covariance"] = covariance.reshape(n, 9)
        records["gating_gain"] = gain
    else:
        raise ValueError(f"record must be '3d', '3d-compact' or '2d', got {record!r}")
    records["target_id"] = targets.target_id
    return records.tobytes()


def encode_target_association(target_ids: ArrayLike) -> bytes:
    """Encode one ``uint8`` target ID per point of the previous frame."""
    values = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    if values.size and (values.min() < 0 or values.max() > 255):
        raise ValueError("per-point target IDs must fit in one byte")
    return values.astype(np.uint8).tobytes()


def encode_target_heights(heights: TargetHeights) -> bytes:
    """Encode 12-byte ``{uint32 tid, float32 maxZ, minZ}`` records."""
    return b"".join(
        struct.pack("<I2f", int(tid), float(top), float(bottom))
        for tid, top, bottom in zip(
            heights.target_id, heights.max_z_m, heights.min_z_m, strict=True
        )
    )


def encode_presence(occupied: bool | int) -> bytes:
    """Encode the ``uint32`` presence word."""
    return struct.pack("<I", int(occupied))
