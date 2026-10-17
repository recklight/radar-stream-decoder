"""Payload data models and the decoders of the out-of-box demo TLVs.

Every decoder receives exactly the bytes its TLV's length field covers (a
``memoryview`` slice, never the rest of the packet) together with a
:class:`DecodeContext` that knows the header, the schema, the decoder settings
and, when available, the radar configuration. A payload whose length does not
match its contents is decoded as far as the bytes allow and reported as a
:class:`~radar_stream_decoder.errors.DecodeIssue`; nothing is read past the
slice.

Formats (little-endian):

Points, SDK 1.x/2.x
    ``uint16 count, uint16 xyzQFormat`` followed by ``count`` records
    ``{uint16 rangeIdx, int16 dopplerIdx, uint16 peakVal, int16 x, int16 y,
    int16 z}``; coordinates in metres are ``value / 2**Q``. SDK 1.0 firmware
    writes the Doppler index unsigned.
Points, SDK 3.x
    Records ``{float32 x, y, z, velocity}``; the optional side info TLV adds
    ``{int16 snr, int16 noise}`` per point in 0.1 dB.
Range / noise profile
    One ``uint16`` log2 magnitude per range bin, taken at zero Doppler and at
    the highest Doppler bin respectively.
Range-Doppler heatmap
    The whole detection matrix, range-major ``uint16`` log2 magnitudes (shape
    ``(range bins, Doppler bins)``).
Statistics
    Six ``uint32``: processing time, transmit time, frame and chirp margins
    (microseconds), active and inter-frame CPU load (percent).
Temperature, SDK 3.x
    ``int32 status, uint32 time_ms`` and ten ``int16`` sensor readings in
    degrees Celsius.

The people counting and people tracking point clouds, target lists and
per-point target IDs live in :mod:`radar_stream_decoder.tracking`; they decode
into the same :class:`PointCloud`.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import NDArray

from .constants import DB_PER_LOG2_UNIT, DemoTlvType
from .errors import ConfigError, DecodeIssue, IssueCode, Severity
from .units import (
    cartesian_to_spherical,
    log2_magnitude_scale,
    range_index_to_m,
    signed_doppler,
)

if TYPE_CHECKING:
    from .config import DecoderConfig
    from .header import FrameHeader
    from .schema import PacketSchema
    from .sensor_config import SensorConfig

_T = TypeVar("_T")


class PayloadKind(str, Enum):
    """Keys under which decoded payloads are stored in a frame."""

    POINTS = "points"
    RANGE_PROFILE = "range_profile"
    NOISE_PROFILE = "noise_profile"
    AZIMUTH_HEATMAP = "azimuth_heatmap"
    RANGE_DOPPLER = "range_doppler"
    STATS = "stats"
    POINT_SIDE_INFO = "point_side_info"
    AZIMUTH_ELEVATION_HEATMAP = "azimuth_elevation_heatmap"
    TEMPERATURE = "temperature"
    VITAL_SIGNS = "vital_signs"
    RANGE_SEGMENT = "range_segment"
    TARGETS = "targets"
    TARGET_ASSOCIATION = "target_association"
    TARGET_HEIGHTS = "target_heights"
    PRESENCE = "presence"


class SnrMethod(str, Enum):
    """Where a point's SNR came from."""

    DETECTION_MATRIX = "detection_matrix"
    NOISE_PROFILE = "noise_profile"
    SIDE_INFO = "side_info"
    UNAVAILABLE = "unavailable"


@dataclass
class DecodeContext:
    """Everything a payload decoder may need besides its own bytes.

    The ``num_*`` and ``*_m`` methods resolve a quantity from the best source
    available: the radar configuration first, then other TLVs of the same
    packet, then the decoder's fall-back settings. They return ``None`` when no
    source knows the answer.
    """

    header: FrameHeader
    schema: PacketSchema
    config: DecoderConfig
    sensor: SensorConfig | None = None
    tlv_lengths: dict[int, int] = field(default_factory=dict)
    issues: list[DecodeIssue] = field(default_factory=list)

    def report(
        self,
        code: IssueCode,
        message: str,
        severity: Severity = Severity.WARNING,
        offset: int | None = None,
    ) -> None:
        """Record an issue against the current frame."""
        self.issues.append(
            DecodeIssue(code, message, severity, frame=self.header.frame_number, offset=offset)
        )

    @property
    def platform(self) -> int:
        """Platform code, the decoder override taking precedence over the header."""
        if self.config.platform is not None:
            return int(self.config.platform)
        return self.header.platform

    def from_sensor(self, getter: Callable[[SensorConfig], _T]) -> _T | None:
        """Evaluate ``getter`` on the radar configuration, ``None`` if it cannot answer."""
        if self.sensor is None:
            return None
        try:
            return getter(self.sensor)
        except (ConfigError, ZeroDivisionError):
            return None

    def range_bin_candidates(self) -> list[int]:
        """Possible range-bin counts, most trustworthy first, without duplicates."""
        candidates: list[int | None] = [self.from_sensor(lambda s: s.num_range_bins)]
        if self.schema != "vital-signs":
            for tlv_type in (DemoTlvType.RANGE_PROFILE, DemoTlvType.NOISE_PROFILE):
                length = self.tlv_lengths.get(int(tlv_type))
                if length:
                    candidates.append(length // 2)
        candidates.append(self.config.num_range_bins)
        result: list[int] = []
        for value in candidates:
            if value and value not in result:
                result.append(int(value))
        return result

    def num_range_bins(self) -> int | None:
        """Range-bin count from the best available source."""
        candidates = self.range_bin_candidates()
        return candidates[0] if candidates else None

    def num_doppler_bins(self) -> int | None:
        """Doppler-bin count from the radar configuration or the settings."""
        value = self.from_sensor(lambda s: s.num_doppler_bins)
        return value if value else self.config.num_doppler_bins

    def doppler_bin_mps(self) -> float | None:
        """Velocity of one Doppler bin (needs the radar configuration)."""
        return self.from_sensor(lambda s: s.doppler_bin_mps)

    def range_bin_m(self) -> float | None:
        """Range of one range bin (needs the radar configuration)."""
        return self.from_sensor(lambda s: s.range_bin_m)

    def range_bias_m(self) -> float:
        """Range bias, the decoder override taking precedence."""
        if self.config.range_bias_m is not None:
            return float(self.config.range_bias_m)
        value = self.from_sensor(lambda s: s.range_bias_m)
        return float(value) if value is not None else 0.0

    def num_azimuth_antennas(self) -> int | None:
        """Virtual antennas of the azimuth array."""
        value = self.from_sensor(lambda s: s.num_virtual_antennas_azimuth)
        return value if value else self.config.num_virtual_antennas

    def num_virtual_antennas(self) -> int | None:
        """All virtual antennas."""
        value = self.from_sensor(lambda s: s.num_virtual_antennas)
        return value if value else self.config.num_virtual_antennas

    def unsigned_doppler(self) -> bool:
        """Whether Doppler indices are written unsigned (SDK 1.0 firmware)."""
        if self.config.unsigned_doppler is not None:
            return bool(self.config.unsigned_doppler)
        return self.header.version.major_minor < (1, 1)

    def db_per_lsb(self) -> float | None:
        """Decibels of one log2-magnitude step, or ``None`` when it cannot be known.

        Summed magnitudes (xWR16xx, SDK 3.x) need the antenna counts, which come
        from the radar configuration or ``num_virtual_antennas``. Firmware
        families without log2-magnitude arrays have no scale.
        """
        if self.config.db_per_lsb is not None:
            return float(self.config.db_per_lsb)
        if self.schema not in ("oob-sdk2", "oob-sdk3"):
            return None
        num_rx = self.from_sensor(lambda s: s.num_rx)
        num_tx = self.from_sensor(lambda s: s.num_tx)
        elevation = bool(self.from_sensor(lambda s: s.num_tx_elevation))
        if not num_rx or not num_tx:
            virtual = self.config.num_virtual_antennas
            num_rx, num_tx = (virtual, 1) if virtual else (None, None)
        try:
            scale = log2_magnitude_scale(
                self.platform,
                self.schema,
                num_rx,
                num_tx,
                elevation,
                fraction_bits=self.config.log2_fraction_bits,
                combining=self.config.log2_combining,
            )
        except ValueError:
            return None
        return scale * DB_PER_LOG2_UNIT


# -- points -------------------------------------------------------------------------------

#: Point record of SDK 1.1-2.x firmware (12 bytes).
QFORMAT_POINT_DTYPE = np.dtype(
    [
        ("range_idx", "<u2"),
        ("doppler_idx", "<i2"),
        ("peak", "<u2"),
        ("x", "<i2"),
        ("y", "<i2"),
        ("z", "<i2"),
    ]
)

#: The same record as SDK 1.0 firmware writes it, with an unsigned Doppler bin.
UNSIGNED_DOPPLER_POINT_DTYPE = np.dtype(
    [
        ("range_idx", "<u2"),
        ("doppler_idx", "<u2"),
        ("peak", "<u2"),
        ("x", "<i2"),
        ("y", "<i2"),
        ("z", "<i2"),
    ]
)

#: Point record of SDK 3.x firmware (16 bytes).
FLOAT_POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("velocity", "<f4")])

#: Per-point side information of SDK 3.x firmware (4 bytes, 0.1 dB units).
#:
#: TI documents and firmware disagree on the type: the Radar Toolbox UART page
#: and the SDK's ``parser_mmw_demo.py`` read both values as ``uint16``, while
#: ``DPIF_PointCloudSideInfo`` in ``dpif_pointcloud.h`` of TI mmWave SDK
#: 03.05.00.04 declares ``int16_t snr`` and ``int16_t noise``. The decoder
#: follows the firmware (the two readings differ only for negative values).
SIDE_INFO_DTYPE = np.dtype([("snr", "<i2"), ("noise", "<i2")])

assert QFORMAT_POINT_DTYPE.itemsize == 12
assert FLOAT_POINT_DTYPE.itemsize == 16
assert SIDE_INFO_DTYPE.itemsize == 4

_POINT_DESCRIPTOR = struct.Struct("<2H")
_MAX_Q_FORMAT = 30

#: Columns of :meth:`PointCloud.to_records` (all float64; NaN where unknown).
POINT_RECORD_FIELDS: tuple[str, ...] = (
    "x_m",
    "y_m",
    "z_m",
    "range_m",
    "azimuth_deg",
    "elevation_deg",
    "range_idx",
    "doppler_idx",
    "velocity_mps",
    "peak",
    "snr_db",
    "noise_db",
    "snr_linear",
    "target_id",
)


def _float_column(values: object, n: int, name: str) -> NDArray[np.float64]:
    if values is None:
        return np.full(n, np.nan)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != n:
        raise ValueError(f"{name} has {array.size} entries for {n} points")
    return array


def _optional_column(values: object, n: int, name: str, dtype: type) -> NDArray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.size != n:
        raise ValueError(f"{name} has {array.size} entries for {n} points")
    return array


@dataclass
class PointCloud:
    """Detected points of one frame as parallel NumPy columns.

    Attributes
    ----------
    x_m, y_m, z_m : numpy.ndarray
        Coordinates in metres (``y`` along the boresight).
    velocity_mps : numpy.ndarray
        Radial velocity; NaN when the radar configuration is unknown (SDK 2.x)
        or always filled (SDK 3.x).
    range_idx, doppler_idx : numpy.ndarray or None
        Bin indices (SDK 1.x/2.x only); the Doppler index is signed.
    peak : numpy.ndarray or None
        Peak value written by the firmware (SDK 1.x/2.x only).
    snr_db, noise_db : numpy.ndarray
        Per-point SNR and noise level; NaN when unavailable.
    snr_linear : numpy.ndarray or None
        SNR as the people counting and tracking firmware reports it, a ratio
        (TI documents it as linear), without conversion to decibels.
    target_id : numpy.ndarray or None
        Tracker target each point was associated with (0-249), or 253 / 254 /
        255 for points left unassociated (SNR too weak, outside the boundary,
        noise). Filled from the next frame's association TLV by
        :func:`radar_stream_decoder.tracking.link_target_associations`.
    q_format : int or None
        Fixed-point format the coordinates were sent in.
    snr_method : SnrMethod
        Source of ``snr_db``.
    """

    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    z_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64] | None = None
    range_idx: NDArray[np.int32] | None = None
    doppler_idx: NDArray[np.int32] | None = None
    peak: NDArray[np.float64] | None = None
    snr_db: NDArray[np.float64] | None = None
    noise_db: NDArray[np.float64] | None = None
    q_format: int | None = None
    snr_method: SnrMethod = SnrMethod.UNAVAILABLE
    snr_linear: NDArray[np.float64] | None = None
    target_id: NDArray[np.int32] | None = None

    def __post_init__(self) -> None:
        self.x_m = np.asarray(self.x_m, dtype=np.float64).reshape(-1)
        n = self.x_m.size
        self.y_m = _float_column(self.y_m, n, "y_m")
        self.z_m = _float_column(self.z_m, n, "z_m")
        self.velocity_mps = _float_column(self.velocity_mps, n, "velocity_mps")
        self.snr_db = _float_column(self.snr_db, n, "snr_db")
        self.noise_db = _float_column(self.noise_db, n, "noise_db")
        self.range_idx = _optional_column(self.range_idx, n, "range_idx", np.int32)
        self.doppler_idx = _optional_column(self.doppler_idx, n, "doppler_idx", np.int32)
        self.peak = _optional_column(self.peak, n, "peak", np.float64)
        self.snr_linear = _optional_column(self.snr_linear, n, "snr_linear", np.float64)
        self.target_id = _optional_column(self.target_id, n, "target_id", np.int32)
        self.snr_method = SnrMethod(self.snr_method)

    @classmethod
    def empty(cls) -> PointCloud:
        """A cloud without points."""
        nothing = np.zeros(0)
        return cls(nothing, nothing, nothing)

    def __len__(self) -> int:
        return int(self.x_m.size)

    @property
    def range_m(self) -> NDArray[np.float64]:
        """Euclidean distance from the sensor."""
        return cartesian_to_spherical(self.x_m, self.y_m, self.z_m)[0]

    @property
    def azimuth_deg(self) -> NDArray[np.float64]:
        """Angle from the boresight towards ``+x``."""
        return cartesian_to_spherical(self.x_m, self.y_m, self.z_m)[1]

    @property
    def elevation_deg(self) -> NDArray[np.float64]:
        """Angle above the ``x``-``y`` plane."""
        return cartesian_to_spherical(self.x_m, self.y_m, self.z_m)[2]

    def range_from_index_m(self, range_bin_m: float, range_bias_m: float = 0.0) -> NDArray:
        """Range from the bin index, ``range_idx * bin - bias`` (SDK 1.x/2.x only)."""
        if self.range_idx is None:
            raise ValueError("this point format carries no range index")
        return range_index_to_m(self.range_idx, range_bin_m, range_bias_m)

    def select(self, mask: NDArray[np.bool_] | NDArray[np.integer]) -> PointCloud:
        """Subset by a boolean mask or an index array."""

        def pick(values: NDArray | None) -> NDArray | None:
            return None if values is None else values[mask]

        return PointCloud(
            x_m=self.x_m[mask],
            y_m=self.y_m[mask],
            z_m=self.z_m[mask],
            velocity_mps=pick(self.velocity_mps),
            range_idx=pick(self.range_idx),
            doppler_idx=pick(self.doppler_idx),
            peak=pick(self.peak),
            snr_db=pick(self.snr_db),
            noise_db=pick(self.noise_db),
            q_format=self.q_format,
            snr_method=self.snr_method,
            snr_linear=pick(self.snr_linear),
            target_id=pick(self.target_id),
        )

    def to_records(self) -> np.ndarray:
        """Structured array with the columns of :data:`POINT_RECORD_FIELDS`.

        Examples
        --------
        >>> cloud = PointCloud([0.0], [2.0], [0.0], doppler_idx=[-3])
        >>> rec = cloud.to_records()
        >>> float(rec["range_m"][0]), float(rec["doppler_idx"][0]), bool(np.isnan(rec["peak"][0]))
        (2.0, -3.0, True)
        """
        n = len(self)
        distance, azimuth, elevation = cartesian_to_spherical(self.x_m, self.y_m, self.z_m)

        def column(values: NDArray | None) -> NDArray[np.float64]:
            return np.full(n, np.nan) if values is None else values.astype(np.float64)

        records = np.zeros(n, dtype=[(name, "<f8") for name in POINT_RECORD_FIELDS])
        records["x_m"], records["y_m"], records["z_m"] = self.x_m, self.y_m, self.z_m
        records["range_m"], records["azimuth_deg"] = distance, azimuth
        records["elevation_deg"] = elevation
        records["range_idx"] = column(self.range_idx)
        records["doppler_idx"] = column(self.doppler_idx)
        records["velocity_mps"] = column(self.velocity_mps)
        records["peak"] = column(self.peak)
        records["snr_db"] = column(self.snr_db)
        records["noise_db"] = column(self.noise_db)
        records["snr_linear"] = column(self.snr_linear)
        records["target_id"] = column(self.target_id)
        return records


def decode_points_qformat(payload: memoryview, ctx: DecodeContext) -> PointCloud:
    """Decode the SDK 1.x/2.x point list (descriptor plus 12-byte records).

    The point count is taken from the descriptor but never trusted beyond the
    payload: a TLV must be exactly ``4 + 12 N`` bytes long, and only the
    records that lie inside it are read.
    """
    if len(payload) < _POINT_DESCRIPTOR.size:
        ctx.report(
            IssueCode.LENGTH_MISMATCH,
            f"point TLV of {len(payload)} bytes cannot hold its 4-byte descriptor",
            Severity.ERROR,
        )
        return PointCloud.empty()
    count, q_format = _POINT_DESCRIPTOR.unpack_from(payload, 0)
    room, leftover = divmod(len(payload) - _POINT_DESCRIPTOR.size, QFORMAT_POINT_DTYPE.itemsize)
    if leftover or room != count:
        ctx.report(
            IssueCode.LENGTH_MISMATCH,
            f"point descriptor declares {count} points but the {len(payload)}-byte TLV "
            f"holds {room} whole records",
            Severity.ERROR,
        )
    if q_format > _MAX_Q_FORMAT:
        ctx.report(
            IssueCode.INVALID_VALUE,
            f"coordinate Q-format {q_format} is implausible; points dropped",
            Severity.ERROR,
        )
        return PointCloud.empty()
    n = min(count, room)
    unsigned = ctx.unsigned_doppler()
    dtype = UNSIGNED_DOPPLER_POINT_DTYPE if unsigned else QFORMAT_POINT_DTYPE
    records = np.frombuffer(payload, dtype=dtype, count=n, offset=_POINT_DESCRIPTOR.size)
    scale = 1.0 / float(1 << q_format)
    doppler = records["doppler_idx"].astype(np.int32)
    if unsigned:
        bins = ctx.num_doppler_bins()
        if bins:
            doppler = signed_doppler(doppler, bins)
        elif n:
            ctx.report(
                IssueCode.MISSING_DOPPLER_BINS,
                "SDK 1.0 Doppler indices are unsigned FFT bins; without the Doppler-bin "
                "count they cannot be made signed",
            )
    resolution = ctx.doppler_bin_mps()
    velocity = doppler * resolution if resolution else None
    return PointCloud(
        x_m=records["x"] * scale,
        y_m=records["y"] * scale,
        z_m=records["z"] * scale,
        velocity_mps=velocity,
        range_idx=records["range_idx"],
        doppler_idx=doppler,
        peak=records["peak"],
        q_format=q_format,
    )


def decode_points_float(payload: memoryview, ctx: DecodeContext) -> PointCloud:
    """Decode the SDK 3.x point list (16-byte float records)."""
    n, leftover = divmod(len(payload), FLOAT_POINT_DTYPE.itemsize)
    if leftover:
        ctx.report(
            IssueCode.LENGTH_MISMATCH,
            f"float point TLV of {len(payload)} bytes is not a whole number of 16-byte records",
            Severity.ERROR,
        )
    if n != ctx.header.declared_point_count:
        ctx.report(
            IssueCode.POINT_COUNT_MISMATCH,
            f"header declares {ctx.header.declared_point_count} points, the TLV holds {n}",
        )
    records = np.frombuffer(payload, dtype=FLOAT_POINT_DTYPE, count=n)
    return PointCloud(
        x_m=records["x"],
        y_m=records["y"],
        z_m=records["z"],
        velocity_mps=records["velocity"],
    )


@dataclass
class PointSideInfo:
    """SDK 3.x per-point SNR and noise level, in decibels."""

    snr_db: NDArray[np.float64]
    noise_db: NDArray[np.float64]

    def __len__(self) -> int:
        return int(np.asarray(self.snr_db).size)


def decode_side_info(payload: memoryview, ctx: DecodeContext) -> PointSideInfo:
    """Decode ``{int16 snr, int16 noise}`` records in units of 0.1 dB."""
    n, leftover = divmod(len(payload), SIDE_INFO_DTYPE.itemsize)
    if leftover:
        ctx.report(
            IssueCode.LENGTH_MISMATCH,
            f"side-info TLV of {len(payload)} bytes is not a whole number of 4-byte records",
            Severity.ERROR,
        )
    records = np.frombuffer(payload, dtype=SIDE_INFO_DTYPE, count=n)
    return PointSideInfo(
        snr_db=records["snr"].astype(np.float64) * 0.1,
        noise_db=records["noise"].astype(np.float64) * 0.1,
    )


# -- log2-magnitude arrays --------------------------------------------------------------------


def _uint16_values(payload: memoryview, ctx: DecodeContext, what: str) -> NDArray[np.uint16]:
    if len(payload) % 2:
        ctx.report(
            IssueCode.LENGTH_MISMATCH,
            f"{what} TLV has an odd length ({len(payload)} bytes); the last byte is ignored",
            Severity.ERROR,
        )
    return np.frombuffer(payload, dtype="<u2", count=len(payload) // 2).astype(np.uint16)


def _require_scale(own: float | None, given: float | None) -> float:
    scale = given if given is not None else own
    if scale is None:
        raise ValueError(
            "the decibel scale is unknown: decode with a radar configuration, set "
            "num_virtual_antennas or db_per_lsb in DecoderConfig, or pass db_per_lsb"
        )
    return float(scale)


@dataclass
class MagnitudeProfile:
    """Range profile (zero Doppler) or noise profile (highest Doppler bin).

    Attributes
    ----------
    kind : str
        ``"range"`` or ``"noise"``.
    raw : numpy.ndarray of uint16
        Fixed-point log2 magnitudes, one per range bin.
    db_per_lsb : float or None
        Decibels of one raw step, filled in by the decoder when it is known.
    """

    kind: str
    raw: NDArray[np.uint16]
    db_per_lsb: float | None = None

    def __len__(self) -> int:
        return int(self.raw.size)

    def to_db(self, db_per_lsb: float | None = None, offset_db: float = 0.0) -> NDArray:
        """Profile in decibels.

        Examples
        --------
        >>> profile = MagnitudeProfile("range", np.array([0, 512, 1024], dtype=np.uint16))
        >>> profile.to_db(db_per_lsb=6.0206 / 512).round(2).tolist()
        [0.0, 6.02, 12.04]
        """
        scale = _require_scale(self.db_per_lsb, db_per_lsb)
        return self.raw.astype(np.float64) * scale + offset_db

    def range_axis(
        self,
        range_bin_m: float | None = None,
        *,
        sensor: SensorConfig | None = None,
        range_bias_m: float | None = None,
    ) -> NDArray[np.float64]:
        """Range of every bin in metres.

        Parameters
        ----------
        range_bin_m : float, optional
            Range of one bin; taken from ``sensor`` when omitted.
        sensor : SensorConfig, optional
            Radar configuration supplying bin size and range bias.
        range_bias_m : float, optional
            Overrides the configuration's range bias.
        """
        bias = range_bias_m
        if range_bin_m is None:
            if sensor is None:
                raise ValueError("pass range_bin_m or a radar configuration")
            range_bin_m = sensor.range_bin_m
            bias = sensor.range_bias_m if bias is None else bias
        return range_index_to_m(np.arange(len(self)), range_bin_m, bias or 0.0)


def decode_range_profile(payload: memoryview, ctx: DecodeContext) -> MagnitudeProfile:
    """Decode the zero-Doppler range profile."""
    return MagnitudeProfile("range", _uint16_values(payload, ctx, "range profile"))


def decode_noise_profile(payload: memoryview, ctx: DecodeContext) -> MagnitudeProfile:
    """Decode the noise profile (highest Doppler bin of every range bin)."""
    return MagnitudeProfile("noise", _uint16_values(payload, ctx, "noise profile"))


@dataclass
class RangeDopplerMap:
    """The detection matrix: log2 magnitude for every range and Doppler bin.

    Attributes
    ----------
    raw : numpy.ndarray of uint16
        Shape ``(range bins, Doppler bins)`` in FFT order (Doppler bin 0 is zero
        velocity); one-dimensional when the shape could not be resolved.
    db_per_lsb : float or None
        Decibels of one raw step, when known.
    """

    raw: NDArray[np.uint16]
    db_per_lsb: float | None = None

    @property
    def resolved(self) -> bool:
        """Whether the matrix has its two-dimensional shape."""
        return self.raw.ndim == 2

    def _need_resolved(self) -> NDArray[np.uint16]:
        if not self.resolved:
            raise ValueError(
                "the heatmap shape is unresolved; decode with a radar configuration or "
                "set num_range_bins in DecoderConfig"
            )
        return self.raw

    @property
    def num_range_bins(self) -> int:
        """Rows of the matrix."""
        return int(self._need_resolved().shape[0])

    @property
    def num_doppler_bins(self) -> int:
        """Columns of the matrix."""
        return int(self._need_resolved().shape[1])

    def centered(self) -> NDArray[np.uint16]:
        """Matrix with zero Doppler moved to the middle column (``fftshift``)."""
        return np.fft.fftshift(self._need_resolved(), axes=1)

    def to_db(self, db_per_lsb: float | None = None, *, centered: bool = False) -> NDArray:
        """Matrix in decibels, optionally Doppler-centred."""
        scale = _require_scale(self.db_per_lsb, db_per_lsb)
        values = self.centered() if centered else self._need_resolved()
        return values.astype(np.float64) * scale

    def values_at(self, range_idx: NDArray, doppler_idx: NDArray) -> NDArray[np.float64]:
        """Raw values at signed ``(range, Doppler)`` indices (Doppler wraps)."""
        matrix = self._need_resolved()
        rows = np.clip(np.asarray(range_idx, dtype=np.int64), 0, matrix.shape[0] - 1)
        cols = np.asarray(doppler_idx, dtype=np.int64) % matrix.shape[1]
        return matrix[rows, cols].astype(np.float64)

    def axes(
        self,
        range_bin_m: float | None = None,
        doppler_bin_mps: float | None = None,
        *,
        sensor: SensorConfig | None = None,
        range_bias_m: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Range (metres) and velocity (m/s) axes of the Doppler-centred matrix.

        The velocity of centred column ``j`` is ``(j - D/2)`` times the Doppler
        bin spacing (:attr:`SensorConfig.doppler_bin_mps`).
        """
        matrix = self._need_resolved()
        bias = range_bias_m
        if sensor is not None:
            range_bin_m = sensor.range_bin_m if range_bin_m is None else range_bin_m
            if doppler_bin_mps is None:
                doppler_bin_mps = sensor.doppler_bin_mps
            bias = sensor.range_bias_m if bias is None else bias
        if range_bin_m is None or doppler_bin_mps is None:
            raise ValueError("pass the bin sizes or a radar configuration")
        rows, cols = matrix.shape
        ranges = range_index_to_m(np.arange(rows), range_bin_m, bias or 0.0)
        velocities = (np.arange(cols) - cols // 2) * float(doppler_bin_mps)
        return ranges, velocities


def decode_range_doppler(payload: memoryview, ctx: DecodeContext) -> RangeDopplerMap:
    """Decode the detection matrix and recover its ``(range, Doppler)`` shape.

    The TLV only says how many cells there are. The range-bin count is taken
    from the radar configuration, else from the range profile of the same
    packet, else from ``DecoderConfig.num_range_bins``; the first candidate that
    divides the cell count wins. Failing all of them the flat array is kept and
    an issue is reported.
    """
    values = _uint16_values(payload, ctx, "range-Doppler heatmap")
    for rows in ctx.range_bin_candidates():
        if values.size % rows == 0 and values.size >= rows:
            return RangeDopplerMap(values.reshape(rows, values.size // rows))
    doppler = ctx.num_doppler_bins()
    if doppler and values.size % doppler == 0 and values.size >= doppler:
        return RangeDopplerMap(values.reshape(values.size // doppler, doppler))
    ctx.report(
        IssueCode.UNRESOLVED_SHAPE,
        f"cannot split {values.size} heatmap cells into range x Doppler bins; pass a radar "
        "configuration or set num_range_bins",
    )
    return RangeDopplerMap(values)


# -- statistics and temperature -----------------------------------------------------------

_STATS = struct.Struct("<6I")

# Temperature report: TI documents and firmware disagree. The Radar Toolbox UART
# page lists a uint32 "tempReportValid" and uint16 readings; TI mmWave SDK
# 03.05.00.04 declares MmwDemo_temperatureStats as int32_t tempReportValid
# (mmw_mss.h; zero means the readings are valid, it carries the return value of
# rlRfGetTemperatureReport) followed by rlRfTempData_t (rl_sensor.h: uint32 time
# in ms, then ten rlInt16_t readings in degrees Celsius). The decoder follows the
# firmware, so a negative status and sub-zero readings decode correctly.
_TEMPERATURE = struct.Struct("<iI10h")


@dataclass(frozen=True)
class ProcessingStats:
    """Timing and load figures the firmware reports per frame."""

    inter_frame_processing_time_us: int
    transmit_output_time_us: int
    inter_frame_processing_margin_us: int
    inter_chirp_processing_margin_us: int
    active_frame_cpu_load_pct: int
    inter_frame_cpu_load_pct: int

    def as_dict(self) -> dict[str, int]:
        """Field names to values."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def decode_stats(payload: memoryview, ctx: DecodeContext) -> ProcessingStats | None:
    """Decode the six ``uint32`` statistics."""
    if len(payload) != _STATS.size:
        ctx.report(
            IssueCode.LENGTH_MISMATCH,
            f"statistics TLV has {len(payload)} bytes instead of {_STATS.size}",
            Severity.ERROR,
        )
        if len(payload) < _STATS.size:
            return None
    return ProcessingStats(*_STATS.unpack_from(payload, 0))


@dataclass(frozen=True)
class TemperatureReport:
    """SDK 3.x temperature readings in degrees Celsius.

    Attributes
    ----------
    status : int
        Status of the temperature query; zero means the readings are valid.
    time_ms : int
        Time stamp of the readings.
    rx_c, tx_c : tuple of int
        Receiver (4) and transmitter (3) sensors.
    power_management_c : int
        Power-management sensor.
    digital_c : tuple of int
        The two digital-core sensors.
    """

    status: int
    time_ms: int
    rx_c: tuple[int, int, int, int]
    tx_c: tuple[int, int, int]
    power_management_c: int
    digital_c: tuple[int, int]

    @property
    def valid(self) -> bool:
        """Whether the readings are valid."""
        return self.status == 0


def decode_temperature(payload: memoryview, ctx: DecodeContext) -> TemperatureReport | None:
    """Decode the 28-byte temperature report."""
    if len(payload) != _TEMPERATURE.size:
        ctx.report(
            IssueCode.LENGTH_MISMATCH,
            f"temperature TLV has {len(payload)} bytes instead of {_TEMPERATURE.size}",
            Severity.ERROR,
        )
        if len(payload) < _TEMPERATURE.size:
            return None
    status, time_ms, *readings = _TEMPERATURE.unpack_from(payload, 0)
    return TemperatureReport(
        status=status,
        time_ms=time_ms,
        rx_c=(readings[0], readings[1], readings[2], readings[3]),
        tx_c=(readings[4], readings[5], readings[6]),
        power_management_c=readings[7],
        digital_c=(readings[8], readings[9]),
    )


@dataclass(frozen=True)
class UnknownTlv:
    """A TLV whose type the schema does not define, kept verbatim."""

    tlv_type: int
    raw: bytes

    def __len__(self) -> int:
        return len(self.raw)
