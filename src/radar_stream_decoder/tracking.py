"""People counting and people tracking output: point clouds, targets, per-point target IDs.

Three TI firmware families run a tracker on the sensor and send its results
instead of (or besides) the out-of-box demo's arrays:

=========================  ==================  ===========================================
Schema                     Header              TLVs
=========================  ==================  ===========================================
``people-counting-3d``     48 B, checksum      6 compressed spherical points, 7 targets,
                                               8 per-point target IDs, 11 presence
``people-counting-2d``     52 B, checksum      6 polar points (float), 7 targets (68 B),
                                               8 per-point target IDs
``people-tracking``        40 B, padded        1020 compressed spherical points,
                                               1010 targets (112 B), 1011 per-point target
                                               IDs, 1012 target heights, 1021 presence
=========================  ==================  ===========================================

Formats (little-endian), taken from TI's firmware sources and documents:

* Compressed spherical points (3D People Counting lab of the mmWave Industrial
  Toolbox 4.6.0, ``Pcount3DDemo_output_message_UARTpointCloud`` in
  ``pcount3D_output.h``; TLV 1020 of the Radar Toolbox people tracking demos):
  five ``float32`` units for elevation (rad), azimuth (rad), Doppler (m/s),
  range (m) and SNR, then 8 bytes per point ``{int8 elevation, int8 azimuth,
  int16 doppler, uint16 range, uint16 snr}``; each value times its unit. The
  3D lab's ``mss_main.c`` uses 0.01 rad, 0.01 rad, 0.00028 m/s, 0.00025 m and
  0.04.
* Polar points (2D people counting lab, Industrial Toolbox 2.x): 16 bytes per
  point ``{float32 range, azimuth, doppler, snr}``.
* 3D targets: 112 bytes ``{uint32 tid, float32 pos[3], vel[3], acc[3],
  ec[16], g, confidence}`` (Industrial Toolbox 4.6-4.12 documents, Radar
  Toolbox TLV 1010), or the 40-byte variant of the Industrial Toolbox 4.5.1
  documentation ``{uint32 tid, float32 posX, posY, velX, velY, accX, accY,
  posZ, velZ, accZ}``. The record size is told apart by the payload length and
  by every target ID lying in 0-249.
* 2D targets: 68 bytes ``{uint32 tid, float32 posX, posY, velX, velY, accX,
  accY, ec[9], g}``.
* Per-point target IDs: one ``uint8`` per point *of the previous frame* (TI:
  "A targetID at index i is the target to which point i of the previous
  frame's point cloud was associated"); 253, 254 and 255 mark points left
  unassociated (SNR too weak, outside the boundary, noise).
  :func:`link_target_associations` attaches them to the right frame.
* Target heights (TLV 1012): 12 bytes per target ``{uint32 tid, float32 maxZ,
  minZ}``.
* Presence (TLV 11 / 1021): one ``uint32``, non-zero when someone is present.

Spherical coordinates become Cartesian with
:func:`~radar_stream_decoder.units.spherical_to_cartesian`
(:math:`x = R\\cos\\phi\\sin\\theta`, :math:`y = R\\cos\\phi\\cos\\theta`,
:math:`z = R\\sin\\phi`), the same convention TI's visualisers use. For a sensor
mounted on the ceiling and looking down, ``y`` points at the floor.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .constants import MAX_TARGET_ID
from .errors import DecodeIssue, IssueCode, Severity
from .payloads import PayloadKind, PointCloud
from .units import spherical_to_cartesian

if TYPE_CHECKING:
    from .frame import RadarFrame
    from .payloads import DecodeContext

# -- point clouds ---------------------------------------------------------------------------

_POINT_UNITS = struct.Struct("<5f")

#: One compressed spherical point (8 bytes).
COMPRESSED_POINT_DTYPE = np.dtype(
    [
        ("elevation", "i1"),
        ("azimuth", "i1"),
        ("doppler", "<i2"),
        ("range", "<u2"),
        ("snr", "<u2"),
    ]
)

#: One point of the 2D people counting lab (16 bytes).
POLAR_POINT_DTYPE = np.dtype(
    [("range", "<f4"), ("azimuth", "<f4"), ("doppler", "<f4"), ("snr", "<f4")]
)

assert COMPRESSED_POINT_DTYPE.itemsize == 8
assert POLAR_POINT_DTYPE.itemsize == 16
assert _POINT_UNITS.size == 20


@dataclass(frozen=True)
class PointUnits:
    """Scale factors of a compressed point cloud (one integer step of each field)."""

    elevation_rad: float
    azimuth_rad: float
    doppler_mps: float
    range_m: float
    snr: float


def _spherical_cloud(
    range_m: NDArray,
    azimuth_rad: NDArray,
    elevation_rad: NDArray,
    doppler_mps: NDArray,
    snr: NDArray,
) -> PointCloud:
    x, y, z = spherical_to_cartesian(range_m, np.degrees(azimuth_rad), np.degrees(elevation_rad))
    return PointCloud(x, y, z, velocity_mps=doppler_mps, snr_linear=snr)


def decode_compressed_points(payload: memoryview, ctx: DecodeContext) -> PointCloud:
    """Decode a compressed spherical point cloud (point units, then 8-byte points).

    Examples
    --------
    >>> import struct
    >>> units = struct.pack("<5f", 0.01, 0.01, 0.5, 0.25, 0.1)
    >>> raw = units + struct.pack("<bbhHH", 0, 0, -2, 8, 30)
    >>> cloud = decode_compressed_points(memoryview(raw), None)
    >>> float(cloud.y_m[0]), float(cloud.velocity_mps[0]), round(float(cloud.snr_linear[0]), 3)
    (2.0, -1.0, 3.0)
    """
    if len(payload) < _POINT_UNITS.size:
        _report(
            ctx,
            IssueCode.LENGTH_MISMATCH,
            f"point cloud TLV of {len(payload)} bytes cannot hold its 20-byte point units",
            Severity.ERROR,
        )
        return PointCloud.empty()
    units = PointUnits(*(float(v) for v in _POINT_UNITS.unpack_from(payload, 0)))
    count, leftover = divmod(len(payload) - _POINT_UNITS.size, COMPRESSED_POINT_DTYPE.itemsize)
    if leftover:
        _report(
            ctx,
            IssueCode.LENGTH_MISMATCH,
            f"point cloud TLV of {len(payload)} bytes is not 20 bytes of units plus whole "
            "8-byte points",
            Severity.ERROR,
        )
    records = np.frombuffer(payload, COMPRESSED_POINT_DTYPE, count=count, offset=_POINT_UNITS.size)
    return _spherical_cloud(
        records["range"] * units.range_m,
        records["azimuth"] * units.azimuth_rad,
        records["elevation"] * units.elevation_rad,
        records["doppler"] * units.doppler_mps,
        records["snr"] * units.snr,
    )


def decode_polar_points(payload: memoryview, ctx: DecodeContext) -> PointCloud:
    """Decode the 2D people counting point cloud (``float32`` range, azimuth, Doppler, SNR)."""
    count, leftover = divmod(len(payload), POLAR_POINT_DTYPE.itemsize)
    if leftover:
        _report(
            ctx,
            IssueCode.LENGTH_MISMATCH,
            f"point cloud TLV of {len(payload)} bytes is not a whole number of 16-byte points",
            Severity.ERROR,
        )
    records = np.frombuffer(payload, POLAR_POINT_DTYPE, count=count)
    return _spherical_cloud(
        records["range"].astype(np.float64),
        records["azimuth"].astype(np.float64),
        np.zeros(count),
        records["doppler"].astype(np.float64),
        records["snr"].astype(np.float64),
    )


# -- targets --------------------------------------------------------------------------------

#: 3D target record of the Industrial Toolbox 4.6-4.12 labs and Radar Toolbox TLV 1010.
TARGET_3D_DTYPE = np.dtype(
    [
        ("target_id", "<u4"),
        ("position", "<f4", (3,)),
        ("velocity", "<f4", (3,)),
        ("acceleration", "<f4", (3,)),
        ("covariance", "<f4", (16,)),
        ("gating_gain", "<f4"),
        ("confidence", "<f4"),
    ]
)

#: 3D target record as the Industrial Toolbox 4.5.1 documentation lists it.
TARGET_3D_COMPACT_DTYPE = np.dtype(
    [
        ("target_id", "<u4"),
        ("pos_x", "<f4"),
        ("pos_y", "<f4"),
        ("vel_x", "<f4"),
        ("vel_y", "<f4"),
        ("acc_x", "<f4"),
        ("acc_y", "<f4"),
        ("pos_z", "<f4"),
        ("vel_z", "<f4"),
        ("acc_z", "<f4"),
    ]
)

#: 2D target record of the 2D people counting lab.
TARGET_2D_DTYPE = np.dtype(
    [
        ("target_id", "<u4"),
        ("position", "<f4", (2,)),
        ("velocity", "<f4", (2,)),
        ("acceleration", "<f4", (2,)),
        ("covariance", "<f4", (9,)),
        ("gating_gain", "<f4"),
    ]
)

assert TARGET_3D_DTYPE.itemsize == 112
assert TARGET_3D_COMPACT_DTYPE.itemsize == 40
assert TARGET_2D_DTYPE.itemsize == 68

#: Columns of :meth:`TargetList.to_records` (all float64; NaN where unknown).
TARGET_RECORD_FIELDS: tuple[str, ...] = (
    "target_id",
    "x_m",
    "y_m",
    "z_m",
    "vx_mps",
    "vy_mps",
    "vz_mps",
    "ax_mps2",
    "ay_mps2",
    "az_mps2",
    "gating_gain",
    "confidence",
    "max_z_m",
    "min_z_m",
)


def _vectors(values: object, n: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n, 3):
        raise ValueError(f"{name} must have shape ({n}, 3), got {array.shape}")
    return array


def _optional(values: object, n: int, name: str) -> NDArray[np.float64] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != n:
        raise ValueError(f"{name} has {array.size} entries for {n} targets")
    return array


@dataclass
class TargetList:
    """Targets the on-sensor tracker reports for one frame.

    Attributes
    ----------
    target_id : numpy.ndarray of int64
        Tracker IDs (0-249).
    position_m, velocity_mps, acceleration_mps2 : numpy.ndarray
        Shape ``(n, 3)``; ``z`` is zero for 2D trackers.
    covariance : numpy.ndarray or None
        Error covariance per target, row-major (3x3 for 2D, 4x4 for 3D trackers).
    gating_gain, confidence : numpy.ndarray or None
        Gating function gain and tracker confidence, when the record has them.
    max_z_m, min_z_m : numpy.ndarray or None
        Height extent (Radar Toolbox TLV 1012), matched by target ID.
    record_bytes : int
        Size of one record on the wire (112, 40 or 68).
    """

    target_id: NDArray[np.int64]
    position_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]
    acceleration_mps2: NDArray[np.float64]
    covariance: NDArray[np.float64] | None = None
    gating_gain: NDArray[np.float64] | None = None
    confidence: NDArray[np.float64] | None = None
    max_z_m: NDArray[np.float64] | None = None
    min_z_m: NDArray[np.float64] | None = None
    record_bytes: int = 0

    def __post_init__(self) -> None:
        self.target_id = np.asarray(self.target_id, dtype=np.int64).reshape(-1)
        n = self.target_id.size
        self.position_m = _vectors(self.position_m, n, "position_m")
        self.velocity_mps = _vectors(self.velocity_mps, n, "velocity_mps")
        self.acceleration_mps2 = _vectors(self.acceleration_mps2, n, "acceleration_mps2")
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=np.float64)
            width = covariance.shape[-1] if covariance.ndim > 1 else covariance.size // max(n, 1)
            self.covariance = covariance.reshape(n, width)
        self.gating_gain = _optional(self.gating_gain, n, "gating_gain")
        self.confidence = _optional(self.confidence, n, "confidence")
        self.max_z_m = _optional(self.max_z_m, n, "max_z_m")
        self.min_z_m = _optional(self.min_z_m, n, "min_z_m")

    def __len__(self) -> int:
        return int(self.target_id.size)

    @classmethod
    def empty(cls) -> TargetList:
        """A list without targets."""
        nothing = np.zeros((0, 3))
        return cls(np.zeros(0, dtype=np.int64), nothing, nothing, nothing)

    def to_records(self) -> np.ndarray:
        """Structured array with the columns of :data:`TARGET_RECORD_FIELDS`."""
        n = len(self)
        records = np.zeros(n, dtype=[(name, "<f8") for name in TARGET_RECORD_FIELDS])
        records["target_id"] = self.target_id
        for axis, name in enumerate("xyz"):
            records[f"{name}_m"] = self.position_m[:, axis]
            records[f"v{name}_mps"] = self.velocity_mps[:, axis]
            records[f"a{name}_mps2"] = self.acceleration_mps2[:, axis]
        for name in ("gating_gain", "confidence", "max_z_m", "min_z_m"):
            values = getattr(self, name)
            records[name] = np.full(n, np.nan) if values is None else values
        return records


def _plausible_ids(payload: memoryview, record_bytes: int) -> bool:
    words = np.frombuffer(payload, dtype="<u4")
    return bool(np.all(words[:: record_bytes // 4] <= MAX_TARGET_ID))


def decode_target_list_3d(payload: memoryview, ctx: DecodeContext) -> TargetList | None:
    """Decode 3D target records of 112 bytes (or the 40-byte variant).

    The size is the one that divides the payload and gives every record a
    target ID within 0-249; when both do, 112 bytes (the size of every TI
    document from Industrial Toolbox 4.6 on and of the Radar Toolbox) is taken
    and the ambiguity is reported.
    """
    if len(payload) == 0:
        return TargetList.empty()
    sizes = [
        size
        for size in (TARGET_3D_DTYPE.itemsize, TARGET_3D_COMPACT_DTYPE.itemsize)
        if len(payload) % size == 0 and _plausible_ids(payload, size)
    ]
    if not sizes:
        _report(
            ctx,
            IssueCode.LENGTH_MISMATCH,
            f"target TLV of {len(payload)} bytes is neither whole 112-byte nor whole 40-byte "
            "records with target IDs in 0-249",
            Severity.ERROR,
        )
        return None
    size = sizes[0]
    if len(sizes) > 1:
        _report(
            ctx,
            IssueCode.AMBIGUOUS_RECORD_SIZE,
            f"{len(payload)} target bytes fit 112- and 40-byte records alike; read as "
            "112-byte records",
            Severity.INFO,
        )
    if size == TARGET_3D_DTYPE.itemsize:
        records = np.frombuffer(payload, TARGET_3D_DTYPE)
        return TargetList(
            target_id=records["target_id"],
            position_m=records["position"],
            velocity_mps=records["velocity"],
            acceleration_mps2=records["acceleration"],
            covariance=records["covariance"],
            gating_gain=records["gating_gain"],
            confidence=records["confidence"],
            record_bytes=size,
        )
    compact = np.frombuffer(payload, TARGET_3D_COMPACT_DTYPE)

    def stack(prefix: str) -> NDArray[np.float64]:
        return np.column_stack([compact[f"{prefix}_{axis}"] for axis in "xyz"])

    return TargetList(
        target_id=compact["target_id"],
        position_m=stack("pos"),
        velocity_mps=stack("vel"),
        acceleration_mps2=stack("acc"),
        record_bytes=size,
    )


def decode_target_list_2d(payload: memoryview, ctx: DecodeContext) -> TargetList:
    """Decode 68-byte 2D target records (``z`` set to zero)."""
    count, leftover = divmod(len(payload), TARGET_2D_DTYPE.itemsize)
    if leftover:
        _report(
            ctx,
            IssueCode.LENGTH_MISMATCH,
            f"target TLV of {len(payload)} bytes is not a whole number of 68-byte records",
            Severity.ERROR,
        )
    records = np.frombuffer(payload, TARGET_2D_DTYPE, count=count)

    def planar(field: str) -> NDArray[np.float64]:
        return np.column_stack((records[field].astype(np.float64), np.zeros(count)))

    return TargetList(
        target_id=records["target_id"],
        position_m=planar("position"),
        velocity_mps=planar("velocity"),
        acceleration_mps2=planar("acceleration"),
        covariance=records["covariance"],
        gating_gain=records["gating_gain"],
        record_bytes=TARGET_2D_DTYPE.itemsize,
    )


# -- per-point target IDs, heights, presence ---------------------------------------------------


@dataclass
class TargetAssociation:
    """Per-point target IDs that refer to the *previous* frame's point cloud.

    Attributes
    ----------
    target_id : numpy.ndarray of uint8
        Target of every point of the previous frame, or 253/254/255.
    """

    target_id: NDArray[np.uint8]

    def __len__(self) -> int:
        return int(np.asarray(self.target_id).size)

    @property
    def associated(self) -> NDArray[np.bool_]:
        """Which points belong to a target."""
        return np.asarray(self.target_id) <= MAX_TARGET_ID


def decode_target_association(payload: memoryview, ctx: DecodeContext) -> TargetAssociation:
    """Decode one ``uint8`` target ID per point of the previous frame."""
    return TargetAssociation(np.frombuffer(payload, dtype=np.uint8).copy())


_HEIGHT_DTYPE = np.dtype([("target_id", "<u4"), ("max_z", "<f4"), ("min_z", "<f4")])


@dataclass
class TargetHeights:
    """Height extent of each target (Radar Toolbox TLV 1012)."""

    target_id: NDArray[np.int64]
    max_z_m: NDArray[np.float64]
    min_z_m: NDArray[np.float64]

    def __len__(self) -> int:
        return int(np.asarray(self.target_id).size)


def decode_target_heights(payload: memoryview, ctx: DecodeContext) -> TargetHeights:
    """Decode 12-byte ``{uint32 tid, float32 maxZ, minZ}`` records."""
    count, leftover = divmod(len(payload), _HEIGHT_DTYPE.itemsize)
    if leftover:
        _report(
            ctx,
            IssueCode.LENGTH_MISMATCH,
            f"target height TLV of {len(payload)} bytes is not a whole number of 12-byte records",
            Severity.ERROR,
        )
    records = np.frombuffer(payload, _HEIGHT_DTYPE, count=count)
    return TargetHeights(
        records["target_id"].astype(np.int64),
        records["max_z"].astype(np.float64),
        records["min_z"].astype(np.float64),
    )


@dataclass(frozen=True)
class PresenceIndication:
    """The presence word: non-zero while the firmware considers the area occupied."""

    value: int

    @property
    def occupied(self) -> bool:
        """Whether someone is present."""
        return self.value != 0


def decode_presence(payload: memoryview, ctx: DecodeContext) -> PresenceIndication | None:
    """Decode the single ``uint32`` presence word."""
    if len(payload) != 4:
        _report(
            ctx,
            IssueCode.LENGTH_MISMATCH,
            f"presence TLV has {len(payload)} bytes instead of 4",
            Severity.ERROR,
        )
        if len(payload) < 4:
            return None
    (value,) = struct.unpack_from("<I", payload, 0)
    return PresenceIndication(int(value))


def merge_target_heights(targets: TargetList, heights: TargetHeights) -> int:
    """Copy the height extents into the target list by target ID; returns the matches."""
    lookup = {
        int(tid): (float(top), float(bottom))
        for tid, top, bottom in zip(
            heights.target_id, heights.max_z_m, heights.min_z_m, strict=True
        )
    }
    n = len(targets)
    top = np.full(n, np.nan)
    bottom = np.full(n, np.nan)
    matched = 0
    for row, tid in enumerate(targets.target_id):
        extent = lookup.get(int(tid))
        if extent is not None:
            top[row], bottom[row] = extent
            matched += 1
    targets.max_z_m, targets.min_z_m = top, bottom
    return matched


# -- linking per-point target IDs to the frame they describe -------------------------------------


@dataclass
class AssociationStats:
    """Outcome of :func:`link_target_associations`.

    Attributes
    ----------
    linked : int
        Association TLVs attached to the previous frame's points.
    mismatched : int
        Association TLVs whose length differs from the previous frame's point count.
    orphaned : int
        Association TLVs whose previous frame is missing (dropped, or the start
        of the stream).
    """

    linked: int = 0
    mismatched: int = 0
    orphaned: int = 0

    def as_dict(self) -> dict[str, int]:
        """Counters by name."""
        return {"linked": self.linked, "mismatched": self.mismatched, "orphaned": self.orphaned}


def link_target_association(
    previous: RadarFrame | None, current: RadarFrame, stats: AssociationStats | None = None
) -> bool:
    """Attach ``current``'s per-point target IDs to the points of ``previous``.

    The IDs describe the frame before; they are attached only when
    ``previous`` is exactly that frame (same sub-frame, frame number one lower)
    and holds as many points as there are IDs. A count mismatch is recorded as
    an issue on ``current``, once however often the same pair is linked.

    Returns
    -------
    bool
        Whether the IDs were attached.
    """
    association = current.payloads.get(PayloadKind.TARGET_ASSOCIATION)
    if association is None:
        return False
    stats = stats if stats is not None else AssociationStats()
    if previous is None or (
        previous.frame_number + 1 != current.frame_number
        or (previous.subframe_number or 0) != (current.subframe_number or 0)
    ):
        stats.orphaned += 1
        return False
    points: PointCloud | None = previous.points
    count = len(points) if points is not None else 0
    if count != len(association):
        stats.mismatched += 1
        issue = DecodeIssue(
            IssueCode.ASSOCIATION_MISMATCH,
            f"{len(association)} per-point target IDs for frame {previous.frame_number}, "
            f"which holds {count} points; not attached",
            Severity.WARNING,
            current.frame_number,
        )
        if issue not in current.issues:  # linking the same frames again adds nothing
            current.issues.append(issue)
        return False
    if points is not None and count:
        points.target_id = np.asarray(association.target_id, dtype=np.int32).copy()
    stats.linked += 1
    return True


def link_target_associations(frames: Sequence[RadarFrame]) -> AssociationStats:
    """Attach every frame's per-point target IDs to the previous frame's points.

    ``frames`` must be in stream order and free of repeats (see
    :func:`radar_stream_decoder.capture.deduplicate`).
    """
    stats = AssociationStats()
    previous: RadarFrame | None = None
    for frame in frames:
        link_target_association(previous, frame, stats)
        previous = frame
    return stats


def _report(ctx: DecodeContext | None, code: IssueCode, message: str, severity: Severity) -> None:
    if ctx is not None:
        ctx.report(code, message, severity)
