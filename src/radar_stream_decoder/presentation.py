"""What the viewer draws, computed without any GUI toolkit.

The viewer's widgets only display. Everything they show is prepared here from
a decoded frame: scatter positions, colours and marker sizes of the point
cloud, the positions of the tracker's targets, the range-profile curve and the
Doppler-centred range-Doppler image, each with axis labels in the viewer's
language. Kept apart from the Qt code, it can be tested headless and reused
by plotting scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .config import ViewerConfig
from .errors import ConfigError
from .frame import RadarFrame
from .sensor_config import SensorConfig
from .snr import peak_level_db


class ColorMode(str, Enum):
    """What the point colours encode."""

    SNR = "snr"
    DOPPLER = "doppler"


@dataclass(frozen=True)
class ScatterData:
    """Point cloud in the ``x``-``y`` plane, ready for a scatter plot."""

    xy: NDArray[np.float64]
    color: NDArray[np.float64]
    size: NDArray[np.float64]
    color_label: str
    color_limits: tuple[float, float]
    colormap: str
    fixed_limits: bool = False


@dataclass(frozen=True)
class CurveData:
    """A line plot."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x_label: str
    y_label: str


@dataclass(frozen=True)
class HeatmapData:
    """An image with its extent, rows along ``y`` and columns along ``x``."""

    image: NDArray[np.float64]
    extent: tuple[float, float, float, float]
    x_label: str
    y_label: str
    color_label: str


@dataclass(frozen=True)
class TargetMarkers:
    """Tracker targets in the ``x``-``y`` plane."""

    xy: NDArray[np.float64]
    target_id: NDArray[np.int64]


@dataclass(frozen=True)
class FrameDisplay:
    """Everything the viewer shows for one frame."""

    frame_number: int
    point_count: int
    issue_count: int
    scatter: ScatterData
    profile: CurveData | None
    heatmap: HeatmapData | None
    targets: TargetMarkers | None = None

    @property
    def target_count(self) -> int:
        """Number of targets shown."""
        return 0 if self.targets is None else int(self.targets.target_id.size)


def _sensor_value(sensor: SensorConfig | None, name: str) -> float | None:
    if sensor is None:
        return None
    try:
        return float(getattr(sensor, name))
    except (ConfigError, ZeroDivisionError):
        return None


def _scaled_sizes(level_db: NDArray[np.float64], viewer: ViewerConfig) -> NDArray[np.float64]:
    smallest, largest = viewer.point_size_range
    if level_db.size == 0:
        return np.zeros(0)
    finite = np.isfinite(level_db)
    if not np.any(finite):
        return np.full(level_db.size, 0.5 * (smallest + largest))
    low, high = np.min(level_db[finite]), np.max(level_db[finite])
    span = high - low if high > low else 1.0
    fraction = np.where(finite, (level_db - low) / span, 0.5)
    return smallest + np.clip(fraction, 0.0, 1.0) * (largest - smallest)


def point_scatter(
    frame: RadarFrame,
    color_mode: ColorMode | str = ColorMode.SNR,
    viewer: ViewerConfig | None = None,
) -> ScatterData:
    """Scatter data of a frame's points.

    Colour shows the SNR or the radial velocity. Without an SNR in decibels
    the colour scale is the firmware's peak level, or the linear SNR of the
    people counting firmware on a log scale; without a radar configuration
    the velocity is replaced by the Doppler index. Marker area grows with the
    reflection strength. At most ``viewer.max_points`` of the strongest points
    are kept.
    """
    viewer = viewer or ViewerConfig()
    mode = ColorMode(color_mode)
    points = frame.points
    if points is None or not len(points):
        empty = np.zeros(0)
        label = "SNR (dB)" if mode is ColorMode.SNR else "徑向速度 (m/s)"
        limits = viewer.snr_limits_db if mode is ColorMode.SNR else (-1.0, 1.0)
        cmap = "viridis" if mode is ColorMode.SNR else "coolwarm"
        return ScatterData(
            np.zeros((0, 2)), empty, empty, label, limits, cmap, mode is ColorMode.SNR
        )

    strength_label = "峰值強度 (dB)"
    if points.peak is not None:
        strength = peak_level_db(points.peak, frame.platform)
    else:
        strength = points.snr_db.copy()
    if not np.any(np.isfinite(strength)):
        strength = points.snr_db.copy()
    if not np.any(np.isfinite(strength)) and points.snr_linear is not None:
        # People counting firmware reports a linear SNR; shown on a log scale.
        with np.errstate(divide="ignore", invalid="ignore"):
            strength = 10.0 * np.log10(points.snr_linear)
        strength_label = "韌體 SNR (10·log10)"

    order = np.arange(len(points))
    if len(points) > viewer.max_points:
        ranking = np.where(np.isfinite(strength), strength, -np.inf)
        order = np.sort(np.argsort(ranking)[::-1][: viewer.max_points])
    xy = np.column_stack((points.x_m[order], points.y_m[order]))
    sizes = _scaled_sizes(strength[order], viewer)

    if mode is ColorMode.SNR:
        fixed = False
        if np.any(np.isfinite(points.snr_db)):
            color, label = points.snr_db[order], "SNR (dB)"
            limits, fixed = viewer.snr_limits_db, True
        else:
            color, label = strength[order], strength_label
            finite = color[np.isfinite(color)]
            limits = (
                (float(finite.min()), float(max(finite.max(), finite.min() + 1.0)))
                if finite.size
                else viewer.snr_limits_db
            )
        cmap = "viridis"
    else:
        if np.any(np.isfinite(points.velocity_mps)):
            color, label = points.velocity_mps[order], "徑向速度 (m/s)"
        elif points.doppler_idx is not None:
            color, label = points.doppler_idx[order].astype(np.float64), "Doppler bin"
        else:
            color, label = np.zeros(order.size), "Doppler bin"
        finite = np.abs(color[np.isfinite(color)])
        bound = float(max(finite.max(), 1.0)) if finite.size else 1.0
        limits = (-bound, bound)
        cmap = "coolwarm"
        fixed = False
    color = np.where(np.isfinite(color), color, limits[0])
    return ScatterData(xy, color.astype(np.float64), sizes, label, limits, cmap, fixed)


def target_markers(frame: RadarFrame) -> TargetMarkers | None:
    """Positions and IDs of the frame's tracker targets, ``None`` without a target list.

    Examples
    --------
    >>> from radar_stream_decoder.synthetic import SyntheticScene, generate_packets
    >>> from radar_stream_decoder.frame import decode_packet
    >>> scene = SyntheticScene(schema="people-counting-3d", seed=1)
    >>> markers = target_markers(decode_packet(next(generate_packets(scene, 1))))
    >>> markers.xy.shape[1], markers.target_id.size == markers.xy.shape[0]
    (2, True)
    """
    targets = frame.targets
    if targets is None:
        return None
    return TargetMarkers(targets.position_m[:, :2].copy(), targets.target_id.copy())


def _segment_curve(frame: RadarFrame, sensor: SensorConfig | None) -> CurveData | None:
    segment = frame.range_segment
    if segment is None or not len(segment):
        return None
    level = 20.0 * np.log10(np.maximum(segment.magnitude(), 1e-3))
    bins = segment.bins if segment.start_bin is not None else np.arange(len(segment))
    bin_m = _sensor_value(sensor, "range_bin_m")
    if bin_m is not None:
        return CurveData(bins * bin_m, level, "距離 (m)", "強度 (dB，相對值)")
    return CurveData(bins.astype(np.float64), level, "距離 bin", "強度 (dB，相對值)")


def range_profile_curve(frame: RadarFrame, sensor: SensorConfig | None = None) -> CurveData | None:
    """The range profile in decibels over metres, or raw units over bins.

    Frames of the Vital Signs lab carry no range profile; for them the
    magnitude of the complex range segment is shown instead.
    """
    profile = frame.range_profile
    if profile is None or not len(profile):
        return _segment_curve(frame, sensor)
    bin_m = _sensor_value(sensor, "range_bin_m")
    if bin_m is not None:
        x = profile.range_axis(bin_m, range_bias_m=sensor.range_bias_m if sensor else 0.0)
        x_label = "距離 (m)"
    else:
        x = np.arange(len(profile), dtype=np.float64)
        x_label = "距離 bin"
    if profile.db_per_lsb is not None:
        y, y_label = profile.to_db(), "強度 (dB)"
    else:
        y, y_label = profile.raw.astype(np.float64), "log2 強度（原始值）"
    return CurveData(x, y, x_label, y_label)


def range_doppler_image(
    frame: RadarFrame, sensor: SensorConfig | None = None
) -> HeatmapData | None:
    """Doppler-centred detection matrix with range along ``y`` and velocity along ``x``."""
    matrix = frame.range_doppler
    if matrix is None or not matrix.resolved:
        return None
    rows, cols = matrix.raw.shape
    centred = matrix.centered().astype(np.float64)
    if matrix.db_per_lsb is not None:
        image, color_label = centred * matrix.db_per_lsb, "強度 (dB)"
    else:
        image, color_label = centred, "log2 強度（原始值）"
    bin_m = _sensor_value(sensor, "range_bin_m")
    velocity_m = _sensor_value(sensor, "doppler_bin_mps")
    if bin_m is not None and velocity_m is not None:
        half = 0.5 * velocity_m
        extent = (
            (-(cols // 2)) * velocity_m - half,
            (cols - 1 - cols // 2) * velocity_m + half,
            -0.5 * bin_m,
            (rows - 0.5) * bin_m,
        )
        x_label, y_label = "徑向速度 (m/s)", "距離 (m)"
    else:
        extent = (-(cols // 2) - 0.5, cols - 1 - cols // 2 + 0.5, -0.5, rows - 0.5)
        x_label, y_label = "Doppler bin", "距離 bin"
    return HeatmapData(image, extent, x_label, y_label, color_label)


def prepare_frame_display(
    frame: RadarFrame,
    sensor: SensorConfig | None = None,
    color_mode: ColorMode | str = ColorMode.SNR,
    viewer: ViewerConfig | None = None,
) -> FrameDisplay:
    """Everything the viewer needs to show one frame.

    Examples
    --------
    >>> from radar_stream_decoder.synthetic import SyntheticScene, generate_packets
    >>> from radar_stream_decoder.frame import decode_packet
    >>> scene = SyntheticScene(seed=4)
    >>> frame = decode_packet(next(generate_packets(scene, 1)), sensor=scene.radar())
    >>> display = prepare_frame_display(frame, scene.radar())
    >>> display.point_count == len(display.scatter.xy), display.profile.x_label
    (True, '距離 (m)')
    """
    viewer = viewer or ViewerConfig()
    scatter = point_scatter(frame, color_mode, viewer)
    return FrameDisplay(
        frame_number=frame.frame_number,
        point_count=frame.num_points,
        issue_count=len(frame.issues),
        scatter=scatter,
        profile=range_profile_curve(frame, sensor),
        heatmap=range_doppler_image(frame, sensor),
        targets=target_markers(frame),
    )
