"""Structured output of decoded frames.

* Points go into a long table, one row per point: frame, sub-frame,
  timestamp, point index, coordinates, range, azimuth, elevation, bin
  indices, velocity, peak, SNR, the people counting firmware's linear SNR, the
  tracker target the point was associated with, and where the SNR came from
  (:func:`write_points_csv` with the standard library ``csv``,
  :func:`to_dataframe` for pandas). Per-point target IDs arrive one frame
  late; both writers attach them to the frame they describe before writing it.
* The targets of the on-sensor trackers get their own table, one row per
  target and frame (:func:`write_targets_csv`).
* Range profiles, noise profiles and detection matrices are stacked per frame
  into an ``.npz`` file with the frame numbers alongside (:func:`save_npz`).
  Frames whose arrays differ in size are stored under separate keys, so the
  file never needs pickled object arrays.
* Records such as frame summaries, processing statistics, Vital Signs
  summaries and capture reports go out as JSON Lines (:func:`write_jsonl`).

Paths always come from the caller; parent folders are created.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .frame import RadarFrame
from .tracking import TARGET_RECORD_FIELDS, link_target_association, link_target_associations

#: Columns of the point table, in order.
POINT_COLUMNS: tuple[str, ...] = (
    "frame",
    "subframe",
    "timestamp_s",
    "point",
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
    "snr_method",
)

#: Columns of the target table, in order.
TARGET_COLUMNS: tuple[str, ...] = ("frame", "subframe", "timestamp_s", *TARGET_RECORD_FIELDS)

_INTEGER_COLUMNS = frozenset({"range_idx", "doppler_idx", "target_id"})


def _timestamps(frames: Sequence[RadarFrame], timestamps: ArrayLike | None) -> list[float | None]:
    if timestamps is None:
        return [None] * len(frames)
    values = [float(v) for v in np.asarray(timestamps, dtype=np.float64).reshape(-1)]
    if len(values) != len(frames):
        raise ValueError(f"{len(values)} timestamps for {len(frames)} frames")
    return values


def _clean(value: float) -> float | None:
    return None if math.isnan(value) else value


def points_to_rows(
    frames: Iterable[RadarFrame], timestamps: ArrayLike | None = None
) -> list[dict[str, Any]]:
    """Long-format point records, one dictionary per point.

    Unknown values are ``None``.

    Parameters
    ----------
    frames : iterable of RadarFrame
    timestamps : array_like, optional
        Seconds per frame (e.g. ``CaptureAnalysis.timestamps_s``).
    """
    frame_list = list(frames)
    times = _timestamps(frame_list, timestamps)
    rows: list[dict[str, Any]] = []
    for frame, time_s in zip(frame_list, times, strict=True):
        points = frame.points
        if points is None or not len(points):
            continue
        records = points.to_records()
        for index, record in enumerate(records):
            row: dict[str, Any] = {
                "frame": frame.frame_number,
                "subframe": frame.subframe_number,
                "timestamp_s": time_s,
                "point": index,
            }
            for name in records.dtype.names or ():
                value = _clean(float(record[name]))
                if value is not None and name in _INTEGER_COLUMNS:
                    value = int(value)
                row[name] = value
            row["snr_method"] = points.snr_method.value
            rows.append(row)
    return rows


def _csv_cells(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: "" if value is None else value for key, value in row.items()}


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> int:
    """Write dictionaries as CSV with a fixed column order; returns the row count."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_cells(row))
            count += 1
    return count


def write_points_csv(
    path: str | Path, frames: Iterable[RadarFrame], timestamps: ArrayLike | None = None
) -> int:
    """Write the point table as CSV; returns the number of points written.

    Per-point target IDs are attached to the frames they describe first (see
    :func:`~radar_stream_decoder.tracking.link_target_associations`).
    """
    frame_list = list(frames)
    link_target_associations(frame_list)
    return write_csv(path, points_to_rows(frame_list, timestamps), POINT_COLUMNS)


def targets_to_rows(
    frames: Iterable[RadarFrame], timestamps: ArrayLike | None = None
) -> list[dict[str, Any]]:
    """Long-format target records, one dictionary per target and frame."""
    frame_list = list(frames)
    times = _timestamps(frame_list, timestamps)
    rows: list[dict[str, Any]] = []
    for frame, time_s in zip(frame_list, times, strict=True):
        targets = frame.targets
        if targets is None or not len(targets):
            continue
        for record in targets.to_records():
            row: dict[str, Any] = {
                "frame": frame.frame_number,
                "subframe": frame.subframe_number,
                "timestamp_s": time_s,
            }
            for name in record.dtype.names or ():
                value = _clean(float(record[name]))
                if value is not None and name in _INTEGER_COLUMNS:
                    value = int(value)
                row[name] = value
            rows.append(row)
    return rows


def write_targets_csv(
    path: str | Path, frames: Iterable[RadarFrame], timestamps: ArrayLike | None = None
) -> int:
    """Write the tracker targets as CSV; returns the number of rows written."""
    return write_csv(path, targets_to_rows(frames, timestamps), TARGET_COLUMNS)


class PointCsvWriter:
    """The point table of :func:`write_points_csv`, written one frame at a time.

    Live streams use it so that no decoded frame has to be kept until the end.
    Each frame is held back until the next one arrives, because the per-point
    target IDs of a frame travel in the next packet; :meth:`close` writes the
    last frame.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file; parent folders are created.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from radar_stream_decoder.frame import decode_packet
    >>> from radar_stream_decoder.synthetic import SyntheticScene, generate_packets
    >>> frame = decode_packet(next(generate_packets(SyntheticScene(seed=1), 1)))
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     with PointCsvWriter(pathlib.Path(folder) / "points.csv") as writer:
    ...         count = writer.write_frame(frame, timestamp_s=0.0)
    >>> count == frame.num_points == writer.points_written
    True
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.points_written = 0
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=list(POINT_COLUMNS), extrasaction="ignore"
        )
        self._writer.writeheader()
        self._held: tuple[RadarFrame, float | None] | None = None

    def write_frame(self, frame: RadarFrame, timestamp_s: float | None = None) -> int:
        """Queue the points of one frame; returns its point count.

        The previous frame is written now, after the target IDs that ``frame``
        carries for it have been attached.
        """
        previous = self._held[0] if self._held is not None else None
        link_target_association(previous, frame)
        self._flush_held()
        self._held = (frame, timestamp_s)
        return frame.num_points

    def _flush_held(self) -> None:
        if self._held is None:
            return
        frame, timestamp_s = self._held
        self._held = None
        times = None if timestamp_s is None else [timestamp_s]
        rows = points_to_rows([frame], times)
        for row in rows:
            self._writer.writerow(_csv_cells(row))
        self.points_written += len(rows)

    def close(self) -> None:
        """Write the frame still held back, then close the file (idempotent)."""
        if not self._handle.closed:
            self._flush_held()
            self._handle.close()

    def __enter__(self) -> PointCsvWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class TargetCsvWriter:
    """The target table of :func:`write_targets_csv`, written one frame at a time.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file; parent folders are created.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.targets_written = 0
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=list(TARGET_COLUMNS), extrasaction="ignore"
        )
        self._writer.writeheader()

    def write_frame(self, frame: RadarFrame, timestamp_s: float | None = None) -> int:
        """Append the targets of one frame; returns how many were written."""
        times = None if timestamp_s is None else [timestamp_s]
        rows = targets_to_rows([frame], times)
        for row in rows:
            self._writer.writerow(_csv_cells(row))
        self.targets_written += len(rows)
        return len(rows)

    def close(self) -> None:
        """Flush and close the file (idempotent)."""
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> TargetCsvWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def frame_summary_rows(
    frames: Iterable[RadarFrame], timestamps: ArrayLike | None = None
) -> list[dict[str, Any]]:
    """One summary record per frame (see :meth:`RadarFrame.summary`)."""
    frame_list = list(frames)
    times = _timestamps(frame_list, timestamps)
    return [
        {**frame.summary(), "timestamp_s": t} for frame, t in zip(frame_list, times, strict=True)
    ]


def stats_rows(
    frames: Iterable[RadarFrame], timestamps: ArrayLike | None = None
) -> list[dict[str, Any]]:
    """Processing statistics of every frame that carries them."""
    frame_list = list(frames)
    times = _timestamps(frame_list, timestamps)
    return [
        {"frame": frame.frame_number, "timestamp_s": t, **frame.stats.as_dict()}
        for frame, t in zip(frame_list, times, strict=True)
        if frame.stats is not None
    ]


def vital_signs_rows(
    frames: Iterable[RadarFrame], timestamps: ArrayLike | None = None
) -> list[dict[str, Any]]:
    """Vital Signs summaries of every frame that carries one."""
    frame_list = list(frames)
    times = _timestamps(frame_list, timestamps)
    return [
        {"frame": frame.frame_number, "timestamp_s": t, **frame.vital_signs.as_dict()}
        for frame, t in zip(frame_list, times, strict=True)
        if frame.vital_signs is not None
    ]


_ROW_BUILDERS = {
    "points": points_to_rows,
    "targets": targets_to_rows,
    "frames": frame_summary_rows,
    "stats": stats_rows,
    "vital_signs": vital_signs_rows,
}


def to_dataframe(
    frames: Iterable[RadarFrame], kind: str = "points", timestamps: ArrayLike | None = None
) -> Any:
    """Records of one kind as a ``pandas.DataFrame``.

    Parameters
    ----------
    frames : iterable of RadarFrame
    kind : {"points", "targets", "frames", "stats", "vital_signs"}
    timestamps : array_like, optional

    Raises
    ------
    ImportError
        If pandas is not installed (``pip install -e ".[pandas]"`` in the project folder).
    """
    if kind not in _ROW_BUILDERS:
        raise ValueError(f"kind must be one of {sorted(_ROW_BUILDERS)}, got {kind!r}")
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            'to_dataframe needs pandas: run pip install -e ".[pandas]" in the project folder'
        ) from exc
    rows = _ROW_BUILDERS[kind](frames, timestamps)
    columns = {"points": list(POINT_COLUMNS), "targets": list(TARGET_COLUMNS)}.get(kind)
    return pd.DataFrame(rows, columns=columns)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write one JSON object per line; returns the record count."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with file_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(dict(record)), ensure_ascii=False) + "\n")
            count += 1
    return count


def _stack(name: str, items: list[tuple[int, np.ndarray]], arrays: dict[str, np.ndarray]) -> None:
    if not items:
        return
    by_shape: dict[tuple[int, ...], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for frame_number, values in items:
        by_shape[values.shape].append((frame_number, values))
    for shape, group in by_shape.items():
        key = name if len(by_shape) == 1 else f"{name}__{'x'.join(str(n) for n in shape)}"
        arrays[key] = np.stack([values for _, values in group])
        arrays[f"{key}__frame"] = np.array([number for number, _ in group], dtype=np.int64)


def save_npz(
    path: str | Path, frames: Iterable[RadarFrame], timestamps: ArrayLike | None = None
) -> list[str]:
    """Stack the array payloads of all frames into one compressed ``.npz``.

    Keys: ``range_profile``, ``noise_profile``, ``range_doppler`` (raw
    log2-magnitude words, frames first) each with a ``<key>__frame`` array of
    frame numbers; ``frame_number`` and ``timestamp_s`` for every frame;
    ``db_per_lsb`` when the scale is known. Arrays whose shape changes within
    the capture are split into one key per shape.

    Returns
    -------
    list of str
        The keys written.
    """
    frame_list = list(frames)
    times = _timestamps(frame_list, timestamps)
    arrays: dict[str, np.ndarray] = {
        "frame_number": np.array([f.frame_number for f in frame_list], dtype=np.int64),
        "timestamp_s": np.array([np.nan if t is None else t for t in times], dtype=np.float64),
    }
    ranges = [
        (f.frame_number, f.range_profile.raw) for f in frame_list if f.range_profile is not None
    ]
    noises = [
        (f.frame_number, f.noise_profile.raw) for f in frame_list if f.noise_profile is not None
    ]
    matrices = [
        (f.frame_number, f.range_doppler.raw)
        for f in frame_list
        if f.range_doppler is not None and f.range_doppler.resolved
    ]
    _stack("range_profile", ranges, arrays)
    _stack("noise_profile", noises, arrays)
    _stack("range_doppler", matrices, arrays)
    scales = [
        f.range_profile.db_per_lsb
        for f in frame_list
        if f.range_profile is not None and f.range_profile.db_per_lsb is not None
    ]
    if scales:
        arrays["db_per_lsb"] = np.array(scales[0])
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file_path, **arrays)
    return sorted(arrays)
