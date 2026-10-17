"""Reading capture files and examining what the recording did to the stream.

A capture is whatever bytes a recorder wrote: ideally the raw data-port
stream, in practice often worse. Recorders that dump their whole working
buffer after every read write each packet several times, slow readers lose
frames, and sensors restart mid-recording. Captures are read here in bounded
chunks and each of these effects is measured.

* A frame identical to one of the recent frames (same frame and sub-frame
  number, same CRC-32) is a repeat and is dropped. The same numbers with
  different content are a *conflict* (kept once, reported) when the copy
  directly follows the first one, carries the same header timestamp or is
  damaged itself. An intact copy replaces a damaged one, which happens when
  a buffer dump cut a packet short and wrote it whole again in the next
  dump.
* Frame-number steps above one are gaps. The report gives their histogram,
  the number of missing frames and the drop ratio
  ``missing / (unique + missing)``.
* Frame numbers that run backwards and are not conflicts count as restarts.
* The header's 32-bit CPU-cycle counter wraps every :math:`2^{32}/f_{clk}`
  seconds (21.5 s at 200 MHz, 7.2 s at 600 MHz). For every step the number
  of wraps :math:`m` is chosen so that :math:`\\Delta c + m\\,2^{32}` comes
  closest to the frame-number step times the frame period, which also
  bridges gaps longer than one wrap period. The 48-byte header of the 3D
  People Counting lab has no timestamp, so its frames are timed by frame
  number and the configured frame period.
* For tracking firmware: target counts, presence, and how many per-point
  target-ID TLVs could be attached to the frame they describe.
"""

from __future__ import annotations

import logging
from collections import Counter, OrderedDict, deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .assembler import AssemblerStats, FrameAssembler
from .config import DecoderConfig
from .constants import CYCLE_COUNTER_MODULUS, cpu_clock_hz
from .errors import ConfigError, DecodeIssue, IssueCode, Severity
from .frame import RadarFrame
from .sensor_config import SensorConfig
from .tracking import AssociationStats, link_target_associations

_LOGGER = logging.getLogger(__name__)

#: Largest frame-number step trusted when estimating the frame period without
#: a nominal value (four frames stay below one counter wrap for any period up
#: to 1.7 s at 600 MHz).
_PERIOD_ESTIMATE_MAX_STEP = 4

#: Payload bytes per second of a 921600-baud 8N1 data port (10 bits per byte).
DATA_PORT_BYTES_PER_S = 921_600 / 10


def iter_capture(
    path: str | Path,
    config: DecoderConfig | None = None,
    sensor: SensorConfig | None = None,
    *,
    chunk_size: int | None = None,
    stats: AssemblerStats | None = None,
) -> Iterator[RadarFrame]:
    """Decode a capture file chunk by chunk, every repeat included.

    Parameters
    ----------
    path : str or pathlib.Path
        Capture file of raw data-port bytes.
    config : DecoderConfig, optional
        Decoder settings.
    sensor : SensorConfig, optional
        Radar configuration for physical units.
    chunk_size : int, optional
        Bytes per read; ``config.chunk_size`` by default.
    stats : AssemblerStats, optional
        Receives the assembler's final counters when the iteration completes.

    Yields
    ------
    RadarFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    config = config or DecoderConfig()
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"capture file not found: {file_path}")
    size = chunk_size or config.chunk_size
    assembler = FrameAssembler(config, sensor)
    with file_path.open("rb") as handle:
        while chunk := handle.read(size):
            yield from assembler.feed(chunk)
    yield from assembler.finish()
    if stats is not None:
        final = assembler.stats
        for name in final.as_dict():
            setattr(stats, name, getattr(final, name))
        stats.issues = list(final.issues)


@dataclass
class DedupeStats:
    """Outcome of de-duplication.

    Attributes
    ----------
    duplicates : int
        Byte-identical repeats dropped.
    conflicts : int
        Repeated frame numbers with different content (the later copy dropped).
    replaced : int
        Damaged copies (cut short or with error-level issues) superseded by a
        later intact copy of the same frame.
    resets : int
        Times the frame number ran backwards (sensor restarts).
    issues : list of DecodeIssue
    """

    duplicates: int = 0
    conflicts: int = 0
    replaced: int = 0
    resets: int = 0
    issues: list[DecodeIssue] = field(default_factory=list)


class FrameVerdict(str, Enum):
    """What :meth:`FrameDeduplicator.judge` decided about a frame."""

    NEW = "new"
    REPEAT = "repeat"
    CONFLICT = "conflict"
    REPLACEMENT = "replacement"


def _is_intact(frame: RadarFrame) -> bool:
    return not frame.truncated and not frame.has_errors


class FrameDeduplicator:
    """Streaming filter that drops repeated frames.

    A frame whose number was seen recently is a repeat when its bytes are
    identical. When they differ, it is a conflict (dropped and reported) if
    it directly follows the first copy, carries the same header timestamp, or
    is damaged itself (cut short, or with error-level issues); a buffer dump
    writes the same packet again, timestamp included. Otherwise an older frame
    number with new content means the sensor restarted, whose counter starts
    over at an unrelated timestamp. One exception comes first: an intact copy
    that follows a damaged one *replaces* it. In a stream the damaged copy has
    already been handed on, so the replacement is passed on as well;
    :func:`deduplicate` swaps it in place.

    Parameters
    ----------
    window : int, optional
        Number of recent frames a repeat is looked for in.
    """

    def __init__(self, window: int = 64) -> None:
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self.window = window
        self.stats = DedupeStats()
        # Frame key -> (CRC-32, intact, header timestamp or None).
        self._recent: OrderedDict[tuple[int, int], tuple[int, bool, int | None]] = OrderedDict()
        self._last_key: tuple[int, int] | None = None

    def accept(self, frame: RadarFrame) -> bool:
        """Return ``True`` for a new frame or a replacement, ``False`` otherwise."""
        return self.judge(frame) in (FrameVerdict.NEW, FrameVerdict.REPLACEMENT)

    def judge(self, frame: RadarFrame) -> FrameVerdict:
        """Classify a frame against the recent ones and update the statistics."""
        key = frame.key
        cycles = frame.header.cpu_cycles
        known = self._recent.get(key)
        if known is not None:
            known_crc, known_intact, known_cycles = known
            if known_crc == frame.crc32:
                self.stats.duplicates += 1
                return FrameVerdict.REPEAT
            if not known_intact and _is_intact(frame):
                self.stats.replaced += 1
                self.stats.issues.append(
                    DecodeIssue(
                        IssueCode.FRAME_REPLACED,
                        f"frame {key[0]} (sub-frame {key[1]}) arrived intact after a damaged "
                        "copy; the intact copy replaces it",
                        Severity.INFO,
                        key[0],
                    )
                )
                self._recent[key] = (frame.crc32, True, cycles)
                return FrameVerdict.REPLACEMENT
            same_packet = cycles is not None and cycles == known_cycles
            if key == self._last_key or same_packet or not _is_intact(frame):
                self.stats.conflicts += 1
                self.stats.issues.append(
                    DecodeIssue(
                        IssueCode.FRAME_CONFLICT,
                        f"frame {key[0]} (sub-frame {key[1]}) arrived twice with different "
                        "content; the first copy is kept",
                        Severity.WARNING,
                        key[0],
                    )
                )
                return FrameVerdict.CONFLICT
        if self._last_key is not None and key < self._last_key:
            self.stats.resets += 1
            self.stats.issues.append(
                DecodeIssue(
                    IssueCode.FRAME_RESET,
                    f"frame number fell from {self._last_key[0]} to {key[0]}: the sensor "
                    "restarted or the recording was spliced",
                    Severity.INFO,
                    key[0],
                )
            )
            self._recent.clear()
        self._recent[key] = (frame.crc32, _is_intact(frame), cycles)
        self._recent.move_to_end(key)
        while len(self._recent) > self.window:
            self._recent.popitem(last=False)
        self._last_key = key
        return FrameVerdict.NEW


def deduplicate(
    frames: Iterable[RadarFrame], window: int = 64
) -> tuple[list[RadarFrame], DedupeStats]:
    """Drop repeated frames; see :class:`FrameDeduplicator`.

    A replacement takes the place of the damaged copy it supersedes, so every
    frame number keeps its position in stream order.
    """
    dedupe = FrameDeduplicator(window)
    unique: list[RadarFrame] = []
    position: dict[tuple[int, int], int] = {}
    for frame in frames:
        verdict = dedupe.judge(frame)
        if verdict is FrameVerdict.NEW:
            position[frame.key] = len(unique)
            unique.append(frame)
        elif verdict is FrameVerdict.REPLACEMENT:
            unique[position[frame.key]] = frame
    return unique, dedupe.stats


def _frame_steps(frame_numbers: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.diff(frame_numbers) if frame_numbers.size > 1 else np.zeros(0, dtype=np.int64)


def estimate_frame_period_s(
    cycles: ArrayLike, frame_numbers: ArrayLike, clock_hz: float
) -> float | None:
    """Median frame period from timestamp steps over small frame-number steps.

    Only steps of at most four frames are used, short enough that the 32-bit
    counter cannot have wrapped more than the modulo arithmetic accounts for.

    Returns
    -------
    float or None
        ``None`` when no usable step exists.

    Examples
    --------
    >>> estimate_frame_period_s([0, 10_000_000, 30_000_000], [1, 2, 4], 200e6)
    0.05
    """
    counts = np.asarray(cycles, dtype=np.int64)
    frames = np.asarray(frame_numbers, dtype=np.int64)
    steps = _frame_steps(frames)
    usable = (steps > 0) & (steps <= _PERIOD_ESTIMATE_MAX_STEP)
    if not np.any(usable):
        return None
    deltas = np.diff(counts) % CYCLE_COUNTER_MODULUS
    per_frame = deltas[usable] / steps[usable]
    return float(np.median(per_frame) / clock_hz)


def unwrap_cpu_cycles(
    cycles: ArrayLike,
    frame_numbers: ArrayLike,
    clock_hz: float,
    nominal_period_s: float | None = None,
) -> NDArray[np.float64]:
    """Seconds since the first frame from wrapping 32-bit cycle counts.

    Parameters
    ----------
    cycles : array_like of int
        Header timestamps, in stream order.
    frame_numbers : array_like of int
        Frame numbers of the same frames.
    clock_hz : float
        Counter clock.
    nominal_period_s : float, optional
        Frame period; estimated with :func:`estimate_frame_period_s` when
        omitted. Without either, steps are taken modulo :math:`2^{32}` only.

    Returns
    -------
    numpy.ndarray of float64
        Starts at zero. Across a frame-number reset the plain modular step is
        used, since the counter's relation to the frame number is lost.

    Examples
    --------
    >>> wrap = 2**32
    >>> cycles = [wrap - 5_000_000, 5_000_000, (5_000_000 + 30 * 10_000_000) % wrap]
    >>> unwrap_cpu_cycles(cycles, [10, 11, 41], 200e6, nominal_period_s=0.05).round(3).tolist()
    [0.0, 0.05, 1.55]
    """
    counts = np.asarray(cycles, dtype=np.int64)
    frames = np.asarray(frame_numbers, dtype=np.int64)
    if counts.shape != frames.shape:
        raise ValueError(f"{counts.size} timestamps for {frames.size} frame numbers")
    if clock_hz <= 0:
        raise ValueError(f"clock_hz must be positive, got {clock_hz}")
    if counts.size == 0:
        return np.zeros(0)
    deltas = (np.diff(counts) % CYCLE_COUNTER_MODULUS).astype(np.float64)
    steps = _frame_steps(frames)
    period = nominal_period_s or estimate_frame_period_s(counts, frames, clock_hz)
    if period is not None:
        expected = steps * period * clock_hz
        wraps = np.rint((expected - deltas) / CYCLE_COUNTER_MODULUS)
        wraps = np.where(steps > 0, np.maximum(wraps, 0.0), 0.0)
        deltas = deltas + wraps * CYCLE_COUNTER_MODULUS
    return np.concatenate(([0.0], np.cumsum(deltas))) / clock_hz


def frame_clock_hz(frames: Sequence[RadarFrame], config: DecoderConfig | None = None) -> float:
    """Timestamp clock for a frame sequence: the override, else the platform default."""
    if config is not None and config.clock_hz:
        return float(config.clock_hz)
    if not frames:
        return cpu_clock_hz(0)
    first = frames[0]
    return cpu_clock_hz(first.platform, first.schema)


def has_cycle_timestamps(frames: Sequence[RadarFrame]) -> bool:
    """Whether every frame's header carries a CPU-cycle timestamp."""
    return all(frame.header.cpu_cycles is not None for frame in frames)


def frame_number_timestamps(
    frame_numbers: ArrayLike, frame_period_s: float | None
) -> NDArray[np.float64]:
    """Seconds since the first frame from frame numbers alone.

    Used for headers without a timestamp. A frame number that does not
    advance (a sensor restart) counts as one frame period.

    Examples
    --------
    >>> frame_number_timestamps([5, 6, 9, 1], 0.1).round(3).tolist()
    [0.0, 0.1, 0.4, 0.5]
    >>> bool(np.isnan(frame_number_timestamps([1, 2], None)).all())
    True
    """
    numbers = np.asarray(frame_numbers, dtype=np.int64)
    if numbers.size == 0:
        return np.zeros(0)
    if not frame_period_s:
        return np.full(numbers.size, np.nan)
    steps = np.diff(numbers)
    steps = np.where(steps > 0, steps, 1)
    return np.concatenate(([0.0], np.cumsum(steps))) * float(frame_period_s)


def frame_timestamps(
    frames: Sequence[RadarFrame],
    config: DecoderConfig | None = None,
    frame_period_s: float | None = None,
) -> NDArray[np.float64]:
    """Seconds since the first frame for a de-duplicated frame sequence.

    Parameters
    ----------
    frames : sequence of RadarFrame
    config : DecoderConfig, optional
        Supplies the clock and nominal-period overrides.
    frame_period_s : float, optional
        Frame period of the radar configuration; times frames whose headers
        carry no timestamp (NaN without any period).
    """
    if not frames:
        return np.zeros(0)
    if not has_cycle_timestamps(frames):
        nominal = config.nominal_frame_period_s if config is not None else None
        return frame_number_timestamps(
            [frame.frame_number for frame in frames], nominal or frame_period_s
        )
    clock = frame_clock_hz(frames, config)
    nominal = config.nominal_frame_period_s if config is not None else None
    return unwrap_cpu_cycles(
        [frame.header.cpu_cycles for frame in frames],
        [frame.frame_number for frame in frames],
        clock,
        nominal,
    )


class FrameTimestamper:
    """Seconds since the first frame, computed one frame at a time.

    The streaming counterpart of :func:`frame_timestamps` for live streams,
    whose frames are not kept: every step of the 32-bit cycle counter is
    resolved as in :func:`unwrap_cpu_cycles`, with the nominal frame period
    when one is configured and otherwise the median of the recent steps over
    at most four frames.

    Parameters
    ----------
    config : DecoderConfig, optional
        Supplies the ``clock_hz`` and ``nominal_frame_period_s`` overrides.
    history : int, optional
        Recent steps the period estimate is taken from.
    frame_period_s : float, optional
        Frame period of the radar configuration, for headers without a
        timestamp (their frames are timed by frame number).

    Examples
    --------
    >>> from radar_stream_decoder.frame import decode_packet
    >>> from radar_stream_decoder.synthetic import SyntheticScene, generate_packets
    >>> packets = list(generate_packets(SyntheticScene(schema="vital-signs", seed=2), 5))
    >>> stamp = FrameTimestamper()
    >>> [round(stamp(decode_packet(p)), 3) for p in packets]
    [0.0, 0.05, 0.1, 0.15, 0.2]
    """

    def __init__(
        self,
        config: DecoderConfig | None = None,
        *,
        history: int = 64,
        frame_period_s: float | None = None,
    ) -> None:
        if history < 1:
            raise ValueError(f"history must be at least 1, got {history}")
        self.config = config or DecoderConfig()
        self.frame_period_s = frame_period_s
        self.clock_hz: float | None = None
        self._steps: deque[float] = deque(maxlen=history)
        self._last: tuple[int, int] | None = None
        self._elapsed_cycles = 0.0
        self._elapsed_frames = 0

    def _period_s(self, clock_hz: float) -> float | None:
        if self.config.nominal_frame_period_s:
            return float(self.config.nominal_frame_period_s)
        if not self._steps:
            return None
        return float(np.median(self._steps)) / clock_hz

    def __call__(self, frame: RadarFrame) -> float:
        """Timestamp of ``frame``, which must follow the frames passed before it.

        NaN for a frame without a header timestamp when no frame period is known.
        """
        if frame.header.cpu_cycles is None:
            return self._by_frame_number(frame.frame_number)
        if self.clock_hz is None:
            self.clock_hz = frame_clock_hz([frame], self.config)
        clock = self.clock_hz
        number, cycles = frame.frame_number, frame.header.cpu_cycles
        if self._last is not None:
            step = number - self._last[0]
            delta = float((cycles - self._last[1]) % CYCLE_COUNTER_MODULUS)
            if 0 < step <= _PERIOD_ESTIMATE_MAX_STEP:
                self._steps.append(delta / step)
            period = self._period_s(clock)
            if period is not None and step > 0:
                wraps = np.rint((step * period * clock - delta) / CYCLE_COUNTER_MODULUS)
                delta += max(float(wraps), 0.0) * CYCLE_COUNTER_MODULUS
            self._elapsed_cycles += delta
        self._last = (number, cycles)
        return self._elapsed_cycles / clock

    def _by_frame_number(self, number: int) -> float:
        period = self.config.nominal_frame_period_s or self.frame_period_s
        if self._last is not None:
            self._elapsed_frames += max(number - self._last[0], 1)
        self._last = (number, 0)
        return float(self._elapsed_frames * period) if period else float("nan")


@dataclass(frozen=True)
class CaptureReport:
    """Forensic summary of a capture.

    Attributes
    ----------
    packets : int
        Frames decoded, repeats included.
    unique_frames : int
        Frames after de-duplication.
    duplicates, conflicts, frame_resets : int
        See :class:`DedupeStats`.
    replaced_copies : int
        Damaged copies superseded by a later intact copy (``DedupeStats.replaced``).
    first_frame, last_frame : int or None
        Frame numbers of the first and last unique frame.
    missing_frames : int
        Frames absent between consecutive unique frame numbers.
    drop_ratio : float
        ``missing / (unique + missing)``.
    gap_histogram : dict of int to int
        Frame-number step to occurrences (steps of one included).
    estimated_frame_period_s : float or None
    clock_hz : float
        Timestamp clock used.
    duration_s : float or None
        Time from the first to the last unique frame.
    bytes_in, bytes_skipped, frames_rejected, frames_salvaged, resyncs,
    truncated_tail_bytes : int
        Assembler counters.
    platforms, sdk_versions, schemas : dict of str to int
        Frames per header platform, SDK version and decoding schema.
    tlv_types : dict of int to int
        TLVs per type over unique frames.
    issue_counts : dict of str to int
        Issues per code (frames, assembler and de-duplication).
    mean_packet_bytes : float or None
        Average declared packet length.
    data_port_load : float or None
        Bytes per frame period relative to what a 921600-baud 8N1 data port
        carries (92160 bytes/s); above 1 the sensor cannot deliver every frame.
    timestamp_source : str
        ``"cpu-cycles"`` (header timestamps), ``"frame-number"`` (headers
        without a timestamp, timed with the frame period) or ``"none"``.
    frames_with_targets, max_targets : int
        Frames carrying a tracker target list, and the most targets in one frame.
    mean_targets : float or None
        Mean target count over the frames of tracking firmware.
    presence_frames : int or None
        Frames reporting presence, ``None`` when no frame carries the indication.
    association_linked, association_mismatched, association_orphaned : int
        Per-point target-ID TLVs attached to the frame they describe, rejected
        for a point-count mismatch, or left without their (missing) frame.
    """

    packets: int
    unique_frames: int
    duplicates: int
    conflicts: int
    frame_resets: int
    first_frame: int | None
    last_frame: int | None
    missing_frames: int
    drop_ratio: float
    gap_histogram: dict[int, int]
    estimated_frame_period_s: float | None
    clock_hz: float
    duration_s: float | None
    bytes_in: int
    bytes_skipped: int
    frames_rejected: int
    frames_salvaged: int
    resyncs: int
    truncated_tail_bytes: int
    platforms: dict[str, int]
    sdk_versions: dict[str, int]
    schemas: dict[str, int]
    tlv_types: dict[int, int]
    issue_counts: dict[str, int]
    mean_packet_bytes: float | None = None
    data_port_load: float | None = None
    replaced_copies: int = 0
    timestamp_source: str = "cpu-cycles"
    frames_with_targets: int = 0
    max_targets: int = 0
    mean_targets: float | None = None
    presence_frames: int | None = None
    association_linked: int = 0
    association_mismatched: int = 0
    association_orphaned: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary (integer keys become strings)."""
        result: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, dict):
                value = {str(key): item for key, item in value.items()}
            result[name] = value
        return result

    def summary_lines(self) -> list[str]:
        """Human-readable report, one fact per line."""
        period = (
            f"{self.estimated_frame_period_s * 1e3:.2f} ms"
            if self.estimated_frame_period_s is not None
            else "unknown"
        )
        span = f"{self.first_frame}..{self.last_frame}" if self.first_frame is not None else "none"
        gaps = ", ".join(f"{step}:{count}" for step, count in sorted(self.gap_histogram.items()))
        load = (
            f"{self.mean_packet_bytes:.0f} bytes per packet, {self.data_port_load:.0%} of a "
            "921600-baud data port"
            if self.mean_packet_bytes is not None and self.data_port_load is not None
            else "unknown"
        )
        if self.timestamp_source == "cpu-cycles":
            timing = f"{period} at {self.clock_hz / 1e6:.0f} MHz"
        elif self.timestamp_source == "frame-number":
            timing = f"{period} (no header timestamp; frames timed by frame number)"
        else:
            timing = period
        targets = (
            f"{self.frames_with_targets} frames, mean {self.mean_targets:.2f}, "
            f"max {self.max_targets}"
            if self.mean_targets is not None
            else "none"
        )
        presence = "n/a" if self.presence_frames is None else f"{self.presence_frames} frames"
        associations = (
            f"linked {self.association_linked}, mismatched {self.association_mismatched}, "
            f"orphaned {self.association_orphaned}"
        )
        return [
            f"packets decoded       {self.packets}",
            f"unique frames         {self.unique_frames} (frames {span})",
            f"duplicates dropped    {self.duplicates}",
            f"conflicting repeats   {self.conflicts}",
            f"replaced copies       {self.replaced_copies}",
            f"sensor restarts       {self.frame_resets}",
            f"missing frames        {self.missing_frames} (drop ratio {self.drop_ratio:.1%})",
            f"frame-number steps    {gaps or 'none'}",
            f"frame period          {timing}",
            f"link load             {load}",
            f"bytes read / skipped  {self.bytes_in} / {self.bytes_skipped}",
            f"rejected / salvaged   {self.frames_rejected} / {self.frames_salvaged}",
            f"resynchronisations    {self.resyncs}",
            f"truncated tail bytes  {self.truncated_tail_bytes}",
            f"platforms             {_format_counts(self.platforms)}",
            f"SDK versions          {_format_counts(self.sdk_versions)}",
            f"schemas               {_format_counts(self.schemas)}",
            f"TLV types             {_format_counts(self.tlv_types)}",
            f"tracker targets       {targets}",
            f"presence reported     {presence}",
            f"target-ID links       {associations}",
            f"issues                {_format_counts(self.issue_counts) or 'none'}",
        ]


def _format_counts(counts: dict[Any, int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items(), key=str))


def summarize_capture(
    packets: int,
    unique: Sequence[RadarFrame],
    assembler: AssemblerStats,
    dedupe: DedupeStats,
    config: DecoderConfig | None = None,
    *,
    frame_period_s: float | None = None,
    associations: AssociationStats | None = None,
) -> CaptureReport:
    """Build the :class:`CaptureReport` of a decoded capture.

    Parameters
    ----------
    packets : int
        Frames decoded before de-duplication.
    unique : sequence of RadarFrame
        Frames after de-duplication, in stream order.
    assembler : AssemblerStats
        Final assembler counters.
    dedupe : DedupeStats
        De-duplication outcome.
    config : DecoderConfig, optional
        Supplies the clock and nominal-period overrides.
    frame_period_s : float, optional
        Frame period of the radar configuration, for headers without a timestamp.
    associations : AssociationStats, optional
        Outcome of attaching per-point target IDs.
    """
    numbers = np.array([frame.frame_number for frame in unique], dtype=np.int64)
    collapsed = numbers[np.r_[True, np.diff(numbers) != 0]] if numbers.size else numbers
    steps = _frame_steps(collapsed)
    forward = steps[steps > 0]
    gap_histogram = {int(step): int(count) for step, count in Counter(forward.tolist()).items()}
    missing = int(np.sum(forward - 1)) if forward.size else 0
    unique_numbers = int(collapsed.size)
    drop_ratio = missing / (unique_numbers + missing) if unique_numbers + missing else 0.0
    clock = frame_clock_hz(unique, config)
    period: float | None = None
    duration: float | None = None
    nominal = config.nominal_frame_period_s if config is not None else None
    timestamp_source = "none"
    if numbers.size and has_cycle_timestamps(unique):
        timestamp_source = "cpu-cycles"
        cycles = np.array([frame.header.cpu_cycles for frame in unique], dtype=np.int64)
        period = nominal or estimate_frame_period_s(cycles, numbers, clock)
        times = unwrap_cpu_cycles(cycles, numbers, clock, period)
        duration = float(times[-1] - times[0])
    elif numbers.size:
        timestamp_source = "frame-number"
        period = nominal or frame_period_s
        times = frame_number_timestamps(numbers, period)
        duration = float(times[-1] - times[0]) if period else None
    issues: Counter[str] = Counter()
    for frame in unique:
        issues.update(issue.code.value for issue in frame.issues)
    issues.update(issue.code.value for issue in assembler.issues)
    issues.update(issue.code.value for issue in dedupe.issues)
    tlv_types: Counter[int] = Counter()
    for frame in unique:
        tlv_types.update(frame.tlv_types)
    mean_bytes = float(np.mean([f.header.packet_length for f in unique])) if unique else None
    load = (
        mean_bytes / period / DATA_PORT_BYTES_PER_S if mean_bytes is not None and period else None
    )
    tracking = [frame for frame in unique if frame.schema.tracks_targets]
    target_counts = [frame.num_targets for frame in tracking]
    presence = [frame.presence for frame in unique if frame.presence is not None]
    links = associations or AssociationStats()
    return CaptureReport(
        packets=packets,
        unique_frames=len(unique),
        duplicates=dedupe.duplicates,
        conflicts=dedupe.conflicts,
        frame_resets=dedupe.resets,
        first_frame=int(numbers[0]) if numbers.size else None,
        last_frame=int(numbers[-1]) if numbers.size else None,
        missing_frames=missing,
        drop_ratio=drop_ratio,
        gap_histogram=gap_histogram,
        estimated_frame_period_s=period,
        clock_hz=clock,
        duration_s=duration,
        bytes_in=assembler.bytes_in,
        bytes_skipped=assembler.bytes_skipped,
        frames_rejected=assembler.frames_rejected,
        frames_salvaged=assembler.frames_salvaged,
        resyncs=assembler.resyncs,
        truncated_tail_bytes=assembler.truncated_tail_bytes,
        platforms=dict(Counter(frame.header.platform_name for frame in unique)),
        sdk_versions=dict(Counter(str(frame.header.version) for frame in unique)),
        schemas=dict(Counter(frame.schema.value for frame in unique)),
        tlv_types=dict(tlv_types),
        issue_counts=dict(issues),
        mean_packet_bytes=mean_bytes,
        data_port_load=load,
        replaced_copies=dedupe.replaced,
        timestamp_source=timestamp_source,
        frames_with_targets=sum(1 for frame in unique if frame.targets is not None),
        max_targets=max(target_counts, default=0),
        mean_targets=float(np.mean(target_counts)) if target_counts else None,
        presence_frames=sum(1 for item in presence if item.occupied) if presence else None,
        association_linked=links.linked,
        association_mismatched=links.mismatched,
        association_orphaned=links.orphaned,
    )


@dataclass
class CaptureAnalysis:
    """A decoded capture: unique frames, their timestamps and the forensic report."""

    frames: list[RadarFrame]
    timestamps_s: NDArray[np.float64]
    report: CaptureReport


def analyze_frames(
    frames: Iterable[RadarFrame],
    assembler: AssemblerStats,
    config: DecoderConfig | None = None,
    *,
    frame_period_s: float | None = None,
) -> CaptureAnalysis:
    """De-duplicate (when configured), link target IDs, timestamp and summarise frames.

    Parameters
    ----------
    frames : iterable of RadarFrame
        Decoded frames in stream order.
    assembler : AssemblerStats
        Final assembler counters.
    config : DecoderConfig, optional
        Decoder settings.
    frame_period_s : float, optional
        Frame period of the radar configuration, for headers without a timestamp.
    """
    config = config or DecoderConfig()
    collected = list(frames)
    if config.deduplicate:
        unique, dedupe = deduplicate(collected, config.dedupe_window)
    else:
        unique, dedupe = collected, DedupeStats()
    associations = link_target_associations(unique)
    report = summarize_capture(
        len(collected),
        unique,
        assembler,
        dedupe,
        config,
        frame_period_s=frame_period_s,
        associations=associations,
    )
    timestamps = frame_timestamps(unique, config, frame_period_s)
    return CaptureAnalysis(unique, timestamps, report)


def sensor_frame_period_s(sensor: SensorConfig | None) -> float | None:
    """Frame period of a radar configuration, or ``None`` when it has none."""
    if sensor is None:
        return None
    try:
        return sensor.frame_period_s
    except ConfigError:
        return None


def analyze_capture(
    path: str | Path,
    config: DecoderConfig | None = None,
    sensor: SensorConfig | None = None,
) -> CaptureAnalysis:
    """Decode a capture file and examine it.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from radar_stream_decoder.synthetic import FAULT_PRESETS, SyntheticScene, synthetic_capture
    >>> scene = SyntheticScene(schema="vital-signs")
    >>> stream = synthetic_capture(scene, 40, FAULT_PRESETS["block-dump"])
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = pathlib.Path(folder) / "capture.bin"
    ...     _ = path.write_bytes(stream.data)
    ...     analysis = analyze_capture(path)
    >>> report = analysis.report
    >>> report.packets == 2 * report.unique_frames - 1
    True
    >>> report.duplicates == report.unique_frames - 1
    True
    """
    config = config or DecoderConfig()
    stats = AssemblerStats()
    frames = list(iter_capture(path, config, sensor, stats=stats))
    analysis = analyze_frames(frames, stats, config, frame_period_s=sensor_frame_period_s(sensor))
    _LOGGER.info(
        "%s: %d packets, %d unique frames, %.1f%% dropped",
        Path(path).name,
        analysis.report.packets,
        analysis.report.unique_frames,
        100.0 * analysis.report.drop_ratio,
    )
    return analysis
