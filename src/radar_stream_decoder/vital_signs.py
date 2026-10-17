"""Vital Signs lab output: the per-frame summary and the complex range segment.

The Vital Signs lab firmware (xWR14xx) reuses the out-of-box framing with its
own two TLVs:

* Type 0 is a 128-byte summary: two ``uint16`` range-bin indices, a
  ``float32`` peak magnitude, a ``uint32`` cycle count, the ``uint16`` first
  and last bin of the range segment, then 28 ``float32`` values (20 estimates
  and 8 reserved words).
* Type 1 is the range-FFT output over bins ``first .. last``, one
  ``{int16 real, int16 imag}`` pair per bin (real part first). The firmware's
  copy loop stops one bin early, so the last pair is never written and arrives
  as zero; the decoder keeps it and says so instead of treating it as data.

Chest-wall displacement follows from the phase of the range bin that holds the
subject. A displacement :math:`d` changes the round-trip path by :math:`2d`, so

.. math:: d = \\frac{\\lambda}{4\\pi}\\,\\mathrm{unwrap}(\\angle X_k) .

Unwrapping assumes consecutive samples; across a gap in the frame numbers the
phase may have turned by any amount, so :func:`phase_displacement_mm` unwraps
every gap-free run separately.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .angle_map import decode_complex_iq
from .errors import IssueCode, Severity

if TYPE_CHECKING:
    from .frame import RadarFrame
    from .payloads import DecodeContext
    from .sensor_config import SensorConfig

#: The 128-byte summary structure, with descriptive field names.
VITAL_SUMMARY_DTYPE = np.dtype(
    [
        ("peak_range_bin", "<u2"),
        ("phase_range_bin", "<u2"),
        ("peak_magnitude", "<f4"),
        ("processing_cycles", "<u4"),
        ("first_range_bin", "<u2"),
        ("last_range_bin", "<u2"),
        ("chest_displacement_mm", "<f4"),
        ("breathing_waveform", "<f4"),
        ("heart_waveform", "<f4"),
        ("heart_rate_fft_bpm", "<f4"),
        ("heart_rate_fft_4hz_bpm", "<f4"),
        ("heart_rate_xcorr_bpm", "<f4"),
        ("heart_rate_peak_count_bpm", "<f4"),
        ("breathing_rate_fft_bpm", "<f4"),
        ("breathing_rate_xcorr_bpm", "<f4"),
        ("breathing_rate_peak_count_bpm", "<f4"),
        ("breathing_confidence", "<f4"),
        ("breathing_confidence_xcorr", "<f4"),
        ("heart_confidence", "<f4"),
        ("heart_confidence_4hz", "<f4"),
        ("heart_confidence_xcorr", "<f4"),
        ("breathing_waveform_energy", "<f4"),
        ("heart_waveform_energy", "<f4"),
        ("motion_flag", "<f4"),
        ("breathing_rate_harmonic_bpm", "<f4"),
        ("heart_rate_harmonic_bpm", "<f4"),
        ("reserved", "<f4", (8,)),
    ]
)
assert VITAL_SUMMARY_DTYPE.itemsize == 128


@dataclass(frozen=True)
class VitalSignsSummary:
    """Per-frame estimates of the Vital Signs lab firmware.

    Rates are in breaths or beats per minute, displacement in millimetres;
    confidence and energy values are the firmware's unitless figures and the
    two waveform samples are its band-pass filtered phase signals.
    """

    peak_range_bin: int
    phase_range_bin: int
    peak_magnitude: float
    processing_cycles: int
    first_range_bin: int
    last_range_bin: int
    chest_displacement_mm: float
    breathing_waveform: float
    heart_waveform: float
    heart_rate_fft_bpm: float
    heart_rate_fft_4hz_bpm: float
    heart_rate_xcorr_bpm: float
    heart_rate_peak_count_bpm: float
    breathing_rate_fft_bpm: float
    breathing_rate_xcorr_bpm: float
    breathing_rate_peak_count_bpm: float
    breathing_confidence: float
    breathing_confidence_xcorr: float
    heart_confidence: float
    heart_confidence_4hz: float
    heart_confidence_xcorr: float
    breathing_waveform_energy: float
    heart_waveform_energy: float
    motion_detected: bool
    breathing_rate_harmonic_bpm: float
    heart_rate_harmonic_bpm: float
    reserved: tuple[float, ...] = (0.0,) * 8

    @property
    def segment_bins(self) -> int:
        """Bins the accompanying range segment should hold."""
        return self.last_range_bin - self.first_range_bin + 1

    @classmethod
    def from_record(cls, record: np.void) -> VitalSignsSummary:
        """Build from one element of a :data:`VITAL_SUMMARY_DTYPE` array."""
        values: dict[str, object] = {}
        for name in VITAL_SUMMARY_DTYPE.names or ():
            if name == "motion_flag":
                values["motion_detected"] = bool(record[name] != 0)
            elif name == "reserved":
                values["reserved"] = tuple(float(v) for v in record[name])
            elif VITAL_SUMMARY_DTYPE[name].kind == "u":
                values[name] = int(record[name])
            else:
                values[name] = float(record[name])
        return cls(**values)  # type: ignore[arg-type]

    def to_record(self) -> np.ndarray:
        """One-element structured array in the wire layout."""
        record = np.zeros(1, dtype=VITAL_SUMMARY_DTYPE)
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "motion_detected":
                record["motion_flag"] = 1.0 if value else 0.0
            else:
                record[item.name] = value
        return record

    def as_dict(self) -> dict[str, float | int | bool]:
        """Every estimate by name (reserved words excluded)."""
        return {
            item.name: getattr(self, item.name) for item in fields(self) if item.name != "reserved"
        }


def decode_vital_summary(payload: memoryview, ctx: DecodeContext) -> VitalSignsSummary | None:
    """Decode the 128-byte type-0 summary."""
    if len(payload) != VITAL_SUMMARY_DTYPE.itemsize:
        ctx.report(
            IssueCode.LENGTH_MISMATCH,
            f"vital-signs summary has {len(payload)} bytes instead of "
            f"{VITAL_SUMMARY_DTYPE.itemsize}",
            Severity.ERROR,
        )
        if len(payload) < VITAL_SUMMARY_DTYPE.itemsize:
            return None
    record = np.frombuffer(payload, dtype=VITAL_SUMMARY_DTYPE, count=1)[0]
    return VitalSignsSummary.from_record(record)


@dataclass
class RangeSegment:
    """Complex range-FFT samples over a contiguous block of range bins.

    Attributes
    ----------
    values : numpy.ndarray of complex64
        One sample per bin.
    start_bin : int or None
        Absolute index of the first bin, taken from the summary of the same
        frame; ``None`` when the frame had no summary.
    """

    values: NDArray
    start_bin: int | None = None

    def __len__(self) -> int:
        return int(self.values.size)

    def _need_start(self) -> int:
        if self.start_bin is None:
            raise ValueError("the segment's first range bin is unknown (no summary in the frame)")
        return self.start_bin

    @property
    def bins(self) -> NDArray[np.int64]:
        """Absolute range-bin index of every sample."""
        return np.arange(len(self)) + self._need_start()

    @property
    def last_bin_written(self) -> bool:
        """Whether the final sample is non-zero (the lab firmware never writes it)."""
        return bool(len(self)) and bool(self.values[-1] != 0)

    def magnitude(self) -> NDArray[np.float64]:
        """Absolute value of every sample."""
        return np.abs(self.values).astype(np.float64)

    def peak_bin(self) -> int:
        """Absolute index of the strongest bin."""
        if not len(self):
            raise ValueError("the segment is empty")
        return int(np.argmax(self.magnitude())) + self._need_start()

    def value_at(self, bin_index: int) -> complex:
        """Sample of an absolute range bin."""
        offset = int(bin_index) - self._need_start()
        if not 0 <= offset < len(self):
            raise IndexError(
                f"bin {bin_index} lies outside the segment "
                f"{self.start_bin}..{self._need_start() + len(self) - 1}"
            )
        return complex(self.values[offset])


def decode_range_segment(payload: memoryview, ctx: DecodeContext) -> RangeSegment:
    """Decode the type-1 complex range segment (real part first).

    The first bin is filled in after the whole frame is decoded, from the
    summary's ``first_range_bin``.
    """
    usable = len(payload) - len(payload) % 4
    if usable != len(payload):
        ctx.report(
            IssueCode.LENGTH_MISMATCH,
            f"range segment of {len(payload)} bytes is not a whole number of complex samples",
            Severity.ERROR,
        )
    return RangeSegment(decode_complex_iq(payload[:usable], order="real_imag"))


@dataclass(frozen=True)
class DisplacementTrack:
    """Displacement over frames, unwrapped separately within gap-free runs.

    Attributes
    ----------
    frame_numbers : numpy.ndarray of int64
    displacement_mm : numpy.ndarray of float64
    run_ids : numpy.ndarray of int64
        Index of the gap-free run each sample belongs to; displacements of
        different runs are not comparable (their offsets differ by an unknown
        multiple of half a wavelength).
    """

    frame_numbers: NDArray[np.int64]
    displacement_mm: NDArray[np.float64]
    run_ids: NDArray[np.int64]

    @property
    def num_runs(self) -> int:
        """Number of gap-free runs."""
        return int(self.run_ids.max()) + 1 if self.run_ids.size else 0


def unwrap_runs(phase: ArrayLike, frame_numbers: ArrayLike) -> tuple[NDArray, NDArray]:
    """Unwrap a phase sequence within runs of consecutive frame numbers.

    Parameters
    ----------
    phase : array_like
        Wrapped phase in radians, one value per frame.
    frame_numbers : array_like of int
        Frame number of every value; a step other than +1 starts a new run.

    Returns
    -------
    unwrapped : numpy.ndarray
        Phase continued across every step inside a run, restarted at gaps.
    run_ids : numpy.ndarray of int64

    Examples
    --------
    >>> phase = [3.0, -3.0, -2.9, 3.0, -3.1]
    >>> unwrapped, runs = unwrap_runs(phase, [10, 11, 12, 20, 21])
    >>> np.round(unwrapped, 3).tolist(), runs.tolist()
    ([3.0, 3.283, 3.383, 3.0, 3.183], [0, 0, 0, 1, 1])
    """
    wrapped = np.asarray(phase, dtype=np.float64).reshape(-1)
    frames = np.asarray(frame_numbers, dtype=np.int64).reshape(-1)
    if wrapped.size != frames.size:
        raise ValueError(f"{wrapped.size} phases for {frames.size} frame numbers")
    if wrapped.size == 0:
        return wrapped.copy(), np.zeros(0, dtype=np.int64)
    starts_run = np.ones(wrapped.size, dtype=bool)
    starts_run[1:] = np.diff(frames) != 1
    run_ids = np.cumsum(starts_run) - 1
    steps = np.zeros(wrapped.size)
    steps[1:] = (np.diff(wrapped) + math.pi) % (2.0 * math.pi) - math.pi
    steps[starts_run] = 0.0
    total = np.cumsum(steps)
    run_starts = np.flatnonzero(starts_run)
    offset = total[run_starts][run_ids]
    unwrapped = wrapped[run_starts][run_ids] + (total - offset)
    return unwrapped, run_ids.astype(np.int64)


def phase_displacement_mm(
    segments: Sequence[RangeSegment],
    bin_index: int,
    wavelength_m: float,
    frame_numbers: ArrayLike,
) -> DisplacementTrack:
    """Displacement of the reflector in one range bin from its phase.

    Parameters
    ----------
    segments : sequence of RangeSegment
        One segment per frame, each with its ``start_bin`` set.
    bin_index : int
        Absolute range bin of the subject.
    wavelength_m : float
        Carrier wavelength (``SensorConfig.wavelength_m``).
    frame_numbers : array_like of int
        Frame number of every segment; gaps split the unwrapping.

    Returns
    -------
    DisplacementTrack
    """
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be positive, got {wavelength_m}")
    frames = np.asarray(frame_numbers, dtype=np.int64).reshape(-1)
    if len(segments) != frames.size:
        raise ValueError(f"{len(segments)} segments for {frames.size} frame numbers")
    samples = np.array([segment.value_at(bin_index) for segment in segments], dtype=np.complex128)
    unwrapped, run_ids = unwrap_runs(np.angle(samples), frames)
    displacement = unwrapped * wavelength_m / (4.0 * math.pi) * 1e3
    return DisplacementTrack(frames, displacement, run_ids)


def chest_displacement_track(
    frames: Iterable[RadarFrame],
    *,
    wavelength_m: float | None = None,
    sensor: SensorConfig | None = None,
    bin_index: int | None = None,
) -> DisplacementTrack:
    """Displacement track from decoded Vital Signs frames.

    Parameters
    ----------
    frames : iterable of RadarFrame
        Frames with a range segment; frames without one are skipped.
    wavelength_m : float, optional
        Carrier wavelength; taken from ``sensor`` when omitted.
    sensor : SensorConfig, optional
        Radar configuration supplying the wavelength.
    bin_index : int, optional
        Range bin to track; defaults to the most frequent ``phase_range_bin``
        of the summaries.

    Raises
    ------
    ValueError
        If no wavelength is given or no frame carries a usable segment.
    """
    if wavelength_m is None:
        if sensor is None:
            raise ValueError("pass wavelength_m or a radar configuration")
        wavelength_m = sensor.wavelength_m
    usable = [
        f for f in frames if f.range_segment is not None and f.range_segment.start_bin is not None
    ]
    if not usable:
        raise ValueError("no frame carries a range segment with a known first bin")
    if bin_index is None:
        votes = [f.vital_signs.phase_range_bin for f in usable if f.vital_signs is not None]
        if not votes:
            raise ValueError("pass bin_index: the frames carry no vital-signs summary")
        values, counts = np.unique(votes, return_counts=True)
        bin_index = int(values[np.argmax(counts)])
    segments = [f.range_segment for f in usable if f.range_segment is not None]
    return phase_displacement_mm(
        segments, bin_index, wavelength_m, [f.frame_number for f in usable]
    )
