"""Settings of the decoder and the viewer.

Every tunable number of the pipeline is a field of one of two dataclasses.
:class:`DecoderConfig` covers how packets are assembled, validated and turned
into physical units: schema selection, strictness, de-duplication, the
timestamp clock, the fall-back bin counts for when no radar configuration is
available, and the SNR estimator. :class:`ViewerConfig` holds the playback
rate and display limits of the GUI.

Both are frozen, validate themselves on construction and round-trip through
JSON (:meth:`DecoderConfig.from_json`, :meth:`DecoderConfig.to_dict`), so a run
can be reproduced from the file it was configured with. Neither holds a path;
input and output locations always come from the caller.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

from .constants import ANGLE_FFT_SIZE, DEFAULT_MAX_PACKET_BYTES, MAX_TLVS_PER_PACKET, parse_platform
from .errors import ConfigError, SchemaError
from .schema import PacketSchema, parse_schema
from .units import Log2Combining


class NoiseEstimator(str, Enum):
    """How the local noise level around a detection is estimated."""

    CELL_AVERAGING = "ca"
    SMALLEST_OF = "so"


@dataclass(frozen=True)
class DecoderConfig:
    """Options of packet assembly and decoding.

    Attributes
    ----------
    schema : PacketSchema or None
        Firmware family; ``None`` detects it per packet.
    platform : int or None
        Overrides the header's platform code for the unit rules.
    strict : bool
        Reject packets with any error-level issue, unaligned lengths, excess
        padding or lost bytes, instead of salvaging what is intact.
    deduplicate : bool
        Drop byte-identical repeats of recent frames (recorders that dump
        whole buffers write every packet more than once).
    dedupe_window : int
        How many recent frames a repeat is looked for in.
    max_packet_bytes : int
        Longest packet considered genuine.
    max_tlvs : int
        Largest TLV count considered genuine.
    chunk_size : int
        Bytes read from a file at a time.
    clock_hz : float or None
        Clock of the header timestamp; ``None`` uses the platform default.
    nominal_frame_period_s : float or None
        Frame period used to resolve counter wrap-around; ``None`` estimates it.
    range_bias_m : float or None
        Overrides the range bias of the radar configuration.
    num_range_bins, num_doppler_bins : int or None
        Fall-back FFT sizes when no radar configuration is given.
    num_virtual_antennas : int or None
        Fall-back azimuth virtual-antenna count (heatmap shape and the antenna
        normalisation of summed log2 magnitudes).
    db_per_lsb : float or None
        Forces the decibel value of one log2-magnitude step.
    log2_fraction_bits : int or None
        Forces the Q-format of log2 magnitudes (8 or 9 in TI firmware).
    log2_combining : Log2Combining or None
        Forces how log2 magnitudes are combined over the virtual antennas:
        ``"average"``, ``"sum"`` or ``"shifted-sum"`` (see
        :class:`~radar_stream_decoder.units.Log2Combining`).
    unsigned_doppler : bool or None
        Forces unsigned Doppler indices (SDK 1.0); ``None`` decides by version.
    snr_guard_cells, snr_train_cells : int
        Guard and training cells on each side for the noise estimate.
    snr_estimator : NoiseEstimator
        Cell averaging or smallest-of.
    snr_log_bias_correction : bool
        Remove the 2.51 dB bias of averaging logarithms of Rayleigh noise.
    angle_fft_size : int
        Angle FFT length for azimuth heatmaps.
    keep_raw_packets : bool
        Keep each packet's bytes on its frame (needed for re-recording).

    Examples
    --------
    >>> config = DecoderConfig(schema="vital-signs", strict=True)
    >>> config.schema
    <PacketSchema.VITAL_SIGNS: 'vital-signs'>
    >>> DecoderConfig.from_dict(config.to_dict()) == config
    True
    """

    schema: PacketSchema | None = None
    platform: int | None = None
    strict: bool = False
    deduplicate: bool = True
    dedupe_window: int = 64
    max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES
    max_tlvs: int = MAX_TLVS_PER_PACKET
    chunk_size: int = 65_536
    clock_hz: float | None = None
    nominal_frame_period_s: float | None = None
    range_bias_m: float | None = None
    num_range_bins: int | None = None
    num_doppler_bins: int | None = None
    num_virtual_antennas: int | None = None
    db_per_lsb: float | None = None
    log2_fraction_bits: int | None = None
    log2_combining: Log2Combining | None = None
    unsigned_doppler: bool | None = None
    snr_guard_cells: int = 2
    snr_train_cells: int = 8
    snr_estimator: NoiseEstimator = NoiseEstimator.CELL_AVERAGING
    snr_log_bias_correction: bool = False
    angle_fft_size: int = ANGLE_FFT_SIZE
    keep_raw_packets: bool = True

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "schema", parse_schema(self.schema))
        except SchemaError as exc:
            raise ConfigError(str(exc)) from None
        try:
            object.__setattr__(self, "platform", parse_platform(self.platform))
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
        if self.log2_combining is not None:
            try:
                object.__setattr__(self, "log2_combining", Log2Combining(self.log2_combining))
            except ValueError:
                choices = ", ".join(mode.value for mode in Log2Combining)
                raise ConfigError(
                    f"log2_combining must be one of {choices}, got {self.log2_combining!r}"
                ) from None
        try:
            object.__setattr__(self, "snr_estimator", NoiseEstimator(self.snr_estimator))
        except ValueError:
            raise ConfigError(
                f"snr_estimator must be 'ca' or 'so', got {self.snr_estimator!r}"
            ) from None
        positive = {
            "dedupe_window": self.dedupe_window,
            "max_packet_bytes": self.max_packet_bytes,
            "max_tlvs": self.max_tlvs,
            "chunk_size": self.chunk_size,
            "snr_train_cells": self.snr_train_cells,
            "angle_fft_size": self.angle_fft_size,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ConfigError(f"{name} must be positive, got {value}")
        optional_positive = {
            "clock_hz": self.clock_hz,
            "nominal_frame_period_s": self.nominal_frame_period_s,
            "num_range_bins": self.num_range_bins,
            "num_doppler_bins": self.num_doppler_bins,
            "num_virtual_antennas": self.num_virtual_antennas,
            "db_per_lsb": self.db_per_lsb,
        }
        for name, value in optional_positive.items():
            if value is not None and value <= 0:
                raise ConfigError(f"{name} must be positive when given, got {value}")
        if self.snr_guard_cells < 0:
            raise ConfigError(f"snr_guard_cells cannot be negative, got {self.snr_guard_cells}")
        if self.log2_fraction_bits is not None and not 0 <= self.log2_fraction_bits <= 15:
            raise ConfigError(
                f"log2_fraction_bits must lie in 0..15, got {self.log2_fraction_bits}"
            )
        if self.max_packet_bytes < 64:
            raise ConfigError("max_packet_bytes below 64 would reject every real packet")
        if self.angle_fft_size % 2:
            raise ConfigError(f"angle_fft_size must be even, got {self.angle_fft_size}")

    # -- serialisation -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary of every field."""
        result: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Enum):
                value = value.value
            result[item.name] = value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecoderConfig:
        """Build from a mapping, rejecting unknown keys with the list of valid ones."""
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                f"unknown decoder setting(s): {', '.join(unknown)}; "
                f"valid settings are {', '.join(sorted(known))}"
            )
        return cls(**dict(data))

    @classmethod
    def from_json(cls, path: str | Path) -> DecoderConfig:
        """Load from a JSON object file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ConfigError
            If it is not a JSON object of known settings.
        """
        file_path = Path(path)
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{file_path.name} is not valid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ConfigError(f"{file_path.name} must contain a JSON object")
        return cls.from_dict(data)

    def to_json(self, path: str | Path) -> Path:
        """Write the settings as an indented JSON object and return the path."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        file_path.write_text(text, encoding="utf-8", newline="\n")
        return file_path

    def replace(self, **changes: Any) -> DecoderConfig:
        """Copy with some fields changed (validated again)."""
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class ViewerConfig:
    """Playback and display settings of the GUI.

    Attributes
    ----------
    fps : float
        Playback rate of recorded or synthetic frames.
    max_points : int
        Points drawn per frame at most (the strongest are kept).
    point_size_range : tuple of float
        Marker areas, in points squared, of the weakest and strongest point.
    demo_frames : int
        Frames generated for the synthetic demonstration.
    demo_seed : int
        Seed of the synthetic demonstration.
    snr_limits_db : tuple of float
        Colour range when points are coloured by SNR.
    live_history_frames : int
        Frames a live source keeps for display; older ones are released so a
        session can run indefinitely (60 s at 20 fps by default).
    """

    fps: float = 20.0
    max_points: int = 512
    point_size_range: tuple[float, float] = (12.0, 90.0)
    demo_frames: int = 400
    demo_seed: int = 7
    snr_limits_db: tuple[float, float] = (0.0, 40.0)
    live_history_frames: int = 1200

    def __post_init__(self) -> None:
        if not 0 < self.fps <= 240:
            raise ConfigError(f"fps must lie in (0, 240], got {self.fps}")
        if self.max_points <= 0 or self.demo_frames <= 0 or self.live_history_frames <= 0:
            raise ConfigError("max_points, demo_frames and live_history_frames must be positive")
        low, high = self.point_size_range
        if not 0 < low <= high:
            raise ConfigError(f"point_size_range must be increasing and positive, got {low, high}")
        if self.snr_limits_db[0] >= self.snr_limits_db[1]:
            raise ConfigError("snr_limits_db must be increasing")

    @property
    def frame_interval_ms(self) -> int:
        """Timer interval for the playback rate."""
        return max(1, round(1000.0 / self.fps))
