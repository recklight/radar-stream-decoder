"""Synthetic radar streams for tests, examples, evaluation and the viewer's demo.

Scenes follow a simple FMCW model rather than random bytes, so that every
decoded quantity can be checked against a known truth:

* targets move in straight lines or swing back and forth; their range and
  radial velocity are quantised to range and Doppler bins exactly as the
  radar configuration dictates;
* the detection matrix is Rayleigh noise (exponential power, independent per
  cell) plus each target spread over its 3 x 3 neighbourhood, written as
  log2 magnitudes with the platform's fixed-point scale; the range profile is
  its zero-Doppler column and the noise profile its highest-Doppler column,
  the same cuts the firmware takes;
* the azimuth heatmap holds complex Gaussian noise plus, for every static
  target, the TDM-MIMO steering vector :math:`e^{j\\pi n\\sin\\theta}` of a
  half-wavelength virtual array;
* points are the targets whose SNR clears a detection threshold, quantised
  to the Q-format of the configuration;
* Vital Signs scenes put a breathing, beating chest wall into one range bin,
  with phase :math:`4\\pi (R + d(t))/\\lambda`;
* people counting and tracking scenes scatter a few spherical points around
  every visible target, add stray noise points, report each target with its
  true state and send the previous frame's point-to-target IDs, as the
  on-sensor tracker does.

:func:`inject_faults` then damages a packet sequence the way real links and
recorders do (junk, fake sync words, drops, packets cut short, flipped bits,
repeats, whole-buffer dumps) and reports what happened to every packet, so
the decoder's recovery can be scored. Everything is driven by
``numpy.random.default_rng(seed)``: one seed, one byte-identical stream.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import struct
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .constants import (
    CYCLE_COUNTER_MODULUS,
    DB_PER_LOG2_UNIT,
    MAGIC_WORD,
    UNASSOCIATED_NOISE,
    DemoTlvType,
    PeopleCountingTlvType,
    PeopleTrackingTlvType,
    Platform,
    VitalSignsTlvType,
    cpu_clock_hz,
    parse_platform,
)
from .encoder import (
    HeaderTemplate,
    encode_azimuth_heatmap,
    encode_compressed_points,
    encode_frame,
    encode_magnitude_profile,
    encode_points_float,
    encode_points_qformat,
    encode_polar_points,
    encode_presence,
    encode_range_doppler,
    encode_range_segment,
    encode_side_info,
    encode_stats,
    encode_target_association,
    encode_target_heights,
    encode_target_list,
    encode_temperature,
    encode_vital_summary,
)
from .frame import resolve_header_len
from .header import (
    DEFAULT_LIMITS,
    HeaderLayout,
    HeaderTiming,
    examine_header,
    walk_tlv_chain,
)
from .payloads import PayloadKind, PointCloud, ProcessingStats, TemperatureReport
from .schema import PacketSchema, parse_schema
from .sensor_config import SensorConfig, default_preset_for, preset_sensor_config
from .tracking import TargetHeights, TargetList
from .units import cartesian_to_spherical, log2_magnitude_scale
from .vital_signs import VitalSignsSummary


@dataclass(frozen=True)
class Target:
    """A point reflector.

    Attributes
    ----------
    x_m, y_m, z_m : float
        Position at ``t = 0`` (``y`` along the boresight).
    vx_mps, vy_mps, vz_mps : float
        Velocity. With ``swing_period_s > 0`` this is the peak velocity of a
        back-and-forth motion, otherwise a constant velocity.
    rcs_db : float
        Reflectivity relative to the scene's reference target.
    swing_period_s : float
        Period of the back-and-forth motion; 0 for straight-line motion.
    """

    x_m: float
    y_m: float
    z_m: float = 0.0
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    vz_mps: float = 0.0
    rcs_db: float = 0.0
    swing_period_s: float = 0.0

    def state_at(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Position and velocity at time ``t``."""
        start = np.array([self.x_m, self.y_m, self.z_m])
        velocity = np.array([self.vx_mps, self.vy_mps, self.vz_mps])
        if self.swing_period_s > 0:
            omega = 2.0 * math.pi / self.swing_period_s
            return start + velocity * math.sin(omega * t) / omega, velocity * math.cos(omega * t)
        return start + velocity * t, velocity


#: A small room: two people walking to and fro, a slow drifter and two static reflectors.
DEFAULT_TARGETS: tuple[Target, ...] = (
    Target(-1.2, 3.5, 0.0, 0.5, -0.9, 0.0, rcs_db=4.0, swing_period_s=9.0),
    Target(1.0, 2.2, 0.2, -0.6, 0.7, 0.0, rcs_db=0.0, swing_period_s=6.0),
    Target(-2.5, 6.0, 0.0, 0.25, 0.15, 0.0, rcs_db=6.0, swing_period_s=14.0),
    Target(0.3, 5.0, 0.0, rcs_db=10.0),
    Target(2.4, 7.0, 0.4, rcs_db=13.0),
)

#: Payloads a scene emits unless told otherwise.
DEFAULT_OUTPUTS: frozenset[PayloadKind] = frozenset(
    {
        PayloadKind.POINTS,
        PayloadKind.RANGE_PROFILE,
        PayloadKind.NOISE_PROFILE,
        PayloadKind.AZIMUTH_HEATMAP,
        PayloadKind.RANGE_DOPPLER,
        PayloadKind.STATS,
    }
)

#: Additional SDK 3.x payloads emitted when the scene uses that schema.
SDK3_EXTRA_OUTPUTS: frozenset[PayloadKind] = frozenset(
    {PayloadKind.POINT_SIDE_INFO, PayloadKind.TEMPERATURE}
)

#: Payloads of people counting and tracking scenes whose outputs are the default.
TRACKING_OUTPUTS: frozenset[PayloadKind] = frozenset(
    {
        PayloadKind.POINTS,
        PayloadKind.TARGETS,
        PayloadKind.TARGET_ASSOCIATION,
        PayloadKind.TARGET_HEIGHTS,
        PayloadKind.PRESENCE,
    }
)


@dataclass(frozen=True)
class ChestMotion:
    """Chest-wall motion of a seated subject for Vital Signs scenes."""

    range_m: float = 0.6
    breathing_rate_bpm: float = 15.0
    breathing_amplitude_mm: float = 4.0
    heart_rate_bpm: float = 72.0
    heart_amplitude_mm: float = 0.3

    def displacement_mm(self, t: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Breathing plus heartbeat displacement at time ``t``."""
        times = np.asarray(t, dtype=np.float64)
        breathing = self.breathing_amplitude_mm * np.sin(
            2.0 * math.pi * self.breathing_rate_bpm / 60.0 * times
        )
        heart = self.heart_amplitude_mm * np.sin(
            2.0 * math.pi * self.heart_rate_bpm / 60.0 * times + 0.7
        )
        return breathing + heart


@dataclass(frozen=True)
class SyntheticScene:
    """Everything that determines a synthetic stream.

    Attributes
    ----------
    targets : tuple of Target
        Reflectors of out-of-box demo scenes.
    platform : int
        Platform code written into the headers (forced to xWR14xx for the
        Vital Signs schema and to xWR68xx for the 3D people counting and the
        people tracking schemas, whose firmware writes nothing else).
    schema : PacketSchema
        Firmware family to imitate.
    sensor : SensorConfig or None
        Radar configuration; a matching preset when ``None``.
    outputs : frozenset of PayloadKind
        Payloads to emit (``PayloadKind`` members or their string values).
    seed : int
        Seed of every random draw.
    noise_floor_db : float
        Mean noise power in the detection matrix's decibel scale.
    reference_snr_db : float
        SNR of a 0 dB target at ``reference_range_m``; it falls with the
        fourth power of range.
    reference_range_m : float
    detection_threshold_db : float
        SNR a target needs to appear in the point list.
    padding_byte : int
        Packet padding value (``0x0F`` imitates SDK 2.0 debug builds).
    first_frame_number, first_cpu_cycles : int
        Counters of the first packet.
    timing_jitter_cycles : float
        Standard deviation of the timestamp jitter.
    declared_point_count : int or None
        Header point count; ``None`` writes the true count (the Vital Signs
        schema defaults to the constant 99 its firmware writes).
    chest : ChestMotion
        Subject of Vital Signs scenes.
    zero_last_segment_bin : bool
        Leave the last range-segment bin zero, like the Vital Signs firmware.
    """

    targets: tuple[Target, ...] = DEFAULT_TARGETS
    platform: int = Platform.XWR16XX
    schema: PacketSchema = PacketSchema.OOB_SDK2
    sensor: SensorConfig | None = None
    outputs: frozenset[PayloadKind] = DEFAULT_OUTPUTS
    seed: int = 0
    noise_floor_db: float = 40.0
    reference_snr_db: float = 30.0
    reference_range_m: float = 3.0
    detection_threshold_db: float = 12.0
    padding_byte: int = 0
    first_frame_number: int = 1
    first_cpu_cycles: int = 0
    timing_jitter_cycles: float = 2000.0
    declared_point_count: int | None = None
    chest: ChestMotion = field(default_factory=ChestMotion)
    zero_last_segment_bin: bool = True

    def __post_init__(self) -> None:
        schema = parse_schema(self.schema)
        object.__setattr__(self, "schema", schema or PacketSchema.OOB_SDK2)
        platform = parse_platform(self.platform)
        if self.schema is PacketSchema.VITAL_SIGNS:
            platform = Platform.XWR14XX
        elif self.schema in (PacketSchema.PEOPLE_COUNTING_3D, PacketSchema.PEOPLE_TRACKING):
            platform = Platform.XWR68XX
        elif platform is None:
            platform = (
                Platform.XWR16XX
                if self.sensor is None or self.sensor.platform is None
                else self.sensor.platform
            )
        object.__setattr__(self, "platform", int(platform))
        object.__setattr__(self, "outputs", frozenset(PayloadKind(o) for o in self.outputs))
        object.__setattr__(self, "targets", tuple(self.targets))
        if not 0 <= self.padding_byte <= 0xFF:
            raise ValueError(f"padding_byte must be a byte value, got {self.padding_byte}")

    def radar(self) -> SensorConfig:
        """The radar configuration the scene is rendered with."""
        if self.sensor is not None:
            return self.sensor
        if self.schema is PacketSchema.VITAL_SIGNS:
            return preset_sensor_config("iwr1443_vital_signs")
        return preset_sensor_config(default_preset_for(self.platform))

    def replace(self, **changes: object) -> SyntheticScene:
        """Copy with some fields changed."""
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FrameTruth:
    """Ground truth of one synthetic out-of-box frame.

    Arrays have one entry per scene target, visible or not.
    """

    frame_number: int
    time_s: float
    position_m: NDArray[np.float64]
    radial_velocity_mps: NDArray[np.float64]
    range_m: NDArray[np.float64]
    azimuth_deg: NDArray[np.float64]
    range_idx: NDArray[np.int64]
    doppler_idx: NDArray[np.int64]
    snr_db: NDArray[np.float64]
    visible: NDArray[np.bool_]
    detected: NDArray[np.bool_]


class _DemoRenderer:
    """Renders out-of-box demo packets (SDK 1.x/2.x and 3.x) of a scene."""

    def __init__(self, scene: SyntheticScene) -> None:
        self.scene = scene
        radar = scene.radar()
        self.radar = radar
        self.num_range_bins = radar.num_range_bins
        self.num_doppler_bins = radar.num_doppler_bins
        self.num_antennas = radar.num_virtual_antennas_azimuth
        self.range_bin_m = radar.range_bin_m
        self.doppler_resolution = radar.doppler_resolution_mps
        self.frame_period_s = radar.frame_period_s
        self.max_range_m = min(radar.max_range_m, (self.num_range_bins - 2) * self.range_bin_m)
        self.q_format = radar.xyz_q_format
        self.db_per_lsb = (
            log2_magnitude_scale(
                scene.platform,
                scene.schema,
                radar.num_rx,
                radar.num_tx,
                radar.num_tx_elevation > 0,
            )
            * DB_PER_LOG2_UNIT
        )
        self.clock_hz = cpu_clock_hz(scene.platform, scene.schema)
        outputs = set(scene.outputs)
        if scene.schema is PacketSchema.OOB_SDK3:
            outputs |= SDK3_EXTRA_OUTPUTS
        self.outputs = frozenset(outputs)

    def truth(self, index: int) -> FrameTruth:
        scene = self.scene
        t = index * self.frame_period_s
        if scene.targets:
            states = [target.state_at(t) for target in scene.targets]
            position = np.array([state[0] for state in states])
            velocity = np.array([state[1] for state in states])
        else:
            position = np.zeros((0, 3))
            velocity = np.zeros((0, 3))
        distance = np.linalg.norm(position, axis=1)
        safe = np.maximum(distance, 1e-9)
        radial = np.einsum("ij,ij->i", position, velocity) / safe
        azimuth = np.degrees(np.arctan2(position[:, 0], position[:, 1]))
        range_idx = np.rint(distance / self.range_bin_m).astype(np.int64)
        half = self.num_doppler_bins // 2
        doppler_idx = np.rint(radial / self.doppler_resolution).astype(np.int64)
        doppler_idx = (doppler_idx + half) % self.num_doppler_bins - half
        rcs = np.array([target.rcs_db for target in scene.targets])
        snr = scene.reference_snr_db + rcs - 40.0 * np.log10(safe / scene.reference_range_m)
        visible = (
            (position[:, 1] > 0.0)
            & (range_idx >= 1)
            & (range_idx < self.num_range_bins - 1)
            & (distance < self.max_range_m)
            & (np.abs(azimuth) <= 60.0)
        )
        detected = visible & (snr >= scene.detection_threshold_db)
        return FrameTruth(
            frame_number=scene.first_frame_number + index,
            time_s=t,
            position_m=position,
            radial_velocity_mps=radial,
            range_m=distance,
            azimuth_deg=azimuth,
            range_idx=range_idx,
            doppler_idx=doppler_idx,
            snr_db=snr,
            visible=visible,
            detected=detected,
        )

    def detection_matrix(self, truth: FrameTruth, rng: np.random.Generator) -> NDArray[np.uint16]:
        rows, cols = self.num_range_bins, self.num_doppler_bins
        floor = self.scene.noise_floor_db
        power = 10.0 ** (floor / 10.0) * rng.exponential(1.0, size=(rows, cols))
        visible = np.flatnonzero(truth.visible)
        if visible.size:
            dr, dd = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")
            dr, dd = dr.reshape(-1), dd.reshape(-1)
            target_rows = truth.range_idx[visible][:, None] + dr[None, :]
            target_cols = (truth.doppler_idx[visible][:, None] + dd[None, :]) % cols
            level_db = (
                floor
                + np.minimum(truth.snr_db[visible], 60.0)[:, None]
                - 6.0 * (np.abs(dr) + np.abs(dd))[None, :]
            )
            inside = (target_rows >= 0) & (target_rows < rows)
            np.add.at(
                power,
                (target_rows[inside], target_cols[inside]),
                10.0 ** (level_db[inside] / 10.0),
            )
        level = 10.0 * np.log10(power)
        return np.clip(np.rint(level / self.db_per_lsb), 0, 65535).astype(np.uint16)

    def azimuth_cube(self, truth: FrameTruth, rng: np.random.Generator) -> NDArray:
        rows, antennas = self.num_range_bins, self.num_antennas
        sigma = 64.0
        cube = (rng.normal(size=(rows, antennas)) + 1j * rng.normal(size=(rows, antennas))) * (
            sigma / math.sqrt(2.0)
        )
        static = np.flatnonzero(truth.visible & (truth.doppler_idx == 0))
        elements = np.arange(antennas)
        for index in static:
            amplitude = sigma * 10.0 ** (min(float(truth.snr_db[index]), 45.0) / 20.0)
            steering = np.exp(
                1j * math.pi * math.sin(math.radians(truth.azimuth_deg[index])) * elements
            )
            row = int(truth.range_idx[index])
            cube[row] += amplitude * steering
            for neighbour in (row - 1, row + 1):
                if 0 <= neighbour < rows:
                    cube[neighbour] += 0.5 * amplitude * steering
        return cube

    def stats(self, rng: np.random.Generator) -> ProcessingStats:
        period_us = int(self.frame_period_s * 1e6)
        processing = int(rng.integers(900, 1600))
        transmit = int(rng.integers(200, 900))
        return ProcessingStats(
            inter_frame_processing_time_us=processing,
            transmit_output_time_us=transmit,
            inter_frame_processing_margin_us=max(0, period_us - processing - transmit - 3000),
            inter_chirp_processing_margin_us=int(rng.integers(15, 40)),
            active_frame_cpu_load_pct=int(rng.integers(20, 60)),
            inter_frame_cpu_load_pct=int(rng.integers(5, 25)),
        )

    def packet(self, index: int, rng: np.random.Generator) -> bytes:
        scene = self.scene
        truth = self.truth(index)
        matrix = self.detection_matrix(truth, rng)
        outputs = self.outputs
        detected = np.flatnonzero(truth.detected)
        sdk3 = scene.schema is PacketSchema.OOB_SDK3
        tlvs: list[tuple[int, bytes]] = []

        if PayloadKind.POINTS in outputs and detected.size:
            position = truth.position_m[detected]
            if sdk3:
                cloud = PointCloud(
                    position[:, 0],
                    position[:, 1],
                    position[:, 2],
                    velocity_mps=truth.radial_velocity_mps[detected],
                )
                tlvs.append((DemoTlvType.POINTS, encode_points_float(cloud)))
                if PayloadKind.POINT_SIDE_INFO in outputs:
                    snr = np.minimum(truth.snr_db[detected], 60.0)
                    noise = np.full(detected.size, scene.noise_floor_db)
                    tlvs.append((DemoTlvType.POINT_SIDE_INFO, encode_side_info(snr, noise)))
            else:
                ranges = truth.range_idx[detected]
                dopplers = truth.doppler_idx[detected]
                if scene.platform == Platform.XWR16XX:
                    peak = 100.0 * 10.0 ** (np.minimum(truth.snr_db[detected], 55.0) / 20.0)
                else:
                    peak = matrix[ranges, dopplers % self.num_doppler_bins].astype(np.float64)
                cloud = PointCloud(
                    position[:, 0],
                    position[:, 1],
                    position[:, 2],
                    range_idx=ranges,
                    doppler_idx=dopplers,
                    peak=peak,
                )
                tlvs.append((DemoTlvType.POINTS, encode_points_qformat(cloud, self.q_format)))
        if PayloadKind.RANGE_PROFILE in outputs:
            tlvs.append((DemoTlvType.RANGE_PROFILE, encode_magnitude_profile(matrix[:, 0])))
        if PayloadKind.NOISE_PROFILE in outputs:
            noise_column = self.num_doppler_bins // 2 - 1
            tlvs.append(
                (DemoTlvType.NOISE_PROFILE, encode_magnitude_profile(matrix[:, noise_column]))
            )
        if PayloadKind.AZIMUTH_HEATMAP in outputs:
            cube = self.azimuth_cube(truth, rng)
            tlvs.append((DemoTlvType.AZIMUTH_HEATMAP, encode_azimuth_heatmap(cube)))
        if PayloadKind.RANGE_DOPPLER in outputs:
            tlvs.append((DemoTlvType.RANGE_DOPPLER_HEATMAP, encode_range_doppler(matrix)))
        if PayloadKind.STATS in outputs:
            tlvs.append((DemoTlvType.STATS, encode_stats(self.stats(rng))))
        if sdk3 and PayloadKind.TEMPERATURE in outputs:
            report = TemperatureReport(
                status=0,
                time_ms=int(truth.time_s * 1e3),
                rx_c=tuple(int(v) for v in rng.integers(40, 47, size=4)),  # type: ignore[arg-type]
                tx_c=tuple(int(v) for v in rng.integers(43, 50, size=3)),  # type: ignore[arg-type]
                power_management_c=int(rng.integers(38, 44)),
                digital_c=tuple(int(v) for v in rng.integers(44, 50, size=2)),  # type: ignore[arg-type]
            )
            tlvs.append((DemoTlvType.TEMPERATURE, encode_temperature(report)))

        declared = scene.declared_point_count
        template = HeaderTemplate.for_schema(
            scene.schema,
            scene.platform,
            frame_number=truth.frame_number,
            cpu_cycles=_timestamp(scene, truth.time_s, self.clock_hz, rng),
            declared_point_count=detected.size if declared is None else declared,
        )
        return encode_frame(template, tlvs, scene.padding_byte)


def _timestamp(scene: SyntheticScene, t: float, clock_hz: float, rng: np.random.Generator) -> int:
    jitter = rng.normal(0.0, scene.timing_jitter_cycles) if scene.timing_jitter_cycles else 0.0
    return round(float(scene.first_cpu_cycles + t * clock_hz + jitter)) % CYCLE_COUNTER_MODULUS


class _VitalSignsRenderer:
    """Renders Vital Signs lab packets: summary plus complex range segment."""

    #: Range interval the lab firmware searches for the subject.
    SEARCH_START_M = 0.3
    SEARCH_END_M = 1.0

    def __init__(self, scene: SyntheticScene) -> None:
        self.scene = scene
        radar = scene.radar()
        self.radar = radar
        self.frame_period_s = radar.frame_period_s
        self.wavelength_m = radar.wavelength_m
        bin_m = radar.range_bin_m
        self.first_bin = int(self.SEARCH_START_M / bin_m)
        self.last_bin = int(self.SEARCH_END_M / bin_m)
        self.subject_bin = round(scene.chest.range_m / bin_m)
        if not self.first_bin <= self.subject_bin < self.last_bin:
            raise ValueError(
                f"the subject at {scene.chest.range_m} m lies outside the searched "
                f"{self.SEARCH_START_M}-{self.SEARCH_END_M} m interval"
            )
        self.clock_hz = cpu_clock_hz(scene.platform, scene.schema)
        clutter_rng = np.random.default_rng([scene.seed, 1])
        bins = np.arange(self.first_bin, self.last_bin + 1)
        self.clutter = (
            300.0
            * np.exp(-(bins - self.first_bin) / 6.0)
            * np.exp(2j * math.pi * clutter_rng.random(bins.size))
        )

    def packet(self, index: int, rng: np.random.Generator) -> bytes:
        scene = self.scene
        chest = scene.chest
        t = index * self.frame_period_s
        breathing = chest.breathing_amplitude_mm * math.sin(
            2.0 * math.pi * chest.breathing_rate_bpm / 60.0 * t
        )
        heart = chest.heart_amplitude_mm * math.sin(
            2.0 * math.pi * chest.heart_rate_bpm / 60.0 * t + 0.7
        )
        displacement_mm = breathing + heart
        phase = 4.0 * math.pi * (chest.range_m + displacement_mm * 1e-3) / self.wavelength_m
        size = self.last_bin - self.first_bin + 1
        values = self.clutter.copy()
        values += (rng.normal(size=size) + 1j * rng.normal(size=size)) * 20.0
        centre = self.subject_bin - self.first_bin
        values[centre] += 4000.0 * np.exp(1j * phase)
        for neighbour in (centre - 1, centre + 1):
            values[neighbour] += 1800.0 * np.exp(1j * (phase + 0.3))
        if scene.zero_last_segment_bin:
            values[-1] = 0.0

        def noisy(value: float, spread: float) -> float:
            return float(value + rng.normal(0.0, spread))

        heart_bpm, breathing_bpm = chest.heart_rate_bpm, chest.breathing_rate_bpm
        summary = VitalSignsSummary(
            peak_range_bin=self.subject_bin,
            phase_range_bin=self.subject_bin,
            peak_magnitude=float(abs(values[centre])),
            processing_cycles=int(rng.integers(150_000, 250_000)),
            first_range_bin=self.first_bin,
            last_range_bin=self.last_bin,
            chest_displacement_mm=float(displacement_mm),
            breathing_waveform=float(breathing),
            heart_waveform=float(heart),
            heart_rate_fft_bpm=noisy(heart_bpm, 1.5),
            heart_rate_fft_4hz_bpm=noisy(heart_bpm, 2.0),
            heart_rate_xcorr_bpm=noisy(heart_bpm, 2.5),
            heart_rate_peak_count_bpm=noisy(heart_bpm, 3.0),
            breathing_rate_fft_bpm=noisy(breathing_bpm, 0.5),
            breathing_rate_xcorr_bpm=noisy(breathing_bpm, 0.8),
            breathing_rate_peak_count_bpm=noisy(breathing_bpm, 1.0),
            breathing_confidence=float(rng.uniform(1.0, 4.0)),
            breathing_confidence_xcorr=float(rng.uniform(0.3, 0.9)),
            heart_confidence=float(rng.uniform(0.5, 2.5)),
            heart_confidence_4hz=float(rng.uniform(0.5, 2.5)),
            heart_confidence_xcorr=float(rng.uniform(0.2, 0.8)),
            breathing_waveform_energy=float(rng.uniform(1e3, 5e3)),
            heart_waveform_energy=float(rng.uniform(10.0, 80.0)),
            motion_detected=False,
            breathing_rate_harmonic_bpm=noisy(breathing_bpm, 0.7),
            heart_rate_harmonic_bpm=noisy(heart_bpm, 2.0),
        )
        declared = 99 if scene.declared_point_count is None else scene.declared_point_count
        template = HeaderTemplate.for_schema(
            PacketSchema.VITAL_SIGNS,
            Platform.XWR14XX,
            frame_number=scene.first_frame_number + index,
            cpu_cycles=_timestamp(scene, t, self.clock_hz, rng),
            declared_point_count=declared,
        )
        tlvs = [
            (VitalSignsTlvType.SUMMARY, encode_vital_summary(summary)),
            (VitalSignsTlvType.RANGE_SEGMENT, encode_range_segment(values)),
        ]
        return encode_frame(template, tlvs, scene.padding_byte)


class _TrackingRenderer:
    """Renders people counting and people tracking packets of a scene."""

    #: Points scattered around a target of reference reflectivity.
    POINTS_PER_TARGET = 6
    #: Spread of a target's points around its centre, metres.
    POINT_SPREAD_M = 0.12
    #: Stray points per frame that the tracker labels as noise.
    NOISE_POINTS = 2

    def __init__(self, scene: SyntheticScene) -> None:
        self.scene = scene
        self.demo = _DemoRenderer(scene)
        self.frame_period_s = self.demo.frame_period_s
        self.clock_hz = cpu_clock_hz(scene.platform, scene.schema)
        outputs = scene.outputs if scene.outputs != DEFAULT_OUTPUTS else TRACKING_OUTPUTS
        self.outputs = frozenset(outputs)
        self.planar = scene.schema is PacketSchema.PEOPLE_COUNTING_2D
        self._previous_ids: NDArray[np.int64] | None = None

    def _points(
        self, truth: FrameTruth, rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        """Cartesian positions, radial velocities, target IDs and linear SNR of the points."""
        positions: list[NDArray[np.float64]] = []
        velocities: list[float] = []
        owners: list[int] = []
        strengths: list[float] = []
        for index in np.flatnonzero(truth.detected):
            count = max(2, round(self.POINTS_PER_TARGET + self.scene.targets[index].rcs_db / 4))
            centre = truth.position_m[index]
            spread = rng.normal(0.0, self.POINT_SPREAD_M, size=(count, 3))
            if self.planar:
                spread[:, 2] = 0.0
            positions.extend(centre + spread)
            velocities.extend(truth.radial_velocity_mps[index] + rng.normal(0.0, 0.05, count))
            owners.extend([int(index)] * count)
            level = min(float(truth.snr_db[index]), 30.0)
            strengths.extend(10.0 ** ((level - rng.uniform(0.0, 6.0, count)) / 10.0))
        for _ in range(self.NOISE_POINTS):
            noise = np.array(
                [rng.uniform(-3.0, 3.0), rng.uniform(1.0, 6.0), 0.0 if self.planar else 1.0]
            )
            positions.append(noise)
            velocities.append(float(rng.normal(0.0, 0.3)))
            owners.append(UNASSOCIATED_NOISE)
            strengths.append(float(rng.uniform(5.0, 15.0)))
        return (
            np.array(positions, dtype=np.float64).reshape(-1, 3),
            np.array(velocities, dtype=np.float64),
            np.array(owners, dtype=np.int64),
            np.array(strengths, dtype=np.float64),
        )

    def _targets(self, truth: FrameTruth, rng: np.random.Generator) -> TargetList:
        visible = np.flatnonzero(truth.detected)
        n = visible.size
        position = truth.position_m[visible] + rng.normal(0.0, 0.02, size=(n, 3))
        velocity = np.array([self.scene.targets[i].state_at(truth.time_s)[1] for i in visible])
        velocity = velocity.reshape(n, 3)
        if self.planar:
            position[:, 2] = 0.0
            velocity[:, 2] = 0.0
        size = 9 if self.planar else 16
        side = 3 if self.planar else 4
        covariance = np.tile(np.eye(side).reshape(-1) * 0.01, (n, 1))[:, :size]
        return TargetList(
            target_id=visible,
            position_m=position,
            velocity_mps=velocity,
            acceleration_mps2=np.zeros((n, 3)),
            covariance=covariance,
            gating_gain=np.full(n, 3.0),
            confidence=None if self.planar else np.full(n, 1.0),
        )

    def packet(self, index: int, rng: np.random.Generator) -> bytes:
        scene = self.scene
        truth = self.demo.truth(index)
        xyz, velocity, owners, strength = self._points(truth, rng)
        if self.planar:
            xyz[:, 2] = 0.0
        rng_m, azimuth_deg, elevation_deg = cartesian_to_spherical(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        targets = self._targets(truth, rng)
        schema = scene.schema
        types = PeopleTrackingTlvType if schema is PacketSchema.PEOPLE_TRACKING else None
        tlvs: list[tuple[int, bytes]] = []
        point_type = types.POINTS if types else PeopleCountingTlvType.POINTS
        if PayloadKind.POINTS in self.outputs and len(rng_m):
            if self.planar:
                payload = encode_polar_points(rng_m, np.radians(azimuth_deg), velocity, strength)
            else:
                payload = encode_compressed_points(
                    rng_m, np.radians(azimuth_deg), np.radians(elevation_deg), velocity, strength
                )
            tlvs.append((point_type, payload))
        has_targets = PayloadKind.TARGETS in self.outputs and len(targets) > 0
        if has_targets:
            record = "2d" if self.planar else "3d"
            target_type = types.TARGETS if types else PeopleCountingTlvType.TARGETS
            tlvs.append((target_type, encode_target_list(targets, record)))
        previous = self._previous_ids
        if (
            has_targets
            and PayloadKind.TARGET_ASSOCIATION in self.outputs
            and previous is not None
            and previous.size
        ):
            association_type = (
                types.TARGET_ASSOCIATION if types else PeopleCountingTlvType.TARGET_ASSOCIATION
            )
            tlvs.append((association_type, encode_target_association(previous)))
        if types and has_targets and PayloadKind.TARGET_HEIGHTS in self.outputs:
            heights = TargetHeights(
                targets.target_id,
                targets.position_m[:, 2] + 0.25,
                targets.position_m[:, 2] - 1.4,
            )
            tlvs.append((types.TARGET_HEIGHTS, encode_target_heights(heights)))
        if not self.planar and PayloadKind.PRESENCE in self.outputs:
            presence_type = types.PRESENCE if types else PeopleCountingTlvType.PRESENCE
            tlvs.append((presence_type, encode_presence(len(targets) > 0)))
        self._previous_ids = owners if PayloadKind.POINTS in self.outputs else None

        template = HeaderTemplate.for_schema(
            schema,
            scene.platform,
            frame_number=truth.frame_number,
            cpu_cycles=_timestamp(scene, truth.time_s, self.clock_hz, rng),
            declared_point_count=len(rng_m),
        )
        timing = HeaderTiming(
            chirp_margin_us=int(rng.integers(10, 40)),
            tracking_time_us=int(rng.integers(300, 900)),
            uart_time_us=int(rng.integers(500, 3000)),
            frame_processing_time_us=int(rng.integers(8000, 20000)),
            frame_margin_us=int(rng.integers(20000, 40000)),
        )
        template = dataclasses.replace(template, timing=timing)
        return encode_frame(template, tlvs, scene.padding_byte)


def generate_packets(
    scene: SyntheticScene | None = None,
    n_frames: int | None = 100,
    *,
    schema: PacketSchema | str | None = None,
    outputs: Sequence[PayloadKind | str] | None = None,
) -> Iterator[bytes]:
    """Yield the packets of a synthetic scene, one per frame.

    Parameters
    ----------
    scene : SyntheticScene, optional
        Scene to render; the default room scene when omitted.
    n_frames : int or None, optional
        Number of frames; ``None`` for an endless stream.
    schema : PacketSchema or str, optional
        Overrides the scene's schema.
    outputs : sequence of PayloadKind or str, optional
        Overrides the scene's payload selection.

    Yields
    ------
    bytes
        Complete packets, byte-identical for identical arguments.
    """
    scene = scene or SyntheticScene()
    changes: dict[str, object] = {}
    if schema is not None:
        changes["schema"] = parse_schema(schema)
    if outputs is not None:
        changes["outputs"] = frozenset(PayloadKind(o) for o in outputs)
    if changes:
        scene = scene.replace(**changes)
    rng = np.random.default_rng(scene.seed)
    renderer: _DemoRenderer | _VitalSignsRenderer | _TrackingRenderer
    if scene.schema is PacketSchema.VITAL_SIGNS:
        renderer = _VitalSignsRenderer(scene)
    elif scene.schema.tracks_targets:
        renderer = _TrackingRenderer(scene)
    else:
        renderer = _DemoRenderer(scene)
    indices = itertools.count() if n_frames is None else range(int(n_frames))
    for index in indices:
        yield renderer.packet(index, rng)


def scene_truth(scene: SyntheticScene, frame_index: int) -> FrameTruth:
    """Ground truth of one frame of a scene with targets (all but Vital Signs scenes)."""
    if scene.schema is PacketSchema.VITAL_SIGNS:
        raise ValueError("ground truth of targets exists for scenes with targets only")
    return _DemoRenderer(scene).truth(frame_index)


# -- fault injection -----------------------------------------------------------------------


class PacketFate(str, Enum):
    """What :func:`inject_faults` did to a packet."""

    DELIVERED = "delivered"
    DROPPED = "dropped"
    TRUNCATED = "truncated"
    CORRUPTED = "corrupted"


@dataclass(frozen=True)
class FaultPlan:
    """Which faults to inject, and how often.

    Attributes
    ----------
    leading_junk_bytes : int
        Random bytes before the first packet (a stream joined mid-way).
    junk_probability : float
        Chance of random bytes before each packet.
    junk_max_bytes : int
        Largest junk insertion.
    fake_sync_probability : float
        Chance that a junk insertion starts with a sync word and a plausible
        header whose TLV chain is garbage.
    drop_probability : float
        Chance that a packet is lost (frame-number gaps).
    drop_every : int or None
        Additionally drop every n-th packet.
    truncate_probability : float
        Chance that a packet is cut short inside its TLVs (receive overflow).
    truncate_tail : bool
        Cut the final packet short (stream ends mid-packet).
    bitflip_probability : float
        Chance that one bit of a packet's content flips. The protocol has no
        checksum, so most of these cannot be detected.
    duplicate_probability : float
        Chance that a packet is sent twice in a row.
    block_dump_size : int or None
        Imitate a recorder that writes its whole working buffer after every
        packet: each block holds the previous and the current packet followed
        by zeros. A block too small for two packets grows to the next multiple
        of this size. Junk and duplicate options are ignored in this mode.
    """

    leading_junk_bytes: int = 0
    junk_probability: float = 0.0
    junk_max_bytes: int = 48
    fake_sync_probability: float = 0.0
    drop_probability: float = 0.0
    drop_every: int | None = None
    truncate_probability: float = 0.0
    truncate_tail: bool = False
    bitflip_probability: float = 0.0
    duplicate_probability: float = 0.0
    block_dump_size: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "junk_probability",
            "fake_sync_probability",
            "drop_probability",
            "truncate_probability",
            "bitflip_probability",
            "duplicate_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
        if self.leading_junk_bytes < 0 or self.junk_max_bytes < 1:
            raise ValueError("junk sizes must be non-negative (junk_max_bytes at least 1)")
        if self.drop_every is not None and self.drop_every < 1:
            raise ValueError(f"drop_every must be at least 1, got {self.drop_every}")
        if self.block_dump_size is not None and self.block_dump_size < 64:
            raise ValueError(f"block_dump_size must be at least 64, got {self.block_dump_size}")


#: Named fault plans used by the CLI and the evaluation.
FAULT_PRESETS: dict[str, FaultPlan] = {
    "none": FaultPlan(),
    "light": FaultPlan(
        leading_junk_bytes=23,
        junk_probability=0.05,
        junk_max_bytes=40,
        truncate_probability=0.02,
    ),
    "heavy": FaultPlan(
        leading_junk_bytes=97,
        junk_probability=0.2,
        junk_max_bytes=96,
        fake_sync_probability=0.3,
        drop_probability=0.05,
        truncate_probability=0.08,
        truncate_tail=True,
        duplicate_probability=0.05,
    ),
    "block-dump": FaultPlan(drop_probability=0.75, block_dump_size=2880),
    "bitflip": FaultPlan(bitflip_probability=0.2),
}


@dataclass(frozen=True)
class FaultedStream:
    """A damaged byte stream and what happened to each original packet.

    Attributes
    ----------
    data : bytes
        The stream.
    fates : tuple of PacketFate
        One entry per input packet.
    copies : tuple of int
        How many intact copies of each packet the stream holds.
    junk_bytes : int
        Bytes of inserted junk (fake headers included).
    fake_sync_words : int
        Junk insertions that started with a sync word.
    damaged_crc32 : tuple of (int or None)
        CRC-32 of the bytes actually delivered for every truncated or corrupted
        packet (``None`` for the others), so a damaged packet that decodes
        cleanly can be told apart even when the damage hit its frame number.
    """

    data: bytes
    fates: tuple[PacketFate, ...]
    copies: tuple[int, ...]
    junk_bytes: int
    fake_sync_words: int
    damaged_crc32: tuple[int | None, ...] = ()

    def count(self, fate: PacketFate) -> int:
        """Packets that met a given fate."""
        return sum(1 for item in self.fates if item is fate)


def content_end(packet: bytes) -> int:
    """Offset just past the last TLV of a well-formed packet (padding excluded)."""
    verdict = examine_header(packet, 0, **DEFAULT_LIMITS)
    fields = verdict.fields
    if fields is None:
        raise ValueError(f"not a well-formed packet: {verdict.problem or 'too short'}")
    if fields.layout is HeaderLayout.PADDED:
        header_len, _ = resolve_header_len(packet, fields, None)
    else:
        header_len = fields.layout.min_length
    walk = walk_tlv_chain(
        packet,
        0,
        header_len,
        fields.tlv_count,
        fields.packet_length,
        length_includes_header=fields.layout.checksummed,
    )
    return walk.end_offset


_FAKE_HEADER = struct.Struct("<7I")


def _junk(rng: np.random.Generator, size: int, fake_sync: bool) -> bytes:
    if not fake_sync:
        return rng.bytes(size)
    header = MAGIC_WORD + _FAKE_HEADER.pack(
        0x02000004,
        int(rng.integers(2, 17)) * 32,
        int(Platform.XWR16XX),
        int(rng.integers(0, 2**31)),
        int(rng.integers(0, 2**32)),
        0,
        int(rng.integers(1, 4)),
    )
    # Sub-frame word, then a TLV whose length can never fit: the header is
    # plausible on its own and only the chain gives the forgery away.
    chain = struct.pack("<3I", 0, int(rng.integers(1, 7)), 0xFFFFFF00)
    return header + chain + rng.bytes(max(0, size - len(header) - len(chain)))


def _block_dump(packets: Sequence[bytes], block_size: int) -> bytes:
    # The block must hold the previous and the current packet; a recorder whose
    # buffer is sized in packets would have a proportionally larger one.
    longest_pair = max(
        (len(a) + len(b) for a, b in zip([b"", *packets[:-1]], packets, strict=True)), default=0
    )
    size = block_size * max(1, -(-longest_pair // block_size))
    blocks = []
    previous = b""
    for packet in packets:
        content = previous + packet
        blocks.append(content + bytes(size - len(content)))
        previous = packet
    return b"".join(blocks)


def inject_faults(
    packets: Sequence[bytes],
    plan: FaultPlan,
    rng: np.random.Generator | int | None = None,
) -> FaultedStream:
    """Concatenate packets into a stream with the faults of ``plan``.

    Parameters
    ----------
    packets : sequence of bytes
        Well-formed packets, e.g. from :func:`generate_packets`.
    plan : FaultPlan
        Faults to inject.
    rng : numpy.random.Generator or int, optional
        Random source or seed.

    Returns
    -------
    FaultedStream
    """
    random = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    pieces: list[bytes] = []
    fates: list[PacketFate] = []
    copies: list[int] = []
    damaged_crc32: list[int | None] = []
    delivered: list[bytes] = []
    junk_bytes = 0
    fake_sync_words = 0
    use_junk = plan.block_dump_size is None
    if plan.leading_junk_bytes and use_junk:
        pieces.append(random.bytes(plan.leading_junk_bytes))
        junk_bytes += plan.leading_junk_bytes
    last = len(packets) - 1
    for index, packet in enumerate(packets):
        dropped = plan.drop_every is not None and (index + 1) % plan.drop_every == 0
        if random.random() < plan.drop_probability:
            dropped = True
        if dropped:
            fates.append(PacketFate.DROPPED)
            copies.append(0)
            damaged_crc32.append(None)
            continue
        if use_junk and plan.junk_probability and random.random() < plan.junk_probability:
            fake = bool(plan.fake_sync_probability) and random.random() < plan.fake_sync_probability
            size = int(random.integers(1, plan.junk_max_bytes + 1))
            junk = _junk(random, size, fake)
            pieces.append(junk)
            junk_bytes += len(junk)
            fake_sync_words += int(fake)
        data, fate = packet, PacketFate.DELIVERED
        end = content_end(packet)
        if (plan.truncate_tail and index == last) or random.random() < plan.truncate_probability:
            cut = int(random.integers(len(MAGIC_WORD) + 1, max(len(MAGIC_WORD) + 2, end)))
            data, fate = packet[:cut], PacketFate.TRUNCATED
        elif plan.bitflip_probability and random.random() < plan.bitflip_probability:
            position = int(random.integers(len(MAGIC_WORD), end))
            flipped = bytearray(packet)
            flipped[position] ^= 1 << int(random.integers(0, 8))
            data, fate = bytes(flipped), PacketFate.CORRUPTED
        pieces.append(data)
        delivered.append(data)
        fates.append(fate)
        copies.append(1 if fate is PacketFate.DELIVERED else 0)
        damaged_crc32.append(None if fate is PacketFate.DELIVERED else zlib.crc32(data))
        if (
            use_junk
            and fate is PacketFate.DELIVERED
            and plan.duplicate_probability
            and random.random() < plan.duplicate_probability
        ):
            pieces.append(data)
            copies[-1] += 1
    if plan.block_dump_size is not None:
        stream = _block_dump(delivered, plan.block_dump_size)
        # Every packet is written twice (its own block and the next), the last once.
        copies = [count * 2 if count else 0 for count in copies]
        for position in range(len(copies) - 1, -1, -1):
            if copies[position]:
                copies[position] //= 2
                break
    else:
        stream = b"".join(pieces)
    return FaultedStream(
        stream, tuple(fates), tuple(copies), junk_bytes, fake_sync_words, tuple(damaged_crc32)
    )


def split_into_chunks(
    data: bytes,
    rng: np.random.Generator | int | None = None,
    max_chunk: int = 4096,
    min_chunk: int = 1,
) -> list[bytes]:
    """Split a stream into random-sized chunks, as reads from a port would.

    Examples
    --------
    >>> chunks = split_into_chunks(bytes(100), rng=3, max_chunk=9)
    >>> b"".join(chunks) == bytes(100), max(len(c) for c in chunks) <= 9
    (True, True)
    """
    if not 1 <= min_chunk <= max_chunk:
        raise ValueError(f"need 1 <= min_chunk <= max_chunk, got {min_chunk} and {max_chunk}")
    random = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    chunks: list[bytes] = []
    position = 0
    while position < len(data):
        sizes = random.integers(min_chunk, max_chunk + 1, size=256)
        for size in sizes:
            if position >= len(data):
                break
            chunks.append(data[position : position + int(size)])
            position += int(size)
    return chunks


def packet_crc(packet: bytes) -> int:
    """CRC-32 of a packet, the identity the decoder gives the frame it yields."""
    return zlib.crc32(packet)


def synthetic_capture(
    scene: SyntheticScene | None = None,
    n_frames: int = 200,
    plan: FaultPlan | None = None,
    fault_seed: int | None = None,
) -> FaultedStream:
    """Render a scene and damage it in one call.

    Parameters
    ----------
    scene : SyntheticScene, optional
        Scene to render.
    n_frames : int, optional
        Frames to render.
    plan : FaultPlan, optional
        Faults; none by default.
    fault_seed : int, optional
        Seed of the fault draws; defaults to the scene seed plus one.
    """
    scene = scene or SyntheticScene()
    packets = list(generate_packets(scene, n_frames))
    seed = scene.seed + 1 if fault_seed is None else fault_seed
    return inject_faults(packets, plan or FaultPlan(), seed)
