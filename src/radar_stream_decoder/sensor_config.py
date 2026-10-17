"""Radar parameters from a TI mmWave CLI configuration (``.cfg``).

The UART stream carries bin indices and fixed-point numbers; turning them into
metres, metres per second and decibels needs the chirp configuration that was
sent to the sensor. The parser below reads the text commands of the mmWave
SDK command line and derives the FMCW quantities from them.

Commands understood (argument order of the SDK 1.x-3.x out-of-box demo)::

    profileCfg <id> <startFreq GHz> <idle us> <adcStart us> <rampEnd us>
               <txPower> <txPhase> <slope MHz/us> <txStart us> <numAdcSamples>
               <sampleRate ksps> <hpf1> <hpf2> <rxGain dB>
    channelCfg <rxMask> <txMask> <cascading>
    chirpCfg   <start> <end> <profileId> <startFreqVar> <slopeVar> <idleVar>
               <adcStartVar> <txMask>
    frameCfg   <chirpStart> <chirpEnd> <loops> <frames> <period ms> <trigger> <delay>
    guiMonitor [<subFrameIdx>] <points> <rangeProfile> <noiseProfile>
               <azimuthHeatmap> <rangeDopplerHeatmap> <stats>
    dfeDataOutputMode <1|2|3>
    compRangeBiasAndRxChanPhase <rangeBias m> <re im> ...

Everything else (``cfarCfg``, ``adcCfg`` ...) is kept verbatim in
:attr:`SensorConfig.extras`. Text after ``%`` is a comment.

Derived quantities (with :math:`c = 3\\times10^8` m/s like TI's tools, slope
:math:`S`, sample rate :math:`F_s`, :math:`N` ADC samples, range FFT size
:math:`N_R = 2^{\\lceil\\log_2 N\\rceil}`, chirp period
:math:`T_c = T_{idle} + T_{ramp}`, :math:`N_{tx}` transmitters in the frame,
:math:`N_c` chirps per transmitter, Doppler FFT size
:math:`N_D = 2^{\\lceil\\log_2 N_c\\rceil}` and a reference frequency
:math:`f_{ref}`):

.. math::

    \\Delta R = \\frac{c F_s}{2 S N},\\quad
    \\Delta R_{bin} = \\frac{c F_s}{2 S N_R},\\quad
    R_{max} = k\\,\\frac{c F_s}{2 S},\\quad
    \\Delta v_{bin} = \\frac{c}{2 f_{ref} T_c N_{tx} N_D},\\quad
    v_{max} = \\frac{c}{4 f_{ref} T_c N_{tx}}

Three figures depend on a convention rather than on physics alone, and
published tools differ (see :class:`FmcwConvention`): the frequency whose
wavelength turns phase into motion (the chirp start frequency in TI's mmWave
Demo Visualizer, :math:`f_0 + B/2 + T_{adc} S` in the SDK 3.5 firmware), the
usable share :math:`k` of the IF-bandwidth range limit (0.8 in TI's
visualiser) and whether the velocity resolution counts the zero-padded Doppler
FFT bins.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypeVar

from .constants import SPEED_OF_LIGHT_MPS, Platform, parse_platform, platform_label
from .errors import ConfigError
from .units import next_power_of_two, xyz_q_format

_LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Transmitters that resolve azimuth (bit mask over TX0..TX2). On the xWR16xx
#: both transmitters are azimuth antennas; on the three-transmitter devices
#: TX0 and TX2 form the azimuth aperture and TX1 is raised for elevation.
_AZIMUTH_TX_MASK: dict[int, int] = {Platform.XWR16XX: 0b011}
_ELEVATION_TX_MASK: dict[int, int] = {Platform.XWR16XX: 0b000}
_DEFAULT_AZIMUTH_TX_MASK = 0b101
_DEFAULT_ELEVATION_TX_MASK = 0b010


@dataclass(frozen=True)
class ChirpProfile:
    """One ``profileCfg`` line: the shape of the FMCW chirp."""

    profile_id: int
    start_freq_ghz: float
    idle_time_us: float
    adc_start_time_us: float
    ramp_end_time_us: float
    tx_power_backoff: int
    tx_phase_shifter: int
    freq_slope_mhz_per_us: float
    tx_start_time_us: float
    num_adc_samples: int
    sample_rate_ksps: float
    hpf1_corner: int
    hpf2_corner: int
    rx_gain_db: float


@dataclass(frozen=True)
class ChannelMasks:
    """``channelCfg``: which receivers and transmitters are enabled."""

    rx_mask: int
    tx_mask: int
    cascading: int = 0


@dataclass(frozen=True)
class ChirpEntry:
    """One ``chirpCfg`` line: chirp indices, their profile and transmitters."""

    start_index: int
    end_index: int
    profile_id: int
    start_freq_var_hz: float
    freq_slope_var_khz_per_us: float
    idle_time_var_us: float
    adc_start_time_var_us: float
    tx_mask: int


@dataclass(frozen=True)
class FrameTiming:
    """``frameCfg``: which chirps form a frame, how often, how many times."""

    chirp_start_index: int
    chirp_end_index: int
    num_loops: int
    num_frames: int
    period_ms: float
    trigger_select: int
    trigger_delay_ms: float


@dataclass(frozen=True)
class OutputSelection:
    """``guiMonitor``: which TLVs the firmware is asked to send.

    ``point_side_info`` reflects the SDK 3.x convention (detected-objects flag
    1 = points and side info, 2 = points only).
    """

    points: bool
    range_profile: bool
    noise_profile: bool
    azimuth_heatmap: bool
    range_doppler_heatmap: bool
    stats: bool
    point_side_info: bool = False
    subframe_index: int | None = None


def _popcount(mask: int) -> int:
    return bin(int(mask) & 0xFFFFFFFF).count("1")


class WavelengthReference(str, Enum):
    """Frequency whose wavelength converts Doppler phase into velocity.

    ``start``
        The chirp start frequency :math:`f_0`, as TI's mmWave Demo Visualizer
        computes its velocity figures.
    ``center``
        :math:`f_0 + B/2 + T_{adc} S` with the sampled bandwidth
        :math:`B = S N / F_s`: the ``centerFreq`` that ``mmwdemo_rfparser.c`` of
        TI mmWave SDK 03.05.00.04 uses for the firmware's Doppler step, and
        therefore for the velocities SDK 3.x firmware writes into its point cloud.
    """

    START = "start"
    CENTER = "center"


@dataclass(frozen=True)
class FmcwConvention:
    """The conventions behind the convention-dependent FMCW figures.

    Attributes
    ----------
    wavelength_reference : WavelengthReference
        Frequency of the wavelength used for velocities (and for the phase-to-
        displacement conversion of Vital Signs data).
    max_range_factor : float
        Share of the IF-bandwidth range limit :math:`c F_s / (2 S)` reported as
        the maximum range. TI's visualiser keeps 0.8 as margin for the IF
        filter roll-off; 1.0 reports the limit itself.
    doppler_zero_padding : bool
        Report the velocity resolution as the spacing of the zero-padded Doppler
        FFT bins (``True``, the firmware's ``dopplerStep``) or from the chirps
        actually transmitted (``False``, the firmware's ``dopplerResolution``).
        Bin indices are always converted with the bin spacing.

    Examples
    --------
    >>> FmcwConvention().describe()
    {'wavelength_reference': 'start', 'max_range_factor': 0.8, 'doppler_zero_padding': True}
    """

    wavelength_reference: WavelengthReference = WavelengthReference.START
    max_range_factor: float = 0.8
    doppler_zero_padding: bool = True

    def __post_init__(self) -> None:
        try:
            reference = WavelengthReference(self.wavelength_reference)
        except ValueError:
            raise ConfigError(
                "wavelength_reference must be 'start' or 'center', "
                f"got {self.wavelength_reference!r}"
            ) from None
        object.__setattr__(self, "wavelength_reference", reference)
        if not 0.0 < float(self.max_range_factor) <= 1.0:
            raise ConfigError(f"max_range_factor must lie in (0, 1], got {self.max_range_factor}")
        object.__setattr__(self, "max_range_factor", float(self.max_range_factor))
        object.__setattr__(self, "doppler_zero_padding", bool(self.doppler_zero_padding))

    def describe(self) -> dict[str, str | float | bool]:
        """The convention as plain values, for reports."""
        return {
            "wavelength_reference": self.wavelength_reference.value,
            "max_range_factor": self.max_range_factor,
            "doppler_zero_padding": self.doppler_zero_padding,
        }


@dataclass(frozen=True)
class SensorConfig:
    """Parsed radar configuration with FMCW quantities as properties.

    Build one with :func:`parse_cli_config` or :func:`load_cli_config`.
    Properties that need a command the configuration lacks raise
    :class:`~radar_stream_decoder.errors.ConfigError` naming that command.

    Attributes
    ----------
    platform : int or None
        Device the configuration is meant for; decides the antenna geometry.
    profiles : tuple of ChirpProfile
    channels : ChannelMasks or None
    chirps : tuple of ChirpEntry
    frame : FrameTiming or None
    outputs : OutputSelection or None
    data_output_mode : int
        ``dfeDataOutputMode`` (1 = frame based, 3 = advanced frame).
    range_bias_m : float
        Range bias from ``compRangeBiasAndRxChanPhase``.
    rx_phase_compensation : tuple of complex
        Per-virtual-antenna phase compensation from the same command.
    extras : tuple of (str, tuple of str)
        Commands kept verbatim, in file order.
    convention : FmcwConvention
        Conventions of the convention-dependent figures (velocity wavelength,
        maximum-range margin, Doppler zero padding).
    """

    platform: int | None = None
    profiles: tuple[ChirpProfile, ...] = ()
    channels: ChannelMasks | None = None
    chirps: tuple[ChirpEntry, ...] = ()
    frame: FrameTiming | None = None
    outputs: OutputSelection | None = None
    data_output_mode: int = 1
    range_bias_m: float = 0.0
    rx_phase_compensation: tuple[complex, ...] = ()
    extras: tuple[tuple[str, tuple[str, ...]], ...] = ()
    convention: FmcwConvention = field(default_factory=FmcwConvention)

    def with_convention(self, convention: FmcwConvention) -> SensorConfig:
        """Copy of the configuration that derives its figures under ``convention``."""
        return dataclasses.replace(self, convention=convention)

    # -- configuration sections -------------------------------------------------

    def _need_channels(self) -> ChannelMasks:
        if self.channels is None:
            raise ConfigError("the configuration has no channelCfg command")
        return self.channels

    def _need_frame(self) -> FrameTiming:
        if self.data_output_mode == 3 and self.frame is None:
            raise ConfigError(
                "advanced-frame configurations (dfeDataOutputMode 3) are kept in extras; "
                "frame-dependent quantities need a frameCfg command"
            )
        if self.frame is None:
            raise ConfigError("the configuration has no frameCfg command")
        return self.frame

    def chirps_in_frame(self) -> tuple[ChirpEntry, ...]:
        """``chirpCfg`` entries whose index range overlaps the frame's chirps."""
        if self.frame is None:
            return self.chirps
        first, last = self.frame.chirp_start_index, self.frame.chirp_end_index
        return tuple(c for c in self.chirps if c.start_index <= last and c.end_index >= first)

    @property
    def profile(self) -> ChirpProfile:
        """The chirp profile the frame uses (that of its first chirp)."""
        if not self.profiles:
            raise ConfigError("the configuration has no profileCfg command")
        in_frame = self.chirps_in_frame()
        if in_frame:
            wanted = min(in_frame, key=lambda c: c.start_index).profile_id
            for candidate in self.profiles:
                if candidate.profile_id == wanted:
                    return candidate
        return self.profiles[0]

    @property
    def platform_name(self) -> str:
        """Readable platform name, ``"unknown"`` when not set."""
        return "unknown" if self.platform is None else platform_label(self.platform)

    # -- antennas ---------------------------------------------------------------

    @property
    def num_rx(self) -> int:
        """Enabled receivers."""
        return _popcount(self._need_channels().rx_mask)

    @property
    def tx_mask_in_frame(self) -> int:
        """Transmitters used by the chirps of a frame (falls back to ``channelCfg``)."""
        mask = 0
        for chirp in self.chirps_in_frame():
            mask |= chirp.tx_mask
        return mask if mask else self._need_channels().tx_mask

    @property
    def num_tx(self) -> int:
        """Transmitters time-multiplexed within a frame."""
        return _popcount(self.tx_mask_in_frame)

    @property
    def num_tx_azimuth(self) -> int:
        """Transmitters forming the azimuth aperture.

        With a single transmitter in the frame it is the azimuth aperture by
        definition. Without a known platform every transmitter counts as azimuth.
        """
        if self.num_tx == 1:
            return 1
        if self.platform is None:
            return self.num_tx
        mask = _AZIMUTH_TX_MASK.get(self.platform, _DEFAULT_AZIMUTH_TX_MASK)
        return _popcount(self.tx_mask_in_frame & mask)

    @property
    def num_tx_elevation(self) -> int:
        """Transmitters displaced in elevation (xWR14xx TX1 and similar)."""
        if self.num_tx == 1 or self.platform is None:
            return 0
        mask = _ELEVATION_TX_MASK.get(self.platform, _DEFAULT_ELEVATION_TX_MASK)
        return _popcount(self.tx_mask_in_frame & mask)

    @property
    def num_virtual_antennas(self) -> int:
        """All virtual antennas, ``num_tx * num_rx``."""
        return self.num_tx * self.num_rx

    @property
    def num_virtual_antennas_azimuth(self) -> int:
        """Virtual antennas of the azimuth array, ``num_tx_azimuth * num_rx``."""
        return self.num_tx_azimuth * self.num_rx

    # -- FFT sizes ------------------------------------------------------------------

    @property
    def chirps_per_frame(self) -> int:
        """Chirps transmitted per frame (all transmitters)."""
        frame = self._need_frame()
        return (frame.chirp_end_index - frame.chirp_start_index + 1) * frame.num_loops

    @property
    def num_range_bins(self) -> int:
        """Range FFT size, the ADC sample count rounded up to a power of two."""
        return next_power_of_two(self.profile.num_adc_samples)

    @property
    def num_doppler_bins(self) -> int:
        """Doppler FFT size, chirps per transmitter rounded up to a power of two."""
        chirps, tx = self.chirps_per_frame, self.num_tx
        if chirps % tx:
            raise ConfigError(
                f"{chirps} chirps per frame cannot be split evenly over {tx} transmitters"
            )
        return next_power_of_two(chirps // tx)

    # -- physical quantities --------------------------------------------------------

    @property
    def start_frequency_hz(self) -> float:
        """Chirp start frequency."""
        return self.profile.start_freq_ghz * 1e9

    @property
    def center_frequency_hz(self) -> float:
        """``f0 + B/2 + adcStart * S``, the firmware's ``centerFreq`` (SDK 3.5 rfparser)."""
        profile = self.profile
        return (
            self.start_frequency_hz
            + 0.5 * self.sampled_bandwidth_hz
            + profile.adc_start_time_us * 1e-6 * self.slope_hz_per_s
        )

    @property
    def reference_frequency_hz(self) -> float:
        """Frequency of the wavelength used for velocities (see :class:`FmcwConvention`)."""
        if self.convention.wavelength_reference is WavelengthReference.CENTER:
            return self.center_frequency_hz
        return self.start_frequency_hz

    @property
    def wavelength_m(self) -> float:
        """Wavelength at the reference frequency of the convention."""
        return SPEED_OF_LIGHT_MPS / self.reference_frequency_hz

    @property
    def slope_hz_per_s(self) -> float:
        """Frequency slope."""
        return self.profile.freq_slope_mhz_per_us * 1e12

    @property
    def sample_rate_hz(self) -> float:
        """ADC sample rate."""
        return self.profile.sample_rate_ksps * 1e3

    @property
    def sampled_bandwidth_hz(self) -> float:
        """Bandwidth swept while the ADC samples, ``S N / F_s``."""
        profile = self.profile
        return self.slope_hz_per_s * profile.num_adc_samples / self.sample_rate_hz

    @property
    def range_resolution_m(self) -> float:
        """Range resolution ``c / (2 B)`` of the sampled bandwidth."""
        return SPEED_OF_LIGHT_MPS / (2.0 * self.sampled_bandwidth_hz)

    @property
    def range_bin_m(self) -> float:
        """Range spanned by one range-FFT bin (``rangeIdxToMeters``)."""
        return (
            SPEED_OF_LIGHT_MPS
            * self.sample_rate_hz
            / (2.0 * self.slope_hz_per_s * self.num_range_bins)
        )

    @property
    def if_range_limit_m(self) -> float:
        """Range at which the beat frequency reaches the sample rate, ``c F_s / (2 S)``."""
        return SPEED_OF_LIGHT_MPS * self.sample_rate_hz / (2.0 * self.slope_hz_per_s)

    @property
    def max_range_m(self) -> float:
        """Usable range: ``max_range_factor`` of the convention times :attr:`if_range_limit_m`."""
        return self.convention.max_range_factor * self.if_range_limit_m

    @property
    def chirp_period_s(self) -> float:
        """Idle plus ramp time of one chirp."""
        profile = self.profile
        return (profile.idle_time_us + profile.ramp_end_time_us) * 1e-6

    @property
    def max_velocity_mps(self) -> float:
        """Unambiguous radial velocity ``c / (4 f_ref Tc Ntx)``."""
        return SPEED_OF_LIGHT_MPS / (
            4.0 * self.reference_frequency_hz * self.chirp_period_s * self.num_tx
        )

    @property
    def doppler_bin_mps(self) -> float:
        """Velocity spanned by one Doppler FFT bin, ``c / (2 f_ref Tc Ntx ND)``.

        Doppler bin indices are converted with this spacing whatever the
        convention says about resolution: the firmware's FFT has ``ND`` bins.
        """
        return SPEED_OF_LIGHT_MPS / (
            2.0
            * self.reference_frequency_hz
            * self.chirp_period_s
            * self.num_tx
            * self.num_doppler_bins
        )

    @property
    def doppler_resolution_mps(self) -> float:
        """Velocity resolution under the convention.

        With zero padding it equals :attr:`doppler_bin_mps`; without, it is
        ``c / (2 f_ref Tc Nchirps)`` over the chirps actually transmitted in a
        frame, the firmware's ``dopplerResolution``.
        """
        if self.convention.doppler_zero_padding:
            return self.doppler_bin_mps
        return SPEED_OF_LIGHT_MPS / (
            2.0 * self.reference_frequency_hz * self.chirp_period_s * self.chirps_per_frame
        )

    @property
    def frame_period_s(self) -> float:
        """Frame periodicity."""
        return self._need_frame().period_ms * 1e-3

    @property
    def xyz_q_format(self) -> int:
        """Q-format of point coordinates for this range-bin size."""
        return xyz_q_format(self.range_bin_m)

    def summary(self) -> dict[str, float | int | str]:
        """Every derived quantity that the configuration supports, by name.

        Quantities whose commands are missing are left out instead of raising;
        the conventions in effect follow as ``convention_*`` entries.
        """

        def attempt(getter: Callable[[], _T]) -> _T | None:
            try:
                return getter()
            except ConfigError:
                return None

        items: dict[str, Callable[[], float | int | str]] = {
            "platform": lambda: self.platform_name,
            "num_rx": lambda: self.num_rx,
            "num_tx": lambda: self.num_tx,
            "num_tx_azimuth": lambda: self.num_tx_azimuth,
            "num_tx_elevation": lambda: self.num_tx_elevation,
            "num_virtual_antennas": lambda: self.num_virtual_antennas,
            "num_virtual_antennas_azimuth": lambda: self.num_virtual_antennas_azimuth,
            "num_range_bins": lambda: self.num_range_bins,
            "chirps_per_frame": lambda: self.chirps_per_frame,
            "num_doppler_bins": lambda: self.num_doppler_bins,
            "start_frequency_ghz": lambda: self.start_frequency_hz / 1e9,
            "center_frequency_ghz": lambda: self.center_frequency_hz / 1e9,
            "reference_frequency_ghz": lambda: self.reference_frequency_hz / 1e9,
            "wavelength_mm": lambda: self.wavelength_m * 1e3,
            "sampled_bandwidth_mhz": lambda: self.sampled_bandwidth_hz / 1e6,
            "range_resolution_m": lambda: self.range_resolution_m,
            "range_bin_m": lambda: self.range_bin_m,
            "if_range_limit_m": lambda: self.if_range_limit_m,
            "max_range_m": lambda: self.max_range_m,
            "chirp_period_us": lambda: self.chirp_period_s * 1e6,
            "doppler_bin_mps": lambda: self.doppler_bin_mps,
            "doppler_resolution_mps": lambda: self.doppler_resolution_mps,
            "max_velocity_mps": lambda: self.max_velocity_mps,
            "frame_period_ms": lambda: self.frame_period_s * 1e3,
            "xyz_q_format": lambda: self.xyz_q_format,
            "range_bias_m": lambda: self.range_bias_m,
        }
        result: dict[str, float | int | str] = {}
        for name, getter in items.items():
            value = attempt(getter)
            if value is not None:
                result[name] = value
        for name, value in self.convention.describe().items():
            result[f"convention_{name}"] = value
        return result


# -- parsing ---------------------------------------------------------------------------


def _as_float(token: str, line: int, command: str, position: int) -> float:
    try:
        return float(token)
    except ValueError:
        raise ConfigError(
            f"{command}: argument {position} ({token!r}) is not a number", line
        ) from None


def _as_int(token: str, line: int, command: str, position: int) -> int:
    try:
        return int(token, 0)
    except ValueError:
        value = _as_float(token, line, command, position)
        if value != int(value):
            raise ConfigError(
                f"{command}: argument {position} ({token!r}) must be an integer", line
            ) from None
        return int(value)


def _check_arity(command: str, args: list[str], expected: tuple[int, ...], line: int) -> None:
    if len(args) not in expected:
        wanted = " or ".join(str(n) for n in expected)
        raise ConfigError(f"{command} takes {wanted} arguments, got {len(args)}", line)


def _parse_profile(args: list[str], line: int) -> ChirpProfile:
    _check_arity("profileCfg", args, (14,), line)

    def real(position: int) -> float:
        return _as_float(args[position], line, "profileCfg", position + 1)

    def integer(position: int) -> int:
        return _as_int(args[position], line, "profileCfg", position + 1)

    profile = ChirpProfile(
        profile_id=integer(0),
        start_freq_ghz=real(1),
        idle_time_us=real(2),
        adc_start_time_us=real(3),
        ramp_end_time_us=real(4),
        tx_power_backoff=integer(5),
        tx_phase_shifter=integer(6),
        freq_slope_mhz_per_us=real(7),
        tx_start_time_us=real(8),
        num_adc_samples=integer(9),
        sample_rate_ksps=real(10),
        hpf1_corner=integer(11),
        hpf2_corner=integer(12),
        rx_gain_db=real(13),
    )
    if profile.start_freq_ghz <= 0 or profile.freq_slope_mhz_per_us == 0:
        raise ConfigError("profileCfg: start frequency and slope must be non-zero", line)
    if profile.num_adc_samples <= 0 or profile.sample_rate_ksps <= 0:
        raise ConfigError("profileCfg: ADC sample count and rate must be positive", line)
    if profile.idle_time_us + profile.ramp_end_time_us <= 0:
        raise ConfigError("profileCfg: idle plus ramp time must be positive", line)
    return profile


def _parse_channels(args: list[str], line: int) -> ChannelMasks:
    _check_arity("channelCfg", args, (3,), line)
    rx, tx, cascade = (_as_int(t, line, "channelCfg", k + 1) for k, t in enumerate(args))
    if not 0 < rx <= 0xF:
        raise ConfigError(f"channelCfg: receive mask {rx} must enable 1-4 receivers", line)
    if not 0 < tx <= 0x7:
        raise ConfigError(f"channelCfg: transmit mask {tx} must enable 1-3 transmitters", line)
    return ChannelMasks(rx, tx, cascade)


def _parse_chirp(args: list[str], line: int) -> ChirpEntry:
    _check_arity("chirpCfg", args, (8,), line)
    start, end, profile_id = (_as_int(args[k], line, "chirpCfg", k + 1) for k in range(3))
    freq_var, slope_var, idle_var, adc_var = (
        _as_float(args[k], line, "chirpCfg", k + 1) for k in range(3, 7)
    )
    tx_mask = _as_int(args[7], line, "chirpCfg", 8)
    if end < start:
        raise ConfigError(f"chirpCfg: end index {end} precedes start index {start}", line)
    return ChirpEntry(start, end, profile_id, freq_var, slope_var, idle_var, adc_var, tx_mask)


def _parse_frame(args: list[str], line: int) -> FrameTiming:
    _check_arity("frameCfg", args, (7,), line)
    start, end, loops, frames = (_as_int(args[k], line, "frameCfg", k + 1) for k in range(4))
    period = _as_float(args[4], line, "frameCfg", 5)
    trigger = _as_int(args[5], line, "frameCfg", 6)
    delay = _as_float(args[6], line, "frameCfg", 7)
    if end < start:
        raise ConfigError(f"frameCfg: chirp end {end} precedes chirp start {start}", line)
    if loops <= 0:
        raise ConfigError(f"frameCfg: loop count must be positive, got {loops}", line)
    if period <= 0:
        raise ConfigError(f"frameCfg: frame period must be positive, got {period}", line)
    return FrameTiming(start, end, loops, frames, period, trigger, delay)


def _parse_outputs(args: list[str], line: int, platform: int | None) -> OutputSelection:
    if platform == Platform.XWR14XX:
        _check_arity("guiMonitor", args, (6,), line)
    elif platform is not None:
        _check_arity("guiMonitor", args, (7,), line)
    else:
        _check_arity("guiMonitor", args, (6, 7), line)
    values = [_as_int(token, line, "guiMonitor", k + 1) for k, token in enumerate(args)]
    subframe = values.pop(0) if len(values) == 7 else None
    # TI documents disagree on which value of <detected objects> adds the side-info
    # TLV 7: the Radar Toolbox and Industrial Toolbox UART pages say "set to 2".
    # The firmware says otherwise: mmw_config.h of TI mmWave SDK 03.05.00.04
    # documents 1 as "list of detected objects ... and side info" and 2 as points
    # only, and mss_main.c sends TLV 7 only for 1 (as does the SDK 3.06 user
    # guide). The decoder follows the firmware.
    points, range_profile, noise, azimuth, range_doppler, stats = values
    return OutputSelection(
        points=points != 0,
        range_profile=range_profile != 0,
        noise_profile=noise != 0,
        azimuth_heatmap=azimuth != 0,
        range_doppler_heatmap=range_doppler != 0,
        stats=stats != 0,
        point_side_info=points == 1,
        subframe_index=subframe,
    )


def parse_cli_config(
    text: str,
    platform: int | str | None = None,
    convention: FmcwConvention | None = None,
) -> SensorConfig:
    """Parse the text of a TI mmWave CLI configuration.

    Parameters
    ----------
    text : str
        Contents of a ``.cfg`` file.
    platform : int or str, optional
        Device the configuration is for (``"14xx"``, ``"IWR1642"``,
        ``0xA1642`` ...). It decides the ``guiMonitor`` argument count and the
        azimuth/elevation split of the transmitters.
    convention : FmcwConvention, optional
        Conventions of the convention-dependent figures; TI's mmWave Demo
        Visualizer conventions by default.

    Returns
    -------
    SensorConfig

    Raises
    ------
    ConfigError
        With the offending line number, for a wrong argument count, a
        non-numeric argument, an impossible value or a chirp referring to an
        undefined profile.

    Examples
    --------
    >>> config = parse_cli_config('''
    ... channelCfg 15 3 0
    ... profileCfg 0 77 7 6 60 0 0 40 1 256 5000 0 0 30
    ... chirpCfg 0 0 0 0 0 0 0 1
    ... chirpCfg 1 1 0 0 0 0 0 2
    ... frameCfg 0 1 16 0 50 1 0
    ... ''', platform="16xx")
    >>> config.num_virtual_antennas, config.num_range_bins, config.num_doppler_bins
    (8, 256, 16)
    >>> round(config.range_bin_m, 4), round(config.max_velocity_mps, 3)
    (0.0732, 7.269)
    >>> centre = config.with_convention(FmcwConvention(wavelength_reference="center"))
    >>> round(centre.max_velocity_mps, 3)
    7.151
    """
    try:
        platform_code = parse_platform(platform)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None

    profiles: dict[int, ChirpProfile] = {}
    chirps: list[tuple[int, ChirpEntry]] = []
    channels: ChannelMasks | None = None
    frame: FrameTiming | None = None
    outputs: OutputSelection | None = None
    output_mode = 1
    range_bias = 0.0
    phases: tuple[complex, ...] = ()
    extras: list[tuple[str, tuple[str, ...]]] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        content = raw_line.split("%", 1)[0].strip()
        if not content:
            continue
        command, *args = content.split()
        if command == "profileCfg":
            profile = _parse_profile(args, line_no)
            if profile.profile_id in profiles:
                _LOGGER.debug("line %d redefines profile %d", line_no, profile.profile_id)
            profiles[profile.profile_id] = profile
        elif command == "channelCfg":
            channels = _parse_channels(args, line_no)
        elif command == "chirpCfg":
            chirps.append((line_no, _parse_chirp(args, line_no)))
        elif command == "frameCfg":
            frame = _parse_frame(args, line_no)
        elif command == "guiMonitor":
            outputs = _parse_outputs(args, line_no, platform_code)
        elif command == "dfeDataOutputMode":
            _check_arity(command, args, (1,), line_no)
            output_mode = _as_int(args[0], line_no, command, 1)
            if output_mode not in (1, 2, 3):
                raise ConfigError(
                    f"dfeDataOutputMode must be 1, 2 or 3, got {output_mode}", line_no
                )
        elif command == "compRangeBiasAndRxChanPhase":
            if not args or len(args) % 2 == 0:
                raise ConfigError(
                    f"{command} takes a range bias followed by real/imaginary pairs, "
                    f"got {len(args)} arguments",
                    line_no,
                )
            numbers = [_as_float(t, line_no, command, k + 1) for k, t in enumerate(args)]
            range_bias = numbers[0]
            phases = tuple(
                complex(re, im) for re, im in zip(numbers[1::2], numbers[2::2], strict=True)
            )
        else:
            extras.append((command, tuple(args)))

    for line_no, chirp in chirps:
        if profiles and chirp.profile_id not in profiles:
            raise ConfigError(f"chirpCfg refers to undefined profile {chirp.profile_id}", line_no)
        if channels is not None and chirp.tx_mask & ~channels.tx_mask:
            raise ConfigError(
                f"chirpCfg transmit mask {chirp.tx_mask} uses transmitters that channelCfg "
                f"({channels.tx_mask}) does not enable",
                line_no,
            )

    return SensorConfig(
        platform=platform_code,
        profiles=tuple(profiles[key] for key in sorted(profiles)),
        channels=channels,
        chirps=tuple(chirp for _, chirp in chirps),
        frame=frame,
        outputs=outputs,
        data_output_mode=output_mode,
        range_bias_m=range_bias,
        rx_phase_compensation=phases,
        extras=tuple(extras),
        convention=convention or FmcwConvention(),
    )


def load_cli_config(
    path: str | Path,
    platform: int | str | None = None,
    convention: FmcwConvention | None = None,
) -> SensorConfig:
    """Read and parse a ``.cfg`` file; see :func:`parse_cli_config`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ConfigError
        If its contents are invalid; the message names the file and line.
    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        return parse_cli_config(text, platform, convention)
    except ConfigError as exc:
        raise ConfigError(f"{file_path.name}: {exc}") from None


#: Example configurations shipped with the package (also in ``configs/``).
#: They are written for this project; detection tuning commands (CFAR, peak
#: grouping, clutter removal) differ between firmware releases and are left
#: to the SDK's own profiles.
PRESET_CLI_CONFIGS: dict[str, str] = {
    "iwr1443_oob_2tx4rx": """\
% IWR1443, out-of-box demo output format.
% Two azimuth transmitters (TX0, TX2) x four receivers, 77 GHz,
% 0.117 m range bins up to 24 m, +/-7.9 m/s, 20 frames per second.
% Detection tuning commands (cfarCfg, peakGrouping, ...) are firmware-release
% specific: append those of the SDK release running on the device.
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 5 0
adcCfg 2 1
adcbufCfg 0 1 0 1
profileCfg 0 77 10 6 52 0 0 30 1 256 6000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
frameCfg 0 1 32 0 50 1 0
guiMonitor 1 1 1 1 1 1
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
sensorStart
""",
    "iwr1642_oob_2tx4rx": """\
% IWR1642, out-of-box demo output format (SDK 1.1 or later).
% Two transmitters x four receivers, 77 GHz,
% 0.073 m range bins up to 15 m, +/-7.3 m/s, 20 frames per second.
% Detection tuning commands (cfarCfg, peakGrouping, ...) are firmware-release
% specific: append those of the SDK release running on the device.
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 3 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 77 7 6 60 0 0 40 1 256 5000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
frameCfg 0 1 16 0 50 1 0
guiMonitor -1 1 1 1 1 1 1
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
sensorStart
""",
    "iwr6843_3d_3tx4rx": """\
% IWR6843, three transmitters x four receivers (TX1 raised for elevation),
% 60 GHz, 0.100 m range bins up to 10.3 m, +/-6.5 m/s, 10 frames per second.
% Detection and tracking commands are firmware-release specific: append those
% of the firmware running on the device.
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60 7 6 57 0 0 70 1 128 6000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 64 0 100 1 0
guiMonitor -1 1 0 0 0 0 0
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
sensorStart
""",
    "iwr1443_vital_signs": """\
% IWR1443, chirp parameters for short-range vital-signs sensing.
% One transmitter x four receivers, 77 GHz, 70 MHz/us slope,
% 200 samples at 4 Msps (0.033 m range bins), 20 frames per second.
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 1 0
adcCfg 2 1
adcbufCfg 0 1 0 1
profileCfg 0 77 100 6 57 0 0 70 1 200 4000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
frameCfg 0 0 2 0 50 1 0
sensorStart
""",
}

#: Platform each preset is written for.
PRESET_PLATFORMS: dict[str, Platform] = {
    "iwr1443_oob_2tx4rx": Platform.XWR14XX,
    "iwr1642_oob_2tx4rx": Platform.XWR16XX,
    "iwr6843_3d_3tx4rx": Platform.XWR68XX,
    "iwr1443_vital_signs": Platform.XWR14XX,
}


def preset_sensor_config(name: str) -> SensorConfig:
    """Parse one of :data:`PRESET_CLI_CONFIGS`.

    Examples
    --------
    >>> round(preset_sensor_config("iwr1443_vital_signs").range_bin_m, 4)
    0.0335
    """
    try:
        text = PRESET_CLI_CONFIGS[name]
    except KeyError:
        names = ", ".join(sorted(PRESET_CLI_CONFIGS))
        raise ConfigError(f"unknown preset {name!r}; choose one of {names}") from None
    return parse_cli_config(text, PRESET_PLATFORMS[name])


def default_preset_for(platform: int) -> str:
    """Name of the preset matching a platform (the xWR16xx one for other devices).

    Examples
    --------
    >>> default_preset_for(0xA1443), default_preset_for(0xA6843), default_preset_for(0xA1843)
    ('iwr1443_oob_2tx4rx', 'iwr6843_3d_3tx4rx', 'iwr1642_oob_2tx4rx')
    """
    if platform == Platform.XWR14XX:
        return "iwr1443_oob_2tx4rx"
    if platform == Platform.XWR68XX:
        return "iwr6843_3d_3tx4rx"
    return "iwr1642_oob_2tx4rx"
