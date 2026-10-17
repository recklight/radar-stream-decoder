"""Conversions between the integers on the wire and physical units.

Indices to metres and metres per second, the sign of Doppler indices, the
log2-magnitude scale of range/noise profiles and the detection matrix, and
Cartesian-to-spherical geometry in TI's sensor frame (``y`` along the
boresight, ``x`` to the right, ``z`` up).
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import DB_PER_LOG2_UNIT, Platform


def next_power_of_two(value: int) -> int:
    """Smallest power of two that is at least ``value``.

    Examples
    --------
    >>> next_power_of_two(200), next_power_of_two(256), next_power_of_two(1)
    (256, 256, 1)
    """
    if value < 1:
        raise ValueError(f"expected a positive integer, got {value}")
    return 1 << (int(value) - 1).bit_length()


def signed_doppler(indices: ArrayLike, num_doppler_bins: int) -> NDArray[np.int32]:
    """Map unsigned Doppler FFT bins onto signed indices.

    Bins ``0 .. D/2-1`` are non-negative velocities and ``D/2 .. D-1`` are the
    negative ones (``-D/2 .. -1``), the usual circular-shift convention.
    Firmware built with SDK 1.0 writes the unsigned bin; later releases write
    the signed value directly.

    Parameters
    ----------
    indices : array_like of int
        Unsigned bins in ``[0, D)``.
    num_doppler_bins : int
        Doppler FFT size ``D``.

    Returns
    -------
    numpy.ndarray of int32

    Examples
    --------
    >>> signed_doppler([0, 1, 7, 8, 15], 16).tolist()
    [0, 1, 7, -8, -1]
    """
    if num_doppler_bins < 1:
        raise ValueError(f"num_doppler_bins must be positive, got {num_doppler_bins}")
    values = np.asarray(indices, dtype=np.int64) % num_doppler_bins
    half = num_doppler_bins // 2
    return np.where(values >= half, values - num_doppler_bins, values).astype(np.int32)


def doppler_to_velocity(indices: ArrayLike, doppler_resolution_mps: float) -> NDArray[np.float64]:
    """Signed Doppler index times the velocity of one Doppler bin.

    Examples
    --------
    >>> doppler_to_velocity([-2, 0, 3], 0.25).tolist()
    [-0.5, 0.0, 0.75]
    """
    return np.asarray(indices, dtype=np.float64) * float(doppler_resolution_mps)


def range_index_to_m(
    indices: ArrayLike, range_bin_m: float, range_bias_m: float = 0.0
) -> NDArray[np.float64]:
    """Range-FFT bin to metres, ``k * bin - bias`` clipped at zero.

    Examples
    --------
    >>> range_index_to_m([0, 1, 10], 0.05, range_bias_m=0.07).round(3).tolist()
    [0.0, 0.0, 0.43]
    """
    metres = np.asarray(indices, dtype=np.float64) * float(range_bin_m) - float(range_bias_m)
    return np.maximum(metres, 0.0)


class Log2Combining(str, Enum):
    """How firmware folds the per-antenna log2 magnitudes into one word.

    ``average``
        Mean over the virtual antennas (xWR14xx out-of-box demo, SDK 1.x).
    ``sum``
        Plain sum over the virtual antennas (xWR16xx out-of-box demo, SDK 1.x/2.x;
        ``MmwDemo_log2AbsAccum`` in SDK 2.0 ``dss_data_path.c``).
    ``shifted-sum``
        Every antenna's value is shifted right by :math:`\\lceil\\log_2 N\\rceil`
        before it is added (out-of-box demo of SDK 3.x). The sum therefore stays
        within 16 bits for any antenna count, at the cost of up to
        :math:`2^{\\lceil\\log_2 N\\rceil}` LSB of truncation.
    """

    AVERAGE = "average"
    SUM = "sum"
    SHIFTED_SUM = "shifted-sum"


def default_log2_fraction_bits(platform: int, schema: str) -> int:
    """Fixed-point fraction bits of log2 magnitudes for a firmware family.

    The xWR16xx out-of-box demo up to SDK 2.x and the SDK 3.x demo write Q8;
    the xWR14xx demo writes Q9.

    TI documents and firmware disagree for SDK 3.x: the Radar Toolbox page
    "Understanding UART Data Output Format" and the SDK 3.x demo's doxygen call
    the range profile a "sum of log2 magnitudes ... in Q9 format", but the
    firmware of TI mmWave SDK 03.05.00.04 writes Q8: ``mmwavelib_log2Abs32``
    returns ``round(log2(abs(x)) * 2**8)`` (``mmwavelib_log2abs.c``) and
    ``dpif_detmatrix.h`` defines ``DPIF_DETMATRIX_DATA_QFORMAT`` as 8. The
    decoder follows the firmware.

    Examples
    --------
    >>> default_log2_fraction_bits(0xA6843, "oob-sdk3")
    8
    >>> default_log2_fraction_bits(0xA1443, "oob-sdk2")
    9
    """
    if schema == "oob-sdk3":
        return 8
    return 8 if platform == Platform.XWR16XX else 9


def default_log2_combining(platform: int, schema: str) -> Log2Combining:
    """How a firmware family combines log2 magnitudes over the virtual antennas.

    Examples
    --------
    >>> default_log2_combining(0xA6843, "oob-sdk3").value
    'shifted-sum'
    >>> default_log2_combining(0xA1642, "oob-sdk2").value
    'sum'
    >>> default_log2_combining(0xA1443, "oob-sdk2").value
    'average'
    """
    if schema == "oob-sdk3":
        return Log2Combining.SHIFTED_SUM
    return Log2Combining.SUM if platform == Platform.XWR16XX else Log2Combining.AVERAGE


def antenna_shift_bits(num_virtual_antennas: int) -> int:
    """The right shift SDK 3.x applies per antenna, :math:`\\lceil\\log_2 N\\rceil`.

    Examples
    --------
    >>> [antenna_shift_bits(n) for n in (1, 4, 8, 12, 16)]
    [0, 2, 3, 4, 4]
    """
    if num_virtual_antennas < 1:
        raise ValueError(f"expected at least one virtual antenna, got {num_virtual_antennas}")
    return (int(num_virtual_antennas) - 1).bit_length()


def log2_magnitude_scale(
    platform: int,
    schema: str,
    num_rx: int | None = None,
    num_tx: int | None = None,
    elevation_tx: bool = False,
    *,
    fraction_bits: int | None = None,
    combining: Log2Combining | str | None = None,
) -> float:
    """Factor that turns a raw log2-magnitude word into log2 units per antenna.

    The profiles and the detection matrix carry ``Q(f)`` fixed-point log2
    magnitudes combined over the :math:`N = N_{rx} N_{tx}` virtual antennas
    (see :class:`Log2Combining`). Multiplying a word :math:`v` by this factor,
    then by :data:`~radar_stream_decoder.constants.DB_PER_LOG2_UNIT`, gives
    the mean per-antenna level in decibels:

    .. math::

        \\text{average:}\\ \\frac{v}{2^{f}}, \\qquad
        \\text{sum:}\\ \\frac{v}{2^{f} N}, \\qquad
        \\text{shifted-sum:}\\ \\frac{v\\,2^{\\lceil\\log_2 N\\rceil}}{2^{f} N}

    The shifted sum is what TI mmWave SDK 03.05.00.04 writes:
    ``DPU_DopplerProcDSP_log2AbsAccum`` (``dopplerprocdsp.c``) shifts the first
    antenna's Q8 log2 magnitude right by ``log2VirtAnt = ceil(log2(N))`` and adds
    the others with ``mmwavelib_accum16shift`` (``out += in >> shift``,
    ``mmwavelib_accum.c``). :math:`N` counts every virtual antenna, elevation
    transmitters included. TI's own threshold conversion in the demo's
    ``mss_main.c`` uses the same factor, ``dB * (256/6) * (N / 2^ceil(log2 N))``,
    with 6 dB in place of :math:`20\\log_{10}2 \\approx 6.02` dB.

    The xWR14xx demo with its elevation transmitter enabled needs an extra
    factor 4/3 on its averaged values, which TI's visualiser applies as well.

    Parameters
    ----------
    platform : int
        Platform code.
    schema : str
        Packet schema value.
    num_rx, num_tx : int, optional
        Receive and transmit antennas in use; required unless the values are
        averages.
    elevation_tx : bool, optional
        xWR14xx only: the elevation transmitter is part of the frame.
    fraction_bits : int, optional
        Override of the fixed-point fraction bits.
    combining : Log2Combining or str, optional
        Override of the antenna combining convention.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the values are combined sums and the antenna counts are unknown.

    Examples
    --------
    >>> round(log2_magnitude_scale(0xA1642, "oob-sdk2", num_rx=4, num_tx=2) * 2048, 6)
    1.0
    >>> log2_magnitude_scale(0xA1443, "oob-sdk2") == 1 / 512
    True
    >>> log2_magnitude_scale(0xA6843, "oob-sdk3", num_rx=4, num_tx=3) == 16 / (256 * 12)
    True
    """
    bits = default_log2_fraction_bits(platform, schema) if fraction_bits is None else fraction_bits
    mode = (
        default_log2_combining(platform, schema) if combining is None else Log2Combining(combining)
    )
    scale = 1.0 / float(1 << int(bits))
    if mode is Log2Combining.AVERAGE:
        if elevation_tx and platform == Platform.XWR14XX:
            scale *= 4.0 / 3.0
        return scale
    if not num_rx or not num_tx:
        raise ValueError(
            f"{mode.value} log2 magnitudes need the receive and transmit antenna counts"
        )
    antennas = int(num_rx) * int(num_tx)
    scale /= float(antennas)
    if mode is Log2Combining.SHIFTED_SUM:
        scale *= float(1 << antenna_shift_bits(antennas))
    return scale


def log2_to_db(values: ArrayLike, scale: float, offset_db: float = 0.0) -> NDArray[np.float64]:
    """Raw log2-magnitude words to decibels.

    Parameters
    ----------
    values : array_like
        Raw words.
    scale : float
        Output of :func:`log2_magnitude_scale`.
    offset_db : float, optional
        Added afterwards, for example :func:`fft_scale_compensation_db`.

    Examples
    --------
    >>> log2_to_db([512], 1 / 512).round(4).tolist()
    [6.0206]
    """
    return np.asarray(values, dtype=np.float64) * (float(scale) * DB_PER_LOG2_UNIT) + offset_db


def _fft_stage_scale(size: int) -> float:
    # log4 via log2: math.log(64, 4) is 3.0000000000000004, which ceil() would ruin.
    stages = math.ceil(math.log2(size) / 2.0 - 1.0)
    return float(stages * stages) / float(size)


def fft_scale_compensation_db(platform: int, num_range_bins: int, num_doppler_bins: int) -> float:
    """Offset that refers log2 magnitudes to an absolute level.

    The fixed-point FFTs scale their output by a size-dependent amount. TI's
    demo visualiser undoes this with

    * xWR16xx: :math:`20\\log_{10}(16/D) + 20\\log_{10}(s(R)/s(64))`, where
      :math:`s(N) = \\lceil\\log_4 N - 1\\rceil^2 / N`;
    * xWR14xx: :math:`20\\log_{10}(32/R)`.

    The decoder never adds this offset on its own; relative quantities (SNR,
    peak heights) do not need it. It is provided for comparisons with plots
    from TI's tools.

    Examples
    --------
    >>> round(fft_scale_compensation_db(0xA1443, 256, 32), 3)
    -18.062
    >>> round(fft_scale_compensation_db(0xA1642, 64, 16), 3)
    0.0
    """
    if num_range_bins < 1 or num_doppler_bins < 1:
        raise ValueError("bin counts must be positive")
    if platform == Platform.XWR16XX:
        doppler_term = 20.0 * math.log10(16.0 / num_doppler_bins)
        range_term = 20.0 * math.log10(_fft_stage_scale(num_range_bins) / _fft_stage_scale(64))
        return doppler_term + range_term
    return 20.0 * math.log10(32.0 / num_range_bins)


def cartesian_to_spherical(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Range, azimuth and elevation of points in the sensor frame.

    Azimuth is measured from the boresight (``y``) towards ``x``; elevation
    from the ``x``-``y`` plane towards ``z``. Both are in degrees.

    Examples
    --------
    >>> r, az, el = cartesian_to_spherical([1.0], [1.0], [0.0])
    >>> round(float(r[0]), 4), round(float(az[0]), 1), float(el[0])
    (1.4142, 45.0, 0.0)
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    zs = np.asarray(z, dtype=np.float64)
    ground = np.hypot(xs, ys)
    distance = np.hypot(ground, zs)
    azimuth = np.degrees(np.arctan2(xs, ys))
    elevation = np.degrees(np.arctan2(zs, ground))
    return distance, azimuth, elevation


def spherical_to_cartesian(
    distance: ArrayLike, azimuth_deg: ArrayLike, elevation_deg: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Inverse of :func:`cartesian_to_spherical`.

    Examples
    --------
    >>> x, y, z = spherical_to_cartesian([2.0], [30.0], [0.0])
    >>> round(float(x[0]), 6), round(float(y[0]), 4)
    (1.0, 1.7321)
    """
    r = np.asarray(distance, dtype=np.float64)
    az = np.radians(np.asarray(azimuth_deg, dtype=np.float64))
    el = np.radians(np.asarray(elevation_deg, dtype=np.float64))
    ground = r * np.cos(el)
    return ground * np.sin(az), ground * np.cos(az), r * np.sin(el)


def xyz_q_format(range_bin_m: float) -> int:
    """Q-format the demo uses for point coordinates, ``ceil(log2(16 / bin))``.

    One coordinate step is then at most 1/16 of a range bin, so the int16
    coordinates resolve well below one bin and still reach beyond 1000 bins.

    Examples
    --------
    >>> xyz_q_format(0.0732)
    8
    """
    if range_bin_m <= 0:
        raise ValueError(f"range_bin_m must be positive, got {range_bin_m}")
    return math.ceil(math.log2(16.0 / range_bin_m))
