"""Azimuth static heatmaps: complex decoding, angle FFT and Cartesian resampling.

The out-of-box demo's type-4 TLV is the zero-Doppler slice of the radar cube:
for every range bin, one complex sample per azimuth virtual antenna
(``num_tx_azimuth * num_rx`` of them in TDM-MIMO order), each written as
``cmplx16ImRe_t`` with the *imaginary* part first, then the real part. Reading
the pair the other way round yields ``j * conj(z)``, whose angle spectrum is
the mirror image of the true one.

Angle processing is the classic FFT (Bartlett) beamformer of a uniform linear
array with half-wavelength spacing. A target at azimuth :math:`\\theta` adds a
phase of :math:`\\pi\\sin\\theta` per virtual element, so after an
:math:`N`-point zero-padded FFT and ``fftshift``, bin :math:`k` corresponds to

.. math:: \\theta_k = \\arcsin(2k/N), \\qquad k = -N/2+1, \\dots, N/2-1 .

The bin :math:`k = -N/2` (:math:`\\theta = -90^\\circ` exactly) is dropped,
leaving :math:`N - 1` angles. A heatmap over :math:`(r, \\theta)` is finally
resampled onto a Cartesian grid with ``x = r sin(theta)``, ``y = r cos(theta)``
by bilinear interpolation, written in plain NumPy so that SciPy is not needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import ANGLE_FFT_SIZE
from .errors import IssueCode, Severity

if TYPE_CHECKING:
    from .payloads import DecodeContext

_ORDERS = ("imag_real", "real_imag")


def decode_complex_iq(payload: bytes | memoryview, order: str = "imag_real") -> NDArray:
    """Decode interleaved ``int16`` pairs into ``complex64`` samples.

    Parameters
    ----------
    payload : bytes-like
        A multiple of four bytes.
    order : {"imag_real", "real_imag"}
        Which part comes first in each pair. The out-of-box demo heatmaps are
        ``imag_real``; the Vital Signs range segment is ``real_imag``.

    Returns
    -------
    numpy.ndarray of complex64

    Examples
    --------
    >>> import struct
    >>> raw = struct.pack("<4h", 3, 4, -1, 2)
    >>> decode_complex_iq(raw).tolist()
    [(4+3j), (2-1j)]
    >>> decode_complex_iq(raw, order="real_imag").tolist()
    [(3+4j), (-1+2j)]
    """
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {_ORDERS}, got {order!r}")
    if len(payload) % 4:
        raise ValueError(f"complex payload length {len(payload)} is not a multiple of 4")
    pairs = np.frombuffer(payload, dtype="<i2").reshape(-1, 2).astype(np.float32)
    first, second = pairs[:, 0], pairs[:, 1]
    real, imag = (second, first) if order == "imag_real" else (first, second)
    return (real + 1j * imag).astype(np.complex64)


def encode_complex_iq(values: ArrayLike, order: str = "imag_real") -> bytes:
    """Inverse of :func:`decode_complex_iq`, rounding to the nearest ``int16``.

    Values outside the ``int16`` range are clipped.
    """
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {_ORDERS}, got {order!r}")
    samples = np.asarray(values, dtype=np.complex128).reshape(-1)
    real = np.clip(np.rint(samples.real), -32768, 32767).astype("<i2")
    imag = np.clip(np.rint(samples.imag), -32768, 32767).astype("<i2")
    pairs = np.stack((imag, real) if order == "imag_real" else (real, imag), axis=1)
    return pairs.astype("<i2").tobytes()


def angle_bins(n_fft: int = ANGLE_FFT_SIZE) -> NDArray[np.int64]:
    """Signed FFT bins ``-N/2+1 .. N/2-1`` that :func:`angle_spectrum` returns."""
    if n_fft < 2 or n_fft % 2:
        raise ValueError(f"n_fft must be an even number of at least 2, got {n_fft}")
    return np.arange(-n_fft // 2 + 1, n_fft // 2)


def angle_axis_rad(n_fft: int = ANGLE_FFT_SIZE) -> NDArray[np.float64]:
    """Azimuth of every angle bin, ``asin(2k/N)``, in radians."""
    return np.arcsin(2.0 * angle_bins(n_fft) / n_fft)


def angle_axis_deg(n_fft: int = ANGLE_FFT_SIZE) -> NDArray[np.float64]:
    """Azimuth of every angle bin in degrees.

    Examples
    --------
    >>> axis = angle_axis_deg(64)
    >>> len(axis), round(float(axis[31]), 6), round(float(axis[-1]), 2)
    (63, 0.0, 75.64)
    """
    return np.degrees(angle_axis_rad(n_fft))


def angle_spectrum(cube: ArrayLike, n_fft: int = ANGLE_FFT_SIZE) -> NDArray[np.float64]:
    """Magnitude of the zero-padded angle FFT across virtual antennas.

    Parameters
    ----------
    cube : array_like of complex, shape (range bins, antennas) or (antennas,)
        Antenna samples per range bin.
    n_fft : int, optional
        FFT length (at least the antenna count).

    Returns
    -------
    numpy.ndarray, shape (range bins, n_fft - 1)
        ``|FFT|`` after ``fftshift``, without the ``-N/2`` bin, aligned with
        :func:`angle_axis_deg`.
    """
    samples = np.atleast_2d(np.asarray(cube))
    if samples.shape[-1] > n_fft:
        raise ValueError(f"n_fft ({n_fft}) is shorter than the {samples.shape[-1]} antennas")
    angle_bins(n_fft)  # validates n_fft
    spectrum = np.fft.fftshift(np.fft.fft(samples, n=n_fft, axis=-1), axes=-1)
    return np.abs(spectrum[..., 1:])


def peak_angle_deg(
    spectrum: ArrayLike, n_fft: int = ANGLE_FFT_SIZE, refine: bool = True
) -> NDArray[np.float64]:
    """Azimuth of the strongest angle bin of every row.

    Parameters
    ----------
    spectrum : array_like, shape (rows, n_fft - 1) or (n_fft - 1,)
        Output of :func:`angle_spectrum`.
    n_fft : int, optional
        FFT length the spectrum was computed with.
    refine : bool, optional
        Interpolate the peak with a parabola through the log magnitudes of
        the peak bin and its neighbours, for sub-bin accuracy.

    Returns
    -------
    numpy.ndarray
        Degrees, one per row.
    """
    values = np.atleast_2d(np.asarray(spectrum, dtype=np.float64))
    _rows, width = values.shape
    if width != n_fft - 1:
        raise ValueError(f"spectrum rows must have {n_fft - 1} bins, got {width}")
    peak = np.argmax(values, axis=1)
    position = peak.astype(np.float64)
    if refine:
        inner = (peak > 0) & (peak < width - 1)
        idx = np.flatnonzero(inner)
        if idx.size:
            logs = np.log(np.maximum(values, np.finfo(np.float64).tiny))
            left = logs[idx, peak[idx] - 1]
            centre = logs[idx, peak[idx]]
            right = logs[idx, peak[idx] + 1]
            curvature = left - 2.0 * centre + right
            shift = np.zeros(idx.size)
            ok = curvature < 0
            shift[ok] = 0.5 * (left[ok] - right[ok]) / curvature[ok]
            position[idx] += np.clip(shift, -0.5, 0.5)
    signed = position - (n_fft // 2 - 1)
    return np.degrees(np.arcsin(np.clip(2.0 * signed / n_fft, -1.0, 1.0)))


@dataclass
class CartesianHeatmap:
    """A heatmap resampled onto an ``x``-``y`` grid.

    Attributes
    ----------
    x_m : numpy.ndarray, shape (nx,)
    y_m : numpy.ndarray, shape (ny,)
    image : numpy.ndarray, shape (ny, nx)
        NaN outside the covered range and angle interval.
    """

    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    image: NDArray[np.float64]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(left, right, bottom, top)`` for ``matplotlib.imshow``."""
        return (float(self.x_m[0]), float(self.x_m[-1]), float(self.y_m[0]), float(self.y_m[-1]))


def _fractional_index(
    axis: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    """Lower neighbour, interpolation weight and in-range mask on an increasing axis."""
    lower = np.searchsorted(axis, values, side="right") - 1
    lower = np.clip(lower, 0, axis.size - 2)
    span = axis[lower + 1] - axis[lower]
    weight = (values - axis[lower]) / span
    inside = (values >= axis[0]) & (values <= axis[-1])
    return lower, weight, inside


def polar_to_cartesian(
    spectrum: ArrayLike,
    range_axis_m: ArrayLike,
    theta_rad: ArrayLike,
    grid_points: int = 100,
    x_extent: tuple[float, float] | None = None,
    y_extent: tuple[float, float] | None = None,
) -> CartesianHeatmap:
    """Bilinearly resample a ``(range, angle)`` heatmap onto a Cartesian grid.

    Parameters
    ----------
    spectrum : array_like, shape (R, M)
        Values over increasing ranges (rows) and increasing angles (columns).
    range_axis_m : array_like, shape (R,)
        Range of every row; strictly increasing.
    theta_rad : array_like, shape (M,)
        Azimuth of every column in radians; strictly increasing (it need not be
        uniform -- the ``asin`` spacing of the angle FFT is not).
    grid_points : int, optional
        Grid size along each axis.
    x_extent, y_extent : tuple of float, optional
        Grid limits; default to ``(-R_max, R_max)`` and ``(0, R_max)``.

    Returns
    -------
    CartesianHeatmap

    Examples
    --------
    >>> heat = polar_to_cartesian(np.ones((4, 3)), [0.0, 1.0, 2.0, 3.0],
    ...                           np.radians([-60.0, 0.0, 60.0]), grid_points=5)
    >>> heat.image.shape, float(np.nanmax(heat.image))
    ((5, 5), 1.0)
    """
    values = np.asarray(spectrum, dtype=np.float64)
    ranges = np.asarray(range_axis_m, dtype=np.float64).reshape(-1)
    angles = np.asarray(theta_rad, dtype=np.float64).reshape(-1)
    if values.shape != (ranges.size, angles.size):
        raise ValueError(
            f"spectrum shape {values.shape} does not match {ranges.size} ranges x "
            f"{angles.size} angles"
        )
    if ranges.size < 2 or angles.size < 2:
        raise ValueError("need at least two ranges and two angles to interpolate")
    if np.any(np.diff(ranges) <= 0) or np.any(np.diff(angles) <= 0):
        raise ValueError("range and angle axes must be strictly increasing")
    if grid_points < 2:
        raise ValueError(f"grid_points must be at least 2, got {grid_points}")
    r_max = float(ranges[-1])
    xs = np.linspace(*(x_extent or (-r_max, r_max)), grid_points)
    ys = np.linspace(*(y_extent or (0.0, r_max)), grid_points)
    grid_x, grid_y = np.meshgrid(xs, ys)
    radius = np.hypot(grid_x, grid_y)
    azimuth = np.arctan2(grid_x, grid_y)
    r0, rw, r_in = _fractional_index(ranges, radius)
    t0, tw, t_in = _fractional_index(angles, azimuth)
    image = (
        (1.0 - rw) * (1.0 - tw) * values[r0, t0]
        + (1.0 - rw) * tw * values[r0, t0 + 1]
        + rw * (1.0 - tw) * values[r0 + 1, t0]
        + rw * tw * values[r0 + 1, t0 + 1]
    )
    image[~(r_in & t_in)] = np.nan
    return CartesianHeatmap(xs, ys, image)


@dataclass
class AzimuthHeatmap:
    """Zero-Doppler complex samples per range bin and virtual antenna.

    Attributes
    ----------
    cube : numpy.ndarray of complex64
        Shape ``(range bins, antennas)``; one-dimensional when the antenna
        count could not be resolved.
    includes_elevation : bool
        ``True`` for the SDK 3.x azimuth-elevation heatmap, which carries every
        virtual antenna rather than only the azimuth array.
    """

    cube: NDArray
    includes_elevation: bool = False

    @property
    def resolved(self) -> bool:
        """Whether ``cube`` has its two-dimensional shape."""
        return self.cube.ndim == 2

    def _need_resolved(self) -> NDArray:
        if not self.resolved:
            raise ValueError(
                "the antenna count is unresolved; decode with a radar configuration or set "
                "num_virtual_antennas in DecoderConfig"
            )
        return self.cube

    def spectrum(self, n_fft: int = ANGLE_FFT_SIZE) -> NDArray[np.float64]:
        """Angle spectrum of every range bin (see :func:`angle_spectrum`)."""
        return angle_spectrum(self._need_resolved(), n_fft)

    def peak_azimuth_deg(self, n_fft: int = ANGLE_FFT_SIZE, refine: bool = True) -> NDArray:
        """Strongest azimuth of every range bin."""
        return peak_angle_deg(self.spectrum(n_fft), n_fft, refine)

    def to_cartesian(
        self,
        range_axis_m: ArrayLike,
        n_fft: int = ANGLE_FFT_SIZE,
        grid_points: int = 100,
        x_extent: tuple[float, float] | None = None,
        y_extent: tuple[float, float] | None = None,
    ) -> CartesianHeatmap:
        """Angle spectrum resampled onto an ``x``-``y`` grid."""
        return polar_to_cartesian(
            self.spectrum(n_fft),
            range_axis_m,
            angle_axis_rad(n_fft),
            grid_points,
            x_extent,
            y_extent,
        )


def _decode_cube(
    payload: memoryview, ctx: DecodeContext, antennas: int | None, includes_elevation: bool
) -> AzimuthHeatmap:
    what = "azimuth-elevation heatmap" if includes_elevation else "azimuth heatmap"
    usable = len(payload) - len(payload) % 4
    if usable != len(payload):
        ctx.report(
            IssueCode.LENGTH_MISMATCH,
            f"{what} TLV of {len(payload)} bytes is not a whole number of complex samples",
            Severity.ERROR,
        )
    samples = decode_complex_iq(payload[:usable], order="imag_real")
    if antennas and samples.size % antennas == 0 and samples.size >= antennas:
        return AzimuthHeatmap(samples.reshape(-1, antennas), includes_elevation)
    for rows in ctx.range_bin_candidates():
        if samples.size % rows == 0 and samples.size >= rows:
            return AzimuthHeatmap(samples.reshape(rows, -1), includes_elevation)
    ctx.report(
        IssueCode.UNRESOLVED_SHAPE,
        f"cannot split {samples.size} {what} samples into range bins x antennas; pass a "
        "radar configuration or set num_virtual_antennas",
    )
    return AzimuthHeatmap(samples, includes_elevation)


def decode_azimuth_heatmap(payload: memoryview, ctx: DecodeContext) -> AzimuthHeatmap:
    """Decode the type-4 azimuth static heatmap.

    The antenna count comes from the radar configuration or
    ``DecoderConfig.num_virtual_antennas``; failing both, it is inferred from
    the range-bin count (radar configuration or the packet's range profile).
    """
    return _decode_cube(payload, ctx, ctx.num_azimuth_antennas(), includes_elevation=False)


def decode_azimuth_elevation_heatmap(payload: memoryview, ctx: DecodeContext) -> AzimuthHeatmap:
    """Decode the SDK 3.x type-8 heatmap (every virtual antenna per range bin)."""
    return _decode_cube(payload, ctx, ctx.num_virtual_antennas(), includes_elevation=True)
