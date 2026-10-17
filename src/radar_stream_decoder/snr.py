"""Per-point SNR from the log2-magnitude arrays of the same frame.

SDK 1.x/2.x firmware sends no SNR per point. Its ``peakVal`` is not usable as
one: on the xWR16xx it is a linear amplitude (a scaled square root of the
peak power), while the profiles and the detection matrix are log2
magnitudes, so the two cannot be subtracted. The estimate here compares
log2 magnitudes with log2 magnitudes, from the detection matrix when the
frame carries one and from the noise profile otherwise:

* In the detection matrix, the point's cell ``M[r, d]`` is compared with a
  local noise level from the training cells of the same range bin along
  Doppler, excluding guard cells and wrapping around (the Doppler axis is
  circular). The noise level is the cell average (CA, Finn & Johnson 1968
  [1]_) or the smaller of the two one-sided averages (SO, Trunk 1978 [2]_);
  the difference times the decibel scale is the SNR.
* Without the matrix, the range profile (zero Doppler) is compared with the
  noise profile at the point's range bin. This is exact for static points
  and only approximate for moving ones.

SDK 3.x firmware sends the SNR itself in its side info, which the frame
decoder reads directly.

Averaging in the log domain under-estimates the mean noise power: for
Rayleigh-distributed amplitudes the mean of ``10 log10`` of an exponential
variable lies :math:`10\\gamma/\\ln 10 \\approx 2.51` dB below the decibel
value of its mean (:math:`\\gamma` is the Euler-Mascheroni constant). The SNR
is therefore biased upwards by that amount unless ``log_bias_correction`` is
set.

References
----------
.. [1] H. M. Finn and R. S. Johnson, "Adaptive detection mode with threshold
   control as a function of spatially sampled clutter-level estimates,"
   *RCA Review*, vol. 29, no. 3, pp. 414-464, 1968.
.. [2] G. V. Trunk, "Range resolution of targets using automatic detectors,"
   *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-14,
   no. 5, pp. 750-755, 1978.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import NoiseEstimator
from .constants import Platform
from .payloads import SnrMethod

if TYPE_CHECKING:
    from .payloads import MagnitudeProfile, PointCloud, RangeDopplerMap

_LOGGER = logging.getLogger(__name__)

_EULER_GAMMA = 0.5772156649015329

#: Bias of averaging decibel values of exponentially distributed noise power.
LOG_DETECTOR_BIAS_DB = 10.0 * _EULER_GAMMA / math.log(10.0)


def _fit_training_window(num_cells: int, guard: int, train: int, cyclic: bool) -> int:
    """Largest training length (at most ``train``) that does not wrap onto the cell itself."""
    if guard < 0 or train < 1:
        raise ValueError(f"guard must be >= 0 and train >= 1, got {guard} and {train}")
    if not cyclic:
        return train
    room = (num_cells - 1) // 2 - guard
    if room < 1:
        raise ValueError(
            f"{num_cells} cells leave no training cells beside {guard} guard cells per side"
        )
    if room < train:
        _LOGGER.debug("training window shortened from %d to %d cells", train, room)
    return min(train, room)


def ca_noise_floor(
    rows: ArrayLike,
    idx: ArrayLike,
    guard: int = 2,
    train: int = 8,
    mode: NoiseEstimator | str = NoiseEstimator.CELL_AVERAGING,
    cyclic: bool = True,
) -> NDArray[np.float64]:
    """Local noise level around one cell of each row, averaged in the log domain.

    Parameters
    ----------
    rows : array_like, shape (n, cells) or (cells,)
        Log-domain values; one row per cell under test.
    idx : array_like of int, shape (n,) or scalar
        Index of the cell under test in each row.
    guard : int, optional
        Cells skipped on each side of the cell under test.
    train : int, optional
        Training cells averaged on each side. With a cyclic axis it is
        shortened when the row is too short for the full window.
    mode : {"ca", "so"}, optional
        ``"ca"`` averages both sides; ``"so"`` takes the smaller one-sided mean.
    cyclic : bool, optional
        Wrap around the row ends (Doppler); otherwise only cells inside the
        row are used and a missing side falls back to the other one.

    Returns
    -------
    numpy.ndarray, shape (n,)

    Examples
    --------
    >>> row = np.zeros(32); row[10] = 50.0; row[20:24] = 3.0
    >>> float(ca_noise_floor(row, 10, guard=1, train=4)[0])
    0.0
    >>> float(ca_noise_floor(row, 16, guard=1, train=4, mode="ca")[0])
    0.75
    >>> float(ca_noise_floor(row, 16, guard=1, train=4, mode="so")[0])
    0.0
    """
    values = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    centre = np.asarray(idx, dtype=np.int64).reshape(-1)
    if centre.size != values.shape[0]:
        raise ValueError(f"{centre.size} cell indices for {values.shape[0]} rows")
    estimator = NoiseEstimator(mode)
    cells = values.shape[1]
    window = _fit_training_window(cells, guard, train, cyclic)
    offsets = np.arange(guard + 1, guard + window + 1)
    left = centre[:, None] - offsets[None, :]
    right = centre[:, None] + offsets[None, :]
    row_index = np.arange(values.shape[0])[:, None]
    if cyclic:
        left_mean = values[row_index, left % cells].mean(axis=1)
        right_mean = values[row_index, right % cells].mean(axis=1)
    else:
        left_ok = left >= 0
        right_ok = right < cells
        left_vals = np.where(left_ok, values[row_index, np.clip(left, 0, cells - 1)], 0.0)
        right_vals = np.where(right_ok, values[row_index, np.clip(right, 0, cells - 1)], 0.0)
        left_count = left_ok.sum(axis=1)
        right_count = right_ok.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            left_mean = left_vals.sum(axis=1) / left_count
            right_mean = right_vals.sum(axis=1) / right_count
        left_mean = np.where(left_count > 0, left_mean, right_mean)
        right_mean = np.where(right_count > 0, right_mean, left_mean)
    if estimator is NoiseEstimator.SMALLEST_OF:
        return np.minimum(left_mean, right_mean)
    return 0.5 * (left_mean + right_mean)


@dataclass(frozen=True)
class SnrEstimate:
    """Per-point SNR and how it was obtained."""

    snr_db: NDArray[np.float64]
    method: SnrMethod


def estimate_point_snr(
    points: PointCloud,
    *,
    rd_map: RangeDopplerMap | None = None,
    range_profile: MagnitudeProfile | None = None,
    noise_profile: MagnitudeProfile | None = None,
    db_per_lsb: float,
    guard_cells: int = 2,
    train_cells: int = 8,
    estimator: NoiseEstimator | str = NoiseEstimator.CELL_AVERAGING,
    log_bias_correction: bool = False,
) -> SnrEstimate:
    """SNR of SDK 1.x/2.x points from the frame's log2-magnitude arrays.

    Parameters
    ----------
    points : PointCloud
        Points with range and Doppler indices.
    rd_map : RangeDopplerMap, optional
        Detection matrix of the same frame (preferred source).
    range_profile, noise_profile : MagnitudeProfile, optional
        Fallback source when there is no detection matrix.
    db_per_lsb : float
        Decibels of one log2-magnitude step.
    guard_cells, train_cells : int, optional
        Noise-window geometry along Doppler.
    estimator : {"ca", "so"}, optional
        Noise estimator.
    log_bias_correction : bool, optional
        Subtract :data:`LOG_DETECTOR_BIAS_DB`.

    Returns
    -------
    SnrEstimate
        ``method`` is ``unavailable`` (and every SNR NaN) when neither source is
        present or the points carry no indices.
    """
    n = len(points)
    unavailable = SnrEstimate(np.full(n, np.nan), SnrMethod.UNAVAILABLE)
    if n == 0 or points.range_idx is None:
        return unavailable
    correction = LOG_DETECTOR_BIAS_DB if log_bias_correction else 0.0
    ranges = points.range_idx.astype(np.int64)
    if rd_map is not None and rd_map.resolved and points.doppler_idx is not None:
        matrix = rd_map.raw
        if np.all(ranges < matrix.shape[0]):
            cells = points.doppler_idx.astype(np.int64) % matrix.shape[1]
            rows = matrix[ranges].astype(np.float64)
            signal = rows[np.arange(n), cells]
            noise = ca_noise_floor(rows, cells, guard_cells, train_cells, estimator, cyclic=True)
            snr = (signal - noise) * db_per_lsb - correction
            return SnrEstimate(snr, SnrMethod.DETECTION_MATRIX)
    if range_profile is not None and noise_profile is not None:
        bins = min(len(range_profile), len(noise_profile))
        if bins and np.all(ranges < bins):
            signal = range_profile.raw[ranges].astype(np.float64)
            noise = noise_profile.raw[ranges].astype(np.float64)
            snr = (signal - noise) * db_per_lsb - correction
            return SnrEstimate(snr, SnrMethod.NOISE_PROFILE)
    return unavailable


def peak_level_db(peak: ArrayLike, platform: int) -> NDArray[np.float64]:
    """Relative level of the firmware's ``peakVal`` in decibels.

    The xWR16xx writes a linear amplitude, so its level is
    ``20 log10(peak)``; for other platforms the convention of TI's visualiser,
    ``10 log10(1 + peak)``, is followed. Neither is a calibrated SNR: it can
    rank points, but it is no basis for a threshold.

    Examples
    --------
    >>> peak_level_db([1000], 0xA1642).round(1).tolist()
    [60.0]
    >>> peak_level_db([0], 0xA1443).tolist()
    [0.0]
    """
    values = np.asarray(peak, dtype=np.float64)
    if platform == Platform.XWR16XX:
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(np.where(values > 0, values, np.nan))
    return 10.0 * np.log10(1.0 + np.maximum(values, 0.0))
