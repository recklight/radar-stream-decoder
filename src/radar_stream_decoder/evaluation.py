"""How well the decoder copes with damaged streams, and how accurate its estimates are.

Both measurements run on synthetic streams whose content is known.

The first is stream integrity. The padded framing has no checksum, and the
people counting headers carry one over the header only, so the assembler's
structural checks act as a classifier that labels every packet either
*clean* (decoded without error-level issues) or *flagged/rejected*. With the
fault injector's record of what it did to every packet:

==================  =====================  ==========================
                    decoded clean          flagged or lost
==================  =====================  ==========================
intact packet       true positive          false negative (missed)
damaged packet      false positive         true negative
==================  =====================  ==========================

Sensitivity is the share of intact packets recovered, specificity the share of
damaged packets that did not pass as clean, and the unweighted average recall
(UAR) their mean. Precision is the share of clean frames that reproduce an
intact packet byte for byte; frames forged out of junk lower it. Each trial
uses its own seed, so repeated trials give a mean and a standard deviation.

The second is the accuracy of the estimators: on undamaged streams the
decoded SNR, azimuth peak, range and radial velocity are compared with the
scene's ground truth.
"""

from __future__ import annotations

import math
import time
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from .assembler import FrameAssembler
from .config import DecoderConfig
from .frame import RadarFrame, decode_packet
from .synthetic import (
    FaultPlan,
    PacketFate,
    SyntheticScene,
    generate_packets,
    inject_faults,
    scene_truth,
    split_into_chunks,
)


@dataclass(frozen=True)
class RecoveryTrial:
    """Outcome of decoding one damaged stream.

    Attributes
    ----------
    intact, damaged : int
        Packets delivered intact / delivered damaged (dropped ones are neither).
    true_positive, false_negative, false_positive, true_negative : int
        The confusion matrix described in the module documentation.
    spurious : int
        Clean frames that match no packet at all.
    clean_frames : int
        Frames decoded without error-level issues (repeats included).
    matched_frames : int
        Clean frames identical to an intact packet (repeats included).
    stream_bytes, bytes_skipped, resyncs, frames_rejected, frames_salvaged : int
    decode_seconds : float
        Wall time of assembling and decoding.
    """

    intact: int
    damaged: int
    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int
    spurious: int
    clean_frames: int
    matched_frames: int
    stream_bytes: int
    bytes_skipped: int
    resyncs: int
    frames_rejected: int
    frames_salvaged: int
    decode_seconds: float

    @property
    def sensitivity(self) -> float:
        """Recovered share of intact packets."""
        return self.true_positive / self.intact if self.intact else math.nan

    @property
    def specificity(self) -> float:
        """Share of damaged packets that did not pass as clean (NaN without damage)."""
        return self.true_negative / self.damaged if self.damaged else math.nan

    @property
    def uar(self) -> float:
        """Unweighted average recall of the two classes (sensitivity alone without damage)."""
        values = [v for v in (self.sensitivity, self.specificity) if not math.isnan(v)]
        return float(np.mean(values)) if values else math.nan

    @property
    def precision(self) -> float:
        """Share of clean frames that reproduce an intact packet."""
        return self.matched_frames / self.clean_frames if self.clean_frames else math.nan

    @property
    def throughput_mb_s(self) -> float:
        """Stream bytes decoded per second, in megabytes."""
        return self.stream_bytes / self.decode_seconds / 1e6 if self.decode_seconds else math.nan


def run_recovery_trial(
    scene: SyntheticScene,
    n_frames: int,
    plan: FaultPlan,
    seed: int,
    *,
    config: DecoderConfig | None = None,
    max_chunk: int = 4096,
) -> RecoveryTrial:
    """Render, damage, re-chunk and decode one stream, then score the result.

    Parameters
    ----------
    scene : SyntheticScene
        Scene to render (its own seed is used as is).
    n_frames : int
        Packets to render.
    plan : FaultPlan
        Faults to inject.
    seed : int
        Seed of the fault and chunking draws.
    config : DecoderConfig, optional
        Decoder settings (de-duplication does not affect the counts).
    max_chunk : int, optional
        Largest read size.
    """
    packets = list(generate_packets(scene, n_frames))
    rng = np.random.default_rng(seed)
    stream = inject_faults(packets, plan, rng)
    chunks = split_into_chunks(stream.data, rng, max_chunk=max_chunk)
    assembler = FrameAssembler(config or DecoderConfig(), scene.radar())
    started = time.perf_counter()
    frames: list[RadarFrame] = []
    for chunk in chunks:
        frames.extend(assembler.feed(chunk))
    frames.extend(assembler.finish())
    elapsed = time.perf_counter() - started

    intact_crc = {
        zlib.crc32(packet): index
        for index, (packet, fate) in enumerate(zip(packets, stream.fates, strict=True))
        if fate is PacketFate.DELIVERED
    }
    damaged = {
        index
        for index, fate in enumerate(stream.fates)
        if fate in (PacketFate.TRUNCATED, PacketFate.CORRUPTED)
    }
    # A damaged packet that passes as clean is recognised by the CRC of the bytes
    # actually delivered (a flipped bit may have changed its frame number), and
    # by its frame number when the damage changed its length as well.
    damaged_by_crc = {
        crc: index for index, crc in enumerate(stream.damaged_crc32) if crc is not None
    }
    damaged_by_number = {scene.first_frame_number + index: index for index in damaged}
    clean = [frame for frame in frames if not frame.has_errors and not frame.truncated]
    matched = [frame for frame in clean if frame.crc32 in intact_crc]
    unmatched = [frame for frame in clean if frame.crc32 not in intact_crc]
    recovered = {frame.crc32 for frame in matched}
    passed_damaged: set[int] = set()
    spurious = 0
    for frame in unmatched:
        index = damaged_by_crc.get(frame.crc32, damaged_by_number.get(frame.frame_number))
        if index is None:
            spurious += 1
        else:
            passed_damaged.add(index)
    stats = assembler.stats
    return RecoveryTrial(
        intact=len(intact_crc),
        damaged=len(damaged),
        true_positive=len(recovered),
        false_negative=len(intact_crc) - len(recovered),
        false_positive=len(passed_damaged),
        true_negative=len(damaged) - len(passed_damaged),
        spurious=spurious,
        clean_frames=len(clean),
        matched_frames=len(matched),
        stream_bytes=len(stream.data),
        bytes_skipped=stats.bytes_skipped,
        resyncs=stats.resyncs,
        frames_rejected=stats.frames_rejected,
        frames_salvaged=stats.frames_salvaged,
        decode_seconds=elapsed,
    )


def _mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(finite)), "std": float(np.std(finite))}


@dataclass(frozen=True)
class RecoveryReport:
    """Repeated recovery trials of one fault plan."""

    plan_name: str
    trials: tuple[RecoveryTrial, ...]

    def confusion_matrix(self) -> list[list[int]]:
        """``[[TP, FN], [FP, TN]]`` summed over trials."""
        return [
            [sum(t.true_positive for t in self.trials), sum(t.false_negative for t in self.trials)],
            [sum(t.false_positive for t in self.trials), sum(t.true_negative for t in self.trials)],
        ]

    def summary(self) -> dict[str, object]:
        """Mean and standard deviation of every rate, plus summed counts."""
        trials = list(self.trials)
        return {
            "faults": self.plan_name,
            "trials": len(trials),
            "sensitivity": _mean_std([t.sensitivity for t in trials]),
            "specificity": _mean_std([t.specificity for t in trials]),
            "uar": _mean_std([t.uar for t in trials]),
            "precision": _mean_std([t.precision for t in trials]),
            "throughput_mb_s": _mean_std([t.throughput_mb_s for t in trials]),
            "confusion_matrix": self.confusion_matrix(),
            "spurious_frames": sum(t.spurious for t in trials),
            "intact_packets": sum(t.intact for t in trials),
            "damaged_packets": sum(t.damaged for t in trials),
            "bytes_skipped": sum(t.bytes_skipped for t in trials),
            "resyncs": sum(t.resyncs for t in trials),
            "per_trial": [asdict(t) for t in trials],
        }


def evaluate_recovery(
    scene: SyntheticScene,
    plan: FaultPlan,
    *,
    plan_name: str = "custom",
    n_frames: int = 300,
    n_trials: int = 5,
    seed: int = 0,
    config: DecoderConfig | None = None,
) -> RecoveryReport:
    """Repeat :func:`run_recovery_trial` with seeds ``seed, seed + 1, ...``.

    Each trial also re-renders the scene with its own seed, so noise, fault
    positions and chunk boundaries all change from trial to trial while the
    whole evaluation stays reproducible.
    """
    if n_trials < 1 or n_frames < 1:
        raise ValueError("n_trials and n_frames must be positive")
    trials = tuple(
        run_recovery_trial(
            scene.replace(seed=scene.seed + trial),
            n_frames,
            plan,
            seed + trial,
            config=config,
        )
        for trial in range(n_trials)
    )
    return RecoveryReport(plan_name, trials)


@dataclass(frozen=True)
class EstimatorAccuracy:
    """Errors of the decoded quantities against the scene's ground truth.

    Attributes
    ----------
    snr_bias_db, snr_rmse_db : float
        Of the per-point SNR from the detection matrix.
    snr_samples : int
    azimuth_rmse_deg, azimuth_max_error_deg : float
        Of the refined azimuth-heatmap peak of isolated static targets.
    azimuth_samples : int
    range_rmse_m : float
        Of the point coordinates (Q-format quantisation).
    velocity_rmse_mps : float
        Of the point velocity (Doppler-bin quantisation).
    velocity_samples : int
    """

    snr_bias_db: float
    snr_rmse_db: float
    snr_samples: int
    azimuth_rmse_deg: float
    azimuth_max_error_deg: float
    azimuth_samples: int
    range_rmse_m: float
    velocity_rmse_mps: float
    velocity_samples: int

    def as_dict(self) -> dict[str, float | int]:
        """Field names to values."""
        return asdict(self)


#: Azimuth accuracy counts only static reflectors without another static
#: reflector within this many range bins (a heatmap row has one peak).
ISOLATION_BINS = 2


def _rms(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values else math.nan


def evaluate_estimators(
    scene: SyntheticScene, n_frames: int = 100, config: DecoderConfig | None = None
) -> EstimatorAccuracy:
    """Compare decoded SNR, azimuth, range and velocity with ground truth.

    Only out-of-box demo scenes (SDK 1.x/2.x schema) carry everything needed.
    """
    if scene.schema != "oob-sdk2":
        raise ValueError("estimator accuracy is measured on oob-sdk2 scenes")
    sensor = scene.radar()
    config = config or DecoderConfig()
    snr_errors: list[float] = []
    azimuth_errors: list[float] = []
    range_errors: list[float] = []
    velocity_errors: list[float] = []
    for index, packet in enumerate(generate_packets(scene, n_frames)):
        truth = scene_truth(scene, index)
        frame = decode_packet(packet, config, sensor)
        detected = np.flatnonzero(truth.detected)
        points = frame.points
        if points is not None and len(points) == detected.size and detected.size:
            expected_snr = np.minimum(truth.snr_db[detected], 60.0)
            snr_errors.extend((points.snr_db - expected_snr).tolist())
            range_errors.extend((points.range_m - truth.range_m[detected]).tolist())
            velocity_errors.extend(
                (points.velocity_mps - truth.radial_velocity_mps[detected]).tolist()
            )
        heatmap = frame.azimuth_heatmap
        if heatmap is not None and heatmap.resolved:
            static = np.flatnonzero(truth.visible & (truth.doppler_idx == 0))
            if static.size:
                peaks = heatmap.peak_azimuth_deg(config.angle_fft_size)
                bins = truth.range_idx[static]
                for target, row in zip(static, bins, strict=True):
                    # One peak per range bin: only isolated reflectors have a defined answer.
                    if np.sum(np.abs(bins - row) <= ISOLATION_BINS) > 1:
                        continue
                    estimate = peaks[int(row)]
                    azimuth_errors.append(float(estimate - truth.azimuth_deg[target]))
    finite_snr = [e for e in snr_errors if math.isfinite(e)]
    return EstimatorAccuracy(
        snr_bias_db=float(np.mean(finite_snr)) if finite_snr else math.nan,
        snr_rmse_db=_rms(finite_snr),
        snr_samples=len(finite_snr),
        azimuth_rmse_deg=_rms(azimuth_errors),
        azimuth_max_error_deg=float(np.max(np.abs(azimuth_errors))) if azimuth_errors else math.nan,
        azimuth_samples=len(azimuth_errors),
        range_rmse_m=_rms(range_errors),
        velocity_rmse_mps=_rms(velocity_errors),
        velocity_samples=len(velocity_errors),
    )
