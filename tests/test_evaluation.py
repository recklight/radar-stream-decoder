"""Stream-integrity scores and estimator accuracy on synthetic streams."""

from __future__ import annotations

import math

import pytest

from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.evaluation import (
    evaluate_estimators,
    evaluate_recovery,
    run_recovery_trial,
)
from radar_stream_decoder.synthetic import FAULT_PRESETS, SyntheticScene

VITAL = SyntheticScene(schema="vital-signs", seed=40)


def test_clean_stream_is_recovered_completely():
    trial = run_recovery_trial(VITAL, 50, FAULT_PRESETS["none"], seed=1)
    assert trial.intact == 50 and trial.damaged == 0
    assert trial.sensitivity == 1.0 and math.isnan(trial.specificity)
    assert trial.uar == 1.0 and trial.precision == 1.0
    assert trial.throughput_mb_s > 0


@pytest.mark.parametrize("strict", [False, True])
def test_structural_faults_are_all_caught(strict):
    trial = run_recovery_trial(
        VITAL, 150, FAULT_PRESETS["heavy"], seed=3, config=DecoderConfig(strict=strict)
    )
    assert trial.damaged > 0 and trial.intact > 0
    assert trial.sensitivity == 1.0
    assert trial.specificity == 1.0
    assert trial.precision == 1.0 and trial.spurious == 0


def test_bit_flips_expose_the_missing_checksum():
    trial = run_recovery_trial(VITAL, 150, FAULT_PRESETS["bitflip"], seed=2)
    assert trial.sensitivity == 1.0
    assert trial.damaged > 10
    assert trial.specificity < 0.6  # most flipped payload bits cannot be detected


def test_repeated_trials_summary():
    report = evaluate_recovery(
        VITAL, FAULT_PRESETS["light"], plan_name="light", n_frames=60, n_trials=3, seed=5
    )
    summary = report.summary()
    assert summary["trials"] == 3 and summary["faults"] == "light"
    assert summary["sensitivity"]["mean"] == 1.0 and summary["sensitivity"]["std"] == 0.0
    matrix = summary["confusion_matrix"]
    assert sum(map(sum, matrix)) == summary["intact_packets"] + summary["damaged_packets"]
    assert len(summary["per_trial"]) == 3
    with pytest.raises(ValueError):
        evaluate_recovery(VITAL, FAULT_PRESETS["light"], n_trials=0)


def test_estimator_accuracy():
    scene = SyntheticScene(platform="16xx", seed=5)
    accuracy = evaluate_estimators(scene, 80)
    sensor = scene.radar()
    assert accuracy.snr_samples > 100 and accuracy.azimuth_samples > 50
    assert accuracy.snr_bias_db == pytest.approx(2.5, abs=1.0)  # log-domain averaging bias
    assert accuracy.azimuth_rmse_deg < 0.5
    assert accuracy.range_rmse_m < 0.005
    assert accuracy.velocity_rmse_mps < sensor.doppler_resolution_mps / 2
    corrected = evaluate_estimators(scene, 80, DecoderConfig(snr_log_bias_correction=True))
    assert abs(corrected.snr_bias_db) < 0.6
    assert set(accuracy.as_dict()) >= {"snr_bias_db", "azimuth_rmse_deg"}
    with pytest.raises(ValueError):
        evaluate_estimators(VITAL, 5)
