"""Decoder and viewer settings: validation and JSON round trips."""

from __future__ import annotations

from pathlib import Path

import pytest

from radar_stream_decoder.config import DecoderConfig, NoiseEstimator, ViewerConfig
from radar_stream_decoder.errors import ConfigError
from radar_stream_decoder.schema import PacketSchema

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_and_coercion():
    config = DecoderConfig(schema="OOB_SDK3", platform="IWR1642", snr_estimator="so")
    assert config.schema is PacketSchema.OOB_SDK3
    assert config.platform == 0xA1642
    assert config.snr_estimator is NoiseEstimator.SMALLEST_OF
    assert DecoderConfig().schema is None and DecoderConfig(schema="auto").schema is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"schema": "sdk9"},
        {"platform": "toaster"},
        {"snr_estimator": "median"},
        {"dedupe_window": 0},
        {"max_packet_bytes": 32},
        {"clock_hz": -1.0},
        {"num_range_bins": 0},
        {"snr_guard_cells": -1},
        {"log2_fraction_bits": 20},
        {"angle_fft_size": 63},
    ],
)
def test_invalid_config_values(kwargs):
    with pytest.raises(ConfigError):
        DecoderConfig(**kwargs)


def test_json_round_trip(tmp_path):
    config = DecoderConfig(schema="vital-signs", strict=True, clock_hz=200e6, num_range_bins=256)
    path = config.to_json(tmp_path / "decoder.json")
    assert DecoderConfig.from_json(path) == config
    assert path.read_bytes().count(b"\r\n") == 0


def test_from_dict_names_unknown_keys():
    with pytest.raises(ConfigError, match="schmea") as caught:
        DecoderConfig.from_dict({"schmea": "auto"})
    assert "valid settings" in str(caught.value)


def test_bad_json_files(tmp_path):
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON object"):
        DecoderConfig.from_json(tmp_path / "list.json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        DecoderConfig.from_json(tmp_path / "broken.json")
    with pytest.raises(FileNotFoundError):
        DecoderConfig.from_json(tmp_path / "absent.json")


def test_shipped_decoder_configs_load():
    assert DecoderConfig.from_json(CONFIGS / "decoder_default.json") == DecoderConfig()
    strict = DecoderConfig.from_json(CONFIGS / "decoder_strict.json")
    assert strict.strict and strict.snr_log_bias_correction


def test_replace_validates_again():
    config = DecoderConfig().replace(strict=True)
    assert config.strict
    with pytest.raises(ConfigError):
        DecoderConfig().replace(chunk_size=0)


def test_viewer_config():
    viewer = ViewerConfig(fps=25)
    assert viewer.frame_interval_ms == 40
    for bad in (
        {"fps": 0},
        {"max_points": 0},
        {"point_size_range": (5.0, 1.0)},
        {"snr_limits_db": (5.0, 5.0)},
    ):
        with pytest.raises(ConfigError):
            ViewerConfig(**bad)
