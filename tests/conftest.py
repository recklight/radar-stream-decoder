"""Shared fixtures. Every fixture is synthesised in code; no recorded data is read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from radar_stream_decoder.sensor_config import SensorConfig, preset_sensor_config
from radar_stream_decoder.synthetic import SyntheticScene, generate_packets


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240607)


@pytest.fixture
def sensor_16xx() -> SensorConfig:
    return preset_sensor_config("iwr1642_oob_2tx4rx")


@pytest.fixture
def sensor_14xx() -> SensorConfig:
    return preset_sensor_config("iwr1443_oob_2tx4rx")


@pytest.fixture
def scene_16xx() -> SyntheticScene:
    return SyntheticScene(platform="16xx", seed=3)


@pytest.fixture
def scene_14xx() -> SyntheticScene:
    return SyntheticScene(platform="14xx", seed=4)


@pytest.fixture
def scene_sdk3() -> SyntheticScene:
    return SyntheticScene(schema="oob-sdk3", platform="16xx", seed=5)


@pytest.fixture
def scene_vital() -> SyntheticScene:
    return SyntheticScene(schema="vital-signs", seed=6)


@pytest.fixture
def packets_16xx(scene_16xx: SyntheticScene) -> list[bytes]:
    return list(generate_packets(scene_16xx, 12))


@pytest.fixture
def packets_vital(scene_vital: SyntheticScene) -> list[bytes]:
    return list(generate_packets(scene_vital, 30))


@pytest.fixture
def capture_file(tmp_path: Path, packets_16xx: list[bytes]) -> Path:
    """A clean capture of twelve xWR16xx packets with junk in front."""
    path = tmp_path / "capture.bin"
    path.write_bytes(b"\x00\xff junk " + b"".join(packets_16xx))
    return path
