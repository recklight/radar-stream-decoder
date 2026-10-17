"""Radar configuration parsing and the derived FMCW quantities, checked against hand calculation."""

from __future__ import annotations

from pathlib import Path

import pytest

from radar_stream_decoder.constants import Platform
from radar_stream_decoder.errors import ConfigError
from radar_stream_decoder.sensor_config import (
    PRESET_CLI_CONFIGS,
    FmcwConvention,
    load_cli_config,
    parse_cli_config,
    preset_sensor_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
C = 3e8

BASE = """\
% comment line
sensorStop
channelCfg 15 {tx} 0
profileCfg 0 77 7 6 60 0 0 40 1 256 5000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
frameCfg 0 1 16 0 50 1 0
sensorStart
"""


def test_xwr16xx_quantities_by_hand():
    config = preset_sensor_config("iwr1642_oob_2tx4rx")
    fs, slope, n_adc, f0 = 5000e3, 40e12, 256, 77e9
    tc = (7 + 60) * 1e-6
    assert config.platform == Platform.XWR16XX
    assert (config.num_rx, config.num_tx, config.num_virtual_antennas) == (4, 2, 8)
    assert (config.num_tx_azimuth, config.num_tx_elevation) == (2, 0)
    assert config.num_range_bins == 256 and config.chirps_per_frame == 32
    assert config.num_doppler_bins == 16
    assert config.range_resolution_m == pytest.approx(C / (2 * slope * n_adc / fs))
    assert config.range_bin_m == pytest.approx(C * fs / (2 * slope * 256))
    assert config.range_bin_m == pytest.approx(0.0732421875)
    assert config.max_range_m == pytest.approx(0.8 * C * fs / (2 * slope))
    assert config.max_range_m == pytest.approx(15.0)
    assert config.max_velocity_mps == pytest.approx(C / (4 * f0 * tc * 2))
    assert config.doppler_resolution_mps == pytest.approx(C / (2 * f0 * tc * 2 * 16))
    assert config.doppler_resolution_mps * 16 / 2 == pytest.approx(config.max_velocity_mps)
    assert config.frame_period_s == pytest.approx(0.05)
    assert config.wavelength_m == pytest.approx(C / f0)
    assert config.sampled_bandwidth_hz == pytest.approx(2.048e9)
    assert config.xyz_q_format == 8
    assert config.outputs is not None and config.outputs.subframe_index == -1


def test_xwr14xx_antenna_geometry():
    config = preset_sensor_config("iwr1443_oob_2tx4rx")
    assert config.tx_mask_in_frame == 0b101
    assert (config.num_tx_azimuth, config.num_tx_elevation) == (2, 0)
    assert config.num_virtual_antennas_azimuth == 8
    three = parse_cli_config(
        BASE.format(tx=7)
        .replace("chirpCfg 1 1 0 0 0 0 0 2", "chirpCfg 1 1 0 0 0 0 0 2\nchirpCfg 2 2 0 0 0 0 0 4")
        .replace("frameCfg 0 1", "frameCfg 0 2"),
        platform="14xx",
    )
    assert three.num_tx == 3
    assert (three.num_tx_azimuth, three.num_tx_elevation) == (2, 1)
    assert three.num_virtual_antennas == 12 and three.num_virtual_antennas_azimuth == 8
    assert three.num_doppler_bins == 16


def test_single_transmitter_is_the_azimuth_aperture():
    config = parse_cli_config(
        BASE.format(tx=3)
        .replace("chirpCfg 1 1 0 0 0 0 0 2\n", "")
        .replace("frameCfg 0 1", "frameCfg 0 0"),
        platform="14xx",
    )
    assert config.num_tx == 1 and config.num_tx_azimuth == 1 and config.num_tx_elevation == 0
    assert config.num_doppler_bins == 16


def test_vital_signs_preset_matches_its_range_bins():
    config = preset_sensor_config("iwr1443_vital_signs")
    assert config.range_bin_m == pytest.approx(1200 / 35840)  # 0.0335 m
    assert int(0.3 / config.range_bin_m) == 8 and int(1.0 / config.range_bin_m) == 29


@pytest.mark.parametrize("name", sorted(PRESET_CLI_CONFIGS))
def test_shipped_config_files_equal_the_presets(name):
    text = (CONFIGS / f"{name}.cfg").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text == PRESET_CLI_CONFIGS[name]
    assert load_cli_config(CONFIGS / f"{name}.cfg").summary() == parse_cli_config(text).summary()


def test_guimonitor_argument_count_depends_on_the_platform():
    six = BASE.format(tx=3) + "guiMonitor 1 1 0 0 0 1\n"
    seven = BASE.format(tx=3) + "guiMonitor -1 1 1 0 0 0 1\n"
    assert parse_cli_config(six, "14xx").outputs.range_profile
    assert parse_cli_config(seven, "16xx").outputs.subframe_index == -1
    assert parse_cli_config(six).outputs.points and parse_cli_config(seven).outputs.stats
    with pytest.raises(ConfigError, match="line 9"):
        parse_cli_config(seven, "14xx")
    with pytest.raises(ConfigError, match="takes 7 arguments"):
        parse_cli_config(six, "16xx")


def test_sdk3_points_flag():
    flags = parse_cli_config(BASE.format(tx=3) + "guiMonitor -1 1 0 0 0 0 0\n").outputs
    assert flags.points and flags.point_side_info
    flags = parse_cli_config(BASE.format(tx=3) + "guiMonitor -1 2 0 0 0 0 0\n").outputs
    assert flags.points and not flags.point_side_info


@pytest.mark.parametrize(
    ("bad_line", "fragment"),
    [
        ("profileCfg 0 77 7 6 60 0 0 40 1 256 5000 0 0", "takes 14 arguments"),
        ("profileCfg 0 77 7 6 60 0 0 forty 1 256 5000 0 0 30", "not a number"),
        ("profileCfg 0 77 7 6 60 0 0 40 1 256.5 5000 0 0 30", "must be an integer"),
        ("profileCfg 0 77 7 6 60 0 0 0 1 256 5000 0 0 30", "non-zero"),
        ("channelCfg 15 0 0", "1-3 transmitters"),
        ("channelCfg 0 3 0", "1-4 receivers"),
        ("chirpCfg 3 1 0 0 0 0 0 1", "precedes"),
        ("chirpCfg 0 0 5 0 0 0 0 1", "undefined profile 5"),
        ("chirpCfg 0 0 0 0 0 0 0 4", "does not enable"),
        ("frameCfg 0 1 0 0 50 1 0", "loop count"),
        ("frameCfg 0 1 16 0 0 1 0", "frame period"),
        ("dfeDataOutputMode 7", "1, 2 or 3"),
        ("compRangeBiasAndRxChanPhase 0.1 1", "real/imaginary pairs"),
    ],
)
def test_invalid_lines_name_their_line(bad_line, fragment):
    text = BASE.format(tx=3) + bad_line + "\n"
    with pytest.raises(ConfigError, match=fragment) as caught:
        parse_cli_config(text, "16xx")
    assert caught.value.line == 9


def test_extras_comments_and_compensation():
    text = BASE.format(tx=3) + "cfarCfg -1 0 2 8 4 3 0 15 1   % trailing comment\n"
    text += "compRangeBiasAndRxChanPhase 0.07 1 0 0 1\n"
    config = parse_cli_config(text, "16xx")
    assert ("cfarCfg", ("-1", "0", "2", "8", "4", "3", "0", "15", "1")) in config.extras
    assert ("sensorStop", ()) in config.extras
    assert config.range_bias_m == pytest.approx(0.07)
    assert config.rx_phase_compensation == (1 + 0j, 1j)


def test_missing_sections_are_named():
    partial = parse_cli_config("profileCfg 0 77 7 6 60 0 0 40 1 256 5000 0 0 30\n")
    with pytest.raises(ConfigError, match="channelCfg"):
        _ = partial.num_rx
    with pytest.raises(ConfigError, match="frameCfg"):
        _ = partial.frame_period_s
    summary = partial.summary()
    assert "range_bin_m" in summary and "num_rx" not in summary
    with pytest.raises(ConfigError, match="profileCfg"):
        _ = parse_cli_config("channelCfg 15 3 0\n").profile
    advanced = parse_cli_config(
        "dfeDataOutputMode 3\nprofileCfg 0 77 7 6 60 0 0 40 1 256 5000 0 0 30\n"
    )
    with pytest.raises(ConfigError, match="advanced-frame"):
        _ = advanced.chirps_per_frame


def test_uneven_chirp_split_is_rejected():
    text = BASE.format(tx=3).replace("frameCfg 0 1 16", "frameCfg 0 0 15")
    config = parse_cli_config(text.replace("chirpCfg 0 0 0 0 0 0 0 1", "chirpCfg 0 0 0 0 0 0 0 3"))
    assert config.num_tx == 2
    with pytest.raises(ConfigError, match="cannot be split evenly"):
        _ = config.num_doppler_bins


def test_profile_used_by_the_frame():
    text = (
        BASE.format(tx=3)
        .replace(
            "chirpCfg 0 0 0", "profileCfg 1 60 7 6 60 0 0 20 1 128 2500 0 0 30\nchirpCfg 0 0 1"
        )
        .replace("chirpCfg 1 1 0", "chirpCfg 1 1 1")
    )
    config = parse_cli_config(text)
    assert config.profile.profile_id == 1 and config.num_range_bins == 128


def test_load_errors_name_the_file(tmp_path):
    path = tmp_path / "broken.cfg"
    path.write_text("channelCfg 15\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"broken\.cfg: line 1"):
        load_cli_config(path)
    with pytest.raises(FileNotFoundError):
        load_cli_config(tmp_path / "missing.cfg")
    with pytest.raises(ConfigError, match="unknown preset"):
        preset_sensor_config("iwr9999")
    with pytest.raises(ConfigError):
        parse_cli_config("", platform="toaster")


#: The chirp and detection commands of ODS_6m_default.cfg, the 68xx 3D People Counting
#: lab in TI mmWave Industrial Toolbox 4.6.0 (labs/people_counting/68xx_3D_people_counting/
#: chirp_configs). The scene commands (boundary boxes, sensor position, gating and
#: allocation) describe one room and are left out; trackingCfg stays to show that
#: commands the parser does not interpret end up in ``extras``.
ODS_6M_DEFAULT = """\
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
lowPower 0 0
profileCfg 0 60.75 30.00 25.00 59.10 657930 0 54.71 1   96 2950.00 2 1 36
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2  96  0 55.00 1 0
dynamicRACfarCfg -1 4 4 2 2 8 12 4 8 5.00 8.00 0.40 1 1
staticRACfarCfg -1 6 2 2 2 8 8 6 4 8.00 15.00 0.30 0 0
dynamicRangeAngleCfg -1 0.75 0.0010 1 0
dynamic2DAngleCfg -1 1.5 0.0300 1 0 1 0.30 0.85 8.00
staticRangeAngleCfg -1 0 8 8
antGeometry0 0 0 -1 -1 -2 -2 -3 -3 -2 -2 -3 -3
antGeometry1 0 -1 -1 0 0 -1 -1 0 -2 -3 -3 -2
antPhaseRot 1 -1 -1 1 1 -1 -1 1 1 -1 -1 1
fovCfg -1 70.0 20.0
compRangeBiasAndRxChanPhase 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
trackingCfg 1 2 800 30 46 96 55
sensorStart
"""


class TestFmcwConventions:
    """Convention-dependent figures of a public TI configuration, computed independently."""

    F0, IDLE, ADC_START, RAMP_END = 60.75e9, 30e-6, 25e-6, 59.10e-6
    SLOPE, SAMPLES, RATE, TX, LOOPS = 54.71e12, 96, 2950e3, 3, 96

    @property
    def tc(self) -> float:
        return self.IDLE + self.RAMP_END

    @property
    def centre(self) -> float:
        # mmwdemo_rfparser.c (TI mmWave SDK 03.05.00.04): centerFreq = startFreq +
        # bandwidth/2 + adcStart*slope, bandwidth = slope*numAdcSamples/sampleRate.
        bandwidth = self.SLOPE * self.SAMPLES / self.RATE
        return self.F0 + bandwidth / 2 + self.ADC_START * self.SLOPE

    def test_convention_free_figures(self):
        config = parse_cli_config(ODS_6M_DEFAULT, "68xx")
        assert (config.num_tx, config.num_rx, config.chirps_per_frame) == (3, 4, 288)
        assert (config.num_range_bins, config.num_doppler_bins) == (128, 128)
        assert round(config.range_resolution_m, 5) == 0.08425  # c Fs / (2 S N)
        assert config.range_resolution_m == pytest.approx(C * self.RATE / (2 * self.SLOPE * 96))
        assert config.if_range_limit_m == pytest.approx(8.0881, abs=1e-4)
        assert config.chirp_period_s == pytest.approx(89.1e-6)
        assert config.frame_period_s == pytest.approx(0.055)
        assert ("trackingCfg", ("1", "2", "800", "30", "46", "96", "55")) in config.extras

    def test_default_is_the_demo_visualizer_convention(self):
        config = parse_cli_config(ODS_6M_DEFAULT, "68xx")
        assert config.reference_frequency_hz == self.F0
        assert config.max_range_m == pytest.approx(0.8 * C * self.RATE / (2 * self.SLOPE))
        assert config.max_velocity_mps == pytest.approx(C / (4 * self.F0 * self.tc * self.TX))
        assert config.max_velocity_mps == pytest.approx(4.6187, abs=1e-4)
        bin_mps = C / (2 * self.F0 * self.tc * self.TX * 128)
        assert config.doppler_bin_mps == pytest.approx(bin_mps)
        assert config.doppler_resolution_mps == pytest.approx(bin_mps)
        summary = config.summary()
        assert summary["convention_wavelength_reference"] == "start"
        assert summary["convention_max_range_factor"] == 0.8
        assert summary["convention_doppler_zero_padding"] is True

    def test_firmware_convention(self):
        convention = FmcwConvention(
            wavelength_reference="center", max_range_factor=1.0, doppler_zero_padding=False
        )
        config = parse_cli_config(ODS_6M_DEFAULT, "68xx", convention)
        assert config.center_frequency_hz == pytest.approx(self.centre)
        assert config.center_frequency_hz == pytest.approx(63.0077e9, rel=1e-5)
        assert config.max_range_m == pytest.approx(8.0881, abs=1e-4)
        assert config.max_velocity_mps == pytest.approx(C / (4 * self.centre * self.tc * self.TX))
        assert config.max_velocity_mps * 3.6 == pytest.approx(16.03, abs=0.01)  # km/h
        # dopplerResolution of the rfparser: c / (2 numChirpsPerFrame centerFreq Tc).
        assert config.doppler_resolution_mps == pytest.approx(C / (2 * 288 * self.centre * self.tc))
        # Bin indices still convert with the FFT bin spacing.
        assert config.doppler_bin_mps == pytest.approx(
            C / (2 * self.centre * self.tc * self.TX * 128)
        )
        assert config.wavelength_m == pytest.approx(C / self.centre)

    def test_conventions_are_validated(self):
        with pytest.raises(ConfigError, match="wavelength_reference"):
            FmcwConvention(wavelength_reference="middle")
        with pytest.raises(ConfigError, match="max_range_factor"):
            FmcwConvention(max_range_factor=1.5)
        config = preset_sensor_config("iwr1642_oob_2tx4rx")
        centred = config.with_convention(FmcwConvention(wavelength_reference="center"))
        assert centred.max_velocity_mps < config.max_velocity_mps
        assert centred.range_bin_m == config.range_bin_m
