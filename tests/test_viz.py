"""The PySide6 viewer, headless (QT_QPA_PLATFORM=offscreen); skipped without the gui extra."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("matplotlib")

from radar_stream_decoder import viz  # noqa: E402  (after the platform is chosen)
from radar_stream_decoder.port_emulator import EmulatedPortFactory  # noqa: E402
from radar_stream_decoder.sources import SerialByteSource  # noqa: E402
from radar_stream_decoder.synthetic import SyntheticScene, generate_packets  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    created = viz.RadarViewerWindow(interactive=False)
    created.show()
    yield created
    created.close()


def spin(app, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def test_widgets_speak_traditional_chinese(window):
    assert window.windowTitle() == "雷達資料流檢視器"
    sources = [window.source_combo.itemText(i) for i in range(window.source_combo.count())]
    assert sources == ["合成示範", "離線擷取檔（.bin）", "序列埠（即時）"]
    assert [b.text() for b in (window.play_button, window.pause_button, window.step_button)] == [
        "播放",
        "暫停",
        "單步",
    ]
    assert window.export_button.text() == "匯出目前 frame（CSV）"
    assert window.info_panel.title() == "資訊"
    assert not window.play_button.isEnabled() and not window.export_button.isEnabled()
    assert window.canvas.ax_points.get_title() == "點雲（俯視）"


def test_demo_decodes_in_the_background_and_plays(window, app):
    window.load_demo(n_frames=60)
    assert window.wait_until_loaded(60)
    assert len(window.frames) >= 55  # light faults reject a packet now and then
    assert window.current_index == 0
    assert window.frame_label.text() == str(window.frames[0].frame_number)
    assert int(window.loaded_label.text()) == len(window.frames)
    window.step()
    assert window.current_index == 1
    window.frame_slider.setValue(10)
    assert window.current_index == 10
    window.play()
    assert window.timer.isActive() and window.pause_button.isEnabled()
    spin(app, 0.4)
    window.pause()
    assert window.current_index > 10 and not window.timer.isActive()


def test_rendering_keeps_up_with_20_fps(window):
    window.load_demo(n_frames=150)
    assert window.wait_until_loaded(60)
    window.render_times.clear()
    for index in range(len(window.frames)):
        window.show_frame(index)
    median_ms = float(np.median(window.render_times)) * 1e3
    assert median_ms < 50.0, f"median frame takes {median_ms:.1f} ms"


def test_colour_mode_and_export(window, tmp_path):
    window.load_demo(n_frames=20)
    assert window.wait_until_loaded(60)
    window.color_combo.setCurrentIndex(1)
    assert window.color_mode.value == "doppler"
    assert window.canvas.colorbar.ax.get_ylabel() == "徑向速度 (m/s)"
    target = window.export_current_frame(tmp_path / "frame.csv")
    lines = Path(target).read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("frame,") and len(lines) == window.frames[0].num_points + 1


def test_capture_file_without_heatmaps(window, tmp_path):
    path = tmp_path / "vital.bin"
    path.write_bytes(b"".join(generate_packets(SyntheticScene(schema="vital-signs"), 25)))
    window.load_capture(path)
    assert window.wait_until_loaded(30)
    assert len(window.frames) == 25
    assert window.canvas.heatmap_note.get_visible()
    assert window.points_label.text() == "0"
    window.load_capture(tmp_path / "absent.bin")
    assert "找不到擷取檔" in window.last_error


def test_failing_source_is_reported(window):
    class Failing:
        description = "failing"

        def chunks(self):
            raise OSError("port vanished")
            yield b""  # pragma: no cover

        def close(self):
            return None

    window._start(Failing, live=False)
    assert window.wait_until_loaded(10)
    assert window.last_error is not None and "port vanished" in window.last_error


def test_live_serial_source_with_emulated_ports(window, app, monkeypatch):
    data = b"".join(generate_packets(SyntheticScene(schema="vital-signs", seed=2), 40))
    factory = EmulatedPortFactory(data)

    def replaying(data_port, **kwargs):
        return SerialByteSource(data_port, serial_factory=factory, **kwargs)

    monkeypatch.setattr(viz, "SerialByteSource", replaying)
    window.connect_serial("DATA")
    deadline = time.monotonic() + 10
    while len(window.frames) < 40 and time.monotonic() < deadline:
        spin(app, 0.05)
    assert len(window.frames) == 40 and window.live
    spin(app, 0.2)
    assert window.current_index == len(window.frames) - 1  # live view follows the newest frame
    window.stop_loading()
    assert not window.is_loading


def test_live_history_is_bounded_but_counters_are_cumulative(app, monkeypatch):
    packets = list(generate_packets(SyntheticScene(schema="vital-signs", seed=3), 40))
    factory = EmulatedPortFactory(b"".join(packets))
    monkeypatch.setattr(
        viz,
        "SerialByteSource",
        lambda data_port, **kwargs: SerialByteSource(data_port, serial_factory=factory, **kwargs),
    )
    window = viz.RadarViewerWindow(
        viz.DecoderConfig(), interactive=False, viewer=viz.ViewerConfig(live_history_frames=15)
    )
    try:
        window.show()
        window.connect_serial("DATA")
        deadline = time.monotonic() + 10
        while window.received_frames < 40 and time.monotonic() < deadline:
            spin(app, 0.05)
        spin(app, 0.2)
        assert window.received_frames == 40 and len(window.frames) == 15
        assert [f.frame_number for f in window.frames] == list(range(26, 41))
        assert window.loaded_label.text() == "40"
        assert window.current_index == 14 and window.frame_label.text() == "40"
    finally:
        window.close()


def test_serial_configuration_file_supplies_the_units(window, tmp_path):
    config = Path(__file__).resolve().parents[1] / "configs" / "iwr1642_oob_2tx4rx.cfg"
    text, sensor = window.read_sensor_config(config)
    assert text.startswith("%") and "sensorStart" in text
    assert sensor.num_tx == 2 and round(sensor.range_bin_m, 4) == 0.0732
    assert window.read_sensor_config(tmp_path / "absent.cfg") is None
    assert "雷達組態檔" in window.last_error
    broken = tmp_path / "broken.cfg"
    broken.write_text("profileCfg 0 77\n", encoding="utf-8")
    assert window.read_sensor_config(broken) is None and "profileCfg" in window.last_error


def test_connect_dialog_without_a_cli_port_uses_the_cfg_for_units_only(window, app, monkeypatch):
    config = Path(__file__).resolve().parents[1] / "configs" / "iwr1642_oob_2tx4rx.cfg"
    factory = EmulatedPortFactory(b"".join(generate_packets(SyntheticScene(seed=2), 10)))
    opened: list[dict] = []

    def replaying(data_port, **kwargs):
        opened.append(kwargs)
        return SerialByteSource(data_port, serial_factory=factory, **kwargs)

    class Answered:
        """The connect dialog, filled in: data port and .cfg, no CLI port."""

        def __init__(self, parent):
            self.data_port = QtWidgets.QLineEdit("DATA")
            self.cli_port = QtWidgets.QLineEdit("")
            self.config_path = QtWidgets.QLineEdit(str(config))

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(viz, "SerialByteSource", replaying)
    monkeypatch.setattr(viz, "SerialConnectDialog", Answered)
    window.source_combo.setCurrentIndex(2)
    window._open_selected_source()
    deadline = time.monotonic() + 10
    while len(window.frames) < 10 and time.monotonic() < deadline:
        spin(app, 0.05)
    window.stop_loading()
    assert window.last_error is None
    assert len(window.frames) == 10
    assert opened[0]["cli_port"] is None and opened[0]["cli_text"] is None
    assert window.sensor.num_tx == 2


def test_tracking_demo_shows_points_and_targets(app):
    window = viz.RadarViewerWindow(
        viz.DecoderConfig(schema="people-counting-3d"), interactive=False
    )
    try:
        window.show()
        window.load_demo(n_frames=12)
        assert window.wait_until_loaded(60)
        assert window.frames and all(f.schema.value == "people-counting-3d" for f in window.frames)
        window.show_frame(3)
        offsets = window.canvas.target_scatter.get_offsets()
        assert len(offsets) == window.frames[3].num_targets > 0
        np.testing.assert_allclose(offsets, window.frames[3].targets.position_m[:, :2])
        assert window.targets_label.text() == str(window.frames[3].num_targets)
        assert len(window.canvas.scatter.get_offsets()) == window.frames[3].num_points
        assert "people-counting-3d" in window.source_label.text()
    finally:
        window.close()


def test_frames_without_targets_show_a_dash(window):
    window.load_demo(n_frames=5)
    assert window.wait_until_loaded(60)
    assert window.targets_label.text() == "—"
    assert len(window.canvas.target_scatter.get_offsets()) == 0


def test_cjk_font_lookup():
    name = viz.configure_cjk_font()
    assert name is None or isinstance(name, str)
