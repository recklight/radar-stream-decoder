"""PySide6 live viewer.

The point cloud seen from above fills the left of the window, coloured by SNR
or radial velocity, with the marker size following the reflection strength.
When the firmware reports the on-sensor tracker's targets, they are drawn as
crosses. The range profile sits top right and the Doppler-centred
range-Doppler heatmap bottom right. The toolbar picks the data source
(synthetic demo, capture file, live serial port), drives playback (play,
pause, single step, a frame slider) and exports the current frame; the
information panel shows the frame number, the point count and the
parse-error counters.

The viewer itself only displays. Decoding runs in a :class:`FrameLoader`
on a ``QThread`` through the same
:class:`~radar_stream_decoder.sources.FrameStream` the command line uses,
what to draw comes from :mod:`radar_stream_decoder.presentation`, and export
goes through :mod:`radar_stream_decoder.export`.

The artists are created once, updated in place (``set_offsets``,
``set_array``, ``set_sizes``, ``set_data``) and blitted over a cached
background. The whole figure is redrawn only when an axis range, a label or a
colour scale has to change; axis ranges only ever grow, so that happens a
handful of times per capture.

The interface text is Traditional Chinese.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

try:
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover - exercised only without the gui extra
    raise ImportError(
        'the viewer needs PySide6 and matplotlib: run pip install -e ".[gui]" in the project folder'
    ) from exc

import numpy as np

from .assembler import AssemblerStats
from .config import DecoderConfig, ViewerConfig
from .errors import ConfigError
from .export import write_points_csv
from .frame import RadarFrame
from .presentation import ColorMode, FrameDisplay, prepare_frame_display
from .sensor_config import SensorConfig, parse_cli_config
from .sources import ByteSource, FileByteSource, FrameStream, SerialByteSource, SyntheticByteSource
from .synthetic import FAULT_PRESETS, SyntheticScene

_LOGGER = logging.getLogger(__name__)

#: Fonts with Traditional Chinese glyphs, in order of preference.
_CJK_FONTS = (
    "Microsoft JhengHei",
    "Microsoft JhengHei UI",
    "PingFang TC",
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "Source Han Sans TC",
    "Heiti TC",
    "Noto Sans CJK JP",
    "Microsoft YaHei",
    "WenQuanYi Zen Hei",
)

SOURCE_LABELS = {
    "demo": "合成示範",
    "file": "離線擷取檔（.bin）",
    "serial": "序列埠（即時）",
}
COLOR_LABELS = {ColorMode.SNR: "依 SNR 著色", ColorMode.DOPPLER: "依都卜勒速度著色"}

_fonts_configured = False


def configure_cjk_font() -> str | None:
    """Make matplotlib render Traditional Chinese labels; returns the font used.

    When no suitable font is installed the labels fall back to the default
    font and the missing-glyph warnings are silenced.
    """
    global _fonts_configured
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    chosen = [name for name in _CJK_FONTS if name in available]
    if not _fonts_configured:
        _fonts_configured = True
        matplotlib.rcParams["axes.unicode_minus"] = False
        if chosen:
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [
                *chosen,
                *matplotlib.rcParams["font.sans-serif"],
            ]
        else:
            warnings.filterwarnings("ignore", message=r"Glyph \d+ .*missing from")
    return chosen[0] if chosen else None


class FrameLoader(QtCore.QObject):
    """Decodes a byte source on a worker thread and hands frames over in batches.

    Signals
    -------
    frames_ready(list)
        A batch of decoded frames.
    finished(object)
        The final :class:`~radar_stream_decoder.assembler.AssemblerStats`.
    failed(str)
        An error message; nothing more will be emitted.
    """

    frames_ready = QtCore.Signal(object)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        source_factory: Callable[[], ByteSource],
        decoder: DecoderConfig,
        sensor: SensorConfig | None,
        *,
        live: bool = False,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._factory = source_factory
        self._decoder = decoder
        self._sensor = sensor
        self._live = live
        # A live source may pause for any length of time, so its frames are
        # handed over one by one rather than held back for a full batch.
        self._batch_size = 1 if live else max(1, batch_size)
        self._stop = threading.Event()
        self._source: ByteSource | None = None

    @QtCore.Slot()
    def run(self) -> None:
        """Decode until the source ends or :meth:`stop` is called."""
        try:
            self._source = self._factory()
            stream = FrameStream(self._source, self._decoder, self._sensor)
            batch: list[RadarFrame] = []
            for frame in stream:
                if self._stop.is_set():
                    break
                batch.append(frame)
                if len(batch) >= self._batch_size:
                    self.frames_ready.emit(batch)
                    batch = []
            if batch:
                self.frames_ready.emit(batch)
            self.finished.emit(stream.assembler_stats)
        except Exception as exc:  # shown to the user instead of killing the thread
            _LOGGER.debug("frame loader failed", exc_info=True)
            self.failed.emit(str(exc))

    def stop(self) -> None:
        """Ask the loader to finish early."""
        self._stop.set()
        if self._source is not None:
            self._source.close()


class RadarPlotCanvas(FigureCanvasQTAgg):
    """The three plots of the viewer, updated in place and redrawn by blitting."""

    def __init__(self, viewer: ViewerConfig) -> None:
        configure_cjk_font()
        figure = Figure(figsize=(11.0, 6.5), layout="constrained")
        super().__init__(figure)
        self.viewer = viewer
        grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
        self.ax_points = figure.add_subplot(grid[:, 0])
        self.ax_profile = figure.add_subplot(grid[0, 1])
        self.ax_heatmap = figure.add_subplot(grid[1, 1])

        self.ax_points.set_title("點雲（俯視）")
        self.ax_points.set_xlabel("x (m)")
        self.ax_points.set_ylabel("y (m)")
        self.ax_points.set_aspect("equal", adjustable="box")
        self.ax_points.grid(True, alpha=0.3)
        self.scatter = self.ax_points.scatter(
            [],
            [],
            c=[],
            s=[],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            edgecolors="none",
            animated=True,
        )
        self.colorbar = figure.colorbar(self.scatter, ax=self.ax_points, shrink=0.85)
        self.target_scatter = self.ax_points.scatter(
            [],
            [],
            marker="X",
            s=150,
            c="crimson",
            edgecolors="white",
            linewidths=0.8,
            animated=True,
            zorder=3,
            label="追蹤目標",
        )

        self.ax_profile.set_title("距離剖面")
        self.ax_profile.grid(True, alpha=0.3)
        (self.profile_line,) = self.ax_profile.plot([], [], color="tab:blue", lw=1.2, animated=True)

        self.ax_heatmap.set_title("距離-都卜勒熱圖")
        self.heatmap = self.ax_heatmap.imshow(
            np.zeros((2, 2)), aspect="auto", origin="lower", cmap="magma", animated=True
        )
        self.heatmap_bar = figure.colorbar(self.heatmap, ax=self.ax_heatmap)
        self.heatmap_note = self.ax_heatmap.text(
            0.5,
            0.5,
            "此 frame 沒有熱圖",
            ha="center",
            va="center",
            transform=self.ax_heatmap.transAxes,
            color="0.4",
            visible=False,
            animated=True,
        )

        # Everything that changes per frame is animated: drawn over a cached
        # background, so toggling or updating it never needs a full redraw.
        self._animated = (
            self.scatter,
            self.target_scatter,
            self.profile_line,
            self.heatmap,
            self.heatmap_note,
        )
        self._background: Any = None
        self._limits: dict[str, tuple[float, float]] = {}
        self._quanta: dict[str, float] = {}
        self._labels: dict[str, str] = {}
        self._cmap = "viridis"
        self.mpl_connect("draw_event", self._on_draw)

    # -- blitting ------------------------------------------------------------------

    def _on_draw(self, _event: object) -> None:
        self._background = self.copy_from_bbox(self.figure.bbox)
        for artist in self._animated:
            if artist.get_visible():
                self.figure.draw_artist(artist)

    def reset_scales(self) -> None:
        """Forget the axis ranges (for a new data source)."""
        self._limits.clear()
        self._labels.clear()
        self._quanta.clear()

    def _grow(self, key: str, low: float, high: float, *, exact: bool = False) -> bool:
        """Widen a remembered range to cover ``[low, high]``; ``True`` when it changed.

        Ranges grow in coarse steps (a quarter of the first span) with one step
        of headroom, so that a capture triggers only a few full redraws.
        ``exact`` keeps the requested range as it is (fixed colour scales).
        """
        if not (math.isfinite(low) and math.isfinite(high)):
            return False
        if high <= low:
            high = low + 1.0
        current = self._limits.get(key)
        if current is not None and current[0] <= low and current[1] >= high:
            return False
        if current is not None:
            low, high = min(low, current[0]), max(high, current[1])
        if not exact:
            quantum = self._quanta.setdefault(key, max(0.25 * (high - low), 1e-6))
            low = math.floor(low / quantum) * quantum - quantum
            high = math.ceil(high / quantum) * quantum + quantum
        self._limits[key] = (low, high)
        return True

    def _relabel(self, key: str, text: str) -> bool:
        if self._labels.get(key) == text:
            return False
        self._labels[key] = text
        return True

    # -- updating ------------------------------------------------------------------

    def _apply(self, display: FrameDisplay) -> bool:
        """Update the artists; return ``True`` when the static parts must be redrawn."""
        redraw = False
        scatter = display.scatter
        self.scatter.set_offsets(scatter.xy if len(scatter.xy) else np.zeros((0, 2)))
        self.scatter.set_array(scatter.color)
        self.scatter.set_sizes(scatter.size)
        if scatter.colormap != self._cmap:
            self._cmap = scatter.colormap
            self.scatter.set_cmap(scatter.colormap)
            redraw = True
        if self._relabel("color", scatter.color_label):
            self.colorbar.set_label(scatter.color_label)
            self._limits.pop("color", None)
            self._quanta.pop("color", None)
            redraw = True
        if self._grow("color", *scatter.color_limits, exact=scatter.fixed_limits):
            self.scatter.set_clim(*self._limits["color"])
            redraw = True
        markers = display.targets
        target_xy = markers.xy if markers is not None and len(markers.xy) else np.zeros((0, 2))
        self.target_scatter.set_offsets(target_xy)
        shown = np.vstack((scatter.xy, target_xy)) if len(scatter.xy) else target_xy
        if len(shown):
            reach = float(np.max(np.abs(shown[:, 0]), initial=1.0))
            depth = float(np.max(shown[:, 1], initial=1.0))
            if self._grow("points_x", -reach, reach) | self._grow("points_y", 0.0, depth):
                self.ax_points.set_xlim(*self._limits["points_x"])
                self.ax_points.set_ylim(0.0, self._limits["points_y"][1])
                redraw = True

        curve = display.profile
        if curve is None:
            self.profile_line.set_data([], [])
        else:
            self.profile_line.set_data(curve.x, curve.y)
            if self._relabel("profile_x", curve.x_label) | self._relabel(
                "profile_y", curve.y_label
            ):
                self.ax_profile.set_xlabel(curve.x_label)
                self.ax_profile.set_ylabel(curve.y_label)
                for key in ("profile_x", "profile_y"):
                    self._limits.pop(key, None)
                    self._quanta.pop(key, None)
                redraw = True
            finite = curve.y[np.isfinite(curve.y)]
            if finite.size:
                grew_x = self._grow("profile_x", float(curve.x[0]), float(curve.x[-1]), exact=True)
                grew_y = self._grow("profile_y", float(finite.min()), float(finite.max()))
                if grew_x or grew_y:
                    self.ax_profile.set_xlim(*self._limits["profile_x"])
                    self.ax_profile.set_ylim(*self._limits["profile_y"])
                    redraw = True

        heat = display.heatmap
        has_heat = heat is not None
        self.heatmap.set_visible(has_heat)
        self.heatmap_note.set_visible(not has_heat)
        if heat is not None:
            self.heatmap.set_data(heat.image)
            if tuple(self.heatmap.get_extent()) != tuple(heat.extent):
                self.heatmap.set_extent(heat.extent)
                self.ax_heatmap.set_xlim(heat.extent[0], heat.extent[1])
                self.ax_heatmap.set_ylim(heat.extent[2], heat.extent[3])
                redraw = True
            if self._relabel("heat_x", heat.x_label) | self._relabel("heat_y", heat.y_label):
                self.ax_heatmap.set_xlabel(heat.x_label)
                self.ax_heatmap.set_ylabel(heat.y_label)
                self.heatmap_bar.set_label(heat.color_label)
                self._limits.pop("heat", None)
                self._quanta.pop("heat", None)
                redraw = True
            finite = heat.image[np.isfinite(heat.image)]
            if finite.size and self._grow(
                "heat", float(np.percentile(finite, 5)), float(finite.max())
            ):
                self.heatmap.set_clim(*self._limits["heat"])
                redraw = True
        return redraw

    def show_frame(self, display: FrameDisplay) -> float:
        """Draw one frame; returns the time it took in seconds."""
        started = time.perf_counter()
        if self._apply(display) or self._background is None:
            self.draw()
        else:
            self.restore_region(self._background)
            for artist in self._animated:
                if artist.get_visible():
                    self.figure.draw_artist(artist)
            self.blit(self.figure.bbox)
        return time.perf_counter() - started


class SerialConnectDialog(QtWidgets.QDialog):
    """Asks for the data port, the optional CLI port and configuration file."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("連線序列埠")
        self.data_port = QtWidgets.QLineEdit()
        self.data_port.setPlaceholderText("例如 COM4 或 /dev/ttyACM1")
        self.cli_port = QtWidgets.QLineEdit()
        self.cli_port.setPlaceholderText("選填；要先送出雷達組態時才需要")
        self.config_path = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("瀏覽…")
        browse.clicked.connect(self._browse)
        config_row = QtWidgets.QHBoxLayout()
        config_row.addWidget(self.config_path)
        config_row.addWidget(browse)
        form = QtWidgets.QFormLayout()
        form.addRow("資料埠（921600 baud）", self.data_port)
        form.addRow("CLI 埠（115200 baud）", self.cli_port)
        form.addRow("雷達組態檔（.cfg）", config_row)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("連線")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇雷達組態檔", "", "雷達組態 (*.cfg);;所有檔案 (*)"
        )
        if path:
            self.config_path.setText(path)


class RadarViewerWindow(QtWidgets.QMainWindow):
    """Main window: source selection, playback controls, plots and frame information.

    Parameters
    ----------
    decoder : DecoderConfig, optional
        Decoder settings for every source.
    sensor : SensorConfig, optional
        Radar configuration for capture files and serial ports (the demo
        brings its own).
    viewer : ViewerConfig, optional
        Playback and display settings.
    interactive : bool, optional
        Show error dialogs; tests pass ``False`` and read :attr:`last_error`.
    """

    def __init__(
        self,
        decoder: DecoderConfig | None = None,
        sensor: SensorConfig | None = None,
        viewer: ViewerConfig | None = None,
        *,
        interactive: bool = True,
    ) -> None:
        super().__init__()
        self.decoder = decoder or DecoderConfig()
        self.base_sensor = sensor
        self.sensor = sensor
        self.viewer = viewer or ViewerConfig()
        self.interactive = interactive
        self.frames: list[RadarFrame] = []
        self.current_index = -1
        self.color_mode = ColorMode.SNR
        self.last_error: str | None = None
        self.last_render_s = 0.0
        self.render_times: list[float] = []
        self.rejected_packets = 0
        self.received_frames = 0
        self.damaged_frames = 0
        self.live = False
        self._thread: QtCore.QThread | None = None
        self._loader: FrameLoader | None = None
        self._loading = False

        self.setWindowTitle("雷達資料流檢視器")
        self.resize(1400, 820)
        self.canvas = RadarPlotCanvas(self.viewer)
        self._build_toolbar()
        self._build_info_panel()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.info_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("就緒")

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.viewer.frame_interval_ms)
        self.timer.timeout.connect(self._advance)
        self._update_controls()

    # -- construction ----------------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = QtWidgets.QToolBar("控制列", self)
        bar.setMovable(False)
        self.addToolBar(bar)
        bar.addWidget(QtWidgets.QLabel("資料來源："))
        self.source_combo = QtWidgets.QComboBox()
        for key, label in SOURCE_LABELS.items():
            self.source_combo.addItem(label, key)
        bar.addWidget(self.source_combo)
        self.open_button = QtWidgets.QPushButton("開啟…")
        self.open_button.setToolTip("依所選的資料來源開啟擷取檔、連線序列埠或重新產生示範資料")
        self.open_button.clicked.connect(self._open_selected_source)
        bar.addWidget(self.open_button)
        bar.addSeparator()
        self.play_button = QtWidgets.QPushButton("播放")
        self.play_button.clicked.connect(self.play)
        self.pause_button = QtWidgets.QPushButton("暫停")
        self.pause_button.clicked.connect(self.pause)
        self.step_button = QtWidgets.QPushButton("單步")
        self.step_button.setToolTip("前進一個 frame")
        self.step_button.clicked.connect(self.step)
        for button in (self.play_button, self.pause_button, self.step_button):
            bar.addWidget(button)
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setMinimumWidth(260)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setToolTip("frame 進度")
        self.frame_slider.valueChanged.connect(self._slider_moved)
        bar.addWidget(self.frame_slider)
        bar.addSeparator()
        self.color_combo = QtWidgets.QComboBox()
        for mode, label in COLOR_LABELS.items():
            self.color_combo.addItem(label, mode.value)
        self.color_combo.currentIndexChanged.connect(self._color_mode_changed)
        bar.addWidget(self.color_combo)
        bar.addSeparator()
        self.export_button = QtWidgets.QPushButton("匯出目前 frame（CSV）")
        self.export_button.clicked.connect(lambda: self.export_current_frame())
        bar.addWidget(self.export_button)

    def _build_info_panel(self) -> None:
        self.info_panel = QtWidgets.QGroupBox("資訊")
        form = QtWidgets.QFormLayout(self.info_panel)
        self.frame_label = QtWidgets.QLabel("—")
        self.points_label = QtWidgets.QLabel("—")
        self.targets_label = QtWidgets.QLabel("—")
        self.frame_issues_label = QtWidgets.QLabel("—")
        self.errors_label = QtWidgets.QLabel("0")
        self.loaded_label = QtWidgets.QLabel("0")
        self.source_label = QtWidgets.QLabel("—")
        self.source_label.setWordWrap(True)
        self.render_label = QtWidgets.QLabel("—")
        form.addRow("Frame 編號", self.frame_label)
        form.addRow("偵測點數", self.points_label)
        form.addRow("追蹤目標數", self.targets_label)
        form.addRow("本 frame 問題", self.frame_issues_label)
        form.addRow("解析錯誤（累計）", self.errors_label)
        form.addRow("已載入 frame", self.loaded_label)
        form.addRow("資料來源", self.source_label)
        form.addRow("繪圖時間", self.render_label)

    # -- sources -------------------------------------------------------------------

    def load_demo(self, n_frames: int | None = None, seed: int | None = None) -> None:
        """Decode a synthetic scene (with light link faults) in the background.

        The scene imitates the firmware family of the decoder's forced schema,
        the SDK 1.x/2.x out-of-box demo otherwise.
        """
        scene = SyntheticScene(
            schema=self.decoder.schema or "oob-sdk2",
            seed=self.viewer.demo_seed if seed is None else seed,
        )
        frames = n_frames or self.viewer.demo_frames
        self.sensor = scene.radar()
        self.source_label.setText(
            f"合成示範：{scene.schema.value}（{frames} 個 frame，含少量傳輸錯誤）"
        )
        self._start(
            lambda: SyntheticByteSource(scene, frames, faults=FAULT_PRESETS["light"]),
            live=False,
        )

    def load_capture(self, path: str | Path) -> None:
        """Decode a capture file in the background."""
        file_path = Path(path)
        if not file_path.is_file():
            self._report_error(f"找不到擷取檔：{file_path}")
            return
        self.sensor = self.base_sensor
        self.source_label.setText(f"擷取檔 {file_path.name}")
        self._start(lambda: FileByteSource(file_path, self.decoder.chunk_size), live=False)

    def connect_serial(
        self,
        data_port: str,
        cli_port: str | None = None,
        cli_text: str | None = None,
        sensor: SensorConfig | None = None,
    ) -> None:
        """Decode live from a serial port in the background.

        ``sensor`` supplies the physical units; without it the configuration
        given when the viewer was opened is used.
        """
        self.sensor = sensor if sensor is not None else self.base_sensor
        self.source_label.setText(f"序列埠 {data_port}")
        self._start(
            lambda: SerialByteSource(data_port, cli_port=cli_port, cli_text=cli_text),
            live=True,
        )

    def _open_selected_source(self) -> None:
        kind = self.source_combo.currentData()
        if kind == "demo":
            self.load_demo()
        elif kind == "file":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "開啟擷取檔", "", "擷取檔 (*.bin *.dat);;所有檔案 (*)"
            )
            if path:
                self.load_capture(path)
        else:
            dialog = SerialConnectDialog(self)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            port = dialog.data_port.text().strip()
            if not port:
                self._report_error("請輸入資料埠名稱")
                return
            cli_port = dialog.cli_port.text().strip() or None
            config_file = dialog.config_path.text().strip()
            cli_text: str | None = None
            sensor: SensorConfig | None = None
            if config_file:
                loaded = self.read_sensor_config(config_file)
                if loaded is None:
                    return
                cli_text, sensor = loaded
            # Without a CLI port the .cfg only supplies the units; the sensor
            # keeps running whatever configuration it already has.
            self.connect_serial(port, cli_port, cli_text if cli_port else None, sensor)

    def read_sensor_config(self, path: str | Path) -> tuple[str, SensorConfig] | None:
        """Read a ``.cfg`` file for a serial session: its text and its parsed form.

        The text is what gets sent to the sensor; the parsed configuration
        gives the decoder its physical units. Returns ``None`` (after
        reporting the problem) when the file cannot be read or parsed.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
            return text, parse_cli_config(text, self.decoder.platform)
        except (OSError, ConfigError) as exc:
            self._report_error(f"無法使用雷達組態檔：{exc}")
            return None

    def _start(self, factory: Callable[[], ByteSource], live: bool) -> None:
        self.stop_loading()
        self.pause()
        self.frames = []
        self.current_index = -1
        self.rejected_packets = 0
        self.received_frames = 0
        self.damaged_frames = 0
        self.live = live
        self.canvas.reset_scales()
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.blockSignals(False)
        self.loaded_label.setText("0")
        self.errors_label.setText("0")
        self.last_error = None
        loader = FrameLoader(factory, self.decoder, self.sensor, live=live)
        thread = QtCore.QThread(self)
        loader.moveToThread(thread)
        thread.started.connect(loader.run)
        loader.frames_ready.connect(self._frames_arrived)
        loader.finished.connect(self._loading_finished)
        loader.failed.connect(self._loading_failed)
        loader.finished.connect(thread.quit)
        loader.failed.connect(thread.quit)
        self._loader, self._thread, self._loading = loader, thread, True
        self.statusBar().showMessage("即時接收中…" if live else "正在解碼…")
        thread.start()
        self._update_controls()
        if live:
            self.play()

    def stop_loading(self) -> None:
        """Stop the background loader, if any, and wait for its thread."""
        if self._loader is not None:
            self._loader.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
        if self._loader is not None:
            self._loader.deleteLater()
        self._loader, self._thread, self._loading = None, None, False

    @property
    def is_loading(self) -> bool:
        """Whether a loader is still decoding."""
        return self._loading

    def wait_until_loaded(self, timeout_s: float = 30.0) -> bool:
        """Process events until loading has finished; ``False`` on timeout."""
        deadline = time.monotonic() + timeout_s
        app = QtWidgets.QApplication.instance()
        while self._loading and time.monotonic() < deadline:
            if app is not None:
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
            time.sleep(0.005)
        if app is not None:
            app.processEvents()
        return not self._loading

    @QtCore.Slot(object)
    def _frames_arrived(self, batch: Sequence[RadarFrame]) -> None:
        first = not self.frames
        self.frames.extend(batch)
        self.received_frames += len(batch)
        self.damaged_frames += sum(1 for frame in batch if frame.has_errors)
        excess = len(self.frames) - self.viewer.live_history_frames
        if self.live and excess > 0:
            # A live session runs indefinitely: keep only the recent frames.
            del self.frames[:excess]
            self.current_index = max(self.current_index - excess, -1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, len(self.frames) - 1)
        self.frame_slider.blockSignals(False)
        self.loaded_label.setText(str(self.received_frames))
        self._refresh_error_count()
        if first:
            self.show_frame(0)
        self._update_controls()

    @QtCore.Slot(object)
    def _loading_finished(self, stats: AssemblerStats) -> None:
        self._loading = False
        self.rejected_packets = stats.frames_rejected
        self._refresh_error_count()
        self.statusBar().showMessage(
            f"已解碼 {len(self.frames)} 個 frame；略過 {stats.bytes_skipped} 個位元組，"
            f"捨棄 {stats.frames_rejected} 個損壞封包"
        )
        self._update_controls()

    @QtCore.Slot(str)
    def _loading_failed(self, message: str) -> None:
        self._loading = False
        self._report_error(f"載入失敗：{message}")
        self._update_controls()

    def _report_error(self, message: str) -> None:
        self.last_error = message
        self.statusBar().showMessage(message)
        _LOGGER.warning("%s", message)
        if self.interactive:
            box = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Critical, "錯誤", message, parent=self
            )
            box.open()

    def _refresh_error_count(self) -> None:
        self.errors_label.setText(f"{self.damaged_frames + self.rejected_packets}")

    # -- playback ------------------------------------------------------------------

    def play(self) -> None:
        """Start playback (live sources follow the newest frame)."""
        if self.frames or self.live:
            self.timer.start()
        self._update_controls()

    def pause(self) -> None:
        """Stop playback."""
        self.timer.stop()
        self._update_controls()

    def step(self) -> None:
        """Show the next frame."""
        if self.frames and self.current_index < len(self.frames) - 1:
            self.show_frame(self.current_index + 1)

    def _advance(self) -> None:
        if not self.frames:
            return
        if self.live:
            target = len(self.frames) - 1
        elif self.current_index >= len(self.frames) - 1:
            if not self._loading:
                self.pause()
            return
        else:
            target = self.current_index + 1
        if target != self.current_index:
            self.show_frame(target)

    def _slider_moved(self, value: int) -> None:
        if 0 <= value < len(self.frames) and value != self.current_index:
            self.show_frame(value)

    def _color_mode_changed(self, _index: int) -> None:
        self.color_mode = ColorMode(self.color_combo.currentData())
        if 0 <= self.current_index < len(self.frames):
            self.show_frame(self.current_index)

    def show_frame(self, index: int) -> None:
        """Display frame ``index`` of the loaded frames."""
        if not 0 <= index < len(self.frames):
            return
        frame = self.frames[index]
        self.current_index = index
        display = prepare_frame_display(frame, self.sensor, self.color_mode, self.viewer)
        self.last_render_s = self.canvas.show_frame(display)
        self.render_times.append(self.last_render_s)
        if len(self.render_times) > 500:
            del self.render_times[:-500]
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(str(display.frame_number))
        self.points_label.setText(str(display.point_count))
        self.targets_label.setText("—" if display.targets is None else str(display.target_count))
        self.frame_issues_label.setText(str(display.issue_count))
        self.render_label.setText(f"{self.last_render_s * 1e3:.1f} ms")

    def _update_controls(self) -> None:
        has_frames = bool(self.frames)
        playing = self.timer.isActive()
        self.play_button.setEnabled((has_frames or self.live) and not playing)
        self.pause_button.setEnabled(playing)
        self.step_button.setEnabled(has_frames and not playing)
        self.frame_slider.setEnabled(has_frames and not self.live)
        self.export_button.setEnabled(has_frames)

    # -- export --------------------------------------------------------------------

    def export_current_frame(self, path: str | Path | None = None) -> Path | None:
        """Write the points of the displayed frame as CSV; asks for a file without ``path``."""
        if not 0 <= self.current_index < len(self.frames):
            self._report_error("目前沒有可匯出的 frame")
            return None
        frame = self.frames[self.current_index]
        if path is None:
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "匯出目前 frame", f"frame_{frame.frame_number}.csv", "CSV 檔案 (*.csv)"
            )
            if not chosen:
                return None
            path = chosen
        target = Path(path)
        count = write_points_csv(target, [frame])
        self.statusBar().showMessage(
            f"已匯出 frame {frame.frame_number} 的 {count} 個點到 {target}"
        )
        return target

    def closeEvent(self, event: Any) -> None:  # Qt naming
        self.timer.stop()
        self.stop_loading()
        super().closeEvent(event)


def launch_viewer(
    source: str = "demo",
    capture: str | Path | None = None,
    sensor: SensorConfig | None = None,
    decoder: DecoderConfig | None = None,
    data_port: str | None = None,
    cli_port: str | None = None,
    cli_text: str | None = None,
    fps: float = 20.0,
) -> int:
    """Open the viewer and run the Qt event loop; returns its exit code.

    Parameters
    ----------
    source : {"demo", "file", "serial"}
        What to show first.
    capture : str or pathlib.Path, optional
        Capture file for ``source="file"``.
    sensor : SensorConfig, optional
        Radar configuration of the capture or port.
    decoder : DecoderConfig, optional
        Decoder settings.
    data_port, cli_port, cli_text : str, optional
        Serial settings for ``source="serial"``.
    fps : float, optional
        Playback rate.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = RadarViewerWindow(decoder, sensor, ViewerConfig(fps=fps))
    window.show()
    if source == "file" and capture is not None:
        window.source_combo.setCurrentIndex(1)
        window.load_capture(capture)
    elif source == "serial" and data_port:
        window.source_combo.setCurrentIndex(2)
        window.connect_serial(data_port, cli_port, cli_text)
    else:
        window.load_demo()
    return int(app.exec())
