"""Byte sources shared by the command line and the viewer.

A capture file, a synthetic scene and a live serial port are all
:class:`ByteSource` objects: they yield chunks of bytes and can be closed.
:class:`FrameStream` turns any of them into decoded, optionally de-duplicated
frames, so the command line, the viewer and the examples decode the same way
wherever the bytes come from.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .assembler import AssemblerStats, FrameAssembler
from .capture import DedupeStats, FrameDeduplicator
from .config import DecoderConfig
from .frame import RadarFrame
from .sensor_config import SensorConfig
from .serial_link import (
    CLI_PORT_BAUDRATE,
    DATA_PORT_BAUDRATE,
    DataPortReader,
    SerialFactory,
    cli_commands,
    open_port,
    send_cli_config,
)
from .synthetic import FaultPlan, SyntheticScene, generate_packets, inject_faults, split_into_chunks

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class ByteSource(Protocol):
    """Anything that delivers radar bytes in chunks."""

    @property
    def description(self) -> str:
        """Short human-readable name of the source."""
        ...

    def chunks(self) -> Iterator[bytes]:
        """Yield the bytes, in order, in chunks of any size."""
        ...

    def close(self) -> None:
        """Release the underlying resource (idempotent)."""
        ...


class FileByteSource:
    """A capture file read in fixed-size chunks.

    Parameters
    ----------
    path : str or pathlib.Path
        Capture file.
    chunk_size : int, optional
        Bytes per read.
    """

    def __init__(self, path: str | Path, chunk_size: int = 65_536) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"capture file not found: {self.path}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = chunk_size

    @property
    def description(self) -> str:
        return f"file {self.path.name}"

    @property
    def size_bytes(self) -> int:
        """File size."""
        return self.path.stat().st_size

    def chunks(self) -> Iterator[bytes]:
        with self.path.open("rb") as handle:
            while chunk := handle.read(self.chunk_size):
                yield chunk

    def close(self) -> None:
        return None


class SyntheticByteSource:
    """Packets of a synthetic scene, optionally damaged and split at random.

    Parameters
    ----------
    scene : SyntheticScene, optional
        Scene to render.
    n_frames : int or None, optional
        Frames to render; ``None`` streams forever.
    faults : FaultPlan, optional
        Faults injected into every batch of packets.
    max_chunk : int, optional
        Largest chunk handed out (chunk sizes are random).
    realtime : bool, optional
        Pace the packets at the scene's frame period, like a live sensor.
    batch_frames : int, optional
        Packets rendered (and faulted) together.
    """

    def __init__(
        self,
        scene: SyntheticScene | None = None,
        n_frames: int | None = 200,
        *,
        faults: FaultPlan | None = None,
        max_chunk: int = 4096,
        realtime: bool = False,
        batch_frames: int = 32,
    ) -> None:
        self.scene = scene or SyntheticScene()
        self.n_frames = n_frames
        self.faults = faults
        self.max_chunk = max_chunk
        self.realtime = realtime
        self.batch_frames = max(1, batch_frames)
        self._stop = threading.Event()

    @property
    def description(self) -> str:
        return f"synthetic {self.scene.schema.value} scene (seed {self.scene.seed})"

    def chunks(self) -> Iterator[bytes]:
        rng = np.random.default_rng([self.scene.seed, 2])
        packets = generate_packets(self.scene, self.n_frames)
        period = self.scene.radar().frame_period_s
        batch: list[bytes] = []
        for packet in packets:
            if self._stop.is_set():
                return
            batch.append(packet)
            if self.realtime or len(batch) >= self.batch_frames:
                yield from self._emit(batch, rng)
                batch = []
                if self.realtime:
                    time.sleep(period)
        if batch:
            yield from self._emit(batch, rng)

    def _emit(self, batch: list[bytes], rng: np.random.Generator) -> Iterator[bytes]:
        data = inject_faults(batch, self.faults, rng).data if self.faults else b"".join(batch)
        yield from split_into_chunks(data, rng, max_chunk=self.max_chunk)

    def close(self) -> None:
        self._stop.set()


class SerialByteSource:
    """Live bytes from the data port, optionally configuring the sensor first.

    Parameters
    ----------
    data_port : str
        Data (auxiliary) port name.
    baudrate : int, optional
        Data-port baud rate.
    cli_port : str, optional
        CLI port name; required to send a configuration.
    cli_text : str, optional
        ``.cfg`` contents sent before streaming (``sensorStop`` ...
        ``sensorStart``).
    cli_baudrate : int, optional
        CLI-port baud rate.
    record_path : str or pathlib.Path, optional
        File receiving the raw bytes as they arrive.
    duration_s : float, optional
        Stop after this long.
    serial_factory : callable, optional
        Replaces pyserial (tests, dry runs).
    """

    def __init__(
        self,
        data_port: str,
        *,
        baudrate: int = DATA_PORT_BAUDRATE,
        cli_port: str | None = None,
        cli_text: str | None = None,
        cli_baudrate: int = CLI_PORT_BAUDRATE,
        record_path: str | Path | None = None,
        duration_s: float | None = None,
        serial_factory: SerialFactory | None = None,
    ) -> None:
        if cli_text and not cli_port:
            raise ValueError("sending a configuration needs the CLI port")
        self.data_port = data_port
        self.baudrate = baudrate
        self.cli_port = cli_port
        self.cli_text = cli_text
        self.cli_baudrate = cli_baudrate
        self.record_path = record_path
        self.duration_s = duration_s
        self.serial_factory = serial_factory
        self._reader: DataPortReader | None = None
        self._data: Any = None
        self._closed = threading.Event()

    @property
    def description(self) -> str:
        baudrate = getattr(self._data, "baudrate", None) or self.baudrate
        return f"serial port {self.data_port} @ {baudrate} baud"

    @property
    def bytes_read(self) -> int:
        """Bytes received so far."""
        return self._reader.bytes_read if self._reader is not None else 0

    def configure_sensor(self) -> list[str]:
        """Send the configuration over the CLI port; returns the sensor's replies.

        A ``configDataPort`` line switches the open data port to the new line
        rate once the sensor acknowledges it.
        """
        if not self.cli_text or not self.cli_port:
            return []
        port = open_port(self.cli_port, self.cli_baudrate, 0.2, self.serial_factory)
        try:
            return send_cli_config(port, cli_commands(self.cli_text), data_port=self._data)
        finally:
            port.close()

    def chunks(self) -> Iterator[bytes]:
        data = open_port(self.data_port, self.baudrate, 0.05, self.serial_factory)
        self._data = data
        reader = DataPortReader(data, record_path=self.record_path)
        self._reader = reader
        reader.start()
        try:
            self.configure_sensor()
            deadline = None if self.duration_s is None else time.monotonic() + self.duration_s

            def finished() -> bool:
                # Polled even while no bytes arrive, so a silent port (sensor not
                # started, wrong port, end of a replay) cannot block past the
                # deadline or a close() request.
                if self._closed.is_set():
                    return True
                return deadline is not None and time.monotonic() >= deadline

            yield from reader.chunks(stop_when=finished)
        finally:
            reader.stop()
            data.close()
            self._data = None

    def close(self) -> None:
        self._closed.set()
        if self._reader is not None:
            self._reader.stop()


class RecordingByteSource:
    """Pass a source through unchanged while writing every chunk to a file.

    The recording holds exactly the bytes the decoder saw, in order, so it can
    be decoded again later with identical results.

    Parameters
    ----------
    source : ByteSource
        Source to record.
    path : str or pathlib.Path
        Recording file (overwritten).
    """

    def __init__(self, source: ByteSource, path: str | Path) -> None:
        self.source = source
        self.path = Path(path)

    @property
    def description(self) -> str:
        return f"{self.source.description} (recorded to {self.path.name})"

    def chunks(self) -> Iterator[bytes]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as recording:
            for chunk in self.source.chunks():
                recording.write(chunk)
                yield chunk

    def close(self) -> None:
        self.source.close()


class FrameStream:
    """Decoded frames of any :class:`ByteSource`.

    Iterating runs the source through a :class:`FrameAssembler` and, when the
    configuration asks for it, a :class:`FrameDeduplicator`. Afterwards
    :attr:`assembler_stats` and :attr:`dedupe_stats` hold the final counters.

    Parameters
    ----------
    source : ByteSource
        Where the bytes come from.
    config : DecoderConfig, optional
        Decoder settings.
    sensor : SensorConfig, optional
        Radar configuration for physical units.

    Examples
    --------
    >>> stream = FrameStream(SyntheticByteSource(n_frames=5))
    >>> [frame.frame_number for frame in stream]
    [1, 2, 3, 4, 5]
    >>> stream.assembler_stats.frames_decoded
    5
    """

    def __init__(
        self,
        source: ByteSource,
        config: DecoderConfig | None = None,
        sensor: SensorConfig | None = None,
    ) -> None:
        self.source = source
        self.config = config or DecoderConfig()
        self.sensor = sensor
        self.assembler = FrameAssembler(self.config, sensor)
        self.deduplicator = (
            FrameDeduplicator(self.config.dedupe_window) if self.config.deduplicate else None
        )
        self.frames_seen = 0

    @property
    def assembler_stats(self) -> AssemblerStats:
        """Assembler counters (live during iteration)."""
        return self.assembler.stats

    @property
    def dedupe_stats(self) -> DedupeStats:
        """De-duplication counters (all zero when de-duplication is off)."""
        return self.deduplicator.stats if self.deduplicator is not None else DedupeStats()

    def _keep(self, frame: RadarFrame) -> bool:
        self.frames_seen += 1
        return self.deduplicator is None or self.deduplicator.accept(frame)

    def __iter__(self) -> Iterator[RadarFrame]:
        try:
            for chunk in self.source.chunks():
                for frame in self.assembler.feed(chunk):
                    if self._keep(frame):
                        yield frame
            for frame in self.assembler.finish():
                if self._keep(frame):
                    yield frame
        finally:
            self.source.close()
            _LOGGER.debug(
                "%s: %d frames decoded, %d bytes skipped",
                self.source.description,
                self.assembler.stats.frames_decoded,
                self.assembler.stats.bytes_skipped,
            )


def iter_frames(
    source: ByteSource,
    config: DecoderConfig | None = None,
    sensor: SensorConfig | None = None,
) -> Iterator[RadarFrame]:
    """Decoded frames of a byte source; see :class:`FrameStream`."""
    yield from FrameStream(source, config, sensor)
