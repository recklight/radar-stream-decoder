"""Live capture from the radar's two UARTs.

TI evaluation modules expose two serial ports. The CLI port (115200 baud, 8N1)
takes the ``.cfg`` text commands one line at a time. The firmware echoes each
line, may print message lines (``Ignored: ...``, ``Debug: ...``,
``Error: ...``), then answers ``Done`` or ``Error <code>`` and prints the
``mmwDemo:/>`` prompt. The prompt has no newline, so it arrives glued to the
next echo. A command the firmware does not know is answered only with
``'<cmd>' is not recognized as a CLI command``. The data port (921600 baud by
default, 8N1, no flow control) carries the binary TLV stream;
``configDataPort <baud> <ackPing>`` changes its line rate (TI mmWave SDK
03.05.00.04, ``mmw_cli.c``).

:func:`send_cli_config` first clears the CLI port's input buffer, because a
stale banner or a reply left from an earlier session would otherwise be read
as an answer. It then sends one line at a time and waits for its answer:
``Error <code>`` or an unrecognised command fails at once, silence fails after
a timeout, and the error names the line. When a ``configDataPort`` command is
acknowledged it switches the data port to the new line rate, as the sensor
has just done.

:class:`DataPortReader` reads the data port on a background thread with a
large receive buffer, since a reader that falls behind is the usual cause of
dropped frames. It can record the bytes exactly as received. Recording the raw
stream, not snapshots of a working buffer, is what keeps repeated packets out
of a capture.

pyserial is imported only when a real port is opened; every function takes a
``serial_factory`` so tests and dry runs can substitute the emulated ports of
:mod:`radar_stream_decoder.port_emulator`.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

_LOGGER = logging.getLogger(__name__)

#: Baud rate of the data (auxiliary) port.
DATA_PORT_BAUDRATE = 921_600
#: Baud rate of the CLI (user) port.
CLI_PORT_BAUDRATE = 115_200
#: Prompt the out-of-box demo prints before reading every command.
CLI_PROMPT = "mmwDemo:/>"
#: Receive buffer requested from the operating system where supported.
RECEIVE_BUFFER_BYTES = 1 << 20
#: Reply that ends a successful command.
_DONE = "Done"
#: Reply that ends a failed command, with the handler's error code.
_ERROR_CODE = re.compile(r"Error (-?\d+)")
#: Reply to a command the firmware does not know (no Done or Error follows).
_NOT_RECOGNISED = "is not recognized as a CLI command"


class SerialPort(Protocol):
    """The subset of ``serial.Serial`` this module uses."""

    def write(self, data: bytes) -> int | None: ...

    def read(self, size: int = 1) -> bytes: ...

    def readline(self) -> bytes: ...

    def close(self) -> None: ...


#: ``factory(port_name, baudrate, timeout_s) -> SerialPort``
SerialFactory = Callable[[str, int, float], Any]


class SensorCommandError(RuntimeError):
    """The sensor rejected a configuration command or did not answer.

    Attributes
    ----------
    line : int
        Line number of the command in the configuration text.
    command : str
        The command that failed.
    response : str
        What the sensor answered.
    code : int or None
        Error code of an ``Error <code>`` reply.
    """

    def __init__(
        self,
        line: int,
        command: str,
        response: str,
        reason: str = "rejected",
        code: int | None = None,
    ) -> None:
        self.line = line
        self.command = command
        self.response = response
        self.code = code
        # One line, whatever the sensor printed: the command line reports it as is.
        summary = "; ".join(part.strip() for part in response.splitlines() if part.strip())
        super().__init__(f"line {line}: sensor {reason} {command!r}: {summary or '(no reply)'}")


class DataPortError(RuntimeError):
    """Reading the data port failed (the original exception is chained)."""


def require_pyserial() -> Any:
    """Import pyserial or explain how to install it."""
    try:
        import serial
    except ImportError as exc:
        raise ImportError(
            'live capture needs pyserial: run pip install -e ".[serial]" in the project folder'
        ) from exc
    return serial


def open_port(
    name: str,
    baudrate: int,
    timeout_s: float = 0.05,
    serial_factory: SerialFactory | None = None,
) -> Any:
    """Open a serial port (8 data bits, no parity, one stop bit).

    Parameters
    ----------
    name : str
        ``COM4``, ``/dev/ttyACM1`` ...
    baudrate : int
        Line rate.
    timeout_s : float, optional
        Read timeout.
    serial_factory : callable, optional
        Replaces pyserial (tests, dry runs).
    """
    if serial_factory is not None:
        return serial_factory(name, baudrate, timeout_s)
    serial = require_pyserial()
    port = serial.Serial(port=name, baudrate=baudrate, timeout=timeout_s)
    if hasattr(port, "set_buffer_size"):
        try:
            port.set_buffer_size(rx_size=RECEIVE_BUFFER_BYTES)
        except (OSError, ValueError, AttributeError):
            _LOGGER.debug("could not enlarge the receive buffer of %s", name)
    return port


def cli_commands(text: str) -> list[tuple[int, str]]:
    """``(line number, command)`` of every non-empty, non-comment line.

    Examples
    --------
    >>> cli_commands("% comment\\nsensorStop\\n\\nchannelCfg 15 3 0 % trailing\\n")
    [(2, 'sensorStop'), (4, 'channelCfg 15 3 0')]
    """
    commands = []
    for number, raw in enumerate(text.splitlines(), start=1):
        command = raw.split("%", 1)[0].strip()
        if command:
            commands.append((number, command))
    return commands


class _LineReader:
    """Complete lines out of ``readline`` results that may stop mid-line.

    A real port's ``readline`` returns whatever arrived when its timeout
    expires, for example the prompt that waits, without a newline, for the
    next command.
    """

    def __init__(self, port: SerialPort) -> None:
        self.port = port
        self._pending = b""

    def next_line(self) -> str | None:
        """The next complete line without its line end, or ``None`` if none arrived."""
        if b"\n" not in self._pending:
            self._pending += self.port.readline() or b""
        if b"\n" not in self._pending:
            return None
        line, _, self._pending = self._pending.partition(b"\n")
        return line.decode("ascii", errors="replace")


def _clean_reply(line: str) -> str:
    """A reply line without carriage returns and leading prompts."""
    text = line.replace("\r", "").strip()
    while text.startswith(CLI_PROMPT):
        text = text[len(CLI_PROMPT) :].strip()
    return text


def _data_port_baudrate(command: str, reply: Sequence[str]) -> int | None:
    """New data-port line rate of an acknowledged ``configDataPort``, if it took effect."""
    words = command.split()
    if not words or words[0] != "configDataPort" or len(words) < 2:
        return None
    if any(line.startswith("Ignored") for line in reply):
        return None
    try:
        return int(words[1])
    except ValueError:
        return None


def send_cli_config(
    port: SerialPort,
    lines: Sequence[tuple[int, str]] | str,
    *,
    timeout_s: float = 2.0,
    clock: Callable[[], float] = time.monotonic,
    data_port: Any = None,
) -> list[str]:
    """Send configuration commands and wait for each acknowledgement.

    Parameters
    ----------
    port : SerialPort
        Open CLI port.
    lines : str or sequence of (int, str)
        Configuration text, or the output of :func:`cli_commands`.
    timeout_s : float, optional
        Longest wait for one command's ``Done``/``Error``.
    clock : callable, optional
        Monotonic clock (replaceable in tests).
    data_port : serial port, optional
        Open data port. When a ``configDataPort`` command is acknowledged its
        ``baudrate`` is switched to the new line rate, as the sensor's is.

    Returns
    -------
    list of str
        The sensor's reply to every command (echo excluded).

    Raises
    ------
    SensorCommandError
        On an ``Error <code>`` reply, an unrecognised command or a timeout,
        naming the line.
    """
    commands = cli_commands(lines) if isinstance(lines, str) else list(lines)
    reset = getattr(port, "reset_input_buffer", None)
    if callable(reset):
        reset()
    reader = _LineReader(port)
    replies = []
    for number, command in commands:
        port.write((command + "\n").encode("ascii"))
        deadline = clock() + timeout_s
        reply: list[str] = []
        while True:
            raw = reader.next_line()
            if raw is None:
                if clock() > deadline:
                    raise SensorCommandError(
                        number, command, "\n".join(reply), "did not acknowledge"
                    )
                continue
            text = _clean_reply(raw)
            if not text or (text == command and not reply):
                continue  # the echo of the command, or an empty prompt line
            reply.append(text)
            if text == _DONE:
                break
            failure = _ERROR_CODE.fullmatch(text)
            if failure is not None:
                raise SensorCommandError(
                    number, command, "\n".join(reply), code=int(failure.group(1))
                )
            if _NOT_RECOGNISED in text:
                raise SensorCommandError(number, command, "\n".join(reply), "does not know")
        _LOGGER.debug("line %d acknowledged: %s", number, command)
        baudrate = _data_port_baudrate(command, reply)
        if baudrate is not None:
            if data_port is None:
                _LOGGER.warning(
                    "line %d moved the sensor's data port to %d baud; open the data port "
                    "at that rate",
                    number,
                    baudrate,
                )
            else:
                data_port.baudrate = baudrate
                _LOGGER.info("data port switched to %d baud (line %d)", baudrate, number)
        replies.append("\n".join(reply))
    return replies


class DataPortReader(threading.Thread):
    """Background reader of the data port.

    Bytes are handed over through a queue in the order received; an optional
    recording receives exactly the same bytes. Call :meth:`stop` to end the
    thread; :meth:`chunks` then finishes once the queue is drained.

    Parameters
    ----------
    port : SerialPort
        Open data port.
    record_path : str or pathlib.Path, optional
        File that receives the raw bytes; its folder is created when missing.
    read_size : int, optional
        Largest single read.
    """

    _END = b""

    def __init__(
        self, port: SerialPort, *, record_path: str | Path | None = None, read_size: int = 65_536
    ) -> None:
        super().__init__(name="radar-data-port", daemon=True)
        self.port = port
        self.record_path = Path(record_path) if record_path is not None else None
        self.read_size = read_size
        self.bytes_read = 0
        self.error: BaseException | None = None
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._stop_event = threading.Event()

    def run(self) -> None:
        recording = None
        try:
            # Opened inside the try, so a recording that cannot be written ends
            # the stream with a DataPortError instead of a silent, empty run.
            if self.record_path is not None:
                self.record_path.parent.mkdir(parents=True, exist_ok=True)
                recording = self.record_path.open("wb")
            while not self._stop_event.is_set():
                waiting = int(getattr(self.port, "in_waiting", 0) or 0)
                chunk = self.port.read(min(max(waiting, 1), self.read_size))
                if not chunk:
                    continue
                if recording is not None:
                    recording.write(chunk)
                self.bytes_read += len(chunk)
                self._queue.put(bytes(chunk))
        except Exception as exc:  # reported to the consumer, not swallowed
            self.error = exc
            # chunks() raises DataPortError with this message; logging it here as
            # well would print the failure twice.
            _LOGGER.debug("data port reader stopped: %s", exc, exc_info=True)
        finally:
            if recording is not None:
                recording.close()
            self._queue.put(self._END)

    def stop(self, timeout_s: float = 2.0) -> None:
        """Ask the thread to finish and wait for it."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout_s)

    @property
    def stopping(self) -> bool:
        """Whether :meth:`stop` has been called."""
        return self._stop_event.is_set()

    def chunks(
        self, poll_s: float = 0.1, *, stop_when: Callable[[], bool] | None = None
    ) -> Iterator[bytes]:
        """Yield received chunks until the thread has stopped and the queue is empty.

        Parameters
        ----------
        poll_s : float, optional
            Longest wait for data before the stop conditions are checked again.
        stop_when : callable, optional
            Checked before every wait; returning ``True`` ends the iteration.
            Because it is also checked when ``poll_s`` passes without data, a
            deadline or a close request takes effect even on a silent port.

        Raises
        ------
        DataPortError
            If the reader thread failed (the original exception is chained).
        """
        while True:
            if stop_when is not None and stop_when():
                break
            try:
                chunk = self._queue.get(timeout=poll_s)
            except queue.Empty:
                if not self.is_alive() and self._queue.empty():
                    break
                continue
            if chunk == self._END:
                break
            yield chunk
        if self.error is not None:
            raise DataPortError(f"data port reader failed: {self.error}") from self.error
