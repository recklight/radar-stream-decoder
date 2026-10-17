"""Serial ports without hardware: the demo's CLI port and a replaying data port.

The live-capture code talks to ``serial.Serial``-like objects; the classes here
stand in for them in tests, dry runs and the viewer's demo. Their behaviour
follows the firmware of TI mmWave SDK 03.05.00.04, not what the host tools
expect:

* The text-mode UART (``uartsci.c``) echoes every received byte as it
  arrives, echoes a CR as CR LF and takes it as LF; a line ends at LF. Text
  the firmware writes gets a CR *after* every LF (``UartSci_writeChar`` sends
  the LF, then the CR), so a reply such as ``Done`` reaches the host as
  ``Done\\n\\r``.
* The CLI loop (``cli.c``, ``CLI_task``) prints the banner once, then the
  prompt ``mmwDemo:/>`` without a newline before every line it reads. A line
  starting with ``%`` is answered ``Skipped``; an empty line only gets a new
  prompt; a known command runs and is answered ``Done`` or ``Error <code>``;
  an unknown command gets ``'<cmd>' is not recognized as a CLI command`` and
  *no* ``Done`` or ``Error``.
* Among the demo's own commands (``mmw_cli.c``, ``mss_main.c`` of the xwr68xx
  demo), ``sensorStop`` on a stopped sensor prints ``Ignored: Sensor is
  already stopped`` and ``sensorStart`` on a running one ``Ignored: Sensor is
  already started``. The first ``sensorStart`` without the full configuration
  fails with ``Error: Full configuration must be provided ...``; the first
  successful start prints ``Debug: Init Calibration Status = 0x...`` when the
  RF calibration completes. ``configDataPort <baud> <ackPing>`` switches the
  data port's line rate (at most 3125000, only while stopped), prints ``Data
  port baud rate changed to <baud>`` and, with ``ackPing``, sends 16 bytes of
  ``0xFF`` on the data port.

Keep the simplifications in mind when reading test results: the emulator
checks only a configurable subset of the commands the firmware requires before
the first start and does not validate command arguments; it reads the numbers
of ``sensorStart`` and ``configDataPort`` as the firmware's ``atoi`` does, and
answers a ``configDataPort`` without both arguments with ``Error`` (the
firmware reads them unchecked, with undefined results); the banner and the
calibration status value are the SDK 3.5 format with assumed values; nothing
models USB bridges, DTR/RTS or real timing. A data port read at a line rate
other than the sensor's returns zero bytes in place of the data, standing in
for the framing errors a real UART would produce.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterable

from .header import SdkVersion
from .serial_link import CLI_PORT_BAUDRATE, CLI_PROMPT, DATA_PORT_BAUDRATE

#: Highest data-port line rate ``configDataPort`` accepts (``mmw_cli.c``).
MAX_DATA_PORT_BAUDRATE = 3_125_000

#: Commands of the SDK 3.5 xwr68xx demo's own table (``mmw_cli.c``).
DEMO_COMMANDS: frozenset[str] = frozenset(
    {
        "sensorStart",
        "sensorStop",
        "guiMonitor",
        "cfarCfg",
        "multiObjBeamForming",
        "calibDcRangeSig",
        "clutterRemoval",
        "adcbufCfg",
        "compRangeBiasAndRxChanPhase",
        "measureRangeBiasAndRxChanPhase",
        "aoaFovCfg",
        "cfarFovCfg",
        "extendedMaxVelocity",
        "bpmCfg",
        "CQRxSatMonitor",
        "CQSigImgMonitor",
        "analogMonitor",
        "lvdsStreamCfg",
        "configDataPort",
        "queryDemoStatus",
        "calibData",
    }
)

#: Commands of the CLI's mmWave extension table (``cli_mmwave.c``) and its help.
EXTENSION_COMMANDS: frozenset[str] = frozenset(
    {
        "help",
        "version",
        "flushCfg",
        "dfeDataOutputMode",
        "channelCfg",
        "adcCfg",
        "profileCfg",
        "chirpCfg",
        "frameCfg",
        "advFrameCfg",
        "subFrameCfg",
        "lowPower",
        "contModeCfg",
        "bpmCfgAdvanced",
    }
)

#: Commands the emulator requires before the first ``sensorStart`` (a subset of
#: the firmware's ``MmwDemo_isAllCfgInPendingState`` check).
DEFAULT_REQUIRED_COMMANDS: frozenset[str] = frozenset(
    {"channelCfg", "profileCfg", "chirpCfg", "frameCfg"}
)

#: ``MmwDemo_SensorState`` values, as ``queryDemoStatus`` prints them.
SENSOR_STATES: dict[str, int] = {"init": 0, "opened": 1, "started": 2, "stopped": 3}

_LF = 0x0A
_CR = 0x0D


def _atoi(text: str) -> int:
    """C ``atoi``: the leading decimal integer of ``text``, 0 when there is none.

    Examples
    --------
    >>> _atoi("921600"), _atoi("00"), _atoi("-3x"), _atoi("abc")
    (921600, 0, -3, 0)
    """
    match = re.match(r"\s*([+-]?\d+)", text)
    return int(match.group(1)) if match else 0


def _text_mode(text: str) -> bytes:
    """Bytes the firmware's text-mode UART writes for ``text`` (CR after each LF)."""
    return text.encode("ascii", errors="replace").replace(b"\n", b"\n\r")


class _BytePort:
    """Shared read side: a byte queue with a pyserial-like timeout."""

    def __init__(self, timeout_s: float) -> None:
        self.timeout = timeout_s
        self.closed = False
        self.written = bytearray()
        self._lock = threading.Lock()

    def _available(self) -> int:
        raise NotImplementedError

    def _take(self, size: int) -> bytes:
        raise NotImplementedError

    @property
    def in_waiting(self) -> int:
        """Bytes that can be read without waiting."""
        with self._lock:
            return self._available()

    def read(self, size: int = 1) -> bytes:
        """Up to ``size`` bytes; waits ``timeout`` once when nothing is there."""
        with self._lock:
            if self._available():
                return self._take(size)
        time.sleep(self.timeout)
        with self._lock:
            return self._take(size) if self._available() else b""

    def readline(self) -> bytes:
        """Bytes up to and including the next LF, or what is there after the timeout."""
        line = bytearray()
        deadline = time.perf_counter() + self.timeout
        while True:
            with self._lock:
                while self._available():
                    byte = self._take(1)
                    line += byte
                    if byte == b"\n":
                        return bytes(line)
            if time.perf_counter() >= deadline:
                return bytes(line)
            time.sleep(min(self.timeout, 0.001))

    def close(self) -> None:
        """Mark the port closed."""
        self.closed = True


class EmulatedDataPort(_BytePort):
    """The data port: serves recorded or synthetic bytes in order.

    Parameters
    ----------
    data : bytes, optional
        Bytes the sensor sends once it streams.
    bytes_per_second : float, optional
        Pace reads at this rate (921600 baud 8N1 carries 92160 bytes/s).
    timeout_s : float, optional
        Wait of a read that finds no data.
    baudrate : int, optional
        Line rate of both ends at power-up.
    streaming : bool, optional
        Whether the data flows from the start; an emulated CLI port linked to
        this one sets it on ``sensorStart`` and clears it on ``sensorStop``.

    Attributes
    ----------
    baudrate : int
        Line rate of the host end (what ``serial.Serial.baudrate`` is).
    sensor_baudrate : int
        Line rate of the sensor end, changed by ``configDataPort``.
    """

    def __init__(
        self,
        data: bytes = b"",
        *,
        bytes_per_second: float | None = None,
        timeout_s: float = 0.01,
        baudrate: int = DATA_PORT_BAUDRATE,
        streaming: bool = True,
    ) -> None:
        super().__init__(timeout_s)
        self._data = bytes(data)
        self._position = 0
        self._rate = bytes_per_second
        # perf_counter: time.monotonic() ticks only every ~15 ms on Windows.
        self._started: float | None = time.perf_counter() if streaming else None
        self._streamed_before = 0
        self._extra = bytearray()
        self.baudrate = baudrate
        self.sensor_baudrate = baudrate

    @property
    def exhausted(self) -> bool:
        """Whether every data byte has been read."""
        return self._position >= len(self._data)

    @property
    def streaming(self) -> bool:
        """Whether the sensor is sending data."""
        return self._started is not None

    def start_streaming(self) -> None:
        """Let the data flow (``sensorStart``)."""
        with self._lock:
            if self._started is None:
                self._streamed_before = self._position
                self._started = time.perf_counter()

    def stop_streaming(self) -> None:
        """Stop the data (``sensorStop``); bytes not yet released stay queued."""
        with self._lock:
            self._started = None

    def send(self, raw: bytes) -> None:
        """Queue bytes the sensor sends outside the data stream (the ``ackPing``)."""
        with self._lock:
            self._extra += raw

    def _released(self) -> int:
        if self._started is None:
            return self._position
        if self._rate is None:
            return len(self._data)
        paced = self._streamed_before + int((time.perf_counter() - self._started) * self._rate)
        return min(len(self._data), paced)

    def _available(self) -> int:
        return len(self._extra) + max(0, self._released() - self._position)

    def _take(self, size: int) -> bytes:
        chunk = bytearray()
        if self._extra:
            chunk += self._extra[:size]
            del self._extra[: len(chunk)]
        room = size - len(chunk)
        if room > 0:
            count = min(room, max(0, self._released() - self._position))
            chunk += self._data[self._position : self._position + count]
            self._position += count
        if self.baudrate != self.sensor_baudrate:
            return bytes(len(chunk))
        return bytes(chunk)

    def write(self, data: bytes) -> int:
        """Accept and record bytes (the data port ignores them)."""
        self.written += data
        return len(data)

    def reset_input_buffer(self) -> None:
        """Discard what has arrived but not been read."""
        with self._lock:
            self._extra.clear()
            self._position = max(self._position, self._released())


class EmulatedCliPort(_BytePort):
    """The CLI port of the SDK 3.5 out-of-box demo, as the host sees it.

    Parameters
    ----------
    data_port : EmulatedDataPort, optional
        Data port the commands act on (streaming, line rate, ``ackPing``).
    timeout_s : float, optional
        Wait of a read that finds nothing.
    sdk_version : SdkVersion, optional
        Version printed in the banner.
    reject_prefix : str, optional
        Test hook: commands starting with this text fail with ``Error -1``.
    required_commands : iterable of str, optional
        Commands that must precede the first ``sensorStart``.
    calibration_status : int, optional
        Value of the ``Init Calibration Status`` debug line (the firmware prints
        the RF calibration status masked with ``0x1FFF``).
    banner : bool, optional
        Queue the power-up banner (it sits in the input buffer of a port opened
        after power-up; the handshake clears it).

    Attributes
    ----------
    state : str
        ``"init"``, ``"started"`` or ``"stopped"``.
    commands : list of str
        Every line executed, in order.
    """

    def __init__(
        self,
        *,
        data_port: EmulatedDataPort | None = None,
        timeout_s: float = 0.01,
        sdk_version: SdkVersion | None = None,
        reject_prefix: str | None = None,
        required_commands: Iterable[str] = DEFAULT_REQUIRED_COMMANDS,
        calibration_status: int = 0x1FFE,
        banner: bool = True,
    ) -> None:
        super().__init__(timeout_s)
        self.data_port = data_port
        self.reject_prefix = reject_prefix
        self.required_commands = frozenset(required_commands)
        self.calibration_status = calibration_status
        self.state = "init"
        self.commands: list[str] = []
        self.baudrate = CLI_PORT_BAUDRATE
        self._version = sdk_version or SdkVersion(3, 5, 0, 4)
        self._output = bytearray()
        self._line = bytearray()
        self._pending: set[str] = set()
        if banner:
            version = self._version
            self._emit(
                "******************************************\n"
                f"xWR68xx MMW Demo {version.major:02d}.{version.minor:02d}."
                f"{version.bugfix:02d}.{version.build:02d}\n"
                "******************************************\n"
            )
        self._output += CLI_PROMPT.encode("ascii")

    # -- the host's side -------------------------------------------------------------

    def _available(self) -> int:
        return len(self._output)

    def _take(self, size: int) -> bytes:
        chunk = bytes(self._output[:size])
        del self._output[: len(chunk)]
        return chunk

    def reset_input_buffer(self) -> None:
        """Discard what the sensor has sent but the host has not read."""
        with self._lock:
            self._output.clear()

    def write(self, data: bytes) -> int:
        """Receive bytes from the host: echo them and run every completed line."""
        with self._lock:
            self.written += data
            for value in bytes(data):
                if value == _CR:
                    self._output += b"\r\n"
                    value = _LF
                else:
                    self._output.append(value)
                if value == _LF:
                    line = self._line.decode("ascii", errors="replace")
                    self._line.clear()
                    self._execute(line)
                else:
                    self._line.append(value)
        return len(data)

    # -- the firmware's side --------------------------------------------------------------

    def _emit(self, text: str) -> None:
        self._output += _text_mode(text)

    def _execute(self, line: str) -> None:
        if line.startswith("%"):
            self._emit("Skipped\n")
        else:
            tokens = line.replace("\r", " ").split()
            if tokens:
                self._run(tokens, line)
        self._output += CLI_PROMPT.encode("ascii")

    def _run(self, tokens: list[str], line: str) -> None:
        command = tokens[0]
        if command not in DEMO_COMMANDS and command not in EXTENSION_COMMANDS:
            self._emit(f"'{command}' is not recognized as a CLI command\n")
            return
        self.commands.append(line.strip())
        if self.reject_prefix and line.strip().startswith(self.reject_prefix):
            status = -1
        elif command == "sensorStop":
            status = self._sensor_stop()
        elif command == "sensorStart":
            status = self._sensor_start(tokens)
        elif command == "configDataPort":
            status = self._config_data_port(tokens)
        elif command == "queryDemoStatus":
            baud = self.data_port.sensor_baudrate if self.data_port else DATA_PORT_BAUDRATE
            self._emit(f"Sensor State: {SENSOR_STATES[self.state]}\n")
            self._emit(f"Data port baud rate: {baud}\n")
            status = 0
        elif command == "flushCfg":
            self._pending.clear()
            status = 0
        else:
            self._pending.add(command)
            status = 0
        self._emit("Done\n" if status == 0 else f"Error {status}\n")

    def _sensor_stop(self) -> int:
        if self.state != "started":
            self._emit("Ignored: Sensor is already stopped\n")
            return 0
        self.state = "stopped"
        self._pending.clear()
        if self.data_port is not None:
            self.data_port.stop_streaming()
        return 0

    def _sensor_start(self, tokens: list[str]) -> int:
        if len(tokens) == 2 and _atoi(tokens[1]) != 0:
            self._emit(
                "Error: Reconfig is not supported, only argument of 0 is\n"
                "(do not reconfig, just re-start the sensor) valid\n"
            )
            return -1
        reconfigure = len(tokens) != 2
        complete = self.required_commands <= self._pending
        if self.state == "init" and not complete:
            self._emit(
                "Error: Full configuration must be provided before sensor can be started "
                "the first time\n"
            )
            return -1
        if self.state == "started":
            self._emit("Ignored: Sensor is already started\n")
            return 0
        # As in MmwDemo_CLISensorStart these checks apply in every state, so the
        # first start, too, rejects "sensorStart 0" after a complete configuration.
        if not reconfigure and self._pending:
            self._emit(
                "Error: You have provided complete new configuration, issue "
                '"sensorStart" (without argument) if you want it to take effect\n'
                if complete
                else "Error: You have provided partial configuration between stop and "
                "this command and partial configuration cannot be undone.Issue the full "
                'configuration and do "sensorStart" \n'
            )
            return -1
        if reconfigure and not complete:
            self._emit(
                'Error: You have provided no configuration, issue "sensorStart 0" OR '
                'provide full configuration and issue "sensorStart"\n'
                if not self._pending
                else "Error: You have provided partial configuration between stop and "
                "this command and partial configuration cannot be undone.Issue the full "
                'configuration and do "sensorStart" \n'
            )
            return -1
        if self.state == "init":
            self._emit(f"Debug: Init Calibration Status = 0x{self.calibration_status:x}\n")
        self.state = "started"
        self._pending.clear()
        if self.data_port is not None:
            self.data_port.start_streaming()
        return 0

    def _config_data_port(self, tokens: list[str]) -> int:
        if self.state == "started":
            self._emit("Ignored: This command is not allowed after sensor has started\n")
            return 0
        if len(tokens) < 3:
            # The firmware reads both arguments without checking that they exist;
            # the emulator answers like the demo's other argument checks instead.
            self._emit("Error: Invalid usage of the CLI command\n")
            return -1
        baudrate, ack = _atoi(tokens[1]), _atoi(tokens[2])
        if baudrate > MAX_DATA_PORT_BAUDRATE:
            self._emit(f"Ignored: Invalid baud rate ({baudrate}) specified\n")
            return 0
        port = self.data_port
        if port is not None and port.sensor_baudrate != baudrate:
            port.sensor_baudrate = baudrate
            self._emit(f"Data port baud rate changed to {baudrate}\n")
        if port is not None and ack:
            port.send(b"\xff" * 16)
        return 0


class EmulatedPortFactory:
    """``serial_factory`` that opens emulated ports by name.

    The port named ``cli_port`` is an :class:`EmulatedCliPort`; every other name
    opens an :class:`EmulatedDataPort` serving ``data``. The CLI port acts on
    the data port opened before it (or the first one opened afterwards): its
    data flows only after ``sensorStart`` when ``wait_for_start`` is set.

    Parameters
    ----------
    data : bytes
        Bytes the data port serves.
    cli_port : str, optional
        Name of the CLI port.
    bytes_per_second : float, optional
        Pace of the data port.
    reject_prefix : str, optional
        Passed to the CLI port.
    wait_for_start : bool, optional
        Hold the data back until the CLI port receives ``sensorStart`` (only
        when a ``cli_port`` is named).
    cli_options : dict, optional
        Further keyword arguments of :class:`EmulatedCliPort`.
    """

    def __init__(
        self,
        data: bytes,
        *,
        cli_port: str | None = None,
        bytes_per_second: float | None = None,
        reject_prefix: str | None = None,
        wait_for_start: bool = True,
        cli_options: dict[str, object] | None = None,
    ) -> None:
        self.data = data
        self.cli_port = cli_port
        self.bytes_per_second = bytes_per_second
        self.reject_prefix = reject_prefix
        self.wait_for_start = wait_for_start and cli_port is not None
        self.cli_options = dict(cli_options or {})
        self.opened: dict[str, EmulatedCliPort | EmulatedDataPort] = {}
        self._data_port: EmulatedDataPort | None = None
        self._cli: EmulatedCliPort | None = None

    def __call__(
        self, name: str, baudrate: int, timeout_s: float
    ) -> EmulatedCliPort | EmulatedDataPort:
        port: EmulatedCliPort | EmulatedDataPort
        if name == self.cli_port:
            port = EmulatedCliPort(
                data_port=self._data_port,
                timeout_s=min(timeout_s, 0.02),
                reject_prefix=self.reject_prefix,
                **self.cli_options,  # type: ignore[arg-type]
            )
            self._cli = port
        else:
            port = EmulatedDataPort(
                self.data,
                bytes_per_second=self.bytes_per_second,
                timeout_s=min(timeout_s, 0.02),
                streaming=not self.wait_for_start,
            )
            port.baudrate = baudrate
            if self._data_port is None:
                self._data_port = port
                if self._cli is not None:
                    self._cli.data_port = port
        self.opened[name] = port
        return port
