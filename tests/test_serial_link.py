"""Serial handshake, the firmware CLI emulator and background reading (no pyserial, no hardware).

The emulator's expected bytes below are written from the CLI sources of TI mmWave
SDK 03.05.00.04 (``utils/cli/src/cli.c``, ``drivers/uart/src/uartsci.c``,
``demo/xwr68xx/mmw/mss/mmw_cli.c``), not from what the handshake code expects.
"""

from __future__ import annotations

import importlib.util
import threading
import time
import types
import zlib

import pytest
from typer.testing import CliRunner

from radar_stream_decoder import serial_link, sources
from radar_stream_decoder.cli import app
from radar_stream_decoder.port_emulator import (
    EmulatedCliPort,
    EmulatedDataPort,
    EmulatedPortFactory,
)
from radar_stream_decoder.sensor_config import PRESET_CLI_CONFIGS
from radar_stream_decoder.serial_link import (
    DataPortError,
    DataPortReader,
    SensorCommandError,
    cli_commands,
    open_port,
    require_pyserial,
    send_cli_config,
)
from radar_stream_decoder.sources import FrameStream, SerialByteSource

RUNNER = CliRunner()
PROMPT = b"mmwDemo:/>"


def drain(port: EmulatedCliPort) -> bytes:
    return port.read(1 << 16)


# -- the emulator, byte by byte ----------------------------------------------------------


def test_power_up_sends_the_banner_then_a_prompt_without_newline():
    port = EmulatedCliPort(timeout_s=0.001)
    assert drain(port) == (
        b"******************************************\n\r"
        b"xWR68xx MMW Demo 03.05.00.04\n\r"
        b"******************************************\n\r" + PROMPT
    )


def test_a_command_is_echoed_and_answered_with_lf_cr_line_ends():
    port = EmulatedCliPort(timeout_s=0.001, banner=False)
    drain(port)
    port.write(b"channelCfg 15 7 0\n")
    # Echo exactly as received (LF without CR), then Done with LF followed by CR.
    assert drain(port) == b"channelCfg 15 7 0\n" + b"Done\n\r" + PROMPT


def test_carriage_return_is_echoed_as_cr_lf_and_ends_the_line():
    port = EmulatedCliPort(timeout_s=0.001, banner=False)
    drain(port)
    port.write(b"flushCfg\r")
    assert drain(port) == b"flushCfg\r\n" + b"Done\n\r" + PROMPT
    # CR LF from the host is two line ends: the LF runs an empty line, which
    # only earns another prompt.
    port.write(b"flushCfg\r\n")
    assert drain(port) == b"flushCfg\r\n" + b"Done\n\r" + PROMPT + b"\n" + PROMPT


def test_comment_and_unknown_command():
    port = EmulatedCliPort(timeout_s=0.001, banner=False)
    drain(port)
    port.write(b"% a comment\n")
    assert drain(port) == b"% a comment\n" + b"Skipped\n\r" + PROMPT
    port.write(b"sensorWarp 9\n")
    reply = drain(port)
    expected = b"sensorWarp 9\n" + b"'sensorWarp' is not recognized as a CLI command\n\r"
    assert reply == expected + PROMPT
    assert b"Done" not in reply and b"Error" not in reply


def test_sensor_state_messages():
    data = EmulatedDataPort(b"xyz", streaming=False, timeout_s=0.001)
    port = EmulatedCliPort(data_port=data, timeout_s=0.001, banner=False)
    drain(port)
    port.write(b"sensorStop\n")
    stopped = b"Ignored: Sensor is already stopped\n\rDone\n\r"
    assert drain(port) == b"sensorStop\n" + stopped + PROMPT
    port.write(b"sensorStart\n")
    first_start = drain(port)
    assert b"Error: Full configuration must be provided" in first_start
    assert first_start.endswith(b"Error -1\n\r" + PROMPT)
    assert data.read(10) == b""  # the sensor did not start
    for command in ("channelCfg 15 7 0", "profileCfg 0", "chirpCfg 0", "frameCfg 0"):
        port.write(command.encode() + b"\n")
    drain(port)
    port.write(b"sensorStart\n")
    assert drain(port) == (
        b"sensorStart\n" + b"Debug: Init Calibration Status = 0x1ffe\n\rDone\n\r" + PROMPT
    )
    assert port.state == "started" and data.read(10) == b"xyz"
    port.write(b"sensorStart\n")
    assert b"Ignored: Sensor is already started\n\rDone" in drain(port)
    port.write(b"sensorStop\nsensorStart 0\n")
    restart = drain(port)
    assert b"Debug:" not in restart and restart.count(b"Done") == 2
    port.write(b"sensorStart 1\n")
    assert b"Error: Reconfig is not supported" in drain(port)
    port.write(b"queryDemoStatus\n")
    assert b"Sensor State: 2\n\rData port baud rate: 921600\n\rDone" in drain(port)


def test_first_start_without_reconfiguration_is_refused():
    # MmwDemo_CLISensorStart checks "sensorStart 0" against the pending configuration
    # in every state: after a complete configuration it must be a plain sensorStart.
    data = EmulatedDataPort(b"xyz", streaming=False, timeout_s=0.001)
    port = EmulatedCliPort(data_port=data, timeout_s=0.001, banner=False)
    for command in ("channelCfg 15 7 0", "profileCfg 0", "chirpCfg 0", "frameCfg 0"):
        port.write(command.encode() + b"\n")
    drain(port)
    port.write(b"sensorStart 0\n")
    reply = drain(port)
    assert b"Error: You have provided complete new configuration" in reply
    assert reply.endswith(b"Error -1\n\r" + PROMPT)
    assert port.state == "init" and data.read(10) == b""
    port.write(b"sensorStart 00\n")  # atoi("00") == 0: the same request
    assert b"complete new configuration" in drain(port)


def test_config_data_port_changes_the_line_rate_and_pings():
    data = EmulatedDataPort(b"", timeout_s=0.001)
    port = EmulatedCliPort(data_port=data, timeout_s=0.001, banner=False)
    drain(port)
    port.write(b"configDataPort 3125000 1\n")
    assert b"Data port baud rate changed to 3125000\n\rDone" in drain(port)
    assert data.sensor_baudrate == 3_125_000
    data.baudrate = 3_125_000
    assert data.read(64) == b"\xff" * 16
    port.write(b"configDataPort 4000000 0\n")
    assert b"Ignored: Invalid baud rate (4000000) specified" in drain(port)
    port.write(b"configDataPort\n")
    assert b"Error: Invalid usage of the CLI command\n\rError -1" in drain(port)


def test_data_at_the_wrong_line_rate_arrives_as_zeros():
    data = EmulatedDataPort(b"\x02\x01\x04\x03", timeout_s=0.001)
    data.sensor_baudrate = 3_125_000
    assert data.read(4) == bytes(4)


# -- the handshake ----------------------------------------------------------------------------


def test_cli_commands_strip_comments_and_blank_lines():
    text = "% header\n\nsensorStop\nchannelCfg 15 3 0   % note\n   \nsensorStart\n"
    assert cli_commands(text) == [(3, "sensorStop"), (4, "channelCfg 15 3 0"), (6, "sensorStart")]


def test_handshake_sends_every_line():
    port = EmulatedCliPort(timeout_s=0.001)
    replies = send_cli_config(port, PRESET_CLI_CONFIGS["iwr1642_oob_2tx4rx"])
    commands = cli_commands(PRESET_CLI_CONFIGS["iwr1642_oob_2tx4rx"])
    assert len(replies) == len(commands)
    assert port.written.decode("ascii").splitlines() == [command for _, command in commands]
    assert all(reply.splitlines()[-1] == "Done" for reply in replies)
    assert replies[0] == "Ignored: Sensor is already stopped\nDone"
    assert replies[-1] == "Debug: Init Calibration Status = 0x1ffe\nDone"


def test_rejected_command_names_its_line_and_code():
    port = EmulatedCliPort(reject_prefix="frameCfg", timeout_s=0.001)
    with pytest.raises(SensorCommandError) as caught:
        send_cli_config(port, PRESET_CLI_CONFIGS["iwr1642_oob_2tx4rx"])
    error = caught.value
    assert error.command.startswith("frameCfg") and error.line == 15
    assert error.code == -1 and "Error -1" in error.response and "line 15" in str(error)


def test_incomplete_configuration_is_rejected_at_the_first_start():
    port = EmulatedCliPort(timeout_s=0.001)
    with pytest.raises(SensorCommandError) as caught:
        send_cli_config(port, "sensorStop\nflushCfg\nchannelCfg 15 3 0\nsensorStart\n")
    assert caught.value.line == 4 and caught.value.code == -1
    assert "Full configuration must be provided" in caught.value.response
    # The message joins the two reply lines, so the command line prints one line.
    assert len(str(caught.value).splitlines()) == 1 and "; Error -1" in str(caught.value)


def test_unknown_command_fails_at_once_instead_of_timing_out():
    port = EmulatedCliPort(timeout_s=0.001)
    ticks = iter(range(1000))
    with pytest.raises(SensorCommandError, match="does not know") as caught:
        send_cli_config(
            port, "sensorStop\nsensorWarp 9\n", timeout_s=500, clock=lambda: next(ticks)
        )
    assert caught.value.line == 2 and "not recognized" in caught.value.response


def test_stale_replies_are_cleared_before_the_handshake():
    port = EmulatedCliPort(reject_prefix="channelCfg", timeout_s=0.001)
    port.write(b"sensorStop\n")  # a reply ending in Done is left unread
    with pytest.raises(SensorCommandError) as caught:
        send_cli_config(port, "channelCfg 15 3 0\n")
    assert caught.value.code == -1  # the stale Done did not acknowledge the command


def test_silent_sensor_times_out():
    class Silent:
        def write(self, data):
            return len(data)

        def readline(self):
            return b""

    ticks = iter(range(100))
    with pytest.raises(SensorCommandError, match="did not acknowledge"):
        send_cli_config(Silent(), [(1, "sensorStop")], timeout_s=3.0, clock=lambda: next(ticks))


def test_prompt_split_across_reads_is_reassembled():
    class Chopped:
        """A port whose readline returns arbitrary fragments, as a timeout does."""

        def __init__(self) -> None:
            self.pieces = [b"\rmmwDemo:", b"/>sensorStop\n", b"Do", b"ne\n\r", b"mmwDemo:/>"]

        def write(self, data):
            return len(data)

        def readline(self):
            return self.pieces.pop(0) if self.pieces else b""

    assert send_cli_config(Chopped(), [(1, "sensorStop")], timeout_s=1.0) == ["Done"]


def test_handshake_switches_the_data_port_line_rate():
    data = EmulatedDataPort(b"", timeout_s=0.001)
    port = EmulatedCliPort(data_port=data, timeout_s=0.001)
    replies = send_cli_config(port, "sensorStop\nconfigDataPort 1843200 1\n", data_port=data)
    assert data.baudrate == data.sensor_baudrate == 1_843_200
    assert replies[1] == "Data port baud rate changed to 1843200\nDone"
    assert data.read(64) == b"\xff" * 16
    # Refused while the sensor runs: the host keeps its line rate.
    port.state = "started"
    send_cli_config(port, "configDataPort 921600 0\n", data_port=data)
    assert data.baudrate == 1_843_200


def test_serial_source_follows_config_data_port(packets_vital):
    data = b"".join(packets_vital)
    factory = EmulatedPortFactory(data, cli_port="CLI")
    text = PRESET_CLI_CONFIGS["iwr1443_vital_signs"].replace(
        "sensorStart\n", "configDataPort 3125000 1\nsensorStart\n"
    )
    source = SerialByteSource("DATA", cli_port="CLI", cli_text=text, serial_factory=factory)
    stream = FrameStream(source)
    frames = []
    for frame in stream:
        frames.append(frame)
        if len(frames) == len(packets_vital):
            source.close()
    assert [f.crc32 for f in frames] == [zlib.crc32(p) for p in packets_vital]
    assert factory.opened["DATA"].baudrate == 3_125_000
    assert stream.assembler_stats.bytes_skipped == 16  # the ack ping


def test_without_following_the_line_rate_nothing_decodes(packets_vital):
    data = EmulatedDataPort(b"".join(packets_vital), timeout_s=0.001)
    port = EmulatedCliPort(data_port=data, timeout_s=0.001)
    send_cli_config(port, "configDataPort 3125000 0\n")  # no data port: host stays at 921600
    received = b"".join(iter(lambda: data.read(4096), b""))
    assert received and received == bytes(len(received))


# -- the data port reader and the byte source ----------------------------------------------------


def test_reader_delivers_and_records_every_byte(tmp_path, packets_vital):
    data = b"".join(packets_vital)
    port = EmulatedDataPort(data, timeout_s=0.001)
    reader = DataPortReader(port, record_path=tmp_path / "raw.bin", read_size=100)
    reader.start()
    received = bytearray()
    for chunk in reader.chunks(poll_s=0.01):
        received += chunk
        if len(received) >= len(data):
            reader.stop()
    assert bytes(received) == data
    assert (tmp_path / "raw.bin").read_bytes() == data
    assert reader.bytes_read == len(data) and reader.stopping


def test_reader_records_into_a_new_folder_and_reports_a_bad_path(tmp_path, packets_vital):
    data = packets_vital[0]
    target = tmp_path / "captures" / "session.bin"
    reader = DataPortReader(EmulatedDataPort(data, timeout_s=0.001), record_path=target)
    reader.start()
    received = bytearray()
    for chunk in reader.chunks(poll_s=0.01):
        received += chunk
        if len(received) >= len(data):
            reader.stop()
    assert target.read_bytes() == data
    # A folder where the file should go cannot be opened for writing.
    blocked = DataPortReader(EmulatedDataPort(data, timeout_s=0.001), record_path=tmp_path)
    blocked.start()
    with pytest.raises(DataPortError):
        list(blocked.chunks(poll_s=0.01))


def test_reader_surfaces_port_errors():
    class Broken:
        in_waiting = 0

        def read(self, size=1):
            raise OSError("device unplugged")

    reader = DataPortReader(Broken())
    reader.start()
    with pytest.raises(DataPortError, match="device unplugged"):
        list(reader.chunks(poll_s=0.01))


def test_serial_source_end_to_end(tmp_path, packets_vital):
    data = b"junk" + b"".join(packets_vital)
    factory = EmulatedPortFactory(data, cli_port="CLI")
    source = SerialByteSource(
        "DATA",
        cli_port="CLI",
        cli_text=PRESET_CLI_CONFIGS["iwr1443_vital_signs"],
        record_path=tmp_path / "session.bin",
        duration_s=1.0,
        serial_factory=factory,
    )
    frames = []
    for frame in FrameStream(source):
        frames.append(frame)
        if len(frames) == len(packets_vital):
            source.close()
    assert [f.crc32 for f in frames] == [zlib.crc32(p) for p in packets_vital]
    assert factory.opened["CLI"].written.startswith(b"sensorStop\n")
    assert factory.opened["CLI"].state == "started"
    assert factory.opened["DATA"].closed
    assert (tmp_path / "session.bin").read_bytes() == data[
        : len((tmp_path / "session.bin").read_bytes())
    ]
    assert source.bytes_read >= len(data) - 1


def test_data_waits_for_sensor_start(packets_vital):
    factory = EmulatedPortFactory(b"".join(packets_vital), cli_port="CLI")
    data = factory("DATA", 921_600, 0.001)
    cli = factory("CLI", 115_200, 0.001)
    assert data.read(64) == b""
    send_cli_config(cli, PRESET_CLI_CONFIGS["iwr1443_vital_signs"])
    assert data.read(8) == packets_vital[0][:8]


def test_duration_ends_the_capture_when_the_port_goes_silent(packets_vital):
    # The data runs dry long before the deadline; the source must still stop at
    # the deadline instead of waiting forever for the next byte.
    source = SerialByteSource(
        "DATA",
        duration_s=0.3,
        serial_factory=EmulatedPortFactory(b"".join(packets_vital[:3])),
    )
    frames: list = []
    worker = threading.Thread(target=lambda: frames.extend(FrameStream(source)), daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(timeout=10.0)
    assert not worker.is_alive(), "the capture did not end at its deadline"
    assert 0.3 <= time.monotonic() - started < 10.0
    assert [f.crc32 for f in frames] == [zlib.crc32(p) for p in packets_vital[:3]]


def test_reader_stop_condition_is_checked_without_data():
    reader = DataPortReader(EmulatedDataPort(b"", timeout_s=0.001))
    reader.start()
    deadline = time.monotonic() + 0.2
    try:
        assert list(reader.chunks(poll_s=0.01, stop_when=lambda: time.monotonic() > deadline)) == []
    finally:
        reader.stop()
    assert not reader.is_alive()


def test_configuration_needs_the_cli_port():
    with pytest.raises(ValueError, match="CLI port"):
        SerialByteSource("DATA", cli_text="sensorStop")


def test_open_port_uses_the_factory():
    port = open_port("X", 921600, serial_factory=EmulatedPortFactory(b"abc"))
    assert port.read(10) == b"abc"


@pytest.mark.skipif(importlib.util.find_spec("serial") is not None, reason="pyserial is installed")
def test_missing_pyserial_is_explained():
    with pytest.raises(ImportError, match=r'pip install -e "\.\[serial\]"'):
        require_pyserial()
    with pytest.raises(ImportError):
        open_port("COM1", 921600)


def _fake_pyserial(opened: list, *, fail: str | None = None) -> types.SimpleNamespace:
    class FakeSerial:
        def __init__(self, port, baudrate, timeout):
            if fail is not None:
                raise OSError(fail)
            self.port, self.baudrate, self.timeout = port, baudrate, timeout
            self.buffer_size = None
            opened.append(self)

        def set_buffer_size(self, rx_size):
            self.buffer_size = rx_size

    return types.SimpleNamespace(Serial=FakeSerial)


def test_open_port_configures_a_real_port(monkeypatch):
    opened: list = []
    monkeypatch.setattr(serial_link, "require_pyserial", lambda: _fake_pyserial(opened))
    port = open_port("COM4", 921_600, timeout_s=0.05)
    assert (port.port, port.baudrate, port.timeout) == ("COM4", 921_600, 0.05)
    assert port.buffer_size == serial_link.RECEIVE_BUFFER_BYTES


def test_data_port_pacing():
    fast = EmulatedDataPort(bytes(1000), bytes_per_second=1e9, timeout_s=0.001)
    received = b""
    deadline = time.monotonic() + 2.0
    while len(received) < 1000 and time.monotonic() < deadline:
        received += fast.read(400)
    assert len(received) == 1000 and fast.exhausted and fast.in_waiting == 0
    slow = EmulatedDataPort(bytes(1000), bytes_per_second=1.0, timeout_s=0.001)
    assert slow.read(400) == b""  # at one byte per second nothing has arrived yet


# -- the command line reports serial failures in one line ----------------------------------------


def _stream(*args: str):
    return RUNNER.invoke(app, ["stream", *args])


def test_cli_reports_a_port_that_cannot_be_opened(monkeypatch):
    monkeypatch.setattr(
        serial_link,
        "require_pyserial",
        lambda: _fake_pyserial([], fail="could not open port 'COM9': FileNotFoundError"),
    )
    result = _stream("--data-port", "COM9", "--duration", "0.2")
    assert result.exit_code == 2
    assert "error: could not open port 'COM9'" in result.output
    assert "Traceback" not in result.output and len(result.output.strip().splitlines()) == 1


def test_cli_reports_a_rejected_configuration(monkeypatch, tmp_path):
    factory = EmulatedPortFactory(b"", cli_port="CLI", reject_prefix="frameCfg")

    def emulated(name, baudrate, timeout_s=0.05, serial_factory=None):
        return factory(name, baudrate, timeout_s)

    monkeypatch.setattr(sources, "open_port", emulated)
    config = tmp_path / "radar.cfg"
    config.write_text(PRESET_CLI_CONFIGS["iwr1642_oob_2tx4rx"], encoding="utf-8")
    result = _stream("--data-port", "DATA", "--cli-port", "CLI", "--sensor-config", str(config))
    assert result.exit_code == 2
    assert "error: line 15: sensor rejected 'frameCfg" in result.output
    assert "Error -1" in result.output  # the reply, joined into the same line
    assert "Traceback" not in result.output and len(result.output.strip().splitlines()) == 1


def test_cli_reports_a_failing_data_port(monkeypatch):
    class Unplugged(EmulatedDataPort):
        def read(self, size=1):
            raise OSError("device unplugged")

    monkeypatch.setattr(
        sources,
        "open_port",
        lambda name, baudrate, timeout_s=0.05, serial_factory=None: Unplugged(b""),
    )
    result = _stream("--data-port", "DATA", "--duration", "2")
    assert result.exit_code == 2
    assert "error: data port reader failed: device unplugged" in result.output
    assert "Traceback" not in result.output and len(result.output.strip().splitlines()) == 1
