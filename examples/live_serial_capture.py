"""Live capture from a TI evaluation module, with a hardware-free dry run.

With hardware (needs pyserial: ``pip install -e ".[serial]"`` in the project folder)::

    python examples/live_serial_capture.py --data-port COM4 --cli-port COM3 \\
        --sensor-config configs/iwr1642_oob_2tx4rx.cfg --record session.bin --duration 10

Without hardware::

    python examples/live_serial_capture.py --dry-run --duration 3

The dry run replaces pyserial with emulated ports: the CLI port answers like
the SDK 3.5 out-of-box demo's command line (echo, prompt, ``Done``, the
calibration message on the first start) and the data port starts serving a
synthetic stream, paced at the 921600-baud line rate, once ``sensorStart`` is
acknowledged. Only the ports are emulated: the configuration handshake, the
background reader thread, the raw recording and the incremental decoding run
the same code as with a real sensor.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.export import write_points_csv
from radar_stream_decoder.logging_utils import configure_logging
from radar_stream_decoder.port_emulator import EmulatedPortFactory
from radar_stream_decoder.sensor_config import PRESET_CLI_CONFIGS, load_cli_config, parse_cli_config
from radar_stream_decoder.serial_link import DATA_PORT_BAUDRATE
from radar_stream_decoder.sources import FrameStream, SerialByteSource
from radar_stream_decoder.synthetic import SyntheticScene, generate_packets

LOGGER = logging.getLogger("live_serial_capture")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-port", default=None, help="data port, e.g. COM4")
    parser.add_argument("--cli-port", default=None, help="CLI port, e.g. COM3")
    parser.add_argument(
        "--sensor-config", type=Path, default=None, help="radar configuration to send"
    )
    parser.add_argument("--platform", default="16xx")
    parser.add_argument("--record", type=Path, default=None, help="raw recording")
    parser.add_argument("--points", type=Path, default=None, help="point table (CSV)")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds to capture")
    parser.add_argument("--dry-run", action="store_true", help="use emulated ports, no hardware")
    args = parser.parse_args(argv)
    configure_logging("INFO")

    factory = None
    if args.dry_run:
        # A lean output selection, as a real 20 fps session over UART would use:
        # full heatmaps every frame would exceed the 92 kB/s the data port carries.
        scene = SyntheticScene(seed=5, outputs=("points", "range_profile", "stats"))
        data = b"".join(generate_packets(scene, 400))
        factory = EmulatedPortFactory(
            data, cli_port="CLI", bytes_per_second=DATA_PORT_BAUDRATE / 10
        )
        data_port, cli_port = "DATA", "CLI"
        config_text = PRESET_CLI_CONFIGS["iwr1642_oob_2tx4rx"]
        sensor = parse_cli_config(config_text, platform=args.platform)
    else:
        if not args.data_port:
            parser.error("--data-port is required without --dry-run")
        data_port, cli_port = args.data_port, args.cli_port
        config_text = args.sensor_config.read_text(encoding="utf-8") if args.sensor_config else None
        sensor = (
            load_cli_config(args.sensor_config, platform=args.platform)
            if args.sensor_config
            else None
        )

    source = SerialByteSource(
        data_port,
        cli_port=cli_port,
        cli_text=config_text if cli_port else None,
        record_path=args.record,
        duration_s=args.duration,
        serial_factory=factory,
    )
    stream = FrameStream(source, DecoderConfig(), sensor)
    frames = list(stream)
    LOGGER.info(
        "%d frames, %d points, %d bytes received, %d skipped",
        len(frames),
        sum(frame.num_points for frame in frames),
        source.bytes_read,
        stream.assembler_stats.bytes_skipped,
    )
    if factory is not None:
        commands = factory.opened["CLI"].written.decode("ascii").splitlines()
        LOGGER.info("configuration handshake: %d commands acknowledged", len(commands))
    if args.points is not None:
        LOGGER.info("%d points written to %s", write_points_csv(args.points, frames), args.points)
    if args.record is not None:
        LOGGER.info("raw bytes recorded to %s", args.record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
