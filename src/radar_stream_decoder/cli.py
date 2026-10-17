"""Command-line interface.

Commands
--------
``inspect``
    Forensic report of a capture: packets, repeats, drops, restarts, frame
    period, skipped bytes, schemas, TLV types and issues.
``decode``
    Decode a capture into a point table and, for tracking firmware, a target
    table (CSV), stacked arrays (NPZ) and per-frame records (JSON Lines), plus
    the forensic report.
``sensor-config``
    Radar parameters derived from a TI CLI ``.cfg`` file or a built-in preset,
    under selectable FMCW conventions.
``synth``
    Write a synthetic capture, optionally damaged, with a matching ``.cfg``.
``evaluate``
    Decoder robustness (sensitivity, specificity, UAR, precision, confusion
    matrix) and estimator accuracy on synthetic streams.
``stream``
    Decode live from the serial ports, or from a synthetic stand-in.
``viz``
    Launch the PySide6 viewer.

Failures that are expected in normal use (a missing file, an invalid
configuration, a missing optional dependency, a serial port that cannot be
opened or read, a sensor that rejects or ignores a configuration command) end
with a one-line message and exit code 2. The traceback is logged at DEBUG
level, so ``--log-level DEBUG`` shows it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer

from .config import DecoderConfig
from .errors import ConfigError, DecodeError
from .logging_utils import configure_logging
from .schema import PacketSchema
from .sensor_config import (
    PRESET_CLI_CONFIGS,
    FmcwConvention,
    SensorConfig,
    default_preset_for,
    load_cli_config,
    preset_sensor_config,
)
from .serial_link import DataPortError, SensorCommandError

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Decode TI mmWave radar UART streams: captures, live ports and synthetic data.",
)

SCHEMA_HELP = "Packet schema: auto, " + ", ".join(s.value for s in PacketSchema) + "."
FAULT_NAMES = ("none", "light", "heavy", "block-dump", "bitflip")

SchemaOption = typer.Option("auto", "--schema", help=SCHEMA_HELP)
PlatformOption = typer.Option(
    None, "--platform", help="Platform: 14xx, 16xx, 18xx, 68xx or a code such as 0xA1642."
)
SensorOption = typer.Option(
    None, "--sensor-config", "-s", help="TI CLI .cfg file, for physical units."
)
ConfigOption = typer.Option(None, "--config", "-c", help="Decoder settings (JSON object).")
StrictOption = typer.Option(
    None, "--strict/--lenient", help="Reject or salvage damaged packets (default: lenient)."
)
DedupeOption = typer.Option(
    None, "--dedupe/--no-dedupe", help="Drop repeated frames (default: on)."
)


@app.callback()
def _configure(
    log_level: str = typer.Option("WARNING", "--log-level", help="DEBUG, INFO, WARNING or ERROR."),
) -> None:
    try:
        configure_logging(log_level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from None


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


@contextmanager
def _reported_errors() -> Iterator[None]:
    """Turn the library's expected failures into a one-line message and exit code 2."""
    try:
        yield
    except (
        ConfigError,
        DecodeError,
        ValueError,
        ImportError,
        OSError,  # missing files, and pyserial's SerialException (an OSError)
        SensorCommandError,
        DataPortError,
    ) as exc:
        _LOGGER.debug("command failed", exc_info=True)
        raise _fail(str(exc) or type(exc).__name__) from None


def _decoder_config(
    config_path: Path | None,
    schema: str | None = None,
    platform: str | None = None,
    strict: bool | None = None,
    dedupe: bool | None = None,
) -> DecoderConfig:
    config = DecoderConfig.from_json(config_path) if config_path is not None else DecoderConfig()
    changes: dict[str, Any] = {}
    if schema is not None and schema != "auto":
        changes["schema"] = schema
    if platform is not None:
        changes["platform"] = platform
    if strict is not None:
        changes["strict"] = strict
    if dedupe is not None:
        changes["deduplicate"] = dedupe
    return config.replace(**changes) if changes else config


def _sensor(path: Path | None, platform: str | None) -> SensorConfig | None:
    if path is None:
        return None
    return load_cli_config(path, platform)


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


@app.command("inspect")
def inspect_command(
    capture: Path = typer.Argument(..., help="Capture file of raw data-port bytes."),
    schema: str = SchemaOption,
    platform: str | None = PlatformOption,
    sensor_config: Path | None = SensorOption,
    config: Path | None = ConfigOption,
    strict: bool | None = StrictOption,
    dedupe: bool | None = DedupeOption,
    json_out: Path | None = typer.Option(None, "--json", help="Also write the report as JSON."),
) -> None:
    """Print a forensic report of a capture file."""
    from .capture import analyze_capture

    with _reported_errors():
        decoder = _decoder_config(config, schema, platform, strict, dedupe)
        sensor = _sensor(sensor_config, platform)
        analysis = analyze_capture(capture, decoder, sensor)
    typer.echo(f"capture               {capture}")
    for line in analysis.report.summary_lines():
        typer.echo(line)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(analysis.report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        typer.echo(f"report written to {json_out}")


@app.command("decode")
def decode_command(
    capture: Path = typer.Argument(..., help="Capture file of raw data-port bytes."),
    out: Path = typer.Option(..., "--out", "-o", help="Output folder."),
    formats: str = typer.Option(
        "csv,npz,jsonl", "--format", "-f", help="Comma-separated subset of csv, npz, jsonl."
    ),
    schema: str = SchemaOption,
    platform: str | None = PlatformOption,
    sensor_config: Path | None = SensorOption,
    config: Path | None = ConfigOption,
    strict: bool | None = StrictOption,
    dedupe: bool | None = DedupeOption,
) -> None:
    """Decode a capture into CSV / NPZ / JSONL files."""
    from .capture import analyze_capture
    from .export import (
        frame_summary_rows,
        save_npz,
        stats_rows,
        vital_signs_rows,
        write_jsonl,
        write_points_csv,
        write_targets_csv,
    )

    wanted = {item.strip().lower() for item in formats.split(",") if item.strip()}
    unknown = wanted - {"csv", "npz", "jsonl"}
    if unknown or not wanted:
        raise _fail(f"--format takes csv, npz and/or jsonl, got {formats!r}")
    with _reported_errors():
        decoder = _decoder_config(config, schema, platform, strict, dedupe)
        sensor = _sensor(sensor_config, platform)
        if sensor is None:
            _LOGGER.warning("no --sensor-config: velocities and decibel scales may be unknown")
        analysis = analyze_capture(capture, decoder, sensor)
        out.mkdir(parents=True, exist_ok=True)
        frames, times = analysis.frames, analysis.timestamps_s
        written: list[str] = []
        if "csv" in wanted:
            count = write_points_csv(out / "points.csv", frames, times)
            written.append(f"points.csv ({count} points)")
            if any(frame.targets is not None for frame in frames):
                count = write_targets_csv(out / "targets.csv", frames, times)
                written.append(f"targets.csv ({count} target rows)")
        if "jsonl" in wanted:
            write_jsonl(out / "frames.jsonl", frame_summary_rows(frames, times))
            written.append(f"frames.jsonl ({len(frames)} frames)")
            stats = stats_rows(frames, times)
            if stats:
                write_jsonl(out / "stats.jsonl", stats)
                written.append(f"stats.jsonl ({len(stats)} records)")
            vitals = vital_signs_rows(frames, times)
            if vitals:
                write_jsonl(out / "vital_signs.jsonl", vitals)
                written.append(f"vital_signs.jsonl ({len(vitals)} records)")
        if "npz" in wanted:
            keys = save_npz(out / "arrays.npz", frames, times)
            written.append(f"arrays.npz ({', '.join(keys)})")
        report_path = out / "report.json"
        report_path.write_text(
            json.dumps(analysis.report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        written.append("report.json")
    typer.echo(
        f"{analysis.report.unique_frames} frames decoded from {capture.name} "
        f"({analysis.report.duplicates} repeats dropped, drop ratio "
        f"{analysis.report.drop_ratio:.1%})"
    )
    for item in written:
        typer.echo(f"  {out / item.split(' ')[0]}  {item.partition(' ')[2]}".rstrip())


@app.command("sensor-config")
def sensor_config_command(
    config_file: Path | None = typer.Argument(None, help="TI CLI .cfg file."),
    preset: str | None = typer.Option(
        None, "--preset", help="Built-in configuration: " + ", ".join(sorted(PRESET_CLI_CONFIGS))
    ),
    platform: str | None = PlatformOption,
    wavelength_at: str = typer.Option(
        "start",
        "--wavelength-at",
        help="Frequency of the velocity wavelength: start (TI mmWave Demo Visualizer) or "
        "center (f0 + B/2 + adcStart*slope, the SDK 3.5 firmware).",
    ),
    max_range_factor: float = typer.Option(
        0.8,
        "--max-range-factor",
        help="Share of the IF-bandwidth range limit reported as maximum range (0.8: TI's "
        "visualiser; 1.0: the limit itself).",
    ),
    doppler_padding: bool = typer.Option(
        True,
        "--doppler-padding/--no-doppler-padding",
        help="Velocity resolution from the zero-padded Doppler FFT bins, or from the chirps "
        "actually sent.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of a table."),
) -> None:
    """Show the radar parameters derived from a .cfg file or a preset.

    The convention_* entries state the conventions the figures were derived under.
    """
    if (config_file is None) == (preset is None):
        raise _fail("give either a .cfg file or --preset")
    with _reported_errors():
        convention = FmcwConvention(
            wavelength_reference=wavelength_at.strip().lower(),
            max_range_factor=max_range_factor,
            doppler_zero_padding=doppler_padding,
        )
        sensor = (
            load_cli_config(config_file, platform, convention)
            if config_file is not None
            else preset_sensor_config(preset or "").with_convention(convention)
        )
        summary = sensor.summary()
    if as_json:
        typer.echo(json.dumps(summary, indent=2))
        return
    width = max(len(name) for name in summary)
    for name, value in summary.items():
        typer.echo(f"{name:<{width}}  {_format_value(value)}")


@app.command("synth")
def synth_command(
    out: Path = typer.Argument(..., help="Capture file to write (a .cfg is written next to it)."),
    schema: str = typer.Option("oob-sdk2", "--schema", help=SCHEMA_HELP.replace("auto, ", "")),
    platform: str = typer.Option("16xx", "--platform", help="14xx, 16xx, 18xx or 68xx."),
    frames: int = typer.Option(200, "--frames", min=1, help="Frames to render."),
    seed: int = typer.Option(0, "--seed", help="Seed of the scene and the faults."),
    faults: str = typer.Option("none", "--faults", help="Fault preset: " + ", ".join(FAULT_NAMES)),
    padding_byte: int = typer.Option(0, "--padding-byte", min=0, max=255, help="Padding value."),
    sensor_config: Path | None = SensorOption,
) -> None:
    """Write a synthetic capture with a known scene and optional damage."""
    from .synthetic import FAULT_PRESETS, PacketFate, SyntheticScene, synthetic_capture

    if faults not in FAULT_PRESETS:
        raise _fail(f"--faults must be one of {', '.join(FAULT_PRESETS)}, got {faults!r}")
    with _reported_errors():
        sensor = _sensor(sensor_config, platform)
        scene = SyntheticScene(
            schema=schema, platform=platform, seed=seed, padding_byte=padding_byte, sensor=sensor
        )
        stream = synthetic_capture(scene, frames, FAULT_PRESETS[faults], seed + 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(stream.data)
        config_path = out.with_suffix(".cfg")
        if sensor_config is not None:
            config_text = sensor_config.read_text(encoding="utf-8")
        elif scene.schema is PacketSchema.VITAL_SIGNS:
            config_text = PRESET_CLI_CONFIGS["iwr1443_vital_signs"]
        else:
            config_text = PRESET_CLI_CONFIGS[default_preset_for(scene.platform)]
        if config_path != out:
            config_path.write_text(config_text, encoding="utf-8")
    fates = ", ".join(f"{fate.value} {stream.count(fate)}" for fate in PacketFate)
    typer.echo(f"wrote {len(stream.data)} bytes to {out} ({scene.schema.value}, {fates})")
    typer.echo(f"radar configuration written to {config_path}")


@app.command("evaluate")
def evaluate_command(
    schema: str = typer.Option("oob-sdk2", "--schema", help=SCHEMA_HELP.replace("auto, ", "")),
    platform: str = typer.Option("16xx", "--platform", help="14xx, 16xx, 18xx or 68xx."),
    frames: int = typer.Option(300, "--frames", min=10, help="Frames per trial."),
    trials: int = typer.Option(5, "--trials", min=1, help="Repetitions with different seeds."),
    seed: int = typer.Option(0, "--seed", help="First seed."),
    faults: list[str] | None = typer.Option(
        None, "--faults", help="Fault preset to evaluate (repeatable; default: all but none)."
    ),
    strict: bool | None = StrictOption,
    estimator_frames: int = typer.Option(
        150, "--estimator-frames", min=0, help="Frames for estimator accuracy (0 skips it)."
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the full results as JSON."),
) -> None:
    """Score decoder robustness and estimator accuracy on synthetic streams."""
    from .evaluation import evaluate_estimators, evaluate_recovery
    from .synthetic import FAULT_PRESETS, SyntheticScene

    names = faults or [name for name in FAULT_NAMES if name != "none"]
    bad = [name for name in names if name not in FAULT_PRESETS]
    if bad:
        raise _fail(
            f"unknown fault preset(s) {', '.join(bad)}; choose from {', '.join(FAULT_NAMES)}"
        )
    results: dict[str, Any] = {"schema": schema, "platform": platform, "frames": frames}
    with _reported_errors():
        decoder = _decoder_config(None, None, None, strict, None)
        scene = SyntheticScene(schema=schema, platform=platform, seed=seed)
        typer.echo(
            f"stream recovery: {scene.schema.value} on {scene.radar().platform_name}, "
            f"{trials} trial(s) x {frames} frames"
        )
        typer.echo(
            f"{'faults':<11}{'sensitivity':>16}{'specificity':>16}{'UAR':>16}{'precision':>16}"
        )
        recovery = []
        for name in names:
            report = evaluate_recovery(
                scene,
                FAULT_PRESETS[name],
                plan_name=name,
                n_frames=frames,
                n_trials=trials,
                seed=seed,
                config=decoder,
            )
            summary = report.summary()
            recovery.append(summary)
            cells = []
            for metric in ("sensitivity", "specificity", "uar", "precision"):
                stat = summary[metric]
                cells.append(
                    "n/a" if stat["mean"] is None else f"{stat['mean']:.3f} +/- {stat['std']:.3f}"
                )
            typer.echo(f"{name:<11}" + "".join(f"{cell:>16}" for cell in cells))
        typer.echo("confusion matrices [[TP FN] [FP TN]] summed over trials:")
        for summary in recovery:
            typer.echo(f"  {summary['faults']:<11}{summary['confusion_matrix']}")
        results["recovery"] = recovery
        if estimator_frames and scene.schema is PacketSchema.OOB_SDK2:
            accuracy = evaluate_estimators(scene, estimator_frames, decoder)
            corrected = evaluate_estimators(
                scene, estimator_frames, decoder.replace(snr_log_bias_correction=True)
            )
            results["estimators"] = accuracy.as_dict()
            results["estimators_log_bias_corrected"] = corrected.as_dict()
            typer.echo(f"estimator accuracy over {estimator_frames} undamaged frames:")
            typer.echo(
                f"  SNR bias / RMSE        {accuracy.snr_bias_db:+.2f} / "
                f"{accuracy.snr_rmse_db:.2f} dB  (log-bias corrected: "
                f"{corrected.snr_bias_db:+.2f} / {corrected.snr_rmse_db:.2f} dB, "
                f"n={accuracy.snr_samples})"
            )
            typer.echo(
                f"  azimuth RMSE / max     {accuracy.azimuth_rmse_deg:.3f} / "
                f"{accuracy.azimuth_max_error_deg:.3f} deg (n={accuracy.azimuth_samples})"
            )
            typer.echo(f"  range RMSE             {accuracy.range_rmse_m * 1e3:.2f} mm")
            typer.echo(f"  velocity RMSE          {accuracy.velocity_rmse_mps:.3f} m/s")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"results written to {out}")


@app.command("stream")
def stream_command(
    data_port: str | None = typer.Option(None, "--data-port", help="Data port, e.g. COM4."),
    cli_port: str | None = typer.Option(
        None, "--cli-port", help="CLI port; with --sensor-config the sensor is configured first."
    ),
    sensor_config: Path | None = SensorOption,
    baudrate: int = typer.Option(921_600, "--baudrate", help="Data-port baud rate."),
    schema: str = SchemaOption,
    platform: str | None = PlatformOption,
    config: Path | None = ConfigOption,
    record: Path | None = typer.Option(None, "--record", help="Save the raw bytes as received."),
    points_csv: Path | None = typer.Option(None, "--points-csv", help="Write the points here."),
    targets_csv: Path | None = typer.Option(
        None, "--targets-csv", help="Write the tracker targets here."
    ),
    duration: float | None = typer.Option(None, "--duration", min=0.1, help="Seconds to run."),
    max_frames: int | None = typer.Option(None, "--max-frames", min=1, help="Frames to decode."),
    demo: bool = typer.Option(False, "--demo", help="Use a synthetic stand-in for the sensor."),
    paced: bool = typer.Option(False, "--paced", help="With --demo: deliver frames in real time."),
) -> None:
    """Decode live from the radar's serial ports (or a synthetic stand-in).

    Runs until --duration or --max-frames is reached, the source ends, or Ctrl+C;
    the points are written as they arrive, so memory does not grow with the run.
    """
    from .capture import FrameTimestamper, sensor_frame_period_s
    from .export import PointCsvWriter, TargetCsvWriter
    from .sources import FrameStream, RecordingByteSource, SerialByteSource, SyntheticByteSource
    from .synthetic import SyntheticScene

    if not demo and not data_port:
        raise _fail("--data-port is required (or use --demo)")
    with _reported_errors():
        decoder = _decoder_config(config, schema, platform, None, None)
        sensor = _sensor(sensor_config, platform)
        if demo:
            scene = SyntheticScene(
                schema=schema if schema != "auto" else "oob-sdk2",
                platform=platform or "16xx",
                sensor=sensor,
            )
            sensor = sensor or scene.radar()
            source: Any = SyntheticByteSource(scene, max_frames or 200, realtime=paced)
            if record is not None:
                source = RecordingByteSource(source, record)
        else:
            cli_text = (
                sensor_config.read_text(encoding="utf-8") if sensor_config and cli_port else None
            )
            source = SerialByteSource(
                data_port or "",
                baudrate=baudrate,
                cli_port=cli_port,
                cli_text=cli_text,
                record_path=record,
                duration_s=duration,
            )
        stream = FrameStream(source, decoder, sensor)
        timestamp = FrameTimestamper(decoder, frame_period_s=sensor_frame_period_s(sensor))
        writer = PointCsvWriter(points_csv) if points_csv is not None else None
        target_writer = TargetCsvWriter(targets_csv) if targets_csv is not None else None
        frames = points = issues = 0
        interrupted = False
        started = last_report = time.monotonic()
        try:
            for frame in stream:
                time_s = timestamp(frame)
                if writer is not None:
                    writer.write_frame(frame, time_s)
                if target_writer is not None:
                    target_writer.write_frame(frame, time_s)
                frames += 1
                points += frame.num_points
                issues += len(frame.issues)
                now = time.monotonic()
                if now - last_report >= 1.0:
                    _LOGGER.info("%d frames, %d points, %d issues", frames, points, issues)
                    last_report = now
                if max_frames is not None and frames >= max_frames:
                    break
                if duration is not None and now - started >= duration:
                    break
        except KeyboardInterrupt:
            interrupted = True
        finally:
            source.close()
            if writer is not None:
                writer.close()
            if target_writer is not None:
                target_writer.close()
    stats = stream.assembler_stats
    typer.echo(
        f"{frames} frames from {source.description}: {points} points, {issues} issues, "
        f"{stats.bytes_skipped} bytes skipped, {stream.dedupe_stats.duplicates} repeats dropped"
    )
    if interrupted:
        typer.echo("stopped with Ctrl+C")
    if record is not None:
        typer.echo(f"raw bytes recorded to {record}")
    if points_csv is not None:
        typer.echo(f"points written to {points_csv}")
    if targets_csv is not None:
        typer.echo(f"targets written to {targets_csv}")


@app.command("viz")
def viz_command(
    source: str = typer.Option("demo", "--source", help="demo, file or serial."),
    capture: Path | None = typer.Option(None, "--capture", help="Capture file (--source file)."),
    sensor_config: Path | None = SensorOption,
    platform: str | None = PlatformOption,
    schema: str = SchemaOption,
    data_port: str | None = typer.Option(None, "--data-port", help="Data port (--source serial)."),
    cli_port: str | None = typer.Option(
        None,
        "--cli-port",
        help="CLI port (--source serial); with --sensor-config the sensor is configured first.",
    ),
    fps: float = typer.Option(20.0, "--fps", min=1.0, max=120.0, help="Playback rate."),
) -> None:
    """Open the live viewer (needs the gui extra: PySide6 and matplotlib)."""
    if source not in {"demo", "file", "serial"}:
        raise _fail(f"--source must be demo, file or serial, got {source!r}")
    if source == "file" and capture is None:
        raise _fail("--source file needs --capture")
    if source == "serial" and not data_port:
        raise _fail("--source serial needs --data-port")
    with _reported_errors():
        from .viz import launch_viewer

        decoder = _decoder_config(None, schema, platform, None, None)
        sensor = _sensor(sensor_config, platform)
        # As in stream: the .cfg goes to the sensor only through a CLI port.
        cli_text = sensor_config.read_text(encoding="utf-8") if sensor_config and cli_port else None
        code = launch_viewer(
            source=source,
            capture=capture,
            sensor=sensor,
            decoder=decoder,
            data_port=data_port,
            cli_port=cli_port,
            cli_text=cli_text,
            fps=fps,
        )
    raise typer.Exit(code)


def main() -> None:
    """Entry point of the ``radar-stream-decoder`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
