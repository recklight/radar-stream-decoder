"""End to end on synthetic data: render, damage, record, decode, export, report.

    python examples/decode_synthetic_capture.py --out examples/output/demo

1. Render an xWR16xx scene (``--platform 14xx`` for an xWR14xx one) with
   people walking and static reflectors as out-of-box demo packets, and
   damage the stream the way a real link does (junk, fake sync words, drops,
   packets cut short).
2. Write it as a capture file next to the radar configuration it was rendered
   with.
3. Decode the file in chunks, de-duplicate, timestamp and examine it.
4. Export the point table (CSV), the stacked arrays (NPZ), one summary line
   per frame (JSON Lines) and the report (JSON).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from radar_stream_decoder.capture import analyze_capture
from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.export import frame_summary_rows, save_npz, write_jsonl, write_points_csv
from radar_stream_decoder.logging_utils import configure_logging
from radar_stream_decoder.sensor_config import (
    PRESET_CLI_CONFIGS,
    default_preset_for,
    load_cli_config,
)
from radar_stream_decoder.synthetic import (
    FAULT_PRESETS,
    PacketFate,
    SyntheticScene,
    synthetic_capture,
)

LOGGER = logging.getLogger("decode_synthetic_capture")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("examples/output/synthetic_capture"))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--faults", default="heavy", choices=sorted(FAULT_PRESETS))
    parser.add_argument("--platform", default="16xx", choices=("14xx", "16xx"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    configure_logging("INFO")

    scene = SyntheticScene(platform=args.platform, seed=args.seed)
    stream = synthetic_capture(scene, args.frames, FAULT_PRESETS[args.faults])
    args.out.mkdir(parents=True, exist_ok=True)
    capture = args.out / "capture.bin"
    capture.write_bytes(stream.data)
    config_file = args.out / "radar.cfg"
    config_file.write_text(PRESET_CLI_CONFIGS[default_preset_for(scene.platform)], encoding="utf-8")
    LOGGER.info(
        "wrote %d bytes: %s",
        len(stream.data),
        ", ".join(f"{fate.value} {stream.count(fate)}" for fate in PacketFate),
    )

    sensor = load_cli_config(config_file, platform=scene.platform)
    analysis = analyze_capture(capture, DecoderConfig(), sensor)
    for line in analysis.report.summary_lines():
        LOGGER.info("%s", line)

    points = write_points_csv(args.out / "points.csv", analysis.frames, analysis.timestamps_s)
    keys = save_npz(args.out / "arrays.npz", analysis.frames, analysis.timestamps_s)
    write_jsonl(
        args.out / "frames.jsonl", frame_summary_rows(analysis.frames, analysis.timestamps_s)
    )
    (args.out / "report.json").write_text(
        json.dumps(analysis.report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    clean = sum(1 for frame in analysis.frames if not frame.has_errors)
    LOGGER.info(
        "%d frames (%d clean), %d points -> %s; arrays: %s",
        len(analysis.frames),
        clean,
        points,
        args.out,
        ", ".join(keys),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
