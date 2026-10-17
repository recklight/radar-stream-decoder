"""People counting and tracking firmware, end to end on synthetic data.

    python examples/people_counting_capture.py --schema people-counting-3d \
        --out examples/output/pc3d

1. Render a scene as the chosen tracking firmware sends it: a compressed (3D
   lab, Radar Toolbox) or polar (2D lab) point cloud, the tracker's target
   list, the previous frame's per-point target IDs and the presence word,
   behind a checksummed 48- or 52-byte header or the padded 40-byte one.
2. Damage the stream (junk, fake sync words, drops, packets cut short) and
   flip a bit in the header of every tenth packet: the header checksum of the
   people counting labs catches what the padded framing cannot.
3. Decode the capture, attach every frame's target IDs to the frame they
   describe, and export the point table (with the target of every point) and
   the target table.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from radar_stream_decoder.capture import analyze_capture
from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.errors import IssueCode
from radar_stream_decoder.export import write_points_csv, write_targets_csv
from radar_stream_decoder.logging_utils import configure_logging
from radar_stream_decoder.schema import TRACKING_SCHEMAS
from radar_stream_decoder.sensor_config import PRESET_CLI_CONFIGS, default_preset_for
from radar_stream_decoder.synthetic import (
    FAULT_PRESETS,
    SyntheticScene,
    generate_packets,
    inject_faults,
)

LOGGER = logging.getLogger("people_counting_capture")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--schema",
        default="people-counting-3d",
        choices=sorted(schema.value for schema in TRACKING_SCHEMAS),
    )
    parser.add_argument("--out", type=Path, default=Path("examples/output/people_counting"))
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    configure_logging("INFO")

    scene = SyntheticScene(schema=args.schema, seed=args.seed)
    packets = list(generate_packets(scene, args.frames))
    header_len = {"people-counting-3d": 48, "people-counting-2d": 52}.get(args.schema, 40)
    stream = inject_faults(packets, FAULT_PRESETS["heavy"], args.seed + 1)
    data = bytearray(stream.data)
    for index in range(5, len(packets), 10):
        position = data.find(packets[index])
        if position >= 0:
            data[position + 20] ^= 0x04  # frame number (2D lab: packet length)

    args.out.mkdir(parents=True, exist_ok=True)
    capture = args.out / "capture.bin"
    capture.write_bytes(bytes(data))
    config_file = args.out / "radar.cfg"
    config_file.write_text(PRESET_CLI_CONFIGS[default_preset_for(scene.platform)], encoding="utf-8")

    analysis = analyze_capture(capture, DecoderConfig(), scene.radar())
    for line in analysis.report.summary_lines():
        LOGGER.info("%s", line)
    counts = analysis.report.issue_counts
    checksum_failures = counts.get(IssueCode.HEADER_CHECKSUM_MISMATCH.value, 0)
    LOGGER.info(
        "%d-byte headers; %d packets with a damaged header refused by %s",
        header_len,
        checksum_failures,
        "the checksum" if header_len > 40 else "the structural checks",
    )

    points = write_points_csv(args.out / "points.csv", analysis.frames, analysis.timestamps_s)
    targets = write_targets_csv(args.out / "targets.csv", analysis.frames, analysis.timestamps_s)
    LOGGER.info(
        "%d frames: %d points and %d target rows written to %s",
        len(analysis.frames),
        points,
        targets,
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
