"""Stream decoding from a byte source, the way a live application would.

    python examples/stream_from_synthetic_source.py --frames 200 --paced

A :class:`~radar_stream_decoder.sources.SyntheticByteSource` stands in for the
sensor: it delivers packets in random-sized chunks with light link faults,
optionally paced at the frame rate. A
:class:`~radar_stream_decoder.sources.FrameStream` assembles and decodes them
incrementally. The loop below logs progress every 50 frames and keeps the
fastest radial speed seen; reading a serial port looks the same, with a
:class:`~radar_stream_decoder.sources.SerialByteSource` as the source.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from radar_stream_decoder.logging_utils import configure_logging
from radar_stream_decoder.sources import FrameStream, SyntheticByteSource
from radar_stream_decoder.synthetic import FAULT_PRESETS, SyntheticScene

LOGGER = logging.getLogger("stream_from_synthetic_source")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--paced", action="store_true", help="deliver frames in real time")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    configure_logging("INFO")

    scene = SyntheticScene(seed=args.seed)
    sensor = scene.radar()
    source = SyntheticByteSource(
        scene, args.frames, faults=FAULT_PRESETS["light"], max_chunk=2048, realtime=args.paced
    )
    stream = FrameStream(source, sensor=sensor)
    point_counts = []
    fastest = 0.0
    for frame in stream:
        point_counts.append(frame.num_points)
        points = frame.points
        if points is not None and len(points):
            moving = np.abs(points.velocity_mps) > 0.0
            if np.any(moving):
                fastest = max(fastest, float(np.max(np.abs(points.velocity_mps[moving]))))
        if len(point_counts) % 50 == 0:
            LOGGER.info(
                "frame %d: %d points, %d issues so far",
                frame.frame_number,
                frame.num_points,
                stream.assembler_stats.issue_count,
            )
    stats = stream.assembler_stats
    LOGGER.info(
        "%d frames, %.1f points per frame, fastest radial speed %.2f m/s "
        "(unambiguous limit %.2f m/s); %d bytes skipped, %d packets salvaged",
        len(point_counts),
        float(np.mean(point_counts)) if point_counts else 0.0,
        fastest,
        sensor.max_velocity_mps,
        stats.bytes_skipped,
        stats.frames_salvaged,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
