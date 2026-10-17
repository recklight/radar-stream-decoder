"""Vital Signs capture forensics on a synthetic block-dump recording.

    python examples/vital_signs_forensics.py

Some recording tools write their whole working buffer after every read
instead of the bytes they received: each packet then appears twice, the
rest of every block is zero, and a slow reader loses most frames. This
example renders a Vital Signs lab stream (288-byte packets), records it that
way, and shows how the decoder

* recognises the firmware family and its 40-byte header on its own,
* drops the byte-identical repeats and measures the frame loss,
* recovers the frame period from the wrapping 32-bit cycle counter, and
* refuses to unwrap the chest-wall phase across frame gaps, while a gap-free
  recording gives a displacement that follows the firmware's own estimate.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

from radar_stream_decoder.capture import analyze_capture
from radar_stream_decoder.logging_utils import configure_logging
from radar_stream_decoder.synthetic import FAULT_PRESETS, SyntheticScene, synthetic_capture
from radar_stream_decoder.vital_signs import chest_displacement_track

LOGGER = logging.getLogger("vital_signs_forensics")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args(argv)
    configure_logging("INFO")

    scene = SyntheticScene(schema="vital-signs", seed=args.seed)
    sensor = scene.radar()
    with tempfile.TemporaryDirectory() as folder:
        dumped = Path(folder) / "block_dump.bin"
        dumped.write_bytes(synthetic_capture(scene, args.frames, FAULT_PRESETS["block-dump"]).data)
        clean = Path(folder) / "clean.bin"
        clean.write_bytes(synthetic_capture(scene, args.frames).data)
        damaged = analyze_capture(dumped, sensor=sensor)
        intact = analyze_capture(clean, sensor=sensor)

    LOGGER.info("block-dump recording:")
    for line in damaged.report.summary_lines():
        LOGGER.info("  %s", line)
    gaps = chest_displacement_track(damaged.frames, sensor=sensor)
    LOGGER.info(
        "phase unwrapping split the block-dump recording into %d gap-free runs", gaps.num_runs
    )

    track = chest_displacement_track(intact.frames, sensor=sensor)
    firmware = np.array([frame.vital_signs.chest_displacement_mm for frame in intact.frames])
    estimate = track.displacement_mm - track.displacement_mm.mean()
    reference = firmware - firmware.mean()
    correlation = float(np.corrcoef(estimate, reference)[0, 1])
    LOGGER.info(
        "gap-free recording: %d frames in %d run, phase displacement vs firmware estimate: "
        "r = %.3f, RMS difference %.3f mm (wavelength %.2f mm)",
        len(intact.frames),
        track.num_runs,
        correlation,
        float(np.sqrt(np.mean((estimate - reference) ** 2))),
        sensor.wavelength_m * 1e3,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
