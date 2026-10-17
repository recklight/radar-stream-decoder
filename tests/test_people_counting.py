"""People counting and people tracking formats, against packets packed from TI's C structures.

Reference packets are built with ``struct.pack`` from:

* TI mmWave Industrial Toolbox 4.6.0, 3D People Counting lab --
  ``pcount3D_output.h`` (``Pcount3DDemo_output_message_header``, 48 bytes;
  ``Pcount3DDemo_output_message_point_unit``; ``..._UARTpoint``) and the output
  task in ``mss_main.c`` (TLV lengths include the TL pair, no padding, header
  checksum ``~((sum >> 16) + (sum & 0xFFFF))``); target records of 112 bytes
  (``tid, pos[3], vel[3], acc[3], ec[16], g, confidence``) as the toolbox's
  3D People Counting user guide documents them, and the 40-byte variant of the
  Industrial Toolbox 4.5.1 user guide;
* TI mmWave Industrial Toolbox 2.x, 2D people counting lab -- ``mmw_output.h``
  (52-byte ``MmwDemo_output_message_header``, ``MmwDemo_output_message_point``,
  ``MmwDemo_output_message_target``) and ``task_mbox.c``;
* TI Radar Toolbox people tracking -- the 40-byte header of the out-of-box
  demo, TLV 1020 (compressed points), 1010 (112-byte targets), 1011 (per-point
  target IDs), 1012 (``tid, maxZ, minZ``), 1021 (presence).

Expected values are worked out here from the field definitions.
"""

from __future__ import annotations

import csv
import json
import math
import struct

import numpy as np
import pytest
from typer.testing import CliRunner

from radar_stream_decoder.assembler import FrameAssembler
from radar_stream_decoder.capture import analyze_capture, frame_timestamps
from radar_stream_decoder.cli import app
from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.errors import DecodeError, IssueCode, Severity
from radar_stream_decoder.evaluation import evaluate_recovery
from radar_stream_decoder.export import (
    PointCsvWriter,
    TargetCsvWriter,
    points_to_rows,
    targets_to_rows,
)
from radar_stream_decoder.frame import decode_packet
from radar_stream_decoder.header import HeaderLayout, header_checksum
from radar_stream_decoder.schema import PacketSchema
from radar_stream_decoder.synthetic import (
    FAULT_PRESETS,
    FaultPlan,
    SyntheticScene,
    generate_packets,
    inject_faults,
    synthetic_capture,
)
from radar_stream_decoder.tracking import link_target_associations

MAGIC = struct.pack("<4H", 0x0102, 0x0304, 0x0506, 0x0708)
SDK_3_5_0_4 = 0x03050004
SDK_1_2_0_5 = 0x01020005
RUNNER = CliRunner()


def checksum(header_without_checksum: bytes) -> int:
    """The firmware's header checksum over uint16 words (checksum field taken as zero)."""
    words = struct.unpack(f"<{len(header_without_checksum) // 2}H", header_without_checksum)
    total = sum(words)
    return ~((total >> 16) + (total & 0xFFFF)) & 0xFFFF


def tlv_including_header(tlv_type: int, payload: bytes) -> bytes:
    return struct.pack("<2I", tlv_type, 8 + len(payload)) + payload


UNITS = (0.01, 0.01, 0.00028, 0.00025, 0.04)  # elevation, azimuth, Doppler, range, SNR
#: (elevation, azimuth, doppler, range, snr) integers of three points.
RAW_POINTS = [(3, -4, 120, 9000, 300), (0, 10, -500, 12000, 1000), (-2, 0, 0, 4000, 50)]


def compressed_points() -> bytes:
    return struct.pack("<5f", *UNITS) + b"".join(
        struct.pack("<bbhHH", *point) for point in RAW_POINTS
    )


def expected_points() -> dict[str, np.ndarray]:
    f32 = [float(np.float32(u)) for u in UNITS]
    el = np.array([p[0] for p in RAW_POINTS]) * f32[0]
    az = np.array([p[1] for p in RAW_POINTS]) * f32[1]
    r = np.array([p[3] for p in RAW_POINTS]) * f32[3]
    return {
        "x": r * np.cos(el) * np.sin(az),
        "y": r * np.cos(el) * np.cos(az),
        "z": r * np.sin(el),
        "range": r,
        "doppler": np.array([p[2] for p in RAW_POINTS]) * f32[2],
        "snr": np.array([p[4] for p in RAW_POINTS]) * f32[4],
    }


TARGET_112 = struct.pack(
    "<I9f16f2f",
    7,
    0.5, 2.0, 1.1,  # pos x y z
    0.1, -0.2, 0.0,  # vel
    0.0, 0.01, 0.0,  # acc
    *([0.25] * 16),
    3.0,  # gating gain
    0.9,  # confidence
)  # fmt: skip
TARGET_40 = struct.pack(
    "<I9f", 4, 0.5, 2.0, 0.1, -0.2, 0.0, 0.01, 1.1, 0.03, -0.02
)  # tid, posX, posY, velX, velY, accX, accY, posZ, velZ, accZ (IT 4.5.1 guide)


def packet_3d(
    tlvs: list[tuple[int, bytes]],
    *,
    frame: int = 42,
    subframe: int = 0,
    timing: tuple[int, int, int, int] = (25, 9800, 450, 1200),
) -> bytes:
    body = b"".join(tlv_including_header(t, p) for t, p in tlvs)
    head = MAGIC + struct.pack(
        "<9IH",
        SDK_3_5_0_4,
        48 + len(body),
        0xA6843,
        frame,
        subframe,
        *timing,  # chirp margin, frame processing, tracking, UART time (us)
        len(tlvs),
    )
    return head + struct.pack("<H", checksum(head)) + body


def packet_2d(
    tlvs: list[tuple[int, bytes]], *, frame: int = 100, cycles: int = 10_000_000
) -> bytes:
    body = b"".join(tlv_including_header(t, p) for t, p in tlvs)
    head = MAGIC + struct.pack(
        "<10IH",
        SDK_1_2_0_5,
        0xA1642,  # platform at offset 12
        cycles,  # R4F timestamp
        52 + len(body),
        frame,
        0,
        30,
        41000,
        700,
        2100,
        len(tlvs),
    )
    return head + struct.pack("<H", checksum(head)) + body


def packet_tracking(tlvs: list[tuple[int, bytes]], *, frame: int = 5) -> bytes:
    body = b"".join(struct.pack("<2I", t, len(p)) + p for t, p in tlvs)  # length without TL
    length = 40 + len(body)
    total = 32 * ((length + 31) // 32)
    head = struct.pack(
        "<Q8I", 0x0708050603040102, 0x03060000, total, 0xA6843, frame, 99, 3, len(tlvs), 0
    )
    return head + body + bytes(total - length)


# -- reference packets ------------------------------------------------------------------------


class TestPeopleCounting3d:
    def packet(self) -> bytes:
        return packet_3d(
            [
                (6, compressed_points()),
                (7, TARGET_112),
                (8, bytes([7, 7, 255, 253])),
                (11, struct.pack("<I", 1)),
            ]
        )

    def test_header_fields(self):
        frame = decode_packet(self.packet())
        header = frame.header
        assert frame.schema is PacketSchema.PEOPLE_COUNTING_3D
        assert header.layout is HeaderLayout.CHECKSUMMED_48 and header.header_len == 48
        assert str(header.version) == "3.5.0.4" and header.platform == 0xA6843
        assert (header.frame_number, header.subframe_number) == (42, 0)
        assert header.cpu_cycles is None and header.declared_point_count is None
        timing = header.timing
        assert (timing.chirp_margin_us, timing.frame_processing_time_us) == (25, 9800)
        assert (timing.tracking_time_us, timing.uart_time_us) == (450, 1200)
        assert header.checksum == checksum(self.packet()[:46])
        assert not frame.issues

    def test_points_are_decompressed_and_made_cartesian(self):
        points = decode_packet(self.packet()).points
        expected = expected_points()
        np.testing.assert_allclose(points.x_m, expected["x"], rtol=1e-12)
        np.testing.assert_allclose(points.y_m, expected["y"], rtol=1e-12)
        np.testing.assert_allclose(points.z_m, expected["z"], rtol=1e-12)
        np.testing.assert_allclose(points.range_m, expected["range"], rtol=1e-12)
        np.testing.assert_allclose(points.velocity_mps, expected["doppler"], rtol=1e-12)
        np.testing.assert_allclose(points.snr_linear, expected["snr"], rtol=1e-12)
        assert points.range_m[0] == pytest.approx(2.25, abs=1e-6)  # 9000 x 0.00025 m
        assert points.azimuth_deg[0] == pytest.approx(math.degrees(-0.04), abs=1e-5)

    def test_targets_association_and_presence(self):
        frame = decode_packet(self.packet())
        targets = frame.targets
        assert len(targets) == 1 and targets.record_bytes == 112
        assert targets.target_id.tolist() == [7]
        np.testing.assert_allclose(targets.position_m[0], [0.5, 2.0, 1.1], rtol=1e-6)
        np.testing.assert_allclose(targets.velocity_mps[0], [0.1, -0.2, 0.0], rtol=1e-6)
        np.testing.assert_allclose(targets.acceleration_mps2[0], [0.0, 0.01, 0.0], rtol=1e-6)
        assert targets.covariance.shape == (1, 16)
        assert targets.gating_gain[0] == pytest.approx(3.0) and targets.confidence[0] == (
            pytest.approx(0.9)
        )
        assert frame.target_association.target_id.tolist() == [7, 7, 255, 253]
        assert frame.target_association.associated.tolist() == [True, True, False, False]
        assert frame.presence.occupied

    def test_compact_target_records(self):
        frame = decode_packet(packet_3d([(7, TARGET_40 * 2)]))
        targets = frame.targets
        assert targets.record_bytes == 40 and len(targets) == 2
        np.testing.assert_allclose(targets.position_m[0], [0.5, 2.0, 1.1], rtol=1e-6)
        np.testing.assert_allclose(targets.velocity_mps[0], [0.1, -0.2, 0.03], rtol=1e-6)
        np.testing.assert_allclose(targets.acceleration_mps2[0], [0.0, 0.01, -0.02], atol=1e-7)
        assert targets.covariance is None and targets.confidence is None

    def test_record_size_ambiguity_is_settled_and_reported(self):
        # 560 bytes are 5 x 112 or 14 x 40; all-zero compact records keep "IDs" valid both ways.
        frame = decode_packet(packet_3d([(7, struct.pack("<I9f", 0, *([0.0] * 9)) * 14)]))
        assert frame.targets.record_bytes == 112 and len(frame.targets) == 5
        issue = next(i for i in frame.issues if i.code is IssueCode.AMBIGUOUS_RECORD_SIZE)
        assert issue.severity is Severity.INFO
        # Real 112-byte records do not pass as 40-byte ones: their covariance is no valid ID.
        assert decode_packet(packet_3d([(7, TARGET_112 * 5)])).issues == []

    def test_target_bytes_that_fit_no_record_are_an_error(self):
        frame = decode_packet(packet_3d([(7, bytes(50))]))
        assert frame.targets is None and frame.has_errors


class TestPeopleCounting2d:
    POINTS = struct.pack("<4f", 3.0, 0.2, -0.5, 25.0) + struct.pack("<4f", 1.5, -0.4, 0.25, 9.0)
    TARGET = struct.pack("<I6f9ff", 3, -0.6, 2.9, 0.2, -0.3, 0.0, 0.0, *([0.5] * 9), 2.5)

    def packet(self) -> bytes:
        return packet_2d([(6, self.POINTS), (7, self.TARGET), (8, bytes([3, 255]))])

    def test_header_with_platform_at_offset_12(self):
        packet = self.packet()
        assert struct.unpack_from("<I", packet, 12)[0] == 0xA1642
        frame = decode_packet(packet)
        header = frame.header
        assert frame.schema is PacketSchema.PEOPLE_COUNTING_2D and header.header_len == 52
        assert (header.frame_number, header.cpu_cycles, header.packet_length) == (
            100,
            10_000_000,
            len(packet),
        )
        assert header.timing.frame_margin_us == 41000 and header.timing.uart_time_us == 2100
        assert not frame.issues

    def test_points_and_targets(self):
        frame = decode_packet(self.packet())
        points = frame.points
        np.testing.assert_allclose(points.x_m, [3.0 * math.sin(0.2), 1.5 * math.sin(-0.4)], 1e-6)
        np.testing.assert_allclose(points.y_m, [3.0 * math.cos(0.2), 1.5 * math.cos(-0.4)], 1e-6)
        np.testing.assert_array_equal(points.z_m, [0.0, 0.0])
        np.testing.assert_allclose(points.velocity_mps, [-0.5, 0.25])
        np.testing.assert_allclose(points.snr_linear, [25.0, 9.0])
        targets = frame.targets
        assert targets.record_bytes == 68 and targets.target_id.tolist() == [3]
        np.testing.assert_allclose(targets.position_m[0], [-0.6, 2.9, 0.0], rtol=1e-6)
        assert targets.covariance.shape == (1, 9) and targets.confidence is None
        assert frame.target_association.target_id.tolist() == [3, 255]

    def test_empty_tlvs_decode_to_empty_payloads(self):
        frame = decode_packet(packet_2d([(6, b""), (7, b""), (8, b"")]))
        assert frame.num_points == 0 and frame.num_targets == 0
        assert frame.targets.covariance.shape == (0, 9)
        assert len(frame.target_association) == 0 and not frame.has_errors

    def test_timestamps_are_200_mhz_r4f_cycles(self):
        packets = [packet_2d([], frame=n, cycles=10_000_000 * n) for n in range(1, 5)]
        frames = [decode_packet(p) for p in packets]
        np.testing.assert_allclose(frame_timestamps(frames), [0.0, 0.05, 0.1, 0.15])


class TestPeopleTracking:
    def packet(self) -> bytes:
        heights = struct.pack("<I2f", 7, 1.8, 0.1) + struct.pack("<I2f", 9, 1.0, 0.0)
        return packet_tracking(
            [
                (1020, compressed_points()),
                (1010, TARGET_112),
                (1011, bytes([7, 255])),
                (1012, heights),
                (1021, struct.pack("<I", 0)),
            ]
        )

    def test_reference_packet(self):
        frame = decode_packet(self.packet(), DecoderConfig(strict=True))
        assert frame.schema is PacketSchema.PEOPLE_TRACKING
        assert frame.header.header_len == 40 and frame.header.packet_length % 32 == 0
        expected = expected_points()
        np.testing.assert_allclose(frame.points.y_m, expected["y"], rtol=1e-12)
        assert frame.targets.target_id.tolist() == [7]
        # Heights are matched to the target list by target ID.
        assert frame.targets.max_z_m[0] == pytest.approx(1.8)
        assert frame.targets.min_z_m[0] == pytest.approx(0.1)
        assert not frame.presence.occupied
        # The height of target 9 names a target the list does not hold.
        assert [issue.code for issue in frame.issues] == [IssueCode.ASSOCIATION_MISMATCH]

    def test_not_mistaken_for_the_out_of_box_demo(self):
        packet = packet_tracking([(1020, compressed_points()), (1010, TARGET_112)])
        assert decode_packet(packet).schema is PacketSchema.PEOPLE_TRACKING


# -- integrity: checksum and length rules --------------------------------------------------------


@pytest.mark.parametrize("build", [packet_3d, packet_2d])
def test_every_header_bit_flip_is_caught(build):
    packet = build([(8, bytes([1, 2, 3]))])
    header_len = 48 if build is packet_3d else 52
    assert (
        header_checksum(packet[:header_len]) == struct.unpack_from("<H", packet, header_len - 2)[0]
    )
    for byte in range(8, header_len):
        for bit in (0, 5):
            damaged = bytearray(packet)
            damaged[byte] ^= 1 << bit
            assembler = FrameAssembler()
            frames = assembler.feed(bytes(damaged)) + assembler.finish()
            assert not [f for f in frames if not f.has_errors], (byte, bit)


def test_tlv_lengths_must_add_up_to_the_packet_length():
    packet = bytearray(packet_3d([(8, bytes(4))]))
    struct.pack_into("<I", packet, 12, len(packet) + 8)  # claim 8 bytes that never follow ...
    struct.pack_into("<H", packet, 46, checksum(bytes(packet[:46])))  # ... with a valid checksum
    longer = bytes(packet) + bytes(8)
    frame = decode_packet(longer)
    assert IssueCode.EXCESS_PADDING in [issue.code for issue in frame.issues]
    with pytest.raises(DecodeError):
        decode_packet(longer, DecoderConfig(strict=True))
    shorter = bytearray(packet_3d([(8, bytes(4))]))
    struct.pack_into("<I", shorter, 52, 20)  # TLV claims more than the packet holds
    frame = decode_packet(bytes(shorter))
    assert IssueCode.TLV_OVERFLOW in [issue.code for issue in frame.issues]


def test_small_timing_words_do_not_make_a_cut_packet_padded():
    # With processing times of a few microseconds the 48-byte header also passes
    # the padded checks (offset 32 reads as a small TLV count). A packet cut short
    # on the link must still be read with its checksummed layout.
    tlvs = [(6, compressed_points()), (7, TARGET_112), (11, struct.pack("<I", 1))]
    packets = [packet_3d(tlvs, frame=n, timing=(1, 0, 2, 1)) for n in (1, 2, 3)]
    cut = packets[1][: 48 + 8 + 20 + 8 + 30]  # inside the target TLV
    assembler = FrameAssembler()
    frames = assembler.feed(packets[0] + cut + packets[2]) + assembler.finish()
    assert [(f.frame_number, f.truncated) for f in frames] == [(1, False), (2, True), (3, False)]
    assert {f.schema for f in frames} == {PacketSchema.PEOPLE_COUNTING_3D}
    assert frames[1].header.layout is HeaderLayout.CHECKSUMMED_48 and frames[1].num_points == 3


def test_forced_schema_selects_the_header_layout():
    packet = packet_3d([(8, bytes(4))])
    assert decode_packet(packet, DecoderConfig(schema="people-counting-3d")).target_association
    with pytest.raises(DecodeError, match="invalid header"):
        decode_packet(packet, DecoderConfig(schema="people-counting-2d"))


# -- linking per-point target IDs to the frame they describe ---------------------------------------


def test_target_ids_belong_to_the_previous_frame():
    first = decode_packet(packet_3d([(6, compressed_points())], frame=10))
    second = decode_packet(
        packet_3d([(6, compressed_points()), (7, TARGET_112), (8, bytes([7, 255, 7]))], frame=11)
    )
    stats = link_target_associations([first, second])
    assert stats.as_dict() == {"linked": 1, "mismatched": 0, "orphaned": 0}
    assert first.points.target_id.tolist() == [7, 255, 7]
    assert second.points.target_id is None  # waits for frame 12
    rows = points_to_rows([first])
    assert [row["target_id"] for row in rows] == [7, 255, 7]


def test_association_mismatch_and_gaps():
    first = decode_packet(packet_3d([(6, compressed_points())], frame=10))
    wrong = decode_packet(packet_3d([(7, TARGET_112), (8, bytes([7, 7]))], frame=11))
    stats = link_target_associations([first, wrong])
    assert stats.mismatched == 1 and first.points.target_id is None
    assert IssueCode.ASSOCIATION_MISMATCH in [issue.code for issue in wrong.issues]
    # analyze_capture links, and so does write_points_csv: the issue is not doubled.
    link_target_associations([first, wrong])
    assert [issue.code for issue in wrong.issues].count(IssueCode.ASSOCIATION_MISMATCH) == 1
    late = decode_packet(packet_3d([(7, TARGET_112), (8, bytes([7, 7, 7]))], frame=13))
    assert link_target_associations([first, late]).orphaned == 1


def test_streaming_writers(tmp_path):
    first = decode_packet(packet_3d([(6, compressed_points())], frame=10))
    second = decode_packet(
        packet_3d([(6, compressed_points()), (7, TARGET_112), (8, bytes([7, 7, 255]))], frame=11)
    )
    with (
        PointCsvWriter(tmp_path / "points.csv") as points,
        TargetCsvWriter(tmp_path / "targets.csv") as targets,
    ):
        for frame, time_s in ((first, 0.0), (second, 0.055)):
            points.write_frame(frame, time_s)
            targets.write_frame(frame, time_s)
    rows = list(csv.DictReader((tmp_path / "points.csv").open(encoding="utf-8")))
    assert [row["target_id"] for row in rows] == ["7", "7", "255", "", "", ""]
    target_rows = list(csv.DictReader((tmp_path / "targets.csv").open(encoding="utf-8")))
    assert len(target_rows) == 1 and target_rows[0]["target_id"] == "7"
    assert float(target_rows[0]["z_m"]) == pytest.approx(1.1)
    assert targets_to_rows([first]) == []


# -- synthetic scenes, faults, reports, command line ----------------------------------------

SCHEMAS = ["people-counting-3d", "people-counting-2d", "people-tracking"]


@pytest.mark.parametrize("schema", SCHEMAS)
def test_synthetic_scenes_decode_cleanly_and_link(schema):
    scene = SyntheticScene(schema=schema, seed=4)
    frames = [decode_packet(p, DecoderConfig(strict=True)) for p in generate_packets(scene, 12)]
    assert {f.schema.value for f in frames} == {schema}
    assert all(f.num_targets > 0 and f.num_points > 0 for f in frames)
    stats = link_target_associations(frames)
    assert stats.linked == 11 and stats.mismatched == 0
    ids = np.concatenate([f.points.target_id for f in frames[:-1]])
    assert set(ids.tolist()) <= set(range(len(scene.targets))) | {255}


@pytest.mark.parametrize("schema", SCHEMAS)
@pytest.mark.parametrize("preset", ["light", "heavy", "block-dump"])
def test_structural_faults_lose_nothing_and_pass_nothing(schema, preset):
    scene = SyntheticScene(schema=schema, seed=6)
    report = evaluate_recovery(scene, FAULT_PRESETS[preset], n_frames=60, n_trials=2, seed=1)
    summary = report.summary()
    assert summary["sensitivity"]["mean"] == 1.0
    if summary["specificity"]["mean"] is not None:
        assert summary["specificity"]["mean"] == 1.0


@pytest.mark.parametrize("schema", ["people-counting-3d", "people-counting-2d"])
def test_checksum_catches_header_bit_flips(schema):
    packets = list(generate_packets(SyntheticScene(schema=schema, seed=2), 40))
    header_len = 48 if schema.endswith("3d") else 52
    damaged = []
    for index, packet in enumerate(packets):
        raw = bytearray(packet)
        if index % 4 == 1:
            raw[8 + (index * 7) % (header_len - 8)] ^= 0x10
        damaged.append(bytes(raw))
    assembler = FrameAssembler()
    frames = assembler.feed(b"".join(damaged)) + assembler.finish()
    assert [f.frame_number for f in frames] == [n + 1 for n in range(40) if n % 4 != 1]
    codes = [issue.code for issue in assembler.stats.issues]
    assert codes.count(IssueCode.HEADER_CHECKSUM_MISMATCH) >= 5


def test_inspect_report_for_a_tracking_capture(tmp_path):
    scene = SyntheticScene(schema="people-counting-3d", seed=8)
    stream = synthetic_capture(scene, 50, FaultPlan(drop_every=10, leading_junk_bytes=11))
    path = tmp_path / "pc3d.bin"
    path.write_bytes(stream.data)
    analysis = analyze_capture(path, sensor=scene.radar())
    report = analysis.report
    assert report.schemas == {"people-counting-3d": 45}
    assert report.timestamp_source == "frame-number"
    assert report.estimated_frame_period_s == pytest.approx(0.1)
    assert report.frames_with_targets == 45 and report.max_targets >= 1
    # Each dropped frame orphans the IDs of the frame after it.
    assert report.association_orphaned == 4 and report.association_mismatched == 0
    np.testing.assert_allclose(np.diff(analysis.timestamps_s)[:8], 0.1)
    lines = "\n".join(report.summary_lines())
    assert "no header timestamp" in lines and "tracker targets" in lines
    assert "target-ID links       linked 40" in lines


@pytest.mark.parametrize("schema", SCHEMAS)
def test_cli_round_trip(schema, tmp_path):
    capture = tmp_path / "capture.bin"
    result = RUNNER.invoke(
        app, ["synth", str(capture), "--schema", schema, "--frames", "30", "--faults", "light"]
    )
    assert result.exit_code == 0, result.output
    config = capture.with_suffix(".cfg")
    out = tmp_path / "out"
    result = RUNNER.invoke(
        app, ["decode", str(capture), "-o", str(out), "--sensor-config", str(config)]
    )
    assert result.exit_code == 0, result.output
    assert "targets.csv" in result.output
    targets = list(csv.DictReader((out / "targets.csv").open(encoding="utf-8")))
    points = list(csv.DictReader((out / "points.csv").open(encoding="utf-8")))
    assert targets and points
    assert any(row["target_id"] not in ("", "255") for row in points)
    summary = json.loads((out / "frames.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["schema"] == schema and summary["num_targets"] > 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["frames_with_targets"] > 0
    inspect = RUNNER.invoke(app, ["inspect", str(capture), "--schema", schema])
    assert inspect.exit_code == 0 and "tracker targets" in inspect.output


def test_stream_demo_writes_targets(tmp_path):
    targets = tmp_path / "targets.csv"
    result = RUNNER.invoke(
        app,
        [
            "stream",
            "--demo",
            "--schema",
            "people-tracking",
            "--max-frames",
            "6",
            "--targets-csv",
            str(targets),
            "--points-csv",
            str(tmp_path / "points.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(targets.open(encoding="utf-8")))
    times = sorted({float(row["timestamp_s"]) for row in rows})
    assert len(times) == 6 and times[1] == pytest.approx(0.1, abs=1e-3)


def test_faults_on_checksummed_packets_keep_their_bookkeeping():
    packets = list(generate_packets(SyntheticScene(schema="people-counting-2d", seed=1), 10))
    faulted = inject_faults(packets, FaultPlan(truncate_probability=1.0), 3)
    assembler = FrameAssembler(DecoderConfig(strict=True))
    assert assembler.feed(faulted.data) + assembler.finish() == []


def test_tracking_scene_platforms():
    assert SyntheticScene(schema="people-counting-3d", platform="16xx").platform == 0xA6843
    assert SyntheticScene(schema="people-tracking").platform == 0xA6843
    assert SyntheticScene(schema="people-counting-2d").platform == 0xA1642
