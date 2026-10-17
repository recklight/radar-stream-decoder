"""SDK 3.x out-of-box demo packets built from TI's firmware, not from this package's encoder.

Every packet here is packed with ``struct.pack`` field by field from the C
structures and output routine of TI mmWave SDK 03.05.00.04:

* ``demo/xwr68xx/mmw/include/mmw_output.h`` -- ``MmwDemo_output_message_header``
  (40 bytes) and ``MmwDemo_output_message_tl`` (length without the TL pair);
* ``demo/xwr68xx/mmw/mss/mss_main.c`` -- ``MmwDemo_transmitProcessedOutput``:
  ``totalPacketLen = 32 * ((packetLen + 31) / 32)``, the padding sent from an
  uninitialised local array;
* ``common/mmwave_sdk_version.h`` -- version 3.5.0.4.

Expected values come from the firmware's algorithms computed independently
with NumPy (``DPU_DopplerProcDSP_log2AbsAccum`` in ``dopplerprocdsp.c``,
``mmwavelib_log2Abs32`` in ``mmwavelib_log2abs.c``, ``mmwavelib_accum16shift``
in ``mmwavelib_accum.c``), never by running this package's encoder backwards.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from radar_stream_decoder.assembler import FrameAssembler
from radar_stream_decoder.config import DecoderConfig
from radar_stream_decoder.errors import IssueCode, Severity, UnsupportedFormatError
from radar_stream_decoder.frame import decode_packet
from radar_stream_decoder.schema import PacketSchema
from radar_stream_decoder.sensor_config import parse_cli_config

MAGIC = struct.pack("<4H", 0x0102, 0x0304, 0x0506, 0x0708)
SDK_3_5_0_4 = (3 << 24) | (5 << 16) | (0 << 8) | 4
XWR68XX = 0xA6843

#: 3 TX x 4 RX on an IWR6843 (12 virtual antennas), 256 ADC samples -> 256 range bins.
CONFIG_3TX = """\
channelCfg 15 7 0
profileCfg 0 60 7 6 60 0 0 60 1 256 5000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 16 0 100 1 0
"""


def oob_sdk3_packet(
    tlvs: list[tuple[int, bytes]],
    *,
    frame: int = 1,
    points: int = 0,
    padding: bytes | None = None,
    version: int = SDK_3_5_0_4,
) -> bytes:
    """One packet exactly as MmwDemo_transmitProcessedOutput lays it out."""
    body = b"".join(struct.pack("<2I", tlv_type, len(data)) + data for tlv_type, data in tlvs)
    length = 40 + len(body)
    total = 32 * ((length + 31) // 32)
    header = MAGIC + struct.pack(
        "<8I", version, total, XWR68XX, frame, 0x00C0FFEE, points, len(tlvs), 0
    )
    pad = bytes(total - length) if padding is None else padding[: total - length]
    assert len(pad) == total - length
    return header + body + pad


def firmware_log2_word(magnitudes: np.ndarray) -> int:
    """What DPU_DopplerProcDSP_log2AbsAccum writes for one cell over all virtual antennas."""
    q8 = np.rint(np.log2(magnitudes) * 256).astype(np.int64)  # mmwavelib_log2Abs32 output, Q8
    shift = math.ceil(math.log2(magnitudes.size))  # log2VirtAnt = ceil(log2(numVirtualAntennas))
    total = 0
    for value in q8:  # first antenna: x >> shift; the others: accum16shift, saturating at 0xFFFF
        total = min(total + (int(value) >> shift), 0xFFFF)
    return total


class TestLog2MagnitudeScale:
    """SDK 3.x range/noise profiles and the detection matrix are Q8 shifted sums."""

    def test_reference_word_for_equal_antennas(self):
        # Every one of 12 antennas at log2|X| = 15: Q8 3840, shifted by 4 -> 240, times 12.
        word = firmware_log2_word(np.full(12, 2.0**15))
        assert word == 2880
        packet = oob_sdk3_packet([(2, struct.pack("<256H", *([word] * 256)))])
        frame = decode_packet(packet, sensor=parse_cli_config(CONFIG_3TX, "68xx"))
        assert frame.schema is PacketSchema.OOB_SDK3
        # 20 log10(2) x 15 = 90.309 dB per antenna (a Q9 sum, as TI's documents describe
        # it, would give 2.82 dB).
        np.testing.assert_allclose(frame.range_profile.to_db(), 90.30899869919435, rtol=1e-12)

    def test_unequal_antennas_against_the_true_mean_level(self):
        magnitudes = 1000.0 * 2.0 ** (np.arange(12) / 12.0)
        word = firmware_log2_word(magnitudes)
        assert word == 1996
        rows = np.full((256, 16), word, dtype="<u2")
        packet = oob_sdk3_packet(
            [(2, rows[:, 0].tobytes()), (3, rows[:, 7].tobytes()), (5, rows.tobytes())]
        )
        frame = decode_packet(packet, sensor=parse_cli_config(CONFIG_3TX, "68xx"))
        true_mean_db = float(np.mean(20.0 * np.log10(magnitudes)))  # 62.759 dB
        expected = word * 16 / (256 * 12) * 20 * math.log10(2)  # 62.589 dB
        assert frame.range_profile.to_db()[0] == pytest.approx(expected, rel=1e-12)
        assert frame.noise_profile.to_db()[0] == pytest.approx(expected, rel=1e-12)
        assert frame.range_doppler.to_db()[10, 3] == pytest.approx(expected, rel=1e-12)
        # The firmware's truncation costs less than 2^4 Q8 LSB per antenna.
        assert abs(frame.range_profile.to_db()[0] - true_mean_db) < 16 / 256 * 6.03

    @pytest.mark.parametrize(("tx_mask", "chirps", "antennas"), [(1, 1, 4), (3, 2, 8)])
    def test_fewer_transmitters(self, tx_mask, chirps, antennas):
        text = CONFIG_3TX.replace("channelCfg 15 7 0", f"channelCfg 15 {tx_mask} 0")
        if chirps < 3:
            text = text.replace("chirpCfg 2 2 0 0 0 0 0 4\n", "")
        if chirps < 2:
            text = text.replace("chirpCfg 1 1 0 0 0 0 0 2\n", "")
        text = text.replace("frameCfg 0 2", f"frameCfg 0 {chirps - 1}")
        sensor = parse_cli_config(text, "68xx")
        assert sensor.num_virtual_antennas == antennas
        word = firmware_log2_word(np.full(antennas, 2.0**15))
        assert word == 3840  # 4 and 8 are powers of two: the shift undoes the sum exactly
        packet = oob_sdk3_packet([(2, struct.pack("<256H", *([word] * 256)))])
        profile = decode_packet(packet, sensor=sensor).range_profile
        np.testing.assert_allclose(profile.to_db(), 90.30899869919435, rtol=1e-12)


class TestPadding:
    """SDK 3.x padding is an uninitialised buffer, not evidence of damage."""

    STATS = (6, struct.pack("<6I", 1200, 300, 48000, 20, 35, 12))

    def test_uneven_padding_is_information_only(self):
        garbage = bytes([0x13, 0x37, 0x00, 0xA5, 0x5A, 0xFF, 0x01] * 5)
        packet = oob_sdk3_packet([self.STATS], padding=garbage)
        frame = decode_packet(packet, DecoderConfig(strict=True))
        assert [issue.code for issue in frame.issues] == [IssueCode.UNEVEN_PADDING]
        assert frame.issues[0].severity is Severity.INFO and not frame.has_errors

    def test_truncation_is_still_found_by_length_and_next_sync_word(self):
        packets = [oob_sdk3_packet([self.STATS], frame=n) for n in (1, 2, 3)]
        cut = packets[1][: len(packets[1]) - 20]  # lost on the link, next packet follows
        assembler = FrameAssembler()
        frames = assembler.feed(packets[0] + cut + packets[2]) + assembler.finish()
        assert [f.frame_number for f in frames] == [1, 2, 3]
        assert frames[1].truncated and frames[1].has_errors
        assert not frames[0].issues and not frames[2].issues

    def test_sdk2_padding_keeps_its_integrity_check(self):
        packet = bytearray(oob_sdk3_packet([self.STATS], version=0x02000004))
        packet[-3:] = b"\x13\x37\x99"
        frame = decode_packet(bytes(packet))
        assert IssueCode.CORRUPT_PADDING in [issue.code for issue in frame.issues]
        assert frame.has_errors


class TestFormatDetection:
    """An SDK 3.x version word does not make a packet the out-of-box demo."""

    @staticmethod
    def people_counting_3d_packet(frame: int) -> bytes:
        """A 48-byte-header packet as the 3D People Counting lab sends it (pcount3D_output.h)."""
        units = struct.pack("<5f", 0.01, 0.01, 0.00028, 0.00025, 0.04)
        points = b"".join(struct.pack("<bbhHH", 3, -4, 120, 9000 + 37 * k, 300) for k in range(5))
        tlv6 = struct.pack("<2I", 6, 8 + len(units) + len(points)) + units + points
        header = MAGIC + struct.pack(
            "<9IH", SDK_3_5_0_4, 48 + len(tlv6), XWR68XX, frame, 0, 12, 3500, 600, 900, 1
        )
        words = struct.unpack("<23H", header)  # checksum field still zero
        total = sum(words)
        return header + struct.pack("<H", ~((total >> 16) + (total & 0xFFFF)) & 0xFFFF) + tlv6

    def test_3d_people_counting_is_not_taken_for_the_sdk3_demo(self):
        stream = b"".join(self.people_counting_3d_packet(n) for n in (1, 2, 3))
        assembler = FrameAssembler()
        frames = assembler.feed(stream) + assembler.finish()
        assert [f.frame_number for f in frames] == [1, 2, 3]
        assert {f.schema for f in frames} == {PacketSchema.PEOPLE_COUNTING_3D}
        assert assembler.stats.bytes_skipped == 0 and not any(f.issues for f in frames)

    def test_unknown_sdk3_firmware_is_reported_not_forced(self):
        packet = oob_sdk3_packet([(301, bytes(64)), (306, bytes(40))])
        with pytest.raises(UnsupportedFormatError, match="unsupported format") as caught:
            decode_packet(packet)
        assert caught.value.issues[0].code is IssueCode.UNSUPPORTED_FORMAT
        assembler = FrameAssembler()
        assert assembler.feed(packet) + assembler.finish() == []
        codes = [issue.code for issue in assembler.stats.issues]
        assert IssueCode.UNSUPPORTED_FORMAT in codes

    def test_a_forced_schema_still_decodes_it(self):
        packet = oob_sdk3_packet([(301, bytes(64)), (1, struct.pack("<4f", 0.5, 2.0, 0.0, 0.1))])
        frame = decode_packet(packet, DecoderConfig(schema="oob-sdk3"))
        assert frame.schema is PacketSchema.OOB_SDK3 and frame.num_points == 1
        assert [tlv.tlv_type for tlv in frame.unknown_tlvs] == [301]

    def test_a_broken_checksum_is_rejected(self):
        packet = bytearray(self.people_counting_3d_packet(7))
        packet[46] ^= 0x01
        assembler = FrameAssembler()
        assert assembler.feed(bytes(packet)) + assembler.finish() == []
        assert IssueCode.HEADER_CHECKSUM_MISMATCH in [i.code for i in assembler.stats.issues]


class TestDocumentedFieldTypes:
    """Where TI's documents and the SDK 3.5 firmware disagree, the firmware wins."""

    def test_side_info_is_signed(self):
        # DPIF_PointCloudSideInfo: int16_t snr, int16_t noise, 0.1 dB (dpif_pointcloud.h).
        points = struct.pack("<4f", 0.5, 2.0, 0.1, -0.25)
        side = struct.pack("<2h", -15, 412)
        frame = decode_packet(oob_sdk3_packet([(1, points), (7, side)], points=1))
        assert frame.points.snr_db[0] == pytest.approx(-1.5)
        assert frame.points.noise_db[0] == pytest.approx(41.2)

    def test_temperature_status_and_readings_are_signed(self):
        # MmwDemo_temperatureStats: int32_t tempReportValid (0 = valid), then rlRfTempData_t:
        # rlUInt32_t time (ms) and ten rlInt16_t readings in degrees Celsius.
        report = struct.pack("<iI10h", -1, 5000, -5, -4, -3, -2, 40, 41, 42, 30, 45, 46)
        frame = decode_packet(oob_sdk3_packet([(9, report)]))
        temperature = frame.temperature
        assert temperature.status == -1 and not temperature.valid
        assert temperature.rx_c == (-5, -4, -3, -2) and temperature.digital_c == (45, 46)

    def test_detected_objects_one_means_points_and_side_info(self):
        # mmw_config.h: 1 = points and side info, 2 = points only (TI pages say "set to 2").
        base = CONFIG_3TX + "guiMonitor -1 {} 0 0 0 0 0\n"
        assert parse_cli_config(base.format(1), "68xx").outputs.point_side_info
        assert not parse_cli_config(base.format(2), "68xx").outputs.point_side_info

    def test_header_is_forty_bytes(self):
        # Field sum of MmwDemo_output_message_header; one TI page says 44 bytes.
        frame = decode_packet(oob_sdk3_packet([(6, bytes(24))]))
        assert frame.header.header_len == 40 and frame.stats is not None
