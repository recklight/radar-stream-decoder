"""Vital Signs summary layout, range segments and gap-aware phase unwrapping."""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar_stream_decoder.constants import Platform
from radar_stream_decoder.encoder import HeaderTemplate, encode_frame, encode_range_segment
from radar_stream_decoder.errors import IssueCode
from radar_stream_decoder.frame import decode_packet
from radar_stream_decoder.synthetic import ChestMotion, SyntheticScene, generate_packets
from radar_stream_decoder.vital_signs import (
    VITAL_SUMMARY_DTYPE,
    RangeSegment,
    chest_displacement_track,
    phase_displacement_mm,
    unwrap_runs,
)


def test_summary_layout_is_128_bytes():
    assert VITAL_SUMMARY_DTYPE.itemsize == 128
    assert VITAL_SUMMARY_DTYPE.fields["first_range_bin"][1] == 12
    assert VITAL_SUMMARY_DTYPE.fields["chest_displacement_mm"][1] == 16
    assert VITAL_SUMMARY_DTYPE.fields["reserved"][1] == 96


def test_synthetic_packets_have_the_lab_structure(packets_vital):
    frame = decode_packet(packets_vital[0])
    assert len(packets_vital[0]) == 288
    assert frame.header.header_len == 40 and frame.header.declared_point_count == 99
    assert frame.schema.value == "vital-signs" and frame.tlv_types == (0, 1)
    summary = frame.vital_signs
    assert (summary.first_range_bin, summary.last_range_bin) == (8, 29)
    assert summary.segment_bins == 22 == len(frame.range_segment)
    assert not summary.motion_detected
    assert 60 < summary.heart_rate_fft_bpm < 85 and 10 < summary.breathing_rate_fft_bpm < 20
    assert set(summary.as_dict()) >= {"heart_rate_fft_bpm", "motion_detected"}


def test_range_segment_and_the_unwritten_last_bin(packets_vital):
    segment = decode_packet(packets_vital[0]).range_segment
    assert segment.start_bin == 8 and segment.bins[-1] == 29
    assert not segment.last_bin_written
    assert segment.peak_bin() == 18
    assert segment.value_at(18) == segment.values[10]
    with pytest.raises(IndexError):
        segment.value_at(40)
    written = SyntheticScene(schema="vital-signs", seed=2, zero_last_segment_bin=False)
    assert decode_packet(next(generate_packets(written, 1))).range_segment.last_bin_written


def test_segment_without_summary_has_no_start():
    template = HeaderTemplate.for_schema("vital-signs", Platform.XWR14XX)
    frame = decode_packet(encode_frame(template, [(1, encode_range_segment(np.ones(4)))]))
    # Without the type-0 summary the packet is not recognisable as the lab output.
    assert frame.schema.value == "oob-sdk2"
    from radar_stream_decoder.config import DecoderConfig

    forced = decode_packet(
        encode_frame(template, [(1, encode_range_segment(np.ones(4)))]),
        DecoderConfig(schema="vital-signs"),
    )
    assert forced.range_segment.start_bin is None
    assert IssueCode.MISSING_RANGE_START in [issue.code for issue in forced.issues]
    with pytest.raises(ValueError):
        _ = forced.range_segment.bins


def test_segment_length_must_match_the_summary(packets_vital):
    frame = decode_packet(packets_vital[0])
    template = HeaderTemplate.for_schema("vital-signs", Platform.XWR14XX)
    from radar_stream_decoder.encoder import encode_vital_summary

    packet = encode_frame(
        template,
        [(0, encode_vital_summary(frame.vital_signs)), (1, encode_range_segment(np.ones(10)))],
    )
    broken = decode_packet(packet)
    assert broken.has_errors
    assert IssueCode.LENGTH_MISMATCH in [issue.code for issue in broken.issues]


def test_unwrap_runs_restarts_at_gaps():
    true_phase = np.linspace(0, 12 * math.pi, 40)
    wrapped = np.angle(np.exp(1j * true_phase))
    frames = np.arange(40)
    unwrapped, runs = unwrap_runs(wrapped, frames)
    np.testing.assert_allclose(unwrapped, true_phase, atol=1e-9)
    assert runs.max() == 0
    gapped = np.r_[np.arange(20), np.arange(25, 45)]
    _, runs = unwrap_runs(wrapped, gapped)
    assert runs.tolist() == [0] * 20 + [1] * 20
    with pytest.raises(ValueError):
        unwrap_runs([0.0, 1.0], [1])
    empty, empty_runs = unwrap_runs([], [])
    assert empty.size == 0 and empty_runs.size == 0


def test_phase_displacement_follows_the_chest():
    scene = SyntheticScene(
        schema="vital-signs", seed=7, chest=ChestMotion(breathing_amplitude_mm=3.0)
    )
    frames = [decode_packet(p) for p in generate_packets(scene, 200)]
    sensor = scene.radar()
    track = chest_displacement_track(frames, sensor=sensor)
    truth = np.array([f.vital_signs.chest_displacement_mm for f in frames])
    estimate = track.displacement_mm - track.displacement_mm.mean()
    reference = truth - truth.mean()
    assert track.num_runs == 1
    assert np.sqrt(np.mean((estimate - reference) ** 2)) < 0.05  # millimetres
    # The same samples with every other frame missing cannot be unwrapped across the gaps.
    sparse = frames[::2]
    gapped = phase_displacement_mm(
        [f.range_segment for f in sparse], 18, sensor.wavelength_m, [f.frame_number for f in sparse]
    )
    assert gapped.num_runs == len(sparse)
    assert np.ptp(gapped.displacement_mm) <= sensor.wavelength_m * 1e3 / 2 + 1e-9


def test_displacement_input_validation(packets_vital):
    segments = [decode_packet(p).range_segment for p in packets_vital[:3]]
    with pytest.raises(ValueError):
        phase_displacement_mm(segments, 18, 0.0, [1, 2, 3])
    with pytest.raises(ValueError):
        phase_displacement_mm(segments, 18, 0.004, [1, 2])
    with pytest.raises(ValueError):
        chest_displacement_track([decode_packet(p) for p in packets_vital[:3]])
    with pytest.raises(ValueError):
        RangeSegment(np.zeros(0, dtype=np.complex64), 3).peak_bin()
