"""Display data for the viewer, computed without Qt."""

from __future__ import annotations

import numpy as np
import pytest

from radar_stream_decoder.config import ViewerConfig
from radar_stream_decoder.frame import decode_packet
from radar_stream_decoder.presentation import (
    ColorMode,
    point_scatter,
    prepare_frame_display,
    range_doppler_image,
    range_profile_curve,
)
from radar_stream_decoder.synthetic import SyntheticScene, generate_packets


@pytest.fixture
def frame_and_sensor(scene_16xx):
    sensor = scene_16xx.radar()
    return decode_packet(next(generate_packets(scene_16xx, 1)), sensor=sensor), sensor


def test_physical_units_with_a_configuration(frame_and_sensor):
    frame, sensor = frame_and_sensor
    display = prepare_frame_display(frame, sensor)
    assert display.frame_number == 1 and display.point_count == len(display.scatter.xy)
    assert display.profile.x_label == "距離 (m)" and display.profile.y_label == "強度 (dB)"
    assert display.profile.x[-1] == pytest.approx((sensor.num_range_bins - 1) * sensor.range_bin_m)
    heat = display.heatmap
    assert heat.x_label == "徑向速度 (m/s)" and heat.image.shape == (256, 16)
    assert heat.extent[0] < 0 < heat.extent[1]
    assert display.scatter.color_label == "SNR (dB)" and display.scatter.fixed_limits


def test_bins_without_a_configuration(frame_and_sensor):
    frame, _ = frame_and_sensor
    bare = decode_packet(frame.raw)
    display = prepare_frame_display(bare)
    assert display.profile.x_label == "距離 bin" and "原始值" in display.profile.y_label
    assert display.heatmap.x_label == "Doppler bin"
    # Without a decibel scale there is no SNR; colour falls back to the peak level.
    assert display.scatter.color_label == "峰值強度 (dB)"
    doppler = point_scatter(bare, ColorMode.DOPPLER)
    assert doppler.color_label == "Doppler bin" and doppler.colormap == "coolwarm"


def test_doppler_colouring_uses_velocity(frame_and_sensor):
    frame, _ = frame_and_sensor
    scatter = point_scatter(frame, "doppler")
    assert scatter.color_label == "徑向速度 (m/s)" and not scatter.fixed_limits
    assert scatter.color_limits[0] == -scatter.color_limits[1]


def test_marker_sizes_follow_strength(frame_and_sensor):
    frame, _ = frame_and_sensor
    viewer = ViewerConfig(point_size_range=(10.0, 50.0))
    scatter = point_scatter(frame, viewer=viewer)
    assert scatter.size.min() >= 10.0 and scatter.size.max() <= 50.0
    strongest = int(np.argmax(frame.points.peak))
    assert scatter.size[strongest] == pytest.approx(50.0)


def test_point_cap_keeps_the_strongest(frame_and_sensor):
    frame, _ = frame_and_sensor
    scatter = point_scatter(frame, viewer=ViewerConfig(max_points=2))
    assert len(scatter.xy) == 2
    order = np.argsort(frame.points.peak)[::-1][:2]
    expected = {(frame.points.x_m[i], frame.points.y_m[i]) for i in order}
    assert {tuple(row) for row in scatter.xy} == expected


def test_frames_without_points_or_heatmaps(scene_vital, scene_16xx):
    vital = decode_packet(next(generate_packets(scene_vital, 1)))
    display = prepare_frame_display(vital, scene_vital.radar())
    assert display.point_count == 0 and len(display.scatter.xy) == 0
    assert display.heatmap is None
    assert display.profile is not None and "相對值" in display.profile.y_label
    assert display.profile.x[0] == pytest.approx(8 * scene_vital.radar().range_bin_m)
    sparse = decode_packet(next(generate_packets(scene_16xx, 1, outputs=("points",))))
    assert range_profile_curve(sparse) is None and range_doppler_image(sparse) is None
    empty = decode_packet(next(generate_packets(SyntheticScene(targets=()), 1)))
    assert len(point_scatter(empty, "doppler").xy) == 0


def test_people_counting_points_and_targets():
    scene = SyntheticScene(schema="people-counting-2d", seed=3)
    frame = decode_packet(next(generate_packets(scene, 1)), sensor=scene.radar())
    display = prepare_frame_display(frame, scene.radar())
    assert display.scatter.color_label == "韌體 SNR (10·log10)"
    assert np.all(np.isfinite(display.scatter.color))
    assert display.target_count == frame.num_targets > 0
    np.testing.assert_allclose(display.targets.xy, frame.targets.position_m[:, :2])
    assert display.profile is None and display.heatmap is None
