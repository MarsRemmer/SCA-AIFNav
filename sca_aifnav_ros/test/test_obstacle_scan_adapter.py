"""Tests for directional ROS 2 laser scan adaptation."""

import math

import pytest
from sensor_msgs.msg import LaserScan

from sca_aifnav_ros.obstacle_scan_adapter import (
    ObstacleScanAdapter,
)


def scan_message(
    ranges,
    angle_min_deg,
    angle_increment_deg,
    range_min=0.1,
    range_max=10.0,
):
    """Create a laser scan with explicit angular geometry."""
    message = LaserScan()

    message.angle_min = math.radians(
        angle_min_deg
    )

    message.angle_increment = math.radians(
        angle_increment_deg
    )

    message.angle_max = (
        message.angle_min
        + (
            len(ranges) - 1
        )
        * message.angle_increment
    )

    message.range_min = float(
        range_min
    )

    message.range_max = float(
        range_max
    )

    message.ranges = [
        float(value)
        for value in ranges
    ]

    return message


def test_world_angle_sector_centers():
    """Sector-center angles should map to actions 0 through 11."""
    adapter = ObstacleScanAdapter()

    for action_id in range(12):
        center_deg = (
            action_id * 30.0
            + 15.0
        )

        assert (
            adapter
            .action_for_world_angle_deg(
                center_deg
            )
            == action_id
        )


def test_exact_sector_boundaries_use_first_match():
    """Exact shared boundaries should belong to the lower action."""
    adapter = ObstacleScanAdapter()

    assert (
        adapter
        .action_for_world_angle_deg(
            0.0
        )
        == 0
    )

    assert (
        adapter
        .action_for_world_angle_deg(
            30.0
        )
        == 0
    )

    assert (
        adapter
        .action_for_world_angle_deg(
            60.0
        )
        == 1
    )

    assert (
        adapter
        .action_for_world_angle_deg(
            330.0
        )
        == 10
    )


def test_angles_wrap_around_full_circle():
    """World angles should wrap into the 0-to-360 degree range."""
    adapter = ObstacleScanAdapter()

    assert (
        adapter
        .action_for_world_angle_deg(
            360.0
        )
        == 0
    )

    assert (
        adapter
        .action_for_world_angle_deg(
            -1.0
        )
        == 11
    )


def test_one_center_ray_per_action_preserves_distances():
    """One ray at every sector center should produce twelve values."""
    adapter = ObstacleScanAdapter()

    message = scan_message(
        ranges=range(
            1,
            13,
        ),
        angle_min_deg=15.0,
        angle_increment_deg=30.0,
    )

    result = adapter.aggregate(
        message,
        robot_yaw_rad=0.0,
    )

    assert result == pytest.approx(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ]
    )


def test_robot_body_yaw_rotates_scan_into_world_actions():
    """A 90-degree robot rotation should shift rays by three actions."""
    adapter = ObstacleScanAdapter()

    message = scan_message(
        ranges=range(
            1,
            13,
        ),
        angle_min_deg=15.0,
        angle_increment_deg=30.0,
    )

    result = adapter.aggregate(
        message,
        robot_yaw_rad=(
            math.pi / 2.0
        ),
    )

    assert result == pytest.approx(
        [
            10.0,
            11.0,
            12.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
        ]
    )


def test_laser_yaw_offset_rotates_scan():
    """Laser mounting yaw should contribute to world ray direction."""
    adapter = ObstacleScanAdapter()

    message = scan_message(
        ranges=range(
            1,
            13,
        ),
        angle_min_deg=15.0,
        angle_increment_deg=30.0,
    )

    result = adapter.aggregate(
        message,
        robot_yaw_rad=0.0,
        laser_yaw_offset_rad=(
            math.pi / 6.0
        ),
    )

    assert result == pytest.approx(
        [
            12.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
        ]
    )


def test_nan_and_positive_infinity_become_range_max():
    """Invalid scan ranges should use the scan maximum range."""
    adapter = ObstacleScanAdapter()

    message = scan_message(
        ranges=[
            math.inf,
            math.nan,
            1.0,
        ],
        angle_min_deg=5.0,
        angle_increment_deg=10.0,
        range_max=10.0,
    )

    result = adapter.aggregate(
        message,
        robot_yaw_rad=0.0,
    )

    assert result[0] == pytest.approx(
        7.0
    )


def test_empty_direction_retains_nan_semantics():
    """A direction with no contributing ray should remain unknown."""
    adapter = ObstacleScanAdapter()

    message = scan_message(
        ranges=[
            2.0,
        ],
        angle_min_deg=15.0,
        angle_increment_deg=1.0,
    )

    result = adapter.aggregate(
        message,
        robot_yaw_rad=0.0,
    )

    assert result[0] == pytest.approx(
        2.0
    )

    assert math.isnan(
        result[1]
    )

    assert math.isnan(
        result[11]
    )


def test_non_scan_message_is_rejected():
    """The adapter should reject unrelated ROS messages."""
    adapter = ObstacleScanAdapter()

    with pytest.raises(
        TypeError,
        match=(
            "message must be "
            "sensor_msgs.msg.LaserScan"
        ),
    ):
        adapter.aggregate(
            object(),
            robot_yaw_rad=0.0,
        )


def test_non_finite_robot_yaw_is_rejected():
    """World-frame aggregation requires a finite robot yaw."""
    adapter = ObstacleScanAdapter()

    message = scan_message(
        ranges=[
            1.0,
        ],
        angle_min_deg=15.0,
        angle_increment_deg=1.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "robot_yaw_rad must be finite"
        ),
    ):
        adapter.aggregate(
            message,
            robot_yaw_rad=math.nan,
        )
