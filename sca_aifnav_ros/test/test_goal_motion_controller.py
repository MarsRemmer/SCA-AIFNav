"""Tests for planar navigation goal motion control."""

import math

import pytest
from sensor_msgs.msg import LaserScan

from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.goal_motion_controller import (
    DEFAULT_ANGULAR_TOLERANCE_RAD,
    DEFAULT_DISTANCE_TOLERANCE,
    DEFAULT_MAX_ANGULAR_SPEED,
    DEFAULT_MAX_LINEAR_SPEED,
    GoalMotionController,
    normalize_angle,
)


def scan_message(
    ranges,
    angle_min=0.0,
    angle_increment=0.1,
    range_min=0.1,
    range_max=3.5,
):
    """Create one deterministic laser scan."""
    scan = LaserScan()

    scan.angle_min = float(
        angle_min
    )

    scan.angle_increment = float(
        angle_increment
    )

    scan.range_min = float(
        range_min
    )

    scan.range_max = float(
        range_max
    )

    scan.ranges = [
        float(value)
        for value in ranges
    ]

    return scan


def test_default_motion_parameters():
    """Default limits should preserve baseline motion settings."""
    controller = GoalMotionController()

    assert (
        controller.max_linear_speed
        == pytest.approx(
            DEFAULT_MAX_LINEAR_SPEED
        )
    )

    assert (
        controller.max_angular_speed
        == pytest.approx(
            DEFAULT_MAX_ANGULAR_SPEED
        )
    )

    assert (
        controller.angular_tolerance_rad
        == pytest.approx(
            DEFAULT_ANGULAR_TOLERANCE_RAD
        )
    )

    assert (
        controller.distance_tolerance
        == pytest.approx(
            DEFAULT_DISTANCE_TOLERANCE
        )
    )


def test_angle_normalization():
    """Angular errors should remain inside the shortest-turn range."""
    assert normalize_angle(
        3.0 * math.pi
    ) == pytest.approx(
        math.pi
    )

    assert normalize_angle(
        -3.0 * math.pi
    ) == pytest.approx(
        -math.pi
    )


def test_goal_inside_tolerance_stops():
    """A reached target should produce an explicit stop."""
    controller = GoalMotionController()

    command = controller.command(
        current_position=Point2D(
            0.0,
            0.0,
        ),
        physical_yaw_rad=0.0,
        target_position=Point2D(
            0.04,
            0.0,
        ),
    )

    assert command.goal_reached is True
    assert command.linear_speed == 0.0
    assert command.angular_speed == 0.0


def test_aligned_target_drives_forward():
    """An aligned distant target should produce linear motion."""
    controller = GoalMotionController()

    command = controller.command(
        current_position=Point2D(
            0.0,
            0.0,
        ),
        physical_yaw_rad=0.0,
        target_position=Point2D(
            1.0,
            0.0,
        ),
    )

    assert command.goal_reached is False

    assert (
        command.linear_speed
        == pytest.approx(0.3)
    )

    assert command.angular_speed == 0.0


def test_large_heading_error_turns_without_forward_motion():
    """A misaligned robot should rotate before moving forward."""
    controller = GoalMotionController()

    command = controller.command(
        current_position=Point2D(
            0.0,
            0.0,
        ),
        physical_yaw_rad=(
            math.pi / 2.0
        ),
        target_position=Point2D(
            1.0,
            0.0,
        ),
    )

    assert command.linear_speed == 0.0

    assert (
        command.angular_speed
        == pytest.approx(-0.2)
    )


def test_small_heading_error_allows_forward_motion():
    """Angular errors inside tolerance should not block translation."""
    controller = GoalMotionController()

    command = controller.command(
        current_position=Point2D(
            0.0,
            0.0,
        ),
        physical_yaw_rad=0.1,
        target_position=Point2D(
            1.0,
            0.0,
        ),
    )

    assert command.linear_speed > 0.0
    assert command.angular_speed == 0.0


def test_linear_speed_reduces_near_goal():
    """Forward velocity should reduce with remaining distance."""
    controller = GoalMotionController()

    command = controller.command(
        current_position=Point2D(
            0.0,
            0.0,
        ),
        physical_yaw_rad=0.0,
        target_position=Point2D(
            0.2,
            0.0,
        ),
    )

    assert (
        command.linear_speed
        == pytest.approx(0.1)
    )


def test_single_front_obstacle_creates_backward_repulsion():
    """A valid front scan return should repel away from the obstacle."""
    controller = GoalMotionController()

    repulsion = (
        controller.repulsion_from_scan(
            scan_message(
                [1.0],
                angle_min=0.0,
            ),
            physical_yaw_rad=0.0,
        )
    )

    assert repulsion[0] < 0.0
    assert repulsion[1] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_invalid_scan_returns_baseline_fallback_repulsion():
    """A scan with no usable rays should preserve fallback behavior."""
    controller = GoalMotionController()

    repulsion = (
        controller.repulsion_from_scan(
            scan_message(
                [
                    math.inf,
                    math.inf,
                ]
            ),
            physical_yaw_rad=0.0,
        )
    )

    assert repulsion == pytest.approx(
        (
            0.0001,
            0.000000000001,
        )
    )


def test_physical_yaw_changes_world_scan_direction():
    """Repulsion should use physical body yaw when rotating scan rays."""
    controller = GoalMotionController()

    repulsion = (
        controller.repulsion_from_scan(
            scan_message(
                [1.0],
                angle_min=0.0,
            ),
            physical_yaw_rad=(
                math.pi / 2.0
            ),
        )
    )

    assert repulsion[0] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert repulsion[1] < 0.0
