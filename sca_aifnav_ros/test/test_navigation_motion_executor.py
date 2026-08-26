"""Tests for physical navigation action execution."""

import math

import pytest
from sensor_msgs.msg import LaserScan

from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.navigation_core_bridge import (
    NavigationActionTarget,
)
from sca_aifnav_ros.navigation_motion_executor import (
    NavigationMotionExecutor,
)


def target(
    action_id=0,
    source_place_id=0,
    target_place_id=1,
    target_position=None,
    is_stationary=False,
):
    """Create one deterministic navigation target."""
    if target_position is None:
        target_position = Point2D(
            1.0,
            0.0,
        )

    return NavigationActionTarget(
        action_id=action_id,
        source_place_id=source_place_id,
        target_place_id=target_place_id,
        target_position=target_position,
        is_stationary=is_stationary,
    )


def scan():
    """Create one scan without valid obstacle returns."""
    message = LaserScan()

    message.angle_min = 0.0
    message.angle_increment = 0.1
    message.range_min = 0.1
    message.range_max = 3.5

    message.ranges = [
        math.inf,
    ]

    return message


def test_new_executor_is_idle():
    """A new executor should have no active action."""
    executor = NavigationMotionExecutor()

    assert executor.is_active is False
    assert executor.active_target is None
    assert executor.step() is None


def test_start_stores_navigation_target():
    """Starting an action should expose its active target."""
    executor = NavigationMotionExecutor()

    action_target = target()

    executor.start(
        action_target
    )

    assert executor.is_active is True

    assert (
        executor.active_target
        is action_target
    )


def test_second_action_cannot_start_while_active():
    """Only one physical action may execute at a time."""
    executor = NavigationMotionExecutor()

    executor.start(
        target()
    )

    with pytest.raises(
        RuntimeError,
        match="already active",
    ):
        executor.start(
            target(
                action_id=1
            )
        )


def test_stationary_action_completes_without_sensor_data():
    """STAY should complete immediately without moving the robot."""
    executor = NavigationMotionExecutor()

    executor.start(
        target(
            action_id=12,
            source_place_id=2,
            target_place_id=2,
            target_position=Point2D(
                1.0,
                1.0,
            ),
            is_stationary=True,
        )
    )

    update = executor.step()

    assert update.action_id == 12
    assert update.completed_action_id == 12

    assert (
        update.command.goal_reached
        is True
    )

    assert update.command.linear_speed == 0.0
    assert update.command.angular_speed == 0.0

    assert executor.is_active is False


def test_directional_action_generates_motion():
    """A distant aligned target should remain active and move forward."""
    executor = NavigationMotionExecutor()

    executor.start(
        target()
    )

    update = executor.step(
        current_position=Point2D(
            0.0,
            0.0,
        ),
        physical_yaw_rad=0.0,
        scan=scan(),
    )

    assert update.action_id == 0
    assert update.completed_action_id is None

    assert (
        update.command.linear_speed
        > 0.0
    )

    assert (
        update.command.goal_reached
        is False
    )

    assert executor.is_active is True


def test_reached_directional_target_completes_action():
    """Arrival at the physical target should complete the action."""
    executor = NavigationMotionExecutor()

    executor.start(
        target(
            target_position=Point2D(
                1.0,
                0.0,
            )
        )
    )

    update = executor.step(
        current_position=Point2D(
            1.0,
            0.0,
        ),
        physical_yaw_rad=0.0,
        scan=scan(),
    )

    assert update.completed_action_id == 0

    assert (
        update.command.goal_reached
        is True
    )

    assert update.command.linear_speed == 0.0
    assert update.command.angular_speed == 0.0

    assert executor.is_active is False
