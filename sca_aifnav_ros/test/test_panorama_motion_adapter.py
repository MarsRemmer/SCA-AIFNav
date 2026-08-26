"""Tests for panorama ROS motion adaptation."""

import pytest
from geometry_msgs.msg import Twist

from sca_aifnav_ros.panorama_motion_adapter import (
    PanoramaMotionAdapter,
)
from sca_aifnav_ros.panorama_rotation import (
    PanoramaRotationController,
)


def test_rotation_command_converts_to_twist():
    """A rotation command should become a ROS Twist message."""
    controller = PanoramaRotationController()

    command = controller.command(
        current_yaw_rad=0.0,
        goal_yaw_rad=1.0,
    )

    message = PanoramaMotionAdapter.to_twist(
        command
    )

    assert isinstance(
        message,
        Twist,
    )


def test_rotation_twist_contains_only_angular_z():
    """Panorama motion should rotate without translating."""
    controller = PanoramaRotationController()

    command = controller.command(
        current_yaw_rad=0.0,
        goal_yaw_rad=1.0,
    )

    message = PanoramaMotionAdapter.to_twist(
        command
    )

    assert message.linear.x == 0.0
    assert message.linear.y == 0.0
    assert message.linear.z == 0.0

    assert message.angular.x == 0.0
    assert message.angular.y == 0.0

    assert (
        message.angular.z
        == pytest.approx(
            command.angular_speed_rad_s
        )
    )


def test_runtime_speed_limit_is_preserved():
    """The Twist should preserve the controller speed limit."""
    controller = PanoramaRotationController()

    command = controller.command(
        current_yaw_rad=0.0,
        goal_yaw_rad=1.0,
    )

    message = PanoramaMotionAdapter.to_twist(
        command
    )

    assert (
        message.angular.z
        == pytest.approx(0.6)
    )


def test_reached_goal_produces_zero_twist():
    """A reached rotation target should command a full stop."""
    controller = PanoramaRotationController()

    command = controller.command(
        current_yaw_rad=1.0,
        goal_yaw_rad=1.0,
    )

    assert command.goal_reached is True

    message = PanoramaMotionAdapter.to_twist(
        command
    )

    assert message.linear.x == 0.0
    assert message.linear.y == 0.0
    assert message.linear.z == 0.0

    assert message.angular.x == 0.0
    assert message.angular.y == 0.0
    assert message.angular.z == 0.0


def test_stop_twist_is_zero_velocity():
    """Explicit panorama stopping should return a zero Twist."""
    message = (
        PanoramaMotionAdapter.stop_twist()
    )

    assert isinstance(
        message,
        Twist,
    )

    assert message.linear.x == 0.0
    assert message.linear.y == 0.0
    assert message.linear.z == 0.0

    assert message.angular.x == 0.0
    assert message.angular.y == 0.0
    assert message.angular.z == 0.0


def test_non_rotation_command_is_rejected():
    """Only validated panorama rotation commands should be adapted."""
    with pytest.raises(
        TypeError,
        match=(
            "command must be a "
            "PanoramaRotationCommand"
        ),
    ):
        PanoramaMotionAdapter.to_twist(
            object()
        )
