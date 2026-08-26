"""Tests for panorama motion publishing from the navigation node."""

import math

import pytest
import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry

from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)


class CapturingPublisher:
    """Capture ROS messages published during unit tests."""

    def __init__(self):
        self.messages = []

    def publish(
        self,
        message,
    ):
        """Store one published message."""
        self.messages.append(
            message
        )


@pytest.fixture
def ros_context():
    """Provide a fresh ROS 2 context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def quaternion_from_yaw(
    yaw_rad,
):
    """Create a planar quaternion."""
    message = Quaternion()

    message.z = math.sin(
        yaw_rad / 2.0
    )
    message.w = math.cos(
        yaw_rad / 2.0
    )

    return message


def odometry_with_yaw(
    yaw_rad,
):
    """Create one odometry message carrying physical yaw."""
    message = Odometry()

    message.pose.pose.orientation = (
        quaternion_from_yaw(
            yaw_rad
        )
    )

    return message


def capturing_node():
    """Create a navigation node with a capturing command publisher."""
    node = NavigationNode()

    publisher = CapturingPublisher()

    node._cmd_vel_publisher = (
        publisher
    )

    return node, publisher


def test_default_cmd_vel_topic(
    ros_context,
):
    """Navigation commands should default to the robot velocity topic."""
    node = NavigationNode()

    try:
        assert (
            node.get_parameter(
                "cmd_vel_topic"
            ).value
            == "/cmd_vel"
        )
    finally:
        node.destroy_node()


def test_rotation_is_not_published_before_physical_yaw(
    ros_context,
):
    """Panorama motion should wait until physical orientation exists."""
    node, publisher = (
        capturing_node()
    )

    try:
        command = (
            node.publish_panorama_rotation(
                math.pi / 4.0
            )
        )

        assert command is None

        assert (
            publisher.messages
            == []
        )
    finally:
        node.destroy_node()


def test_rotation_publishes_positive_angular_velocity(
    ros_context,
):
    """A distant panorama goal should publish positive rotation."""
    node, publisher = (
        capturing_node()
    )

    try:
        node._odometry_callback(
            odometry_with_yaw(
                0.0
            )
        )

        command = (
            node.publish_panorama_rotation(
                math.pi / 4.0
            )
        )

        assert command is not None
        assert command.goal_reached is False

        assert (
            len(publisher.messages)
            == 1
        )

        message = publisher.messages[0]

        assert message.linear.x == 0.0
        assert message.linear.y == 0.0
        assert message.linear.z == 0.0

        assert message.angular.x == 0.0
        assert message.angular.y == 0.0

        assert (
            message.angular.z
            == pytest.approx(0.2)
        )
    finally:
        node.destroy_node()


def test_reached_goal_publishes_zero_velocity(
    ros_context,
):
    """A reached yaw target should publish a stopping Twist."""
    node, publisher = (
        capturing_node()
    )

    try:
        node._odometry_callback(
            odometry_with_yaw(
                1.0
            )
        )

        command = (
            node.publish_panorama_rotation(
                1.0
            )
        )

        assert command.goal_reached is True

        assert (
            len(publisher.messages)
            == 1
        )

        message = publisher.messages[0]

        assert message.linear.x == 0.0
        assert message.angular.z == 0.0
    finally:
        node.destroy_node()


def test_stop_panorama_rotation_publishes_zero_twist(
    ros_context,
):
    """Explicit panorama stopping should publish zero velocity."""
    node, publisher = (
        capturing_node()
    )

    try:
        node.stop_panorama_rotation()

        assert (
            len(publisher.messages)
            == 1
        )

        message = publisher.messages[0]

        assert message.linear.x == 0.0
        assert message.linear.y == 0.0
        assert message.linear.z == 0.0

        assert message.angular.x == 0.0
        assert message.angular.y == 0.0
        assert message.angular.z == 0.0
    finally:
        node.destroy_node()


def test_rotation_uses_latest_physical_yaw(
    ros_context,
):
    """Successive odometry updates should change panorama control output."""
    node, publisher = (
        capturing_node()
    )

    try:
        goal = math.pi / 4.0

        node._odometry_callback(
            odometry_with_yaw(
                0.0
            )
        )

        first = (
            node.publish_panorama_rotation(
                goal
            )
        )

        node._odometry_callback(
            odometry_with_yaw(
                0.75
            )
        )

        second = (
            node.publish_panorama_rotation(
                goal
            )
        )

        assert first.goal_reached is False
        assert second.goal_reached is True

        assert (
            publisher.messages[0].angular.z
            > 0.0
        )

        assert (
            publisher.messages[1].angular.z
            == 0.0
        )
    finally:
        node.destroy_node()
