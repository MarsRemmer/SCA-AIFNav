"""Tests for physical robot orientation in the ROS navigation node."""

import math

from nav_msgs.msg import Odometry
import pytest
import rclpy

from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)


@pytest.fixture
def ros_context():
    """Provide a fresh ROS 2 context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def odometry_message(
    x,
    y,
    yaw_rad,
):
    """Create planar odometry with position and body yaw."""
    message = Odometry()

    message.pose.pose.position.x = float(x)
    message.pose.pose.position.y = float(y)

    message.pose.pose.orientation.z = math.sin(
        yaw_rad / 2.0
    )

    message.pose.pose.orientation.w = math.cos(
        yaw_rad / 2.0
    )

    return message


def test_node_starts_without_physical_yaw(
    ros_context,
):
    """Physical yaw should be unavailable before odometry arrives."""
    node = NavigationNode()

    try:
        assert node.physical_yaw_rad is None
    finally:
        node.destroy_node()


def test_odometry_callback_stores_physical_yaw(
    ros_context,
):
    """Odometry orientation should update the physical robot yaw."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                0.0,
                0.0,
                math.pi / 2.0,
            )
        )

        assert (
            node.physical_yaw_rad
            == pytest.approx(
                round(
                    math.pi / 2.0,
                    4,
                )
            )
        )
    finally:
        node.destroy_node()


def test_negative_body_yaw_wraps_positive(
    ros_context,
):
    """Negative physical yaw should use the positive angular convention."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                0.0,
                0.0,
                -math.pi / 2.0,
            )
        )

        expected = (
            round(
                -math.pi / 2.0,
                4,
            )
            + 2.0
            * math.pi
        )

        assert (
            node.physical_yaw_rad
            == pytest.approx(
                expected
            )
        )
    finally:
        node.destroy_node()


def test_physical_yaw_and_travel_heading_are_independent(
    ros_context,
):
    """Body orientation should remain distinct from displacement heading."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                0.0,
                0.0,
                math.pi / 2.0,
            )
        )

        node._odometry_callback(
            odometry_message(
                1.0,
                0.0,
                math.pi / 2.0,
            )
        )

        assert (
            node.physical_yaw_rad
            == pytest.approx(
                round(
                    math.pi / 2.0,
                    4,
                )
            )
        )

        assert (
            node
            .latest_odometry_state
            .travel_heading_rad
            == pytest.approx(
                0.0
            )
        )
    finally:
        node.destroy_node()
