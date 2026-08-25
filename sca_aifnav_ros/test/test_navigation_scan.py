"""Tests for laser scan integration in the ROS navigation node."""

import math

from nav_msgs.msg import Odometry
import pytest
import rclpy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)


class CapturingPublisher:
    """Capture published messages for unit tests."""

    def __init__(self):
        """Initialize an empty message collection."""
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


def odometry_message(
    yaw_rad,
):
    """Create planar odometry containing a physical body yaw."""
    message = Odometry()

    message.pose.pose.orientation.z = math.sin(
        yaw_rad / 2.0
    )

    message.pose.pose.orientation.w = math.cos(
        yaw_rad / 2.0
    )

    return message


def centered_scan():
    """Create one laser ray at the center of each action sector."""
    message = LaserScan()

    message.angle_min = math.radians(
        15.0
    )

    message.angle_increment = math.radians(
        30.0
    )

    message.angle_max = math.radians(
        345.0
    )

    message.range_min = 0.1
    message.range_max = 20.0

    message.ranges = [
        float(value)
        for value in range(
            1,
            13,
        )
    ]

    return message


def test_node_starts_without_obstacle_distances(
    ros_context,
):
    """Obstacle state should be unavailable before a scan is processed."""
    node = NavigationNode()

    try:
        assert (
            node.has_obstacle_distances
            is False
        )

        assert (
            node.latest_obstacle_distances
            is None
        )

        assert node.scan_revision == 0
    finally:
        node.destroy_node()


def test_default_scan_parameters(
    ros_context,
):
    """Default laser topics and yaw offset should be declared."""
    node = NavigationNode()

    try:
        assert (
            node.get_parameter(
                "scan_topic"
            ).value
            == "/scan"
        )

        assert (
            node.get_parameter(
                "obstacle_distances_topic"
            ).value
            == "/sca_aifnav/obstacle_distances"
        )

        assert (
            node.get_parameter(
                "laser_yaw_offset_rad"
            ).value
            == pytest.approx(0.0)
        )
    finally:
        node.destroy_node()


def test_scan_before_odometry_is_ignored(
    ros_context,
):
    """A laser scan should wait until physical orientation is known."""
    node = NavigationNode()

    publisher = CapturingPublisher()

    node._obstacle_distance_publisher = (
        publisher
    )

    try:
        node._scan_callback(
            centered_scan()
        )

        assert (
            node.has_obstacle_distances
            is False
        )

        assert node.scan_revision == 0

        assert publisher.messages == []
    finally:
        node.destroy_node()


def test_scan_after_odometry_is_stored_and_published(
    ros_context,
):
    """A valid scan should produce twelve directional distances."""
    node = NavigationNode()

    publisher = CapturingPublisher()

    node._obstacle_distance_publisher = (
        publisher
    )

    try:
        node._odometry_callback(
            odometry_message(
                0.0
            )
        )

        node._scan_callback(
            centered_scan()
        )

        expected = [
            float(value)
            for value in range(
                1,
                13,
            )
        ]

        assert (
            node.latest_obstacle_distances
            == pytest.approx(
                expected
            )
        )

        assert (
            node.has_obstacle_distances
            is True
        )

        assert node.scan_revision == 1

        assert len(
            publisher.messages
        ) == 1

        message = publisher.messages[0]

        assert isinstance(
            message,
            Float32MultiArray,
        )

        assert list(
            message.data
        ) == pytest.approx(
            expected
        )
    finally:
        node.destroy_node()


def test_physical_yaw_rotates_scan_into_world_directions(
    ros_context,
):
    """Physical body yaw should rotate laser rays into global actions."""
    node = NavigationNode()

    publisher = CapturingPublisher()

    node._obstacle_distance_publisher = (
        publisher
    )

    try:
        node._odometry_callback(
            odometry_message(
                math.pi / 2.0
            )
        )

        node._scan_callback(
            centered_scan()
        )

        expected = [
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

        assert (
            node.latest_obstacle_distances
            == pytest.approx(
                expected
            )
        )

        assert node.scan_revision == 1
    finally:
        node.destroy_node()
