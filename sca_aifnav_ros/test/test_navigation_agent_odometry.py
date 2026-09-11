"""Tests for AIMAPP-compatible aligned odometry publication."""

import pytest
import rclpy
from nav_msgs.msg import Odometry

from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)


class CapturePublisher:
    """Capture ROS messages published by a node."""

    def __init__(self):
        self.messages = []

    def publish(
        self,
        message,
    ):
        self.messages.append(
            message
        )


@pytest.fixture
def ros_context():
    """Provide a clean ROS context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def odometry(
    x,
    y,
):
    """Create deterministic physical odometry."""
    message = Odometry()

    message.header.frame_id = "odom"
    message.child_frame_id = "base_footprint"

    message.pose.pose.position.x = float(x)
    message.pose.pose.position.y = float(y)

    message.pose.pose.orientation.w = 1.0

    return message


def test_first_physical_pose_publishes_zero_agent_origin(
    ros_context,
):
    """
    The first raw physical pose should become cognitive origin (0, 0).

    The original /odom message must remain unchanged.
    """
    node = NavigationNode()

    publisher = CapturePublisher()

    node._agent_odom_publisher = (
        publisher
    )

    raw = odometry(
        5.0,
        -2.0,
    )

    try:
        node._odometry_callback(
            raw
        )

        assert len(
            publisher.messages
        ) == 1

        aligned = publisher.messages[0]

        assert (
            aligned.pose.pose.position.x
            == pytest.approx(0.0)
        )

        assert (
            aligned.pose.pose.position.y
            == pytest.approx(0.0)
        )

        assert (
            aligned.header.frame_id
            == "odom"
        )

        assert (
            aligned.child_frame_id
            == "base_footprint"
        )

        # Raw Gazebo odometry must not be modified.
        assert (
            raw.pose.pose.position.x
            == pytest.approx(5.0)
        )

        assert (
            raw.pose.pose.position.y
            == pytest.approx(-2.0)
        )

    finally:
        node.destroy_node()


def test_agent_odometry_follows_runtime_realign(
    ros_context,
):
    """Posterior realignment should also shift Nav2-facing odometry."""
    node = NavigationNode()

    publisher = CapturePublisher()

    node._agent_odom_publisher = (
        publisher
    )

    try:
        # Physical startup position defines cognitive (0, 0).
        node._odometry_callback(
            odometry(
                5.0,
                -2.0,
            )
        )

        # Simulate a confident posterior correction.
        node._odometry_adapter.realign(
            Point2D(
                1.0,
                2.0,
            )
        )

        # Robot then physically moves +0.2 x and +0.1 y.
        node._odometry_callback(
            odometry(
                5.2,
                -1.9,
            )
        )

        aligned = publisher.messages[-1]

        assert (
            aligned.pose.pose.position.x
            == pytest.approx(1.2)
        )

        assert (
            aligned.pose.pose.position.y
            == pytest.approx(2.1)
        )

    finally:
        node.destroy_node()
