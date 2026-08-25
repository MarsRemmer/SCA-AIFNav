"""Tests for cognitive odometry diagnostic publishing."""

import math

from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
import pytest
import rclpy

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)


class CapturingPublisher:
    """Capture published messages for unit tests."""

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


def odometry_message(
    x,
    y,
):
    """Create a minimal odometry message."""
    message = Odometry()

    message.pose.pose.position.x = float(x)
    message.pose.pose.position.y = float(y)

    return message


def test_default_diagnostic_topic(
    ros_context,
):
    """The cognitive odometry diagnostic topic should be declared."""
    node = NavigationNode()

    try:
        assert (
            node.get_parameter(
                "cognitive_odometry_topic"
            ).value
            == "/sca_aifnav/cognitive_odometry"
        )
    finally:
        node.destroy_node()


def test_cognitive_state_converts_to_vector3():
    """Cognitive coordinates and heading should map directly."""
    state = CognitiveOdomState(
        position=Point2D(
            2.5,
            -1.5,
        ),
        travel_heading_rad=(
            math.pi / 2.0
        ),
    )

    message = (
        NavigationNode
        ._cognitive_odometry_message(
            state
        )
    )

    assert isinstance(
        message,
        Vector3,
    )

    assert message.x == pytest.approx(
        2.5
    )

    assert message.y == pytest.approx(
        -1.5
    )

    assert message.z == pytest.approx(
        math.pi / 2.0
    )


def test_callback_publishes_first_state(
    ros_context,
):
    """The first odometry callback should publish its initialized state."""
    node = NavigationNode()

    publisher = CapturingPublisher()

    node._cognitive_odometry_publisher = (
        publisher
    )

    try:
        node._odometry_callback(
            odometry_message(
                3.0,
                -2.0,
            )
        )

        assert len(
            publisher.messages
        ) == 1

        message = publisher.messages[0]

        assert message.x == pytest.approx(
            3.0
        )

        assert message.y == pytest.approx(
            -2.0
        )

        assert message.z == pytest.approx(
            0.0
        )
    finally:
        node.destroy_node()


def test_second_callback_publishes_travel_heading(
    ros_context,
):
    """The second position should publish displacement heading."""
    node = NavigationNode()

    publisher = CapturingPublisher()

    node._cognitive_odometry_publisher = (
        publisher
    )

    try:
        node._odometry_callback(
            odometry_message(
                3.0,
                -2.0,
            )
        )

        node._odometry_callback(
            odometry_message(
                3.0,
                -1.0,
            )
        )

        assert len(
            publisher.messages
        ) == 2

        message = publisher.messages[-1]

        assert message.x == pytest.approx(
            3.0
        )

        assert message.y == pytest.approx(
            -1.0
        )

        assert message.z == pytest.approx(
            math.pi / 2.0
        )

        assert node.odometry_revision == 2
    finally:
        node.destroy_node()
