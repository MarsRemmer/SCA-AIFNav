"""Tests for automatic panorama state-machine advancement."""

import math

from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
import numpy as np
import pytest
import rclpy

from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)
from sca_aifnav_ros.panorama_coordinator import (
    PanoramaCoordinatorState,
)


class CapturingPublisher:
    """Capture velocity commands published during tests."""

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


@pytest.fixture
def bridge():
    """Provide an OpenCV bridge."""
    return CvBridge()


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
    """Create odometry carrying physical yaw."""
    message = Odometry()

    message.pose.pose.orientation = (
        quaternion_from_yaw(
            yaw_rad
        )
    )

    return message


def image_message(
    bridge,
    value,
):
    """Create one deterministic BGR image."""
    image = np.full(
        (3, 4, 3),
        value,
        dtype=np.uint8,
    )

    return bridge.cv2_to_imgmsg(
        image,
        encoding="bgr8",
    )


def feed_camera_batch(
    node,
    bridge,
    front,
    left,
    right,
):
    """Feed one complete three-camera batch."""
    node._image_callback(
        image_message(
            bridge,
            front,
        )
    )

    node._left_image_callback(
        image_message(
            bridge,
            left,
        )
    )

    node._right_image_callback(
        image_message(
            bridge,
            right,
        )
    )


def ready_node(
    bridge,
):
    """Create a node ready to start panorama acquisition."""
    node = NavigationNode()

    publisher = CapturingPublisher()
    node._cmd_vel_publisher = publisher

    node._odometry_callback(
        odometry_with_yaw(
            0.0
        )
    )

    feed_camera_batch(
        node,
        bridge,
        10,
        20,
        30,
    )

    return node, publisher


def test_default_panorama_control_period(
    ros_context,
):
    """Panorama control should default to a 20 Hz update period."""
    node = NavigationNode()

    try:
        assert (
            node.get_parameter(
                "panorama_control_period_sec"
            ).value
            == pytest.approx(0.05)
        )
    finally:
        node.destroy_node()


def test_timer_does_nothing_without_active_panorama(
    ros_context,
):
    """Idle timer callbacks should not publish motion commands."""
    node = NavigationNode()

    publisher = CapturingPublisher()
    node._cmd_vel_publisher = publisher

    try:
        node._panorama_timer_callback()

        assert publisher.messages == []
    finally:
        node.destroy_node()


def test_timer_advances_active_rotation(
    ros_context,
    bridge,
):
    """An active panorama should automatically publish rotation."""
    node, publisher = ready_node(
        bridge
    )

    try:
        node.start_panorama_acquisition()

        node._panorama_timer_callback()

        assert (
            node.panorama_acquisition_state
            is PanoramaCoordinatorState.ROTATING
        )

        assert len(publisher.messages) == 1

        assert (
            publisher.messages[-1].angular.z
            == pytest.approx(0.2)
        )
    finally:
        node.destroy_node()


def test_timer_detects_reached_rotation_goal(
    ros_context,
    bridge,
):
    """Timer advancement should enter fresh-frame waiting at the goal."""
    node, publisher = ready_node(
        bridge
    )

    try:
        node.start_panorama_acquisition()

        node._odometry_callback(
            odometry_with_yaw(
                math.pi / 4.0
            )
        )

        node._panorama_timer_callback()

        assert (
            node.panorama_acquisition_state
            is PanoramaCoordinatorState.WAIT_FRESH_CAMERAS
        )

        assert (
            publisher.messages[-1].angular.z
            == 0.0
        )
    finally:
        node.destroy_node()


def test_timer_waits_until_all_cameras_are_fresh(
    ros_context,
    bridge,
):
    """The next batch should require a fresh frame from every camera."""
    node, _ = ready_node(
        bridge
    )

    try:
        node.start_panorama_acquisition()

        node._odometry_callback(
            odometry_with_yaw(
                math.pi / 4.0
            )
        )

        node._panorama_timer_callback()

        node._image_callback(
            image_message(
                bridge,
                40,
            )
        )

        node._left_image_callback(
            image_message(
                bridge,
                50,
            )
        )

        node._panorama_timer_callback()

        assert (
            node.panorama_acquisition_state
            is PanoramaCoordinatorState.WAIT_FRESH_CAMERAS
        )

        node._right_image_callback(
            image_message(
                bridge,
                60,
            )
        )

        node._panorama_timer_callback()

        assert (
            node.panorama_acquisition_state
            is PanoramaCoordinatorState.ROTATING
        )
    finally:
        node.destroy_node()


def test_timer_ignores_completed_panorama(
    ros_context,
    bridge,
):
    """A completed panorama should not generate further motion."""
    node, publisher = ready_node(
        bridge
    )

    try:
        node.start_panorama_acquisition()

        goals = (
            math.pi / 4.0,
            math.pi / 2.0,
            3.0 * math.pi / 4.0,
        )

        values = (
            (40, 50, 60),
            (70, 80, 90),
            (100, 110, 120),
        )

        for goal, batch in zip(
            goals,
            values,
        ):
            node._odometry_callback(
                odometry_with_yaw(
                    goal
                )
            )

            node._panorama_timer_callback()

            feed_camera_batch(
                node,
                bridge,
                *batch,
            )

            node._panorama_timer_callback()

        assert (
            node.panorama_acquisition_state
            is PanoramaCoordinatorState.COMPLETE
        )

        message_count = len(
            publisher.messages
        )

        node._panorama_timer_callback()

        assert (
            len(publisher.messages)
            == message_count
        )
    finally:
        node.destroy_node()
