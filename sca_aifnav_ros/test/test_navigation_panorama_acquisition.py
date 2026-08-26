"""Tests for panorama acquisition integration in the navigation node."""

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
        self.messages.append(message)


@pytest.fixture
def ros_context():
    """Provide a fresh ROS 2 context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def bridge():
    """Provide a ROS OpenCV bridge."""
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
    """Create odometry carrying one physical yaw."""
    message = Odometry()
    message.pose.pose.orientation = (
        quaternion_from_yaw(yaw_rad)
    )

    return message


def image_message(
    bridge,
    value,
):
    """Create one deterministic BGR image message."""
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
    """Create a node with yaw and one initial camera batch."""
    node = NavigationNode()

    publisher = CapturingPublisher()
    node._cmd_vel_publisher = publisher

    node._odometry_callback(
        odometry_with_yaw(0.0)
    )

    feed_camera_batch(
        node,
        bridge,
        10,
        20,
        30,
    )

    return node, publisher


def test_panorama_does_not_start_without_required_inputs(
    ros_context,
):
    """Panorama acquisition should wait for yaw and cameras."""
    node = NavigationNode()

    try:
        assert (
            node.start_panorama_acquisition()
            is False
        )
        assert (
            node.panorama_acquisition_state
            is None
        )
    finally:
        node.destroy_node()


def test_panorama_start_captures_initial_batch(
    ros_context,
    bridge,
):
    """Starting should capture the initial three-camera batch."""
    node, publisher = ready_node(
        bridge
    )

    try:
        assert (
            node.start_panorama_acquisition()
            is True
        )

        assert (
            node.panorama_acquisition_state
            is PanoramaCoordinatorState.ROTATING
        )

        assert publisher.messages == []
    finally:
        node.destroy_node()


def test_active_panorama_cannot_be_started_again(
    ros_context,
    bridge,
):
    """An active panorama cycle should not be silently replaced."""
    node, _ = ready_node(
        bridge
    )

    try:
        node.start_panorama_acquisition()

        with pytest.raises(
            RuntimeError,
            match="already active",
        ):
            node.start_panorama_acquisition()
    finally:
        node.destroy_node()


def test_panorama_step_publishes_rotation(
    ros_context,
    bridge,
):
    """A rotation step should command the first panorama goal."""
    node, publisher = ready_node(
        bridge
    )

    try:
        node.start_panorama_acquisition()

        state = (
            node.step_panorama_acquisition()
        )

        assert (
            state
            is PanoramaCoordinatorState.ROTATING
        )

        assert (
            publisher.messages[-1].angular.z
            == pytest.approx(0.2)
        )
    finally:
        node.destroy_node()


def test_reached_goal_waits_for_fresh_cameras(
    ros_context,
    bridge,
):
    """Reached rotation should wait for new frames from all cameras."""
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

        state = (
            node.step_panorama_acquisition()
        )

        assert (
            state
            is PanoramaCoordinatorState.WAIT_FRESH_CAMERAS
        )

        assert (
            publisher.messages[-1].angular.z
            == 0.0
        )

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

        assert (
            node.step_panorama_acquisition()
            is PanoramaCoordinatorState.WAIT_FRESH_CAMERAS
        )

        node._right_image_callback(
            image_message(
                bridge,
                60,
            )
        )

        assert (
            node.step_panorama_acquisition()
            is PanoramaCoordinatorState.ROTATING
        )
    finally:
        node.destroy_node()


def test_complete_panorama_cycle_produces_twelve_images(
    ros_context,
    bridge,
):
    """Three rotation stops should produce twelve ordered images."""
    node, _ = ready_node(
        bridge
    )

    try:
        node.start_panorama_acquisition()

        goals = (
            math.pi / 4.0,
            math.pi / 2.0,
            3.0 * math.pi / 4.0,
        )

        batches = (
            (40, 50, 60),
            (70, 80, 90),
            (100, 110, 120),
        )

        for goal, batch in zip(
            goals,
            batches,
        ):
            node._odometry_callback(
                odometry_with_yaw(
                    goal
                )
            )

            assert (
                node.step_panorama_acquisition()
                is PanoramaCoordinatorState.WAIT_FRESH_CAMERAS
            )

            feed_camera_batch(
                node,
                bridge,
                *batch,
            )

            node.step_panorama_acquisition()

        assert (
            node.panorama_acquisition_state
            is PanoramaCoordinatorState.COMPLETE
        )

        images = (
            node.completed_panorama_images()
        )

        assert len(images) == 12

        values = tuple(
            int(image[0, 0, 0])
            for image in images
        )

        assert values == (
            10,
            40,
            70,
            100,
            20,
            50,
            80,
            110,
            30,
            60,
            90,
            120,
        )
    finally:
        node.destroy_node()


def test_completed_images_are_unavailable_before_completion(
    ros_context,
    bridge,
):
    """Incomplete acquisition should not expose panorama images."""
    node, _ = ready_node(
        bridge
    )

    try:
        node.start_panorama_acquisition()

        assert (
            node.completed_panorama_images()
            is None
        )
    finally:
        node.destroy_node()
