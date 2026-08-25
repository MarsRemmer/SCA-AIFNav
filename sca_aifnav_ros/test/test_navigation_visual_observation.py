"""Tests for visual observation integration in the navigation node."""

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
from sca_aifnav_ros.visual_observation import (
    VisualObservationResult,
)


class CompletePanoramaCoordinator:
    """Provide one deterministic completed panorama."""

    def __init__(
        self,
        images,
    ):
        self._images = tuple(images)
        self.is_complete = True

    def compiled_images(
        self,
    ):
        """Return the completed panorama images."""
        return self._images


class FakeVisualObserver:
    """Capture visual observation processing calls."""

    def __init__(
        self,
        result,
    ):
        self.result = result
        self.calls = []

    def process(
        self,
        images,
    ):
        """Return the configured visual observation result."""
        self.calls.append(
            tuple(images)
        )

        return self.result


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


def observation_result(
    observation_id=2,
):
    """Create one deterministic visual observation result."""
    return VisualObservationResult(
        observation_id=observation_id,
        match_scores=(
            0.1,
            0.2,
            1.0,
        ),
        confidence_threshold=0.9,
        attempt_count=1,
    )


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


def image_message(
    bridge,
    value,
):
    """Create one deterministic camera image."""
    image = np.full(
        (3, 4, 3),
        value,
        dtype=np.uint8,
    )

    return bridge.cv2_to_imgmsg(
        image,
        encoding="bgr8",
    )


def prepare_panorama_inputs(
    node,
    bridge,
):
    """Supply physical yaw and all three camera streams."""
    odometry = Odometry()

    odometry.pose.pose.orientation = (
        quaternion_from_yaw(
            0.0
        )
    )

    node._odometry_callback(
        odometry
    )

    node._image_callback(
        image_message(
            bridge,
            10,
        )
    )

    node._left_image_callback(
        image_message(
            bridge,
            20,
        )
    )

    node._right_image_callback(
        image_message(
            bridge,
            30,
        )
    )


def test_visual_observation_is_empty_on_start(
    ros_context,
):
    """A new node should not expose a visual observation."""
    node = NavigationNode()

    try:
        assert (
            node.has_visual_observation
            is False
        )

        assert (
            node.latest_visual_observation
            is None
        )

        assert (
            node.visual_observation_id
            is None
        )
    finally:
        node.destroy_node()


def test_incomplete_panorama_cannot_be_processed(
    ros_context,
):
    """Visual processing should wait for completed panorama acquisition."""
    node = NavigationNode()

    observer = FakeVisualObserver(
        observation_result()
    )

    node._visual_observer = observer

    try:
        assert (
            node.process_completed_visual_observation()
            is None
        )

        assert observer.calls == []
    finally:
        node.destroy_node()


def test_completed_panorama_is_converted_to_observation(
    ros_context,
):
    """All completed panorama images should reach the visual observer."""
    node = NavigationNode()

    images = tuple(
        f"image-{index}"
        for index in range(12)
    )

    result = observation_result(
        observation_id=4
    )

    observer = FakeVisualObserver(
        result
    )

    node._panorama_coordinator = (
        CompletePanoramaCoordinator(
            images
        )
    )

    node._visual_observer = observer

    try:
        returned = (
            node.process_completed_visual_observation()
        )

        assert returned is result
        assert observer.calls == [images]

        assert (
            node.has_visual_observation
            is True
        )

        assert (
            node.latest_visual_observation
            is result
        )

        assert (
            node.visual_observation_id
            == 4
        )
    finally:
        node.destroy_node()


def test_completed_panorama_is_processed_only_once(
    ros_context,
):
    """Repeated access should reuse the stored visual observation."""
    node = NavigationNode()

    images = tuple(
        range(12)
    )

    result = observation_result()

    observer = FakeVisualObserver(
        result
    )

    node._panorama_coordinator = (
        CompletePanoramaCoordinator(
            images
        )
    )

    node._visual_observer = observer

    try:
        first = (
            node.process_completed_visual_observation()
        )

        second = (
            node.process_completed_visual_observation()
        )

        assert first is result
        assert second is result
        assert len(observer.calls) == 1
    finally:
        node.destroy_node()


def test_new_panorama_clears_previous_visual_observation(
    ros_context,
    bridge,
):
    """Starting another panorama should invalidate the previous result."""
    node = NavigationNode()

    try:
        node._latest_visual_observation = (
            observation_result()
        )

        prepare_panorama_inputs(
            node,
            bridge,
        )

        assert (
            node.start_panorama_acquisition()
            is True
        )

        assert (
            node.has_visual_observation
            is False
        )

        assert (
            node.visual_observation_id
            is None
        )
    finally:
        node.destroy_node()
