"""Tests for camera image integration in the ROS navigation node."""

import numpy as np
import pytest
from cv_bridge import CvBridge
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


@pytest.fixture
def bridge():
    """Provide a ROS OpenCV bridge."""
    return CvBridge()


def test_node_starts_without_camera_image(
    ros_context,
):
    """Camera state should be unavailable before an image arrives."""
    node = NavigationNode()

    try:
        assert node.has_image is False
        assert node.latest_image is None
        assert node.image_revision == 0
    finally:
        node.destroy_node()


def test_default_camera_topic(
    ros_context,
):
    """The default camera topic should match the ROS input contract."""
    node = NavigationNode()

    try:
        assert (
            node.get_parameter(
                "camera_topic"
            ).value
            == "/camera/image_raw"
        )
    finally:
        node.destroy_node()


def test_camera_callback_stores_bgr_image(
    ros_context,
    bridge,
):
    """One BGR image should become the latest camera observation."""
    node = NavigationNode()

    expected = np.array(
        [
            [
                [10, 20, 30],
                [40, 50, 60],
            ],
            [
                [70, 80, 90],
                [100, 110, 120],
            ],
        ],
        dtype=np.uint8,
    )

    message = bridge.cv2_to_imgmsg(
        expected,
        encoding="bgr8",
    )

    try:
        node._image_callback(
            message
        )

        assert node.has_image is True

        assert np.array_equal(
            node.latest_image,
            expected,
        )

        assert node.image_revision == 1
    finally:
        node.destroy_node()


def test_rgb_camera_image_is_stored_as_bgr(
    ros_context,
    bridge,
):
    """RGB camera data should be normalized to BGR representation."""
    node = NavigationNode()

    rgb = np.array(
        [
            [
                [10, 20, 30],
            ],
        ],
        dtype=np.uint8,
    )

    message = bridge.cv2_to_imgmsg(
        rgb,
        encoding="rgb8",
    )

    try:
        node._image_callback(
            message
        )

        expected = np.array(
            [
                [
                    [30, 20, 10],
                ],
            ],
            dtype=np.uint8,
        )

        assert np.array_equal(
            node.latest_image,
            expected,
        )
    finally:
        node.destroy_node()


def test_new_camera_image_replaces_previous_image(
    ros_context,
    bridge,
):
    """The camera cache should retain only the latest converted image."""
    node = NavigationNode()

    first = np.zeros(
        (
            2,
            3,
            3,
        ),
        dtype=np.uint8,
    )

    second = np.full(
        (
            4,
            5,
            3,
        ),
        100,
        dtype=np.uint8,
    )

    try:
        node._image_callback(
            bridge.cv2_to_imgmsg(
                first,
                encoding="bgr8",
            )
        )

        node._image_callback(
            bridge.cv2_to_imgmsg(
                second,
                encoding="bgr8",
            )
        )

        assert node.image_revision == 2

        assert node.latest_image.shape == (
            4,
            5,
            3,
        )

        assert np.array_equal(
            node.latest_image,
            second,
        )
    finally:
        node.destroy_node()
