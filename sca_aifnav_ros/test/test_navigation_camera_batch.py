"""Tests for three-camera navigation capture."""

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


def image_message(
    bridge,
    value,
):
    """Create one deterministic BGR ROS image."""
    image = np.full(
        (
            3,
            4,
            3,
        ),
        value,
        dtype=np.uint8,
    )

    return bridge.cv2_to_imgmsg(
        image,
        encoding="bgr8",
    )


def test_default_three_camera_topics(
    ros_context,
):
    """Three camera streams should use the configured default topics."""
    node = NavigationNode()

    try:
        assert (
            node.get_parameter(
                "camera_topic"
            ).value
            == "/camera_front/image_raw"
        )

        assert (
            node.get_parameter(
                "left_camera_topic"
            ).value
            == "/camera_left/image_raw"
        )

        assert (
            node.get_parameter(
                "right_camera_topic"
            ).value
            == "/camera_right/image_raw"
        )
    finally:
        node.destroy_node()


def test_camera_batch_is_not_ready_on_start(
    ros_context,
):
    """A new node should not expose a complete camera batch."""
    node = NavigationNode()

    try:
        assert (
            node.camera_batch_ready
            is False
        )

        assert (
            node.capture_camera_batch()
            is None
        )
    finally:
        node.destroy_node()


def test_front_camera_alone_is_not_enough(
    ros_context,
    bridge,
):
    """One camera stream should not create a complete batch."""
    node = NavigationNode()

    try:
        node._image_callback(
            image_message(
                bridge,
                10,
            )
        )

        assert (
            node.camera_batch_ready
            is False
        )

        assert (
            node.capture_camera_batch()
            is None
        )
    finally:
        node.destroy_node()


def test_three_camera_callbacks_create_ordered_batch(
    ros_context,
    bridge,
):
    """A complete batch should preserve front-left-right ordering."""
    node = NavigationNode()

    try:
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

        assert (
            node.camera_batch_ready
            is True
        )

        batch = (
            node.capture_camera_batch()
        )

        assert len(batch) == 3

        assert np.all(
            batch[0] == 10
        )

        assert np.all(
            batch[1] == 20
        )

        assert np.all(
            batch[2] == 30
        )
    finally:
        node.destroy_node()


def test_camera_revisions_are_independent(
    ros_context,
    bridge,
):
    """Each camera stream should maintain its own revision count."""
    node = NavigationNode()

    try:
        node._image_callback(
            image_message(
                bridge,
                10,
            )
        )

        node._image_callback(
            image_message(
                bridge,
                11,
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

        assert (
            node.front_image_revision
            == 2
        )

        assert (
            node.image_revision
            == 2
        )

        assert (
            node.left_image_revision
            == 1
        )

        assert (
            node.right_image_revision
            == 1
        )
    finally:
        node.destroy_node()


def test_camera_batch_is_frozen_from_later_updates(
    ros_context,
    bridge,
):
    """A captured batch should not change after new camera messages."""
    node = NavigationNode()

    try:
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

        batch = (
            node.capture_camera_batch()
        )

        node._image_callback(
            image_message(
                bridge,
                100,
            )
        )

        node._left_image_callback(
            image_message(
                bridge,
                110,
            )
        )

        node._right_image_callback(
            image_message(
                bridge,
                120,
            )
        )

        assert np.all(
            batch[0] == 10
        )

        assert np.all(
            batch[1] == 20
        )

        assert np.all(
            batch[2] == 30
        )
    finally:
        node.destroy_node()


def test_existing_latest_image_remains_front_camera_alias(
    ros_context,
    bridge,
):
    """The existing image interface should continue to expose front data."""
    node = NavigationNode()

    try:
        node._image_callback(
            image_message(
                bridge,
                42,
            )
        )

        assert np.array_equal(
            node.latest_image,
            node.latest_front_image,
        )

        assert np.all(
            node.latest_image
            == 42
        )
    finally:
        node.destroy_node()
