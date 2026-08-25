"""Tests for ROS 2 image adaptation."""

import numpy as np
import pytest
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from sca_aifnav_ros.image_adapter import (
    ImageAdapter,
)


@pytest.fixture
def bridge():
    """Provide a ROS OpenCV bridge for test-message creation."""
    return CvBridge()


def test_bgr8_image_is_preserved(
    bridge,
):
    """A BGR8 ROS image should preserve its pixel data."""
    adapter = ImageAdapter()

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

    actual = adapter.to_bgr(
        message
    )

    assert np.array_equal(
        actual,
        expected,
    )


def test_rgb8_image_is_converted_to_bgr(
    bridge,
):
    """An RGB8 ROS image should be converted into BGR channel order."""
    adapter = ImageAdapter()

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

    actual = adapter.to_bgr(
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
        actual,
        expected,
    )


def test_mono8_image_is_expanded_to_bgr(
    bridge,
):
    """A mono8 ROS image should become a three-channel BGR image."""
    adapter = ImageAdapter()

    mono = np.array(
        [
            [10, 20],
            [30, 40],
        ],
        dtype=np.uint8,
    )

    message = bridge.cv2_to_imgmsg(
        mono,
        encoding="mono8",
    )

    actual = adapter.to_bgr(
        message
    )

    assert actual.shape == (
        2,
        2,
        3,
    )

    assert np.array_equal(
        actual[:, :, 0],
        mono,
    )

    assert np.array_equal(
        actual[:, :, 1],
        mono,
    )

    assert np.array_equal(
        actual[:, :, 2],
        mono,
    )


def test_output_is_numpy_array(
    bridge,
):
    """Converted image data should use a NumPy array."""
    adapter = ImageAdapter()

    image = np.zeros(
        (
            3,
            4,
            3,
        ),
        dtype=np.uint8,
    )

    message = bridge.cv2_to_imgmsg(
        image,
        encoding="bgr8",
    )

    actual = adapter.to_bgr(
        message
    )

    assert isinstance(
        actual,
        np.ndarray,
    )


def test_output_dimensions_are_preserved(
    bridge,
):
    """Image height and width should survive ROS conversion."""
    adapter = ImageAdapter()

    image = np.zeros(
        (
            7,
            11,
            3,
        ),
        dtype=np.uint8,
    )

    message = bridge.cv2_to_imgmsg(
        image,
        encoding="bgr8",
    )

    actual = adapter.to_bgr(
        message
    )

    assert actual.shape == (
        7,
        11,
        3,
    )


def test_non_image_input_is_rejected():
    """Objects other than ROS Image messages should be rejected."""
    adapter = ImageAdapter()

    with pytest.raises(
        TypeError,
        match=(
            "message must be sensor_msgs.msg.Image"
        ),
    ):
        adapter.to_bgr(
            object()
        )


def test_empty_ros_image_conversion_fails():
    """An empty ROS image should not silently become visual input."""
    adapter = ImageAdapter()

    message = Image()
    message.height = 0
    message.width = 0
    message.encoding = "bgr8"

    with pytest.raises(
        Exception,
    ):
        adapter.to_bgr(
            message
        )
