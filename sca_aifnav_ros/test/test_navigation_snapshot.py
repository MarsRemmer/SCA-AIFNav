"""Tests for navigation sensor snapshot capture."""

import math

from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import LaserScan

from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)
from sca_aifnav_ros.sensor_snapshot import (
    NavigationSensorSnapshot,
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


def odometry_message(
    x,
    y,
    yaw_rad=0.0,
):
    """Create one planar odometry message."""
    message = Odometry()

    message.pose.pose.position.x = float(
        x
    )
    message.pose.pose.position.y = float(
        y
    )

    message.pose.pose.orientation.z = math.sin(
        yaw_rad / 2.0
    )
    message.pose.pose.orientation.w = math.cos(
        yaw_rad / 2.0
    )

    return message


def centered_scan(
    first_distance=1.0,
):
    """Create twelve rays centered on the cognitive action sectors."""
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
    message.range_max = 100.0

    message.ranges = [
        float(
            first_distance
            + index
        )
        for index in range(
            12
        )
    ]

    return message


def image_message(
    bridge,
    value,
):
    """Create one deterministic BGR ROS image."""
    image = np.full(
        (
            4,
            6,
            3,
        ),
        value,
        dtype=np.uint8,
    )

    return bridge.cv2_to_imgmsg(
        image,
        encoding="bgr8",
    )


def test_snapshot_is_not_ready_on_start(
    ros_context,
):
    """A new node should not expose a complete sensor snapshot."""
    node = NavigationNode()

    try:
        assert (
            node.sensor_snapshot_ready
            is False
        )

        assert (
            node.capture_sensor_snapshot()
            is None
        )
    finally:
        node.destroy_node()


def test_odometry_alone_is_not_enough(
    ros_context,
):
    """Odometry alone should not produce a navigation snapshot."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                1.0,
                2.0,
            )
        )

        assert (
            node.sensor_snapshot_ready
            is False
        )

        assert (
            node.capture_sensor_snapshot()
            is None
        )
    finally:
        node.destroy_node()


def test_odometry_and_scan_still_require_image(
    ros_context,
):
    """A snapshot should wait for camera input as well."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                1.0,
                2.0,
            )
        )

        node._scan_callback(
            centered_scan()
        )

        assert (
            node.sensor_snapshot_ready
            is False
        )

        assert (
            node.capture_sensor_snapshot()
            is None
        )
    finally:
        node.destroy_node()


def test_complete_sensor_state_creates_snapshot(
    ros_context,
    bridge,
):
    """One update from every sensor stream should create a snapshot."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                1.0,
                2.0,
            )
        )

        node._scan_callback(
            centered_scan()
        )

        node._image_callback(
            image_message(
                bridge,
                25,
            )
        )

        assert (
            node.sensor_snapshot_ready
            is True
        )

        snapshot = (
            node.capture_sensor_snapshot()
        )

        assert isinstance(
            snapshot,
            NavigationSensorSnapshot,
        )

        assert (
            snapshot.state.position.x
            == pytest.approx(1.0)
        )
        assert (
            snapshot.state.position.y
            == pytest.approx(2.0)
        )

        assert (
            snapshot.obstacle_distances
            == pytest.approx(
                [
                    float(value)
                    for value in range(
                        1,
                        13,
                    )
                ]
            )
        )

        assert np.all(
            snapshot.image
            == 25
        )

        assert snapshot.odometry_revision == 1
        assert snapshot.scan_revision == 1
        assert snapshot.image_revision == 1
    finally:
        node.destroy_node()


def test_snapshot_remains_frozen_after_new_sensor_updates(
    ros_context,
    bridge,
):
    """Later ROS updates should not mutate an existing snapshot."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                1.0,
                2.0,
            )
        )

        node._scan_callback(
            centered_scan(
                first_distance=1.0
            )
        )

        node._image_callback(
            image_message(
                bridge,
                10,
            )
        )

        first_snapshot = (
            node.capture_sensor_snapshot()
        )

        node._odometry_callback(
            odometry_message(
                5.0,
                6.0,
            )
        )

        node._scan_callback(
            centered_scan(
                first_distance=21.0
            )
        )

        node._image_callback(
            image_message(
                bridge,
                200,
            )
        )

        assert (
            first_snapshot.state.position.x
            == pytest.approx(1.0)
        )
        assert (
            first_snapshot.state.position.y
            == pytest.approx(2.0)
        )

        assert (
            first_snapshot.obstacle_distances[
                0
            ]
            == pytest.approx(1.0)
        )

        assert np.all(
            first_snapshot.image
            == 10
        )

        assert (
            first_snapshot.odometry_revision
            == 1
        )
        assert (
            first_snapshot.scan_revision
            == 1
        )
        assert (
            first_snapshot.image_revision
            == 1
        )
    finally:
        node.destroy_node()


def test_new_snapshot_uses_latest_sensor_updates(
    ros_context,
    bridge,
):
    """A later snapshot should capture the newest available sensor state."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                1.0,
                2.0,
            )
        )
        node._scan_callback(
            centered_scan(
                first_distance=1.0
            )
        )
        node._image_callback(
            image_message(
                bridge,
                10,
            )
        )

        first_snapshot = (
            node.capture_sensor_snapshot()
        )

        node._odometry_callback(
            odometry_message(
                5.0,
                6.0,
            )
        )
        node._scan_callback(
            centered_scan(
                first_distance=21.0
            )
        )
        node._image_callback(
            image_message(
                bridge,
                200,
            )
        )

        second_snapshot = (
            node.capture_sensor_snapshot()
        )

        assert (
            first_snapshot.odometry_revision
            == 1
        )

        assert (
            second_snapshot.odometry_revision
            == 2
        )
        assert (
            second_snapshot.scan_revision
            == 2
        )
        assert (
            second_snapshot.image_revision
            == 2
        )

        assert (
            second_snapshot.state.position.x
            == pytest.approx(5.0)
        )
        assert (
            second_snapshot.state.position.y
            == pytest.approx(6.0)
        )

        assert (
            second_snapshot.obstacle_distances[
                0
            ]
            == pytest.approx(21.0)
        )

        assert np.all(
            second_snapshot.image
            == 200
        )
    finally:
        node.destroy_node()
