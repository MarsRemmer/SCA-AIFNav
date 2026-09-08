"""Tests for the ROS 2 navigation node."""

import math

from nav_msgs.msg import Odometry
import pytest
import rclpy

from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)


@pytest.fixture
def ros_context():
    """Provide a fresh ROS 2 context for each test."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def odometry_message(
    x,
    y,
):
    """Create a minimal planar odometry message."""
    message = Odometry()

    message.pose.pose.position.x = float(x)
    message.pose.pose.position.y = float(y)

    return message


def test_node_starts_without_odometry(
    ros_context,
):
    """The node should distinguish startup from valid odometry."""
    node = NavigationNode()

    try:
        assert node.has_odometry is False
        assert (
            node.latest_odometry_state
            is None
        )
        assert node.odometry_revision == 0
    finally:
        node.destroy_node()


def test_default_odometry_topic_is_declared(
    ros_context,
):
    """The default input topic should be /odom."""
    node = NavigationNode()

    try:
        assert (
            node.get_parameter(
                "odom_topic"
            ).value
            == "/odom"
        )
    finally:
        node.destroy_node()


def test_first_odometry_message_initializes_state(
    ros_context,
):
    """The first message should establish position without motion."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                3.0,
                -2.0,
            )
        )

        state = (
            node.latest_odometry_state
        )

        assert node.has_odometry is True

        assert state.position == Point2D(
            0.0,
            0.0,
        )

        assert (
            state.travel_heading_rad
            == pytest.approx(0.0)
        )

        assert node.odometry_revision == 1
    finally:
        node.destroy_node()


def test_consecutive_messages_update_travel_heading(
    ros_context,
):
    """Successive positions should update cognitive travel direction."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                1.0,
                1.0,
            )
        )

        node._odometry_callback(
            odometry_message(
                1.0,
                2.0,
            )
        )

        state = (
            node.latest_odometry_state
        )

        assert state.position == Point2D(
            0.0,
            1.0,
        )

        assert (
            state.travel_heading_rad
            == pytest.approx(
                math.pi / 2.0
            )
        )

        assert node.odometry_revision == 2
    finally:
        node.destroy_node()


def test_body_orientation_does_not_replace_travel_heading(
    ros_context,
):
    """Robot yaw should not replace displacement-based heading."""
    node = NavigationNode()

    try:
        first = odometry_message(
            0.0,
            0.0,
        )

        first.pose.pose.orientation.z = 1.0
        first.pose.pose.orientation.w = 0.0

        node._odometry_callback(first)

        second = odometry_message(
            1.0,
            0.0,
        )

        second.pose.pose.orientation.z = 1.0
        second.pose.pose.orientation.w = 0.0

        node._odometry_callback(second)

        assert (
            node
            .latest_odometry_state
            .travel_heading_rad
            == pytest.approx(0.0)
        )
    finally:
        node.destroy_node()


def test_revision_counts_every_processed_message(
    ros_context,
):
    """Each accepted callback should advance the odometry revision."""
    node = NavigationNode()

    try:
        for index in range(5):
            node._odometry_callback(
                odometry_message(
                    float(index),
                    0.0,
                )
            )

        assert node.odometry_revision == 5
    finally:
        node.destroy_node()


def test_posterior_correction_realigns_cached_odometry(
    ros_context,
):
    """Posterior correction should align physical and cognitive XY."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                10.0,
                20.0,
            )
        )

        node._odometry_callback(
            odometry_message(
                12.0,
                21.0,
            )
        )

        assert (
            node.latest_odometry_state.position
            == Point2D(
                2.0,
                1.0,
            )
        )

        corrected_position = Point2D(
            5.0,
            -3.0,
        )

        posterior_place_id = (
            node._place_memory.resolve_place(
                corrected_position
            )
        )

        class Result:
            pass

        cycle_result = Result()
        cycle_result.posterior_place_id = (
            posterior_place_id
        )

        decision = Result()
        decision.cycle_result = cycle_result

        corrected = (
            node._apply_posterior_cognitive_correction(
                decision
            )
        )

        assert corrected is True

        assert (
            node.latest_odometry_state.position
            == corrected_position
        )

        assert (
            node._internal_cognitive_state.position
            == corrected_position
        )

        assert (
            node._internal_cognitive_place_id
            == posterior_place_id
        )

        assert (
            node._odometry_adapter.alignment_offset
            == Point2D(
                3.0,
                -4.0,
            )
        )
    finally:
        node.destroy_node()


def test_odometry_continues_from_posterior_correction(
    ros_context,
):
    """Later physical motion should continue from corrected cognitive XY."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_message(
                10.0,
                20.0,
            )
        )

        node._odometry_callback(
            odometry_message(
                12.0,
                21.0,
            )
        )

        corrected_position = Point2D(
            5.0,
            -3.0,
        )

        posterior_place_id = (
            node._place_memory.resolve_place(
                corrected_position
            )
        )

        class Result:
            pass

        cycle_result = Result()
        cycle_result.posterior_place_id = (
            posterior_place_id
        )

        decision = Result()
        decision.cycle_result = cycle_result

        node._apply_posterior_cognitive_correction(
            decision
        )

        node._odometry_callback(
            odometry_message(
                13.0,
                21.0,
            )
        )

        assert (
            node.latest_odometry_state.position
            == Point2D(
                6.0,
                -3.0,
            )
        )

        assert (
            node.latest_odometry_state.travel_heading_rad
            == pytest.approx(
                0.0
            )
        )

        assert (
            node._odometry_adapter.alignment_offset
            == Point2D(
                3.0,
                -4.0,
            )
        )
    finally:
        node.destroy_node()
