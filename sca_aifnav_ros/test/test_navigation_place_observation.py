"""Tests for cognitive place observation integration."""

from nav_msgs.msg import Odometry
import pytest
import rclpy

from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)
from sca_aifnav_ros.visual_observation import (
    VisualObservationResult,
)


@pytest.fixture
def ros_context():
    """Provide a fresh ROS 2 context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def odometry_at(
    x,
    y,
):
    """Create one planar odometry observation."""
    message = Odometry()

    message.pose.pose.position.x = float(
        x
    )
    message.pose.pose.position.y = float(
        y
    )
    message.pose.pose.orientation.w = 1.0

    return message


def visual_result(
    observation_id=0,
):
    """Create one completed visual observation."""
    return VisualObservationResult(
        observation_id=observation_id,
        match_scores=(1.0,),
        confidence_threshold=0.9,
        attempt_count=1,
    )


def begin_observation_cycle(
    node,
    x,
    y,
    visual_id=0,
):
    """Prepare one completed visual observation at a position."""
    node._odometry_callback(
        odometry_at(
            x,
            y,
        )
    )

    node._latest_visual_observation = (
        visual_result(
            visual_id
        )
    )

    node._latest_place_observation_id = None


def test_place_observation_is_empty_on_start(
    ros_context,
):
    """A new node should not contain a resolved place."""
    node = NavigationNode()

    try:
        assert (
            node.has_place_observation
            is False
        )

        assert (
            node.place_observation_id
            is None
        )

        assert (
            node.place_memory_size
            == 0
        )
    finally:
        node.destroy_node()


def test_place_resolution_waits_for_visual_observation(
    ros_context,
):
    """High-frequency odometry alone must not create places."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_at(
                1.0,
                2.0,
            )
        )

        assert (
            node.resolve_current_place_observation()
            is None
        )

        assert (
            node.place_memory_size
            == 0
        )
    finally:
        node.destroy_node()


def test_first_completed_observation_creates_first_place(
    ros_context,
):
    """The first completed observation should create place zero."""
    node = NavigationNode()

    try:
        begin_observation_cycle(
            node,
            1.0,
            2.0,
        )

        place_id = (
            node.resolve_current_place_observation()
        )

        assert place_id == 0
        assert node.place_observation_id == 0
        assert node.has_place_observation
        assert node.place_memory_size == 1

        place = node._place_memory.place(
            0
        )

        assert place.x == pytest.approx(
            1.0
        )
        assert place.y == pytest.approx(
            2.0
        )
    finally:
        node.destroy_node()


def test_nearby_position_reuses_existing_place(
    ros_context,
):
    """Positions inside the fixed radius should reuse a place ID."""
    node = NavigationNode()

    try:
        begin_observation_cycle(
            node,
            1.0,
            2.0,
            visual_id=0,
        )

        assert (
            node.resolve_current_place_observation()
            == 0
        )

        begin_observation_cycle(
            node,
            1.3,
            2.0,
            visual_id=1,
        )

        assert (
            node.resolve_current_place_observation()
            == 0
        )

        assert (
            node.place_memory_size
            == 1
        )
    finally:
        node.destroy_node()


def test_distant_position_creates_new_place(
    ros_context,
):
    """A position outside the fixed radius should create a new ID."""
    node = NavigationNode()

    try:
        begin_observation_cycle(
            node,
            0.0,
            0.0,
        )

        assert (
            node.resolve_current_place_observation()
            == 0
        )

        begin_observation_cycle(
            node,
            1.0,
            0.0,
            visual_id=1,
        )

        assert (
            node.resolve_current_place_observation()
            == 1
        )

        assert (
            node.place_memory_size
            == 2
        )
    finally:
        node.destroy_node()


def test_same_cycle_resolution_is_idempotent(
    ros_context,
):
    """Repeated resolution must not create duplicate places."""
    node = NavigationNode()

    try:
        begin_observation_cycle(
            node,
            5.0,
            -2.0,
        )

        first = (
            node.resolve_current_place_observation()
        )

        second = (
            node.resolve_current_place_observation()
        )

        assert first == 0
        assert second == 0
        assert node.place_memory_size == 1
    finally:
        node.destroy_node()
