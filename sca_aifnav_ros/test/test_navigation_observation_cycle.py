"""Tests for completed navigation observation cycles."""

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
    """Create deterministic planar odometry."""
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
    observation_id=3,
):
    """Create one completed visual observation."""
    return VisualObservationResult(
        observation_id=observation_id,
        match_scores=(1.0,),
        confidence_threshold=0.9,
        attempt_count=1,
    )


def prepare_complete_inputs(
    node,
    x=1.0,
    y=2.0,
    visual_id=3,
):
    """Provide all inputs required by one navigation observation."""
    node._odometry_callback(
        odometry_at(
            x,
            y,
        )
    )

    node._latest_obstacle_distances = tuple(
        float(index)
        for index in range(12)
    )

    node._scan_revision = 1

    node._latest_visual_observation = (
        visual_result(
            visual_id
        )
    )


def test_navigation_observation_is_empty_on_start(
    ros_context,
):
    """A new node should not expose a completed observation."""
    node = NavigationNode()

    try:
        assert (
            node.has_navigation_observation
            is False
        )

        assert (
            node.latest_navigation_observation
            is None
        )
    finally:
        node.destroy_node()


def test_navigation_observation_waits_for_required_sensors(
    ros_context,
):
    """Visual data alone should not create a complete observation."""
    node = NavigationNode()

    try:
        node._latest_visual_observation = (
            visual_result()
        )

        assert (
            node.process_completed_navigation_observation()
            is None
        )

        assert (
            node.place_memory_size
            == 0
        )
    finally:
        node.destroy_node()


def test_completed_cycle_maps_directly_to_core_inputs(
    ros_context,
):
    """One complete cycle should freeze all core-facing observations."""
    node = NavigationNode()

    try:
        prepare_complete_inputs(
            node,
            x=1.0,
            y=2.0,
            visual_id=4,
        )

        observation = (
            node.process_completed_navigation_observation()
        )

        assert observation is not None

        assert (
            observation.sensory_observation
            == 4
        )

        assert (
            observation.place_observation
            == 0
        )

        assert (
            observation.state.position.x
            == pytest.approx(0.0)
        )

        assert (
            observation.state.position.y
            == pytest.approx(0.0)
        )

        assert (
            observation.obstacle_distances
            == tuple(
                float(index)
                for index in range(12)
            )
        )

        assert (
            observation.odometry_revision
            == 1
        )

        assert (
            observation.scan_revision
            == 1
        )

        assert node.has_navigation_observation
        assert (
            node.latest_navigation_observation
            is observation
        )
    finally:
        node.destroy_node()


def test_completed_cycle_is_frozen_from_later_sensor_updates(
    ros_context,
):
    """Later odometry and scans must not alter the completed cycle."""
    node = NavigationNode()

    try:
        prepare_complete_inputs(
            node,
            x=1.0,
            y=2.0,
        )

        first = (
            node.process_completed_navigation_observation()
        )

        node._odometry_callback(
            odometry_at(
                9.0,
                9.0,
            )
        )

        node._latest_obstacle_distances = tuple(
            99.0
            for _ in range(12)
        )

        node._scan_revision += 1

        second = (
            node.process_completed_navigation_observation()
        )

        assert second is first

        assert (
            first.state.position.x
            == pytest.approx(0.0)
        )

        assert (
            first.state.position.y
            == pytest.approx(0.0)
        )

        assert (
            first.obstacle_distances[0]
            == 0.0
        )

        assert first.odometry_revision == 1
        assert first.scan_revision == 1
    finally:
        node.destroy_node()


def test_place_resolution_occurs_at_completed_cycle_boundary(
    ros_context,
):
    """Odometry updates before cycle completion must not create places."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_at(
                0.0,
                0.0,
            )
        )

        node._odometry_callback(
            odometry_at(
                0.2,
                0.0,
            )
        )

        node._odometry_callback(
            odometry_at(
                0.4,
                0.0,
            )
        )

        assert node.place_memory_size == 0

        node._latest_obstacle_distances = tuple(
            1.0
            for _ in range(12)
        )

        node._scan_revision = 1
        node._latest_visual_observation = (
            visual_result()
        )

        observation = (
            node.process_completed_navigation_observation()
        )

        assert (
            observation.place_observation
            == 0
        )

        assert (
            node.place_memory_size
            == 1
        )
    finally:
        node.destroy_node()
