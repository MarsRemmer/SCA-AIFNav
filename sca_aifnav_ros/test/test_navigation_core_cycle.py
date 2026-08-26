"""Tests for navigation-core integration in the ROS node."""

from types import SimpleNamespace

import pytest
import rclpy

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)
from sca_aifnav_ros.navigation_observation import (
    NavigationObservation,
)


class FakeCoreBridge:
    """Capture navigation-core bridge interactions."""

    def __init__(
        self,
        next_action_id=4,
    ):
        self.next_action_id = (
            next_action_id
        )
        self.process_calls = []
        self.executed_actions = []

    def process_observation(
        self,
        observation,
    ):
        """Return one deterministic decision."""
        self.process_calls.append(
            observation
        )

        return SimpleNamespace(
            next_action_id=(
                self.next_action_id
            )
        )

    def record_executed_action(
        self,
        action_id,
    ):
        """Capture one completed physical action."""
        self.executed_actions.append(
            action_id
        )
        self.next_action_id = None


@pytest.fixture
def ros_context():
    """Provide a fresh ROS 2 context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def observation():
    """Create one deterministic completed navigation observation."""
    return NavigationObservation(
        state=CognitiveOdomState(
            position=Point2D(
                1.0,
                2.0,
            ),
            travel_heading_rad=0.0,
        ),
        sensory_observation=3,
        place_observation=0,
        obstacle_distances=tuple(
            1.0
            for _ in range(12)
        ),
        odometry_revision=1,
        scan_revision=1,
    )


def test_navigation_core_shares_node_place_memory(
    ros_context,
):
    """ROS place observations and core learning must share memory."""
    node = NavigationNode()

    try:
        assert (
            node._navigation_core_bridge.memory
            is node._place_memory
        )

        assert (
            node._navigation_core_bridge
            .coordinator
            .memory
            is node._place_memory
        )
    finally:
        node.destroy_node()


def test_core_cycle_waits_for_completed_observation(
    ros_context,
):
    """Core planning should not run before an observation exists."""
    node = NavigationNode()

    bridge = FakeCoreBridge()
    node._navigation_core_bridge = bridge

    try:
        result = (
            node.process_completed_navigation_cycle()
        )

        assert result is None
        assert bridge.process_calls == []
    finally:
        node.destroy_node()


def test_completed_observation_reaches_core_bridge(
    ros_context,
):
    """A completed ROS observation should enter the core once."""
    node = NavigationNode()

    bridge = FakeCoreBridge(
        next_action_id=6
    )

    node._navigation_core_bridge = bridge

    completed = observation()

    node._latest_navigation_observation = (
        completed
    )

    try:
        result = (
            node.process_completed_navigation_cycle()
        )

        assert (
            bridge.process_calls
            == [completed]
        )

        assert (
            result.next_action_id
            == 6
        )

        assert (
            node.latest_navigation_decision
            is result
        )

        assert (
            node.next_navigation_action_id
            == 6
        )
    finally:
        node.destroy_node()


def test_same_observation_is_not_processed_twice(
    ros_context,
):
    """Repeated access should reuse the already computed decision."""
    node = NavigationNode()

    bridge = FakeCoreBridge()
    node._navigation_core_bridge = bridge
    node._latest_navigation_observation = (
        observation()
    )

    try:
        first = (
            node.process_completed_navigation_cycle()
        )

        second = (
            node.process_completed_navigation_cycle()
        )

        assert second is first

        assert (
            len(bridge.process_calls)
            == 1
        )
    finally:
        node.destroy_node()


def test_executed_action_is_forwarded_to_core_bridge(
    ros_context,
):
    """Physical action completion should be recorded explicitly."""
    node = NavigationNode()

    bridge = FakeCoreBridge(
        next_action_id=5
    )

    node._navigation_core_bridge = bridge

    try:
        node.record_executed_navigation_action(
            5
        )

        assert (
            bridge.executed_actions
            == [5]
        )

        assert (
            node.next_navigation_action_id
            is None
        )
    finally:
        node.destroy_node()
