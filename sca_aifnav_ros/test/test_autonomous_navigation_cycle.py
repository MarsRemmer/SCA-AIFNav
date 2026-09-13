"""Tests for repeated autonomous navigation lifecycle control."""

from types import SimpleNamespace

import pytest
import rclpy

from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)


class FakeMotionExecutor:
    """Provide deterministic physical-action state."""

    def __init__(
        self,
        is_active=False,
    ):
        self.is_active = is_active


@pytest.fixture
def ros_context():
    """Provide a fresh ROS 2 context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def test_autonomous_navigation_is_disabled_on_start(
    ros_context,
):
    """A new node should not begin autonomous movement itself."""
    node = NavigationNode()

    try:
        assert (
            node.autonomous_navigation_active
            is False
        )
    finally:
        node.destroy_node()


def test_autonomous_navigation_starts_with_panorama(
    ros_context,
):
    """Starting the loop should begin the first observation."""
    node = NavigationNode()

    calls = []

    node.start_panorama_acquisition = (
        lambda: calls.append(
            "panorama"
        ) or True
    )

    try:
        assert (
            node.start_autonomous_navigation()
            is True
        )

        assert calls == [
            "panorama",
        ]

        assert (
            node.autonomous_navigation_active
            is True
        )
    finally:
        node.destroy_node()


def test_autonomous_start_waits_if_panorama_cannot_start(
    ros_context,
):
    """Navigation should remain inactive when observation cannot start."""
    node = NavigationNode()

    node.start_panorama_acquisition = (
        lambda: False
    )

    try:
        assert (
            node.start_autonomous_navigation()
            is False
        )

        assert (
            node.autonomous_navigation_active
            is False
        )
    finally:
        node.destroy_node()


def test_observation_phase_waits_for_completed_cycle(
    ros_context,
):
    """The loop should wait while panorama observation is incomplete."""
    node = NavigationNode()

    node._autonomous_navigation_active = True

    node._navigation_motion_executor = (
        FakeMotionExecutor(
            is_active=False
        )
    )

    calls = []

    node.process_completed_navigation_cycle = (
        lambda: calls.append(
            "core"
        ) or None
    )

    node.start_planned_navigation_action = (
        lambda: calls.append(
            "action"
        ) or True
    )

    try:
        result = (
            node.step_autonomous_navigation()
        )

        assert result is None

        assert calls == [
            "core",
        ]
    finally:
        node.destroy_node()


def test_completed_observation_starts_planned_action(
    ros_context,
):
    """A completed core decision should begin physical execution."""
    node = NavigationNode()

    node._autonomous_navigation_active = True

    node._navigation_motion_executor = (
        FakeMotionExecutor(
            is_active=False
        )
    )

    decision = SimpleNamespace(
        next_action_id=3
    )

    calls = []

    node.process_completed_navigation_cycle = (
        lambda: decision
    )

    node.start_planned_navigation_action = (
        lambda: calls.append(
            "start_action"
        ) or True
    )

    try:
        result = (
            node.step_autonomous_navigation()
        )

        assert result is decision

        assert calls == [
            "start_action",
        ]

        assert (
            node.autonomous_navigation_active
            is True
        )
    finally:
        node.destroy_node()


def test_active_motion_continues_without_new_observation(
    ros_context,
):
    """The loop should keep advancing an unfinished physical action."""
    node = NavigationNode()

    node._autonomous_navigation_active = True

    node._navigation_motion_executor = (
        FakeMotionExecutor(
            is_active=True
        )
    )

    motion_update = SimpleNamespace(
        completed_action_id=None
    )

    calls = []

    node.step_navigation_action = (
        lambda: calls.append(
            "motion"
        ) or motion_update
    )

    node.start_panorama_acquisition = (
        lambda: calls.append(
            "panorama"
        ) or True
    )

    try:
        result = (
            node.step_autonomous_navigation()
        )

        assert result is motion_update

        assert calls == [
            "motion",
        ]
    finally:
        node.destroy_node()


def test_completed_motion_starts_next_panorama(
    ros_context,
):
    """Physical completion should trigger the next observation cycle."""
    node = NavigationNode()

    node._autonomous_navigation_active = True

    node._navigation_motion_executor = (
        FakeMotionExecutor(
            is_active=True
        )
    )

    motion_update = SimpleNamespace(
        completed_action_id=5
    )

    calls = []

    node.step_navigation_action = (
        lambda: calls.append(
            "motion"
        ) or motion_update
    )

    node.start_panorama_acquisition = (
        lambda: calls.append(
            "panorama"
        ) or True
    )

    try:
        result = (
            node.step_autonomous_navigation()
        )

        assert result is motion_update

        assert calls == [
            "motion",
            "panorama",
        ]

        assert (
            node.autonomous_navigation_active
            is True
        )
    finally:
        node.destroy_node()


def test_failed_next_panorama_stops_automatic_loop(
    ros_context,
):
    """The loop should stop rather than continue with stale data."""
    node = NavigationNode()

    node._autonomous_navigation_active = True

    node._navigation_motion_executor = (
        FakeMotionExecutor(
            is_active=True
        )
    )

    node.step_navigation_action = (
        lambda: SimpleNamespace(
            completed_action_id=2
        )
    )

    node.start_panorama_acquisition = (
        lambda: False
    )

    try:
        node.step_autonomous_navigation()

        assert (
            node.autonomous_navigation_active
            is False
        )
    finally:
        node.destroy_node()


def test_missing_execution_target_is_explicit_error(
    ros_context,
):
    """A core plan without a physical target should never be hidden."""
    node = NavigationNode()

    node._autonomous_navigation_active = True

    node._navigation_motion_executor = (
        FakeMotionExecutor(
            is_active=False
        )
    )

    node.process_completed_navigation_cycle = (
        lambda: SimpleNamespace(
            next_action_id=3
        )
    )

    node.start_planned_navigation_action = (
        lambda: False
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="no executable",
        ):
            node.step_autonomous_navigation()

        assert (
            node.autonomous_navigation_active
            is False
        )
    finally:
        node.destroy_node()


def test_control_timer_autostarts_once_when_sensors_are_ready(
    ros_context,
):
    node = NavigationNode()

    starts = []

    try:
        node._latest_odometry_state = object()
        node._latest_obstacle_distances = (
            [1.0] * 12
        )
        node._latest_image = object()
        node._latest_left_image = object()
        node._latest_right_image = object()
        node._latest_physical_yaw_rad = 0.0

        node._image_revision = 1
        node._left_image_revision = 1
        node._right_image_revision = 1

        node.start_autonomous_navigation = (
            lambda: starts.append(True) or True
        )

        node._navigation_control_timer_callback()

        assert starts == [True]
        assert (
            node._autonomous_navigation_started_once
            is True
        )

        node._autonomous_navigation_active = False

        node._navigation_control_timer_callback()

        assert starts == [True]

    finally:
        node.destroy_node()


def test_goal_reached_stops_before_starting_another_action(
    ros_context,
):
    """Goal completion should stop and wait in the selected goal mode."""
    node = NavigationNode()

    node._autonomous_navigation_active = True

    node._navigation_motion_executor = (
        FakeMotionExecutor(
            is_active=False
        )
    )

    decision = SimpleNamespace(
        next_action_id=None,
        executed_action_id=None,
        goal_reached=True,
    )

    calls = []

    node.process_completed_navigation_cycle = (
        lambda: decision
    )

    node.start_planned_navigation_action = (
        lambda: calls.append(
            "start_action"
        ) or True
    )

    try:
        result = (
            node.step_autonomous_navigation()
        )

        assert result is decision
        assert calls == []

        assert (
            node.autonomous_navigation_active
            is False
        )

        assert (
            node._autonomous_navigation_started_once
            is True
        )

    finally:
        node.destroy_node()


def test_goal_mode_change_updates_node_decision_cache(
    ros_context,
):
    """Node and bridge should expose the same replanned decision."""
    node = NavigationNode()

    replacement = SimpleNamespace(
        next_action_id=4
    )

    calls = []

    node._navigation_core_bridge.set_goal_navigation = (
        lambda **kwargs: calls.append(
            kwargs
        ) or replacement
    )

    try:
        result = node.set_goal_navigation(
            mode="goal_direct",
            place_observation=0,
        )

        assert result is replacement

        assert (
            node.latest_navigation_decision
            is replacement
        )

        assert calls == [
            {
                "mode": "goal_direct",
                "sensory_observation": -1,
                "place_observation": 0,
                "preference_weight": 10.0,
            }
        ]

    finally:
        node.destroy_node()


def test_exploration_mode_change_updates_node_decision_cache(
    ros_context,
):
    """Explicit exploration selection should synchronize node cache."""
    node = NavigationNode()

    replacement = SimpleNamespace(
        next_action_id=7
    )

    node._navigation_core_bridge.set_exploration_navigation = (
        lambda: replacement
    )

    try:
        result = (
            node.set_exploration_navigation()
        )

        assert result is replacement

        assert (
            node.latest_navigation_decision
            is replacement
        )

    finally:
        node.destroy_node()


def test_mode_change_rejected_during_active_physical_action(
    ros_context,
):
    """Do not replace a plan while its physical action is executing."""
    node = NavigationNode()

    node._navigation_motion_executor = (
        FakeMotionExecutor(
            is_active=True
        )
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="physical action is active",
        ):
            node.set_goal_navigation(
                mode="goal_direct",
                place_observation=0,
            )

    finally:
        node.destroy_node()
