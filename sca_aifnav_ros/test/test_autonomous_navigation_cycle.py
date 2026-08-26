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
