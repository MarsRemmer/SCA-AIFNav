"""Tests for experiment-only navigation action limits."""

from types import SimpleNamespace

import pytest
import rclpy

from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)


@pytest.fixture
def ros_context():
    """Provide one clean ROS context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def decision(executed_action_id):
    """Create a minimal navigation decision."""
    return SimpleNamespace(
        executed_action_id=executed_action_id,
    )


def test_default_experiment_limit_is_disabled(
    ros_context,
):
    node = NavigationNode()

    try:
        assert (
            node.get_parameter(
                "experiment_action_limit"
            ).value
            == 0
        )

        assert (
            node.experiment_completed_actions
            == 0
        )
    finally:
        node.destroy_node()


def test_bootstrap_does_not_count_as_action(
    ros_context,
):
    node = NavigationNode()

    try:
        node._experiment_action_limit = 2
        node._autonomous_navigation_active = True

        reached = (
            node._record_experiment_decision(
                decision(None)
            )
        )

        assert reached is False
        assert (
            node.experiment_completed_actions
            == 0
        )
        assert (
            node.autonomous_navigation_active
            is True
        )
    finally:
        node.destroy_node()


def test_limit_stops_after_completed_actions(
    ros_context,
):
    node = NavigationNode()

    try:
        node._experiment_action_limit = 2
        node._autonomous_navigation_active = True

        assert (
            node._record_experiment_decision(
                decision(3)
            )
            is False
        )

        assert (
            node.experiment_completed_actions
            == 1
        )

        assert (
            node.autonomous_navigation_active
            is True
        )

        assert (
            node._record_experiment_decision(
                decision(7)
            )
            is True
        )

        assert (
            node.experiment_completed_actions
            == 2
        )

        assert (
            node.autonomous_navigation_active
            is False
        )

    finally:
        node.destroy_node()


def test_disabled_limit_still_counts_actions(
    ros_context,
):
    node = NavigationNode()

    try:
        node._experiment_action_limit = 0
        node._autonomous_navigation_active = True

        for action_id in (1, 2, 3):
            assert (
                node._record_experiment_decision(
                    decision(action_id)
                )
                is False
            )

        assert (
            node.experiment_completed_actions
            == 3
        )

        assert (
            node.autonomous_navigation_active
            is True
        )

    finally:
        node.destroy_node()
