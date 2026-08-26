"""Tests for the ROS-to-navigation-core bridge."""

from types import SimpleNamespace

import pytest

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)
from sca_aifnav_ros.navigation_core_bridge import (
    NavigationCoreBridge,
)
from sca_aifnav_ros.navigation_observation import (
    NavigationObservation,
)


class FakeCoordinator:
    """Provide deterministic core planning results."""

    def __init__(
        self,
        selected_actions,
    ):
        self.selected_actions = list(
            selected_actions
        )
        self.calls = []

    def step_and_plan(
        self,
        **kwargs,
    ):
        """Capture one call and return the next configured action."""
        self.calls.append(
            kwargs
        )

        selected_action = (
            self.selected_actions.pop(0)
        )

        return SimpleNamespace(
            planning=SimpleNamespace(
                selected_action=(
                    selected_action
                )
            )
        )


def observation(
    place_id=0,
    sensory_id=0,
):
    """Create one deterministic navigation observation."""
    return NavigationObservation(
        state=CognitiveOdomState(
            position=Point2D(
                float(place_id),
                0.0,
            ),
            travel_heading_rad=0.0,
        ),
        sensory_observation=sensory_id,
        place_observation=place_id,
        obstacle_distances=tuple(
            float(index)
            for index in range(12)
        ),
        odometry_revision=1,
        scan_revision=1,
    )


def test_default_bridge_shares_supplied_place_memory():
    """Core coordinator should use the ROS-owned place memory."""
    memory = BaselinePlaceMemory()

    bridge = NavigationCoreBridge(
        memory=memory
    )

    assert bridge.memory is memory
    assert (
        bridge.coordinator.memory
        is memory
    )


def test_first_observation_initializes_without_executed_action():
    """The first observation initializes without a fabricated action."""
    bridge = NavigationCoreBridge()

    result = bridge.process_observation(
        observation()
    )

    assert result.is_bootstrap is True
    assert result.executed_action_id is None
    assert result.cycle_result.learning is None
    assert (
        result.next_action_id
        == result.cycle_result.planning.selected_action
    )


def test_second_observation_requires_completed_action():
    """A new observation cannot be learned before action completion."""
    bridge = NavigationCoreBridge()

    bridge.process_observation(
        observation()
    )

    with pytest.raises(
        RuntimeError,
        match="completed physical action",
    ):
        bridge.process_observation(
            observation()
        )


def test_recorded_action_is_used_by_next_observation():
    """The next cycle learns the physically completed planned action."""
    bridge = NavigationCoreBridge()

    first = bridge.process_observation(
        observation()
    )

    planned_action = (
        first.next_action_id
    )

    bridge.record_executed_action(
        planned_action
    )

    second = bridge.process_observation(
        observation()
    )

    assert second.is_bootstrap is False

    assert (
        second.executed_action_id
        == planned_action
    )


def test_executed_action_must_match_planned_action():
    """The bridge rejects a physical action different from the plan."""
    bridge = NavigationCoreBridge()

    first = bridge.process_observation(
        observation()
    )

    wrong_action = (
        first.next_action_id + 1
    ) % bridge.motion_set.ACTION_COUNT

    with pytest.raises(
        ValueError,
    ):
        bridge.record_executed_action(
            wrong_action
        )


@pytest.mark.parametrize(
    "action_id",
    [
        True,
        3.0,
        "3",
    ],
)
def test_executed_action_id_must_be_integer(
    action_id,
):
    """Executed physical actions require integer IDs."""
    bridge = NavigationCoreBridge()

    bridge.process_observation(
        observation()
    )

    with pytest.raises(
        TypeError,
    ):
        bridge.record_executed_action(
            action_id
        )


def test_no_action_can_be_recorded_before_planning():
    """Execution completion requires an existing planned action."""
    bridge = NavigationCoreBridge(
        coordinator=FakeCoordinator(
            []
        )
    )

    with pytest.raises(
        RuntimeError,
        match="no planned action",
    ):
        bridge.record_executed_action(
            0
        )


def test_default_bridge_uses_exploration_navigation_mode():
    """Default runtime should reproduce reference exploration mode."""
    bridge = NavigationCoreBridge()

    interface = bridge.coordinator.model_interface

    assert interface.use_utility is False
    assert interface.use_state_information_gain is True
    assert interface.use_inductive_inference is False
