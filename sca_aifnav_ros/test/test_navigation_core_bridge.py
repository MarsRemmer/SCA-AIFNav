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
    BOOTSTRAP_ACTION_ID,
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


def test_first_observation_uses_bootstrap_stay_action():
    """The first observation should bootstrap learning with STAY."""
    coordinator = FakeCoordinator(
        [
            3,
        ]
    )

    bridge = NavigationCoreBridge(
        coordinator=coordinator
    )

    result = bridge.process_observation(
        observation(
            place_id=2,
            sensory_id=4,
        )
    )

    call = coordinator.calls[0]

    assert (
        call["executed_action_id"]
        == BOOTSTRAP_ACTION_ID
    )

    assert (
        call["current_place_id"]
        == 2
    )

    assert (
        call["sensory_observation"]
        == 4
    )

    assert result.is_bootstrap is True
    assert result.next_action_id == 3
    assert bridge.next_action_id == 3


def test_second_observation_requires_completed_action():
    """A planned action cannot be learned before physical completion."""
    coordinator = FakeCoordinator(
        [
            3,
        ]
    )

    bridge = NavigationCoreBridge(
        coordinator=coordinator
    )

    bridge.process_observation(
        observation()
    )

    with pytest.raises(
        RuntimeError,
        match="no completed physical action",
    ):
        bridge.process_observation(
            observation(
                place_id=1
            )
        )


def test_recorded_action_is_used_by_next_observation():
    """The next cycle should learn the physically completed action."""
    coordinator = FakeCoordinator(
        [
            3,
            7,
        ]
    )

    bridge = NavigationCoreBridge(
        coordinator=coordinator
    )

    bridge.process_observation(
        observation()
    )

    bridge.record_executed_action(
        3
    )

    assert bridge.next_action_id is None
    assert bridge.completed_action_id == 3

    result = bridge.process_observation(
        observation(
            place_id=1,
            sensory_id=2,
        )
    )

    assert (
        coordinator.calls[1][
            "executed_action_id"
        ]
        == 3
    )

    assert result.is_bootstrap is False
    assert result.next_action_id == 7

    assert bridge.completed_action_id is None
    assert bridge.next_action_id == 7


def test_executed_action_must_match_planned_action():
    """The bridge should reject an unexpected physical action ID."""
    coordinator = FakeCoordinator(
        [
            5,
        ]
    )

    bridge = NavigationCoreBridge(
        coordinator=coordinator
    )

    bridge.process_observation(
        observation()
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        bridge.record_executed_action(
            4
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
    coordinator = FakeCoordinator(
        [
            3,
        ]
    )

    bridge = NavigationCoreBridge(
        coordinator=coordinator
    )

    bridge.process_observation(
        observation()
    )

    with pytest.raises(
        TypeError,
        match="must be an integer",
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
