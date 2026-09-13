"""Tests for the ROS-to-navigation-core bridge."""

from types import SimpleNamespace

import pytest

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.navigation_mode import (
    EXPLORE,
    GOAL_BALANCED,
    GOAL_DIRECT,
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


def test_default_bridge_uses_reference_map_growth_parameters():
    """Default runtime should reproduce reference cognitive-map growth."""
    bridge = NavigationCoreBridge()

    assert (
        bridge.coordinator.learning.robot_dimension
        == pytest.approx(0.3)
    )

    assert (
        bridge.coordinator.learning.max_lookahead_steps
        == 8
    )


def test_default_bridge_uses_reference_mcts_parameters():
    """Default runtime should reproduce reference MCTS parameters."""
    bridge = NavigationCoreBridge()

    assert bridge.coordinator.num_simulations == 30

    assert (
        bridge.coordinator.max_rollout_depth
        == 10
    )

    assert bridge.coordinator.c_param == pytest.approx(
        5.0
    )


def test_default_bridge_uses_exploration_navigation_mode():
    """Default runtime should reproduce reference exploration mode."""
    bridge = NavigationCoreBridge()

    interface = bridge.coordinator.model_interface

    assert bridge.coordinator.navigation_mode == EXPLORE
    assert interface.use_utility is False
    assert interface.use_state_information_gain is True
    assert interface.use_inductive_inference is False


def test_bridge_goal_direct_replans_without_learning():
    """Changing goal mode should replan without adding experience."""
    bridge = NavigationCoreBridge()

    first = bridge.process_observation(
        observation()
    )

    history_size_before = len(
        bridge.coordinator
        .learning
        .history
        .entries()
    )

    cycle_count_before = bridge._cycle_count

    decision = bridge.set_goal_navigation(
        mode=GOAL_DIRECT,
        place_observation=0,
    )

    history_size_after = len(
        bridge.coordinator
        .learning
        .history
        .entries()
    )

    assert bridge.navigation_mode == GOAL_DIRECT

    assert (
        bridge.coordinator
        .preferences
        .preferred_observations
        == (-1, 0)
    )

    assert (
        bridge.coordinator
        .preferences
        .place[0]
        == pytest.approx(10.0)
    )

    assert history_size_after == history_size_before
    assert bridge._cycle_count == cycle_count_before

    assert decision is bridge.latest_decision

    assert (
        decision.cycle_result.planning
        is not first.cycle_result.planning
    )

    assert (
        bridge.next_action_id
        == decision.next_action_id
    )


def test_bridge_goal_balanced_selects_balanced_terms():
    """Balanced goal mode should retain state information gain."""
    bridge = NavigationCoreBridge()

    bridge.process_observation(
        observation()
    )

    bridge.set_goal_navigation(
        mode=GOAL_BALANCED,
        place_observation=0,
    )

    interface = (
        bridge.coordinator.model_interface
    )

    assert bridge.navigation_mode == GOAL_BALANCED
    assert interface.use_utility is True
    assert interface.use_state_information_gain is True
    assert interface.use_inductive_inference is True


def test_bridge_switch_back_to_explore_clears_goal():
    """Explicit EXPLORE selection should remove the goal and replan."""
    bridge = NavigationCoreBridge()

    bridge.process_observation(
        observation()
    )

    bridge.set_goal_navigation(
        mode=GOAL_DIRECT,
        place_observation=0,
    )

    decision = (
        bridge.set_exploration_navigation()
    )

    assert bridge.navigation_mode == EXPLORE

    assert (
        bridge.coordinator
        .preferences
        .preferred_observations
        == (-1, -1)
    )

    interface = (
        bridge.coordinator.model_interface
    )

    assert interface.use_utility is False
    assert interface.use_state_information_gain is True
    assert interface.use_inductive_inference is False

    assert decision is bridge.latest_decision

    assert (
        bridge.next_action_id
        == decision.next_action_id
    )


def test_bridge_can_select_goal_before_initial_observation():
    """A goal mode may be configured before the first observation."""
    bridge = NavigationCoreBridge()

    decision = bridge.set_goal_navigation(
        mode=GOAL_DIRECT,
        place_observation=0,
    )

    assert decision is None
    assert bridge.navigation_mode == GOAL_DIRECT

    first = bridge.process_observation(
        observation()
    )

    assert first.is_bootstrap is True
    assert bridge.navigation_mode == GOAL_DIRECT

    assert (
        bridge.coordinator
        .preferences
        .preferred_observations
        == (-1, 0)
    )


def test_mode_change_rejected_while_action_awaits_observation():
    """Do not break the executed-action to observation learning pair."""
    bridge = NavigationCoreBridge()

    first = bridge.process_observation(
        observation()
    )

    bridge.record_executed_action(
        first.next_action_id
    )

    with pytest.raises(
        RuntimeError,
        match="awaiting observation",
    ):
        bridge.set_goal_navigation(
            mode=GOAL_DIRECT,
            place_observation=0,
        )

    assert bridge.navigation_mode == EXPLORE

    assert (
        bridge.coordinator
        .preferences
        .preferred_observations
        == (-1, -1)
    )


def test_goal_at_current_place_completes_without_next_action():
    """A reached goal should complete without launching another action."""
    bridge = NavigationCoreBridge()

    bridge.set_goal_navigation(
        mode=GOAL_DIRECT,
        place_observation=0,
    )

    decision = bridge.process_observation(
        observation(
            place_id=0,
            sensory_id=0,
        )
    )

    assert decision.goal_reached is True
    assert bridge.goal_reached is True
    assert decision.next_action_id is None
    assert bridge.next_action_id is None

    assert bridge.navigation_mode == GOAL_DIRECT

    assert (
        bridge.coordinator
        .preferences
        .preferred_observations
        == (-1, 0)
    )


def test_explicit_explore_clears_goal_completion_state():
    """Leaving goal mode should clear completion without clearing learning."""
    bridge = NavigationCoreBridge()

    bridge.set_goal_navigation(
        mode=GOAL_DIRECT,
        place_observation=0,
    )

    bridge.process_observation(
        observation()
    )

    assert bridge.goal_reached is True

    decision = (
        bridge.set_exploration_navigation()
    )

    assert bridge.goal_reached is False
    assert bridge.navigation_mode == EXPLORE
    assert decision.goal_reached is False

    assert (
        bridge.coordinator
        .preferences
        .preferred_observations
        == (-1, -1)
    )

    assert bridge.next_action_id is not None


def test_mode_change_replan_clears_executed_action_marker():
    """Planning-only mode changes must not repeat learned action evidence."""
    bridge = NavigationCoreBridge()

    first = bridge.process_observation(
        observation()
    )

    bridge.record_executed_action(
        first.next_action_id
    )

    second = bridge.process_observation(
        observation()
    )

    assert (
        second.executed_action_id
        == first.next_action_id
    )

    cycle_count_before = bridge._cycle_count

    replanned = (
        bridge.set_exploration_navigation()
    )

    assert replanned.executed_action_id is None

    assert (
        bridge._cycle_count
        == cycle_count_before
    )
