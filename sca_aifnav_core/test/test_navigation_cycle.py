"""Tests for end-to-end baseline navigation coordination."""

import numpy as np

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.mcts_planner import (
    DEFAULT_MAX_ROLLOUT_DEPTH,
    DEFAULT_MCTS_EXPLORATION,
    DEFAULT_NUM_SIMULATIONS,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.navigation_cycle import (
    BaselineNavigationCoordinator,
    NavigationCycleResult,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_core.preference_state import (
    BaselinePreferenceState,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)


def make_case(
    num_simulations=3,
):
    model = BaselineGenerativeModel()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    motion_set = BaselineMotionSet()

    preferences = BaselinePreferenceState(
        model
    )

    coordinator = BaselineNavigationCoordinator(
        model=model,
        memory=memory,
        motion_set=motion_set,
        preferences=preferences,
        num_simulations=num_simulations,
    )

    state = CognitiveOdomState(
        position=Point2D(
            0.0,
            0.0,
        ),
        travel_heading_rad=0.0,
    )

    obstacle_distances = np.full(
        motion_set.DIRECTION_COUNT,
        np.nan,
        dtype=float,
    )

    return (
        model,
        memory,
        preferences,
        coordinator,
        state,
        obstacle_distances,
    )


def run_cycle(
    coordinator,
    state,
    obstacle_distances,
):
    return coordinator.step_and_plan(
        state=state,
        sensory_observation=0,
        place_observation=0,
        executed_action_id=12,
        obstacle_distances=(
            obstacle_distances
        ),
        action_selection="deterministic",
    )


def test_default_configuration_is_goal_directed():
    (
        _,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    assert (
        coordinator
        .model_interface
        .use_utility
        is True
    )

    assert (
        coordinator
        .model_interface
        .use_state_information_gain
        is False
    )

    assert (
        coordinator
        .model_interface
        .use_inductive_inference
        is True
    )

    assert DEFAULT_NUM_SIMULATIONS == 30
    assert DEFAULT_MAX_ROLLOUT_DEPTH == 4
    assert DEFAULT_MCTS_EXPLORATION == 5.0


def test_preference_is_set_explicitly():
    (
        _,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    snapshot = coordinator.set_preference(
        place_observation=0,
        preference_weight=2.0,
    )

    assert snapshot.preferred_observations == (
        -1,
        0,
    )

    assert snapshot.place[0] == 2.0


def test_one_cycle_returns_learning_and_planning():
    (
        _,
        _,
        _,
        coordinator,
        state,
        obstacle_distances,
    ) = make_case()

    result = run_cycle(
        coordinator,
        state,
        obstacle_distances,
    )

    assert isinstance(
        result,
        NavigationCycleResult,
    )

    assert result.learning.final_belief.ndim == 1

    assert result.planning.num_simulations == 3


def test_stationary_only_map_selects_stay():
    (
        _,
        _,
        _,
        coordinator,
        state,
        obstacle_distances,
    ) = make_case()

    result = run_cycle(
        coordinator,
        state,
        obstacle_distances,
    )

    assert result.planning.available_actions == (
        12,
    )

    assert result.planning.selected_action == 12

    assert set(
        result.planning.root_node.children
    ) == {
        12,
    }


def test_place_observation_is_default_planning_root():
    (
        _,
        _,
        _,
        coordinator,
        state,
        obstacle_distances,
    ) = make_case()

    result = run_cycle(
        coordinator,
        state,
        obstacle_distances,
    )

    assert (
        result.planning.root_node.place_id
        == 0
    )


def test_planning_preserves_post_learning_belief():
    (
        model,
        _,
        _,
        coordinator,
        state,
        obstacle_distances,
    ) = make_case()

    result = run_cycle(
        coordinator,
        state,
        obstacle_distances,
    )

    np.testing.assert_allclose(
        model.state_belief,
        result.learning.final_belief,
    )


def test_unknown_scan_does_not_grow_place_memory():
    (
        _,
        memory,
        _,
        coordinator,
        state,
        obstacle_distances,
    ) = make_case()

    assert len(memory) == 1

    run_cycle(
        coordinator,
        state,
        obstacle_distances,
    )

    assert len(memory) == 1


def test_clear_preference_removes_goal():
    (
        _,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    coordinator.set_preference(
        place_observation=0,
        preference_weight=2.0,
    )

    snapshot = (
        coordinator.clear_preference()
    )

    assert snapshot.preferred_observations == (
        -1,
        -1,
    )

    assert np.all(
        snapshot.sensory == 0.0
    )

    assert np.all(
        snapshot.place == 0.0
    )

    assert np.all(
        snapshot.preferred_states == 0.0
    )


def test_posterior_place_is_used_as_next_planning_root(
    monkeypatch,
):
    (
        model,
        memory,
        _,
        coordinator,
        state,
        obstacle_distances,
    ) = make_case()

    place_one = memory.resolve_place(
        Point2D(
            1.0,
            0.0,
        )
    )

    assert place_one == 1

    model.register_place_observation(
        place_one
    )

    monkeypatch.setattr(
        model,
        "get_confident_state_index",
        lambda **kwargs: place_one,
    )

    result = coordinator.step_and_plan(
        state=state,
        sensory_observation=0,
        place_observation=0,
        executed_action_id=12,
        obstacle_distances=(
            obstacle_distances
        ),
        current_place_id=0,
        action_selection="deterministic",
    )

    assert (
        result.posterior_place_id
        == 1
    )

    assert (
        result.planning.root_node.place_id
        == 1
    )
