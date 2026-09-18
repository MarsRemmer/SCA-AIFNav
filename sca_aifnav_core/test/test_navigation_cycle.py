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
from sca_aifnav_core.navigation_mode import (
    EXPLORE,
    GOAL_BALANCED,
    GOAL_DIRECT,
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


def test_navigation_modes_select_expected_terms():
    (
        _,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    explore = coordinator.set_navigation_mode(
        EXPLORE
    )

    assert explore.name == EXPLORE
    assert coordinator.navigation_mode == EXPLORE
    assert (
        coordinator.model_interface.use_utility
        is False
    )
    assert (
        coordinator
        .model_interface
        .use_state_information_gain
        is True
    )
    assert (
        coordinator
        .model_interface
        .use_inductive_inference
        is False
    )

    direct = coordinator.set_navigation_mode(
        GOAL_DIRECT
    )

    assert direct.name == GOAL_DIRECT
    assert coordinator.navigation_mode == GOAL_DIRECT
    assert (
        coordinator.model_interface.use_utility
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

    balanced = coordinator.set_navigation_mode(
        GOAL_BALANCED
    )

    assert balanced.name == GOAL_BALANCED
    assert coordinator.navigation_mode == GOAL_BALANCED
    assert (
        coordinator.model_interface.use_utility
        is True
    )
    assert (
        coordinator
        .model_interface
        .use_state_information_gain
        is True
    )
    assert (
        coordinator
        .model_interface
        .use_inductive_inference
        is True
    )


def test_invalid_navigation_mode_is_rejected():
    (
        _,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    try:
        coordinator.set_navigation_mode(
            "unsupported"
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "unsupported navigation mode "
            "must raise ValueError"
        )


def test_goal_direct_sets_known_goal_and_reference_terms():
    (
        _,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    snapshot = coordinator.set_goal_navigation(
        mode=GOAL_DIRECT,
        place_observation=0,
    )

    assert coordinator.navigation_mode == GOAL_DIRECT
    assert snapshot.preferred_observations == (
        -1,
        0,
    )
    assert snapshot.place[0] == 10.0

    interface = coordinator.model_interface

    assert interface.use_utility is True
    assert interface.use_state_information_gain is False
    assert interface.use_inductive_inference is True


def test_goal_balanced_keeps_state_information_gain():
    (
        _,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    coordinator.set_goal_navigation(
        mode=GOAL_BALANCED,
        place_observation=0,
    )

    interface = coordinator.model_interface

    assert coordinator.navigation_mode == GOAL_BALANCED
    assert interface.use_utility is True
    assert interface.use_state_information_gain is True
    assert interface.use_inductive_inference is True


def test_exploration_mode_clears_goal_preference():
    (
        _,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    coordinator.set_goal_navigation(
        mode=GOAL_DIRECT,
        place_observation=0,
    )

    snapshot = (
        coordinator.set_exploration_navigation()
    )

    assert coordinator.navigation_mode == EXPLORE
    assert snapshot.preferred_observations == (
        -1,
        -1,
    )

    interface = coordinator.model_interface

    assert interface.use_utility is False
    assert interface.use_state_information_gain is True
    assert interface.use_inductive_inference is False


def test_goal_navigation_requires_explicit_goal():
    (
        _,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    try:
        coordinator.set_goal_navigation(
            mode=GOAL_DIRECT
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "goal mode must require a goal preference"
        )


def test_goal_navigation_rejects_unknown_goal():
    (
        model,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    try:
        coordinator.set_goal_navigation(
            mode=GOAL_DIRECT,
            sensory_observation=(
                model.sensory_observations
            ),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "unknown goals must not be silently created"
        )


def test_named_navigation_modes_keep_parameter_information_gain_disabled():
    """Keep parameter information gain disabled in every named mode."""
    (
        _,
        _,
        _,
        coordinator,
        _,
        _,
    ) = make_case()

    for mode in (
        EXPLORE,
        GOAL_DIRECT,
        GOAL_BALANCED,
    ):
        # Deliberately enable it first. Selecting any named mode must
        # restore the fixed mode configuration and turn it back off.
        coordinator.model_interface.use_parameter_information_gain = True

        coordinator.set_navigation_mode(
            mode
        )

        assert (
            coordinator
            .model_interface
            .use_parameter_information_gain
            is False
        )
