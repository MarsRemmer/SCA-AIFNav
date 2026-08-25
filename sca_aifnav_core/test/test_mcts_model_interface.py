"""Tests for the baseline-compatible MCTS model interface."""

import numpy as np

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.mcts_model_interface import (
    MCTSModelInterface,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
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


def make_interface(
    use_utility=True,
    use_state_information_gain=True,
):
    motion_set = BaselineMotionSet()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    model = BaselineGenerativeModel()

    preferences = BaselinePreferenceState(
        model
    )

    interface = MCTSModelInterface(
        model=model,
        memory=memory,
        motion_set=motion_set,
        preferences=preferences,
        use_utility=use_utility,
        use_state_information_gain=(
            use_state_information_gain
        ),
    )

    return (
        interface,
        model,
        memory,
        motion_set,
        preferences,
    )


def test_possible_actions_include_all_baseline_actions():
    interface, _, _, _, _ = (
        make_interface()
    )

    assert interface.get_possible_actions() == list(
        range(13)
    )


def test_known_direction_returns_existing_place():
    interface, _, memory, _, _ = (
        make_interface()
    )

    place_id = memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    assert place_id == 1

    assert interface.get_next_place_id(
        current_place_id=0,
        action_id=0,
    ) == 1


def test_unknown_direction_returns_minus_one():
    interface, _, _, _, _ = (
        make_interface()
    )

    assert interface.get_next_place_id(
        current_place_id=0,
        action_id=2,
    ) == -1


def test_mcts_lookup_does_not_create_place():
    interface, _, memory, _, _ = (
        make_interface()
    )

    before = len(memory)

    result = interface.get_next_place_id(
        current_place_id=0,
        action_id=2,
    )

    assert result == -1
    assert len(memory) == before


def test_stay_returns_current_place():
    interface, _, memory, _, _ = (
        make_interface()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    assert interface.get_next_place_id(
        current_place_id=1,
        action_id=12,
    ) == 1


def test_nearby_place_outside_action_range_is_rejected():
    interface, _, memory, _, _ = (
        make_interface()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    # The action-1 projected point is close enough to match place 1 by
    # fixed radius, but place 1 itself belongs to action 0's angular sector.
    assert interface.get_next_place_id(
        current_place_id=0,
        action_id=1,
    ) == -1


def test_next_state_belief_uses_B():
    interface, model, _, _, _ = (
        make_interface()
    )

    model.transition_likelihood[
        :,
        :,
        0,
    ] = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    result = interface.get_next_state_belief(
        current_belief=np.array(
            [1.0, 0.0]
        ),
        action_id=0,
    )

    np.testing.assert_allclose(
        result,
        np.array([0.0, 1.0]),
    )


def test_expected_observation_uses_A():
    interface, model, _, _, _ = (
        make_interface()
    )

    model.sensory_likelihood = np.eye(2)
    model.place_likelihood = np.eye(2)

    result = interface.get_expected_observation(
        np.array([0.25, 0.75])
    )

    np.testing.assert_allclose(
        result.sensory,
        np.array([0.25, 0.75]),
    )

    np.testing.assert_allclose(
        result.place,
        np.array([0.25, 0.75]),
    )


def test_utility_mode_scores_goal_action_higher():
    (
        interface,
        model,
        _,
        _,
        preferences,
    ) = make_interface(
        use_utility=True,
        use_state_information_gain=False,
    )

    model.sensory_likelihood = np.eye(2)
    model.place_likelihood = np.eye(2)

    model.transition_likelihood[
        :,
        :,
        0,
    ] = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    model.transition_likelihood[
        :,
        :,
        1,
    ] = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )

    preferences.update(
        model=model,
        place_observation=1,
        preference_weight=2.0,
    )

    goal = interface.evaluate_action(
        current_belief=np.array(
            [1.0, 0.0]
        ),
        action_id=0,
    )

    wrong = interface.evaluate_action(
        current_belief=np.array(
            [1.0, 0.0]
        ),
        action_id=1,
    )

    assert goal.score > wrong.score


def test_exploration_mode_scores_uncertain_future_higher():
    (
        interface,
        model,
        _,
        _,
        _,
    ) = make_interface(
        use_utility=False,
        use_state_information_gain=True,
    )

    model.sensory_likelihood = np.eye(2)
    model.place_likelihood = np.eye(2)

    model.transition_likelihood[
        :,
        :,
        0,
    ] = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    model.transition_likelihood[
        :,
        :,
        1,
    ] = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )

    exploratory = interface.evaluate_action(
        current_belief=np.array(
            [1.0, 0.0]
        ),
        action_id=0,
    )

    certain = interface.evaluate_action(
        current_belief=np.array(
            [1.0, 0.0]
        ),
        action_id=1,
    )

    assert exploratory.score > certain.score

    assert exploratory.score > 0.69
    assert certain.score < 1e-6


def test_interface_evaluation_does_not_change_physical_belief():
    interface, model, _, _, _ = (
        make_interface()
    )

    before = model.state_belief.copy()

    interface.evaluate_action(
        current_belief=np.array(
            [0.5, 0.5]
        ),
        action_id=0,
    )

    np.testing.assert_allclose(
        model.state_belief,
        before,
    )
