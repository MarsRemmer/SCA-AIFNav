"""Tests for minimal active inference MCTS rollout."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.mcts_model_interface import (
    MCTSModelInterface,
)
from sca_aifnav_core.mcts_node import (
    SearchTreeNode,
)
from sca_aifnav_core.mcts_rollout import (
    minimal_rollout_reward,
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


def make_components(
    use_utility=True,
    use_state_information_gain=False,
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

    root = SearchTreeNode(
        state_belief=np.array(
            [1.0, 0.0]
        ),
        place_id=0,
        action_id=None,
    )

    return (
        root,
        interface,
        model,
        memory,
        preferences,
    )


def configure_goal_case(
    model,
    memory,
    preferences,
):
    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    model.sensory_likelihood = np.eye(
        2
    )

    model.place_likelihood = np.eye(
        2
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

    model.enforce_stationary_transition()

    preferences.update(
        model=model,
        place_observation=1,
        preference_weight=2.0,
    )


def test_unexpanded_node_reviews_all_available_actions():
    (
        root,
        interface,
        model,
        memory,
        preferences,
    ) = make_components()

    configure_goal_case(
        model,
        memory,
        preferences,
    )

    expected = interface.evaluate_action(
        current_belief=(
            root.state_belief
        ),
        action_id=0,
    )

    result = minimal_rollout_reward(
        start_node=root,
        model_interface=interface,
        max_depth=4,
    )

    assert result == pytest.approx(
        expected.score
    )


def test_rollout_keeps_negative_best_reward():
    (
        root,
        interface,
        model,
        memory,
        preferences,
    ) = make_components()

    configure_goal_case(
        model,
        memory,
        preferences,
    )

    result = minimal_rollout_reward(
        start_node=root,
        model_interface=interface,
        max_depth=4,
    )

    assert result < 0.0

    assert result == pytest.approx(
        np.log(
            np.exp(2.0)
            / (
                1.0
                + np.exp(2.0)
            )
        )
    )


def test_goal_action_scores_above_stay():
    (
        root,
        interface,
        model,
        memory,
        preferences,
    ) = make_components()

    configure_goal_case(
        model,
        memory,
        preferences,
    )

    goal = interface.evaluate_action(
        current_belief=root.state_belief,
        action_id=0,
    )

    stay = interface.evaluate_action(
        current_belief=root.state_belief,
        action_id=12,
    )

    assert goal.score > stay.score

    result = minimal_rollout_reward(
        start_node=root,
        model_interface=interface,
        max_depth=4,
    )

    assert result == pytest.approx(
        goal.score
    )


def test_expanded_node_uses_its_action_list():
    (
        root,
        interface,
        model,
        memory,
        preferences,
    ) = make_components()

    configure_goal_case(
        model,
        memory,
        preferences,
    )

    stay_child = SearchTreeNode(
        state_belief=np.array(
            [1.0, 0.0]
        ),
        place_id=0,
        parent=root,
        action_id=12,
    )

    root.possible_actions = [
        12
    ]

    root.children = {
        12: stay_child,
    }

    expected = interface.evaluate_action(
        current_belief=(
            root.state_belief
        ),
        action_id=12,
    )

    result = minimal_rollout_reward(
        start_node=root,
        model_interface=interface,
        max_depth=4,
    )

    assert result == pytest.approx(
        expected.score
    )


def test_invalid_supplied_action_is_skipped():
    (
        root,
        interface,
        _,
        _,
        _,
    ) = make_components()

    fake_child = SearchTreeNode(
        state_belief=np.array(
            [1.0, 0.0]
        ),
        place_id=-1,
        parent=root,
        action_id=1,
    )

    root.possible_actions = [
        1
    ]

    root.children = {
        1: fake_child,
    }

    result = minimal_rollout_reward(
        start_node=root,
        model_interface=interface,
        max_depth=4,
    )

    assert result == pytest.approx(
        0.0
    )


def test_empty_expanded_action_list_returns_zero():
    (
        root,
        interface,
        _,
        _,
        _,
    ) = make_components()

    # An empty list without children is not considered expanded, so use
    # an interface with no possible actions to exercise the dead-end case.
    original = (
        interface.get_possible_actions
    )

    interface.get_possible_actions = (
        lambda: []
    )

    try:
        result = minimal_rollout_reward(
            start_node=root,
            model_interface=interface,
            max_depth=4,
        )
    finally:
        interface.get_possible_actions = (
            original
        )

    assert result == pytest.approx(
        0.0
    )


def test_max_depth_does_not_change_minimal_rollout():
    (
        root,
        interface,
        model,
        memory,
        preferences,
    ) = make_components()

    configure_goal_case(
        model,
        memory,
        preferences,
    )

    shallow = minimal_rollout_reward(
        start_node=root,
        model_interface=interface,
        max_depth=1,
    )

    deep = minimal_rollout_reward(
        start_node=root,
        model_interface=interface,
        max_depth=100,
    )

    assert shallow == pytest.approx(
        deep
    )


def test_exploration_rollout_prefers_informative_future():
    (
        root,
        interface,
        model,
        memory,
        _,
    ) = make_components(
        use_utility=False,
        use_state_information_gain=True,
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    model.sensory_likelihood = np.eye(
        2
    )

    model.place_likelihood = np.eye(
        2
    )

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

    model.enforce_stationary_transition()

    result = minimal_rollout_reward(
        start_node=root,
        model_interface=interface,
        max_depth=4,
    )

    assert result == pytest.approx(
        np.log(2.0),
        abs=1e-6,
    )


def test_rollout_does_not_change_physical_belief():
    (
        root,
        interface,
        model,
        memory,
        preferences,
    ) = make_components()

    configure_goal_case(
        model,
        memory,
        preferences,
    )

    before = model.state_belief.copy()

    minimal_rollout_reward(
        start_node=root,
        model_interface=interface,
        max_depth=4,
    )

    np.testing.assert_allclose(
        model.state_belief,
        before,
    )


def test_rollout_does_not_change_node_statistics():
    (
        root,
        interface,
        model,
        memory,
        preferences,
    ) = make_components()

    configure_goal_case(
        model,
        memory,
        preferences,
    )

    root.visit_count = 7
    root.total_reward = 3.5

    minimal_rollout_reward(
        start_node=root,
        model_interface=interface,
        max_depth=4,
    )

    assert root.visit_count == 7

    assert root.total_reward == pytest.approx(
        3.5
    )
