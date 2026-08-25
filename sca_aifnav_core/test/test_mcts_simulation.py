"""Tests for one complete MCTS simulation cycle."""

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
from sca_aifnav_core.mcts_simulation import (
    run_search_simulation,
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


def make_goal_case():
    motion_set = BaselineMotionSet()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    model = BaselineGenerativeModel()

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

    preferences = BaselinePreferenceState(
        model
    )

    preferences.update(
        model=model,
        place_observation=1,
        preference_weight=2.0,
    )

    interface = MCTSModelInterface(
        model=model,
        memory=memory,
        motion_set=motion_set,
        preferences=preferences,
        use_utility=True,
        use_state_information_gain=False,
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
    )


def test_first_simulation_selects_root():
    (
        root,
        interface,
        _,
        _,
    ) = make_goal_case()

    result = run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table={},
    )

    assert result.selected_node is root

    assert result.ancestor_place_ids == (
        0,
    )


def test_first_simulation_expands_root():
    (
        root,
        interface,
        _,
        _,
    ) = make_goal_case()

    result = run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table={},
    )

    assert result.node_expanded

    assert root.is_fully_expanded()

    assert 0 in root.children
    assert 12 in root.children


def test_first_simulation_updates_only_root_statistics():
    (
        root,
        interface,
        _,
        _,
    ) = make_goal_case()

    run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table={},
    )

    assert root.visit_count == 1

    for child in root.children.values():
        assert child.visit_count == 0


def test_propagated_reward_adds_selected_state_reward():
    (
        root,
        interface,
        _,
        _,
    ) = make_goal_case()

    root.state_reward = 0.25

    result = run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table={},
    )

    assert result.propagated_reward == pytest.approx(
        result.rollout_reward
        + 0.25
    )


def test_backpropagated_reward_is_saved_in_root():
    (
        root,
        interface,
        _,
        _,
    ) = make_goal_case()

    result = run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table={},
    )

    assert root.total_reward == pytest.approx(
        result.propagated_reward
    )


def test_expansion_populates_tree_table():
    (
        root,
        interface,
        _,
        _,
    ) = make_goal_case()

    tree_table = {}

    run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    assert 0 in tree_table
    assert 1 in tree_table

    assert tree_table[
        1
    ] is root.children[
        0
    ]


def test_second_simulation_moves_into_unvisited_child():
    (
        root,
        interface,
        _,
        _,
    ) = make_goal_case()

    tree_table = {}

    first = run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    assert first.selected_node is root

    second = run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    assert second.selected_node is root.children[
        0
    ]

    assert second.selected_node.place_id == 1


def test_second_simulation_backpropagates_to_child_and_root():
    (
        root,
        interface,
        _,
        _,
    ) = make_goal_case()

    tree_table = {}

    run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    child = root.children[
        0
    ]

    result = run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    assert result.selected_node is child

    assert child.visit_count == 1

    assert root.visit_count == 2


def test_simulation_does_not_change_physical_belief():
    (
        root,
        interface,
        model,
        _,
    ) = make_goal_case()

    before = model.state_belief.copy()

    run_search_simulation(
        root_node=root,
        model_interface=interface,
        tree_table={},
    )

    np.testing.assert_allclose(
        model.state_belief,
        before,
    )


def test_rollout_depth_does_not_change_first_simulation_reward():
    (
        shallow_root,
        shallow_interface,
        _,
        _,
    ) = make_goal_case()

    shallow = run_search_simulation(
        root_node=shallow_root,
        model_interface=shallow_interface,
        tree_table={},
        max_rollout_depth=1,
    )

    (
        deep_root,
        deep_interface,
        _,
        _,
    ) = make_goal_case()

    deep = run_search_simulation(
        root_node=deep_root,
        model_interface=deep_interface,
        tree_table={},
        max_rollout_depth=100,
    )

    assert shallow.rollout_reward == pytest.approx(
        deep.rollout_reward
    )

    assert shallow.propagated_reward == pytest.approx(
        deep.propagated_reward
    )
