"""Tests for active inference MCTS tree expansion."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.mcts_expansion import (
    expand_search_node,
)
from sca_aifnav_core.mcts_model_interface import (
    MCTSModelInterface,
)
from sca_aifnav_core.mcts_node import (
    SearchTreeNode,
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


def make_components():
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


def test_root_with_only_origin_discovers_stay():
    root, interface, _, _, _ = (
        make_components()
    )

    tree_table = {}

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    assert root.possible_actions == [
        12
    ]

    assert list(root.children) == [
        12
    ]


def test_known_direction_becomes_child():
    root, interface, _, memory, _ = (
        make_components()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    tree_table = {}

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    assert 0 in root.possible_actions
    assert 0 in root.children

    assert root.children[
        0
    ].place_id == 1


def test_invalid_direction_is_not_child():
    root, interface, _, memory, _ = (
        make_components()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    tree_table = {}

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    assert 1 not in root.possible_actions
    assert 1 not in root.children


def test_expansion_does_not_create_places():
    root, interface, _, memory, _ = (
        make_components()
    )

    before = len(memory)

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table={},
    )

    assert len(memory) == before


def test_child_parent_and_action_are_recorded():
    root, interface, _, memory, _ = (
        make_components()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table={},
    )

    child = root.children[0]

    assert child.parent is root
    assert child.action_id == 0


def test_child_state_belief_comes_from_transition_model():
    root, interface, model, memory, _ = (
        make_components()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
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

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table={},
    )

    np.testing.assert_allclose(
        root.children[
            0
        ].state_belief,
        np.array(
            [0.0, 1.0]
        ),
    )


def test_child_contains_expected_observation():
    root, interface, model, memory, _ = (
        make_components()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
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

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table={},
    )

    child = root.children[0]

    np.testing.assert_allclose(
        child.expected_observation.sensory,
        np.array(
            [0.0, 1.0]
        ),
    )

    np.testing.assert_allclose(
        child.expected_observation.place,
        np.array(
            [0.0, 1.0]
        ),
    )


def test_child_reward_matches_action_evaluation():
    root, interface, _, memory, _ = (
        make_components()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    expected = interface.evaluate_action(
        current_belief=(
            root.state_belief
        ),
        action_id=0,
    )

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table={},
    )

    assert root.children[
        0
    ].state_reward == pytest.approx(
        expected.score
    )


def test_new_children_are_registered_in_tree_table():
    root, interface, _, memory, _ = (
        make_components()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    tree_table = {}

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    assert 1 in tree_table

    assert tree_table[
        1
    ] is root.children[
        0
    ]


def test_existing_tree_node_is_reused():
    root, interface, _, memory, _ = (
        make_components()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    existing = SearchTreeNode(
        state_belief=np.array(
            [0.5, 0.5]
        ),
        place_id=1,
        initial_reward=-100.0,
    )

    tree_table = {
        1: existing,
    }

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    assert root.children[
        0
    ] is existing

    assert existing.state_reward > -100.0


def test_transition_to_earlier_ancestor_is_filtered():
    root, interface, _, memory, _ = (
        make_components()
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    child = SearchTreeNode(
        state_belief=np.array(
            [0.5, 0.5]
        ),
        place_id=1,
        parent=root,
        action_id=0,
    )

    expand_search_node(
        node=child,
        model_interface=interface,
        tree_table={},
        ancestor_place_ids=[
            0,
            1,
        ],
    )

    # From place 1, action 6 points approximately back toward place 0.
    assert 6 not in child.possible_actions
    assert 6 not in child.children


def test_reexpansion_rebuilds_children():
    root, interface, _, _, _ = (
        make_components()
    )

    fake_child = SearchTreeNode(
        state_belief=np.array(
            [1.0, 0.0]
        ),
        place_id=99,
    )

    root.children[
        99
    ] = fake_child

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table={},
    )

    assert 99 not in root.children


def test_supplied_actions_are_trusted():
    root, interface, _, _, _ = (
        make_components()
    )

    # The baseline assumes an explicitly supplied action list has already
    # been filtered by its caller.
    root.possible_actions = [
        1
    ]

    tree_table = {}

    expand_search_node(
        node=root,
        model_interface=interface,
        tree_table=tree_table,
    )

    assert 1 in root.children

    assert root.children[
        1
    ].place_id == -1
