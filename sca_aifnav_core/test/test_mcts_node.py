"""Tests for baseline-compatible MCTS search-tree nodes."""

import math

import numpy as np
import pytest

from sca_aifnav_core.mcts_node import (
    DEFAULT_UCB_EXPLORATION,
    SearchTreeNode,
)


def make_node(
    place_id=0,
    parent=None,
    initial_reward=0.0,
    possible_actions=None,
):
    return SearchTreeNode(
        state_belief=np.array(
            [1.0, 0.0]
        ),
        place_id=place_id,
        parent=parent,
        initial_reward=initial_reward,
        possible_actions=possible_actions,
    )


def test_default_ucb_parameter_matches_source():
    assert DEFAULT_UCB_EXPLORATION == pytest.approx(
        1.41
    )


def test_node_initial_statistics_match_baseline():
    node = make_node(
        place_id=3,
        initial_reward=2.5,
    )

    assert node.id == 3
    assert node.place_id == 3

    assert node.total_reward == pytest.approx(
        0.0
    )

    assert node.visit_count == 0
    assert node.state_reward == pytest.approx(
        2.5
    )


def test_unvisited_average_returns_state_reward():
    node = make_node(
        initial_reward=2.5
    )

    assert node.average_reward() == pytest.approx(
        2.5
    )


def test_visited_average_uses_total_reward():
    node = make_node()

    node.total_reward = 9.0
    node.visit_count = 3

    assert node.average_reward() == pytest.approx(
        3.0
    )


def test_unvisited_node_has_infinite_ucb():
    node = make_node()

    assert math.isinf(
        node.ucb1_score()
    )


def test_ucb_utility_only_is_exploitation():
    parent = make_node()
    parent.visit_count = 10

    child = make_node(
        parent=parent
    )

    child.total_reward = 6.0
    child.visit_count = 2

    score = child.ucb1_score(
        c_param=2.0,
        use_utility=True,
        use_state_information_gain=False,
    )

    assert score == pytest.approx(
        3.0
    )


def test_ucb_info_gain_only_is_exploration():
    parent = make_node()
    parent.visit_count = 10

    child = make_node(
        parent=parent
    )

    child.total_reward = 6.0
    child.visit_count = 2

    score = child.ucb1_score(
        c_param=2.0,
        use_utility=False,
        use_state_information_gain=True,
    )

    expected = (
        2.0
        * math.sqrt(
            math.log(10)
            / 2
        )
    )

    assert score == pytest.approx(
        expected
    )


def test_ucb_with_both_terms_adds_them():
    parent = make_node()
    parent.visit_count = 10

    child = make_node(
        parent=parent
    )

    child.total_reward = 6.0
    child.visit_count = 2

    score = child.ucb1_score(
        c_param=2.0,
        use_utility=True,
        use_state_information_gain=True,
    )

    expected = (
        3.0
        + 2.0
        * math.sqrt(
            math.log(10)
            / 2
        )
    )

    assert score == pytest.approx(
        expected
    )


def test_ucb_with_both_terms_disabled_is_zero():
    parent = make_node()
    parent.visit_count = 10

    child = make_node(
        parent=parent
    )

    child.total_reward = 6.0
    child.visit_count = 2

    score = child.ucb1_score(
        use_utility=False,
        use_state_information_gain=False,
    )

    assert score == pytest.approx(
        0.0
    )


def test_baseline_expansion_flag_requires_only_one_child():
    node = make_node(
        possible_actions=[
            0,
            1,
            2,
        ]
    )

    assert not node.is_fully_expanded()

    node.children[0] = make_node(
        place_id=1,
        parent=node,
    )

    assert node.is_fully_expanded()


def test_ucb_tie_keeps_first_inserted_child():
    root = make_node(
        possible_actions=[
            0,
            1,
        ]
    )

    first = make_node(
        place_id=1,
        parent=root,
    )

    second = make_node(
        place_id=2,
        parent=root,
    )

    root.children[0] = first
    root.children[1] = second

    selected = root.select_best_child_ucb()

    assert selected is first


def test_child_rewards_preserve_insertion_order():
    root = make_node(
        possible_actions=[
            0,
            1,
        ]
    )

    first = make_node(
        place_id=1,
        parent=root,
        initial_reward=1.0,
    )

    second = make_node(
        place_id=2,
        parent=root,
        initial_reward=2.0,
    )

    root.children[0] = first
    root.children[1] = second

    assert root.child_average_rewards() == [
        1.0,
        2.0,
    ]


def test_detach_parent_makes_node_root():
    parent = make_node(
        place_id=0
    )

    child = make_node(
        place_id=1,
        parent=parent,
    )

    child.detach_parent()

    assert child.parent is None
