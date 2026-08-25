"""Tests for active inference MCTS tree selection."""

import numpy as np
import pytest

from sca_aifnav_core.mcts_node import (
    SearchTreeNode,
)
from sca_aifnav_core.mcts_selection import (
    DEFAULT_SELECTION_EXPLORATION,
    MAX_REPEATED_PLACE_VISITS,
    select_search_node,
)


def make_node(
    place_id,
    parent=None,
):
    return SearchTreeNode(
        state_belief=np.array(
            [1.0, 0.0]
        ),
        place_id=place_id,
        parent=parent,
    )


def connect(
    parent,
    child,
    action_id=0,
):
    parent.possible_actions = [
        action_id
    ]

    parent.children = {
        action_id: child,
    }


def test_selection_constants_match_baseline():
    assert DEFAULT_SELECTION_EXPLORATION == pytest.approx(
        5.0
    )

    assert MAX_REPEATED_PLACE_VISITS == 2


def test_unexpanded_root_is_selected():
    root = make_node(0)

    result = select_search_node(
        root
    )

    assert result.selected_node is root

    assert result.ancestor_place_ids == (
        0,
    )


def test_expanded_root_selects_its_child():
    root = make_node(0)
    child = make_node(
        1,
        parent=root,
    )

    connect(
        root,
        child,
    )

    result = select_search_node(
        root
    )

    assert result.selected_node is child

    assert result.ancestor_place_ids == (
        0,
        1,
    )


def test_selection_follows_multiple_levels():
    root = make_node(0)

    child = make_node(
        1,
        parent=root,
    )

    leaf = make_node(
        2,
        parent=child,
    )

    connect(
        root,
        child,
        action_id=0,
    )

    connect(
        child,
        leaf,
        action_id=1,
    )

    result = select_search_node(
        root
    )

    assert result.selected_node is leaf

    assert result.ancestor_place_ids == (
        0,
        1,
        2,
    )


def test_zero_exploration_selects_highest_average_reward():
    root = make_node(0)

    lower = make_node(
        1,
        parent=root,
    )

    higher = make_node(
        2,
        parent=root,
    )

    root.possible_actions = [
        0,
        1,
    ]

    root.children = {
        0: lower,
        1: higher,
    }

    root.visit_count = 10

    lower.visit_count = 2
    lower.total_reward = 2.0

    higher.visit_count = 2
    higher.total_reward = 8.0

    result = select_search_node(
        root,
        c_param=0.0,
        use_utility=True,
    )

    assert result.selected_node is higher


def test_unvisited_child_is_prioritized():
    root = make_node(0)

    visited = make_node(
        1,
        parent=root,
    )

    unvisited = make_node(
        2,
        parent=root,
    )

    root.possible_actions = [
        0,
        1,
    ]

    root.children = {
        0: visited,
        1: unvisited,
    }

    root.visit_count = 10

    visited.visit_count = 5
    visited.total_reward = 100.0

    result = select_search_node(
        root
    )

    assert result.selected_node is unvisited


def test_selection_keeps_ucb_exploration_when_utility_disabled():
    root = make_node(0)

    frequently_visited = make_node(
        1,
        parent=root,
    )

    rarely_visited = make_node(
        2,
        parent=root,
    )

    root.possible_actions = [
        0,
        1,
    ]

    root.children = {
        0: frequently_visited,
        1: rarely_visited,
    }

    root.visit_count = 100

    frequently_visited.visit_count = 50
    rarely_visited.visit_count = 1

    result = select_search_node(
        root,
        use_utility=False,
    )

    assert result.selected_node is rarely_visited


def test_repeated_place_cycle_is_stopped():
    root = make_node(0)

    child = make_node(
        1,
        parent=root,
    )

    connect(
        root,
        child,
    )

    connect(
        child,
        root,
    )

    root.visit_count = 10
    child.visit_count = 10

    root.total_reward = 1.0
    child.total_reward = 1.0

    result = select_search_node(
        root,
        c_param=0.0,
    )

    assert result.ancestor_place_ids == (
        0,
        1,
        0,
        1,
        0,
        1,
    )

    assert result.selected_node is child


def test_selection_does_not_change_node_statistics():
    root = make_node(0)

    child = make_node(
        1,
        parent=root,
    )

    connect(
        root,
        child,
    )

    root.visit_count = 7
    root.total_reward = 3.5

    child.visit_count = 2
    child.total_reward = 1.5

    before = (
        root.visit_count,
        root.total_reward,
        child.visit_count,
        child.total_reward,
    )

    select_search_node(
        root
    )

    after = (
        root.visit_count,
        root.total_reward,
        child.visit_count,
        child.total_reward,
    )

    assert after == before
