"""Tests for MCTS reward backpropagation."""

import numpy as np
import pytest

from sca_aifnav_core.mcts_backpropagation import (
    backpropagate_reward,
)
from sca_aifnav_core.mcts_node import (
    SearchTreeNode,
)


def make_node(
    place_id,
    parent=None,
    initial_reward=0.0,
):
    return SearchTreeNode(
        state_belief=np.array(
            [1.0, 0.0]
        ),
        place_id=place_id,
        parent=parent,
        initial_reward=initial_reward,
    )


def test_root_only_receives_reward():
    root = make_node(0)

    backpropagate_reward(
        root,
        reward=2.5,
    )

    assert root.visit_count == 1

    assert root.total_reward == pytest.approx(
        2.5
    )


def test_child_and_root_are_updated():
    root = make_node(0)

    child = make_node(
        1,
        parent=root,
    )

    backpropagate_reward(
        child,
        reward=1.5,
    )

    assert child.visit_count == 1
    assert root.visit_count == 1

    assert child.total_reward == pytest.approx(
        1.5
    )

    assert root.total_reward == pytest.approx(
        1.5
    )


def test_three_level_parent_chain_is_updated():
    root = make_node(0)

    child = make_node(
        1,
        parent=root,
    )

    leaf = make_node(
        2,
        parent=child,
    )

    backpropagate_reward(
        leaf,
        reward=0.75,
    )

    for node in (
        root,
        child,
        leaf,
    ):
        assert node.visit_count == 1

        assert node.total_reward == pytest.approx(
            0.75
        )


def test_existing_statistics_are_accumulated():
    root = make_node(0)

    root.visit_count = 4
    root.total_reward = 3.0

    backpropagate_reward(
        root,
        reward=2.0,
    )

    assert root.visit_count == 5

    assert root.total_reward == pytest.approx(
        5.0
    )


def test_repeated_backpropagation_accumulates():
    root = make_node(0)

    child = make_node(
        1,
        parent=root,
    )

    backpropagate_reward(
        child,
        reward=1.0,
    )

    backpropagate_reward(
        child,
        reward=2.0,
    )

    assert child.visit_count == 2
    assert root.visit_count == 2

    assert child.total_reward == pytest.approx(
        3.0
    )

    assert root.total_reward == pytest.approx(
        3.0
    )


def test_negative_reward_is_propagated_unchanged():
    root = make_node(0)

    child = make_node(
        1,
        parent=root,
    )

    backpropagate_reward(
        child,
        reward=-0.4,
    )

    assert child.total_reward == pytest.approx(
        -0.4
    )

    assert root.total_reward == pytest.approx(
        -0.4
    )


def test_state_reward_is_not_modified():
    root = make_node(
        0,
        initial_reward=3.0,
    )

    before = root.state_reward

    backpropagate_reward(
        root,
        reward=1.0,
    )

    assert root.state_reward == pytest.approx(
        before
    )


def test_sibling_branch_is_not_updated():
    root = make_node(0)

    first = make_node(
        1,
        parent=root,
    )

    second = make_node(
        2,
        parent=root,
    )

    root.children = {
        0: first,
        1: second,
    }

    backpropagate_reward(
        first,
        reward=2.0,
    )

    assert first.visit_count == 1
    assert root.visit_count == 1

    assert second.visit_count == 0

    assert second.total_reward == pytest.approx(
        0.0
    )
