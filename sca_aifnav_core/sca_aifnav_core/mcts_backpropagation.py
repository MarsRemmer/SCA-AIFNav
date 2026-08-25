"""Reward backpropagation for Monte Carlo tree search."""

from sca_aifnav_core.mcts_node import (
    SearchTreeNode,
)


def backpropagate_reward(
    node: SearchTreeNode,
    reward: float,
) -> None:
    """
    Propagate one rollout reward from a node back to the root.

    Every node on the parent chain receives one additional visit and the
    same rollout reward.
    """
    current = node

    while current is not None:
        current.visit_count += 1

        current.total_reward += reward

        current = current.parent
