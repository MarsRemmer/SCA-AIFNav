"""Minimal rollout for active inference Monte Carlo planning."""

from sca_aifnav_core.mcts_model_interface import (
    MCTSModelInterface,
)
from sca_aifnav_core.mcts_node import (
    SearchTreeNode,
)


def minimal_rollout_reward(
    start_node: SearchTreeNode,
    model_interface: MCTSModelInterface,
    max_depth: int,
) -> float:
    """
    Return the best immediate reward available from one search node.

    The baseline planner passes a rollout-depth argument to this routine,
    but its minimal-rollout path evaluates only the next transition. The
    argument is therefore intentionally retained without affecting the
    returned reward.
    """
    _ = max_depth

    if start_node.is_fully_expanded():
        available_actions = list(
            start_node.possible_actions
        )
    else:
        available_actions = (
            model_interface.get_possible_actions()
        )

    if len(available_actions) == 0:
        return 0.0

    best_reward = -float("inf")

    for action_id in available_actions:
        next_place_id = (
            model_interface.get_next_place_id(
                current_place_id=(
                    start_node.place_id
                ),
                action_id=action_id,
            )
        )

        if next_place_id < 0:
            continue

        evaluation = (
            model_interface.evaluate_action(
                current_belief=(
                    start_node.state_belief
                ),
                action_id=action_id,
            )
        )

        if evaluation.score > best_reward:
            best_reward = evaluation.score

    if best_reward == -float("inf"):
        return 0.0

    return float(
        best_reward
    )
