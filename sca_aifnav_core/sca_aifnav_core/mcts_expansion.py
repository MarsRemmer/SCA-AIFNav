"""Tree expansion for active inference Monte Carlo planning."""

from typing import Dict, Iterable, Optional

from sca_aifnav_core.mcts_model_interface import (
    MCTSModelInterface,
)
from sca_aifnav_core.mcts_node import (
    SearchTreeNode,
)


def expand_search_node(
    node: SearchTreeNode,
    model_interface: MCTSModelInterface,
    tree_table: Dict[int, SearchTreeNode],
    ancestor_place_ids: Optional[Iterable[int]] = None,
) -> SearchTreeNode:
    """
    Expand one search-tree node in all currently available directions.

    When the node does not yet have an action list, valid actions are
    discovered from the cognitive place graph. Invalid transitions and
    transitions back to earlier ancestors are discarded.

    Existing entries in ``tree_table`` are reused to preserve the
    place-indexed search structure.
    """
    if ancestor_place_ids is None:
        ancestor_place_ids = (
            node.place_id,
        )
    else:
        ancestor_place_ids = tuple(
            ancestor_place_ids
        )

    if node.possible_actions is None:
        node.possible_actions = []

        candidate_actions = (
            model_interface.get_possible_actions()
        )
    else:
        candidate_actions = list(
            node.possible_actions
        )

    # The baseline rebuilds the outgoing child dictionary whenever a
    # node is expanded.
    node.children = {}

    for action_id in candidate_actions:
        next_place_id = (
            model_interface.get_next_place_id(
                current_place_id=node.place_id,
                action_id=action_id,
            )
        )

        # If actions are being discovered for the first time, only retain
        # known reachable places that do not return to an earlier ancestor.
        if action_id not in node.possible_actions:
            earlier_ancestors = (
                ancestor_place_ids[:-1]
            )

            if (
                next_place_id < 0
                or next_place_id
                in earlier_ancestors
            ):
                continue

            node.possible_actions.append(
                action_id
            )

        evaluation = (
            model_interface.evaluate_action(
                current_belief=(
                    node.state_belief
                ),
                action_id=action_id,
            )
        )

        child_reward = (
            evaluation.score
        )

        if next_place_id in tree_table:
            child_node = tree_table[
                next_place_id
            ]

            if (
                child_node.state_reward
                < child_reward
            ):
                child_node.state_reward = (
                    child_reward
                )
        else:
            child_node = SearchTreeNode(
                state_belief=(
                    evaluation
                    .predicted_state
                    .copy()
                ),
                place_id=next_place_id,
                parent=node,
                action_id=action_id,
                expected_observation=(
                    evaluation
                    .expected_observations
                ),
                initial_reward=(
                    child_reward
                ),
            )

            tree_table[
                child_node.id
            ] = child_node

        node.children[
            action_id
        ] = child_node

    return node
