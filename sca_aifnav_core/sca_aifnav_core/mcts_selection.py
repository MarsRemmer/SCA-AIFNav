"""Tree selection for active inference Monte Carlo planning."""

from dataclasses import dataclass
from typing import Tuple

from sca_aifnav_core.mcts_node import (
    SearchTreeNode,
)


DEFAULT_SELECTION_EXPLORATION = 5.0
MAX_REPEATED_PLACE_VISITS = 2


@dataclass(frozen=True)
class SelectionResult:
    """Summarize one MCTS tree-selection traversal."""

    selected_node: SearchTreeNode
    ancestor_place_ids: Tuple[int, ...]


def select_search_node(
    root_node: SearchTreeNode,
    c_param: float = DEFAULT_SELECTION_EXPLORATION,
    use_utility: bool = True,
) -> SelectionResult:
    """
    Traverse expanded nodes using UCB1 until a search leaf is reached.

    The returned path includes both the root and the selected node.
    Repeated cognitive places are limited to avoid cycling indefinitely.

    Baseline compatibility intentionally keeps the UCB exploration term
    enabled during selection independently of the active-inference
    state-information-gain setting.
    """
    current = root_node
    ancestor_place_ids = []

    while current.is_fully_expanded():
        ancestor_place_ids.append(
            current.id
        )

        next_node = (
            current.select_best_child_ucb(
                c_param=c_param,
                use_utility=use_utility,
                use_state_information_gain=True,
            )
        )

        if next_node is None:
            break

        current = next_node

        occurrence_counts = {
            place_id: ancestor_place_ids.count(
                place_id
            )
            for place_id in set(
                ancestor_place_ids
            )
        }

        if any(
            count
            > MAX_REPEATED_PLACE_VISITS
            for count in occurrence_counts.values()
        ):
            break

    ancestor_place_ids.append(
        current.id
    )

    return SelectionResult(
        selected_node=current,
        ancestor_place_ids=tuple(
            ancestor_place_ids
        ),
    )
