"""One complete Monte Carlo tree-search simulation cycle."""

from dataclasses import dataclass
from typing import Dict, Tuple

from sca_aifnav_core.mcts_backpropagation import (
    backpropagate_reward,
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
from sca_aifnav_core.mcts_rollout import (
    minimal_rollout_reward,
)
from sca_aifnav_core.mcts_selection import (
    DEFAULT_SELECTION_EXPLORATION,
    select_search_node,
)


@dataclass(frozen=True)
class SimulationResult:
    """Summarize one complete MCTS simulation."""

    selected_node: SearchTreeNode
    ancestor_place_ids: Tuple[int, ...]
    node_expanded: bool
    rollout_reward: float
    propagated_reward: float


def run_search_simulation(
    root_node: SearchTreeNode,
    model_interface: MCTSModelInterface,
    tree_table: Dict[int, SearchTreeNode],
    max_rollout_depth: int = 4,
    c_param: float = DEFAULT_SELECTION_EXPLORATION,
) -> SimulationResult:
    """
    Execute one complete baseline MCTS simulation.

    The cycle performs selection, optional expansion, minimal rollout, and
    reward backpropagation. The selected node's immediate state reward is
    added to the rollout reward before propagation.
    """
    selection = select_search_node(
        root_node=root_node,
        c_param=c_param,
        use_utility=model_interface.use_utility,
    )

    selected_node = (
        selection.selected_node
    )

    node_expanded = False

    if not selected_node.is_fully_expanded():
        expand_search_node(
            node=selected_node,
            model_interface=model_interface,
            tree_table=tree_table,
            ancestor_place_ids=(
                selection.ancestor_place_ids
            ),
        )

        node_expanded = True

    rollout_reward = minimal_rollout_reward(
        start_node=selected_node,
        model_interface=model_interface,
        max_depth=max_rollout_depth,
    )

    propagated_reward = (
        rollout_reward
        + selected_node.state_reward
    )

    backpropagate_reward(
        node=selected_node,
        reward=propagated_reward,
    )

    return SimulationResult(
        selected_node=selected_node,
        ancestor_place_ids=(
            selection.ancestor_place_ids
        ),
        node_expanded=node_expanded,
        rollout_reward=rollout_reward,
        propagated_reward=(
            propagated_reward
        ),
    )
