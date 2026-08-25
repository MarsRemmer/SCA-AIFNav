"""Complete Monte Carlo tree-search planner for active inference navigation."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from sca_aifnav_core.mcts_action_inference import (
    ACTION_PRECISION,
    DEFAULT_ACTION_SELECTION,
    POLICY_PRECISION,
    RootActionInference,
    infer_root_action,
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


DEFAULT_NUM_SIMULATIONS = 30
DEFAULT_MAX_ROLLOUT_DEPTH = 4
DEFAULT_MCTS_EXPLORATION = 5.0


@dataclass(frozen=True)
class MCTSPlan:
    """Result of one complete MCTS planning step."""

    selected_action: Optional[int]
    policy_posterior: np.ndarray
    action_values: np.ndarray
    selection_probabilities: np.ndarray
    available_actions: Tuple[int, ...]
    root_node: SearchTreeNode
    tree_table: Dict[int, SearchTreeNode]
    num_simulations: int


def plan_mcts(
    current_belief,
    current_place_id: int,
    model_interface: MCTSModelInterface,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    max_rollout_depth: int = DEFAULT_MAX_ROLLOUT_DEPTH,
    c_param: float = DEFAULT_MCTS_EXPLORATION,
    action_selection: str = DEFAULT_ACTION_SELECTION,
    gamma: float = POLICY_PRECISION,
    alpha: float = ACTION_PRECISION,
    possible_actions=None,
    rng=None,
) -> MCTSPlan:
    """
    Run one complete active-inference MCTS planning step.

    A fresh search tree is constructed for each call. The root is expanded
    through repeated simulations, and final root-child statistics are
    converted into a posterior over actions.
    """
    _validate_positive_integer(
        num_simulations,
        "num_simulations",
    )

    _validate_nonnegative_integer(
        max_rollout_depth,
        "max_rollout_depth",
    )

    belief = np.asarray(
        current_belief,
        dtype=float,
    ).copy()

    if belief.ndim != 1:
        raise ValueError(
            "current_belief must be one-dimensional"
        )

    if belief.size != model_interface.model.num_states:
        raise ValueError(
            "current_belief size must match model states"
        )

    if not np.all(
        np.isfinite(belief)
    ):
        raise ValueError(
            "current_belief must contain only finite values"
        )

    model_interface.get_place_position(
        current_place_id
    )

    expected_observation = (
        model_interface.get_expected_observation(
            belief
        )
    )

    if possible_actions is None:
        root_possible_actions = None
    else:
        root_possible_actions = list(
            possible_actions
        )

    root_node = SearchTreeNode(
        state_belief=belief,
        place_id=current_place_id,
        parent=None,
        action_id=None,
        expected_observation=(
            expected_observation
        ),
        possible_actions=(
            root_possible_actions
        ),
    )

    # Baseline behavior: each planning call starts with a fresh table.
    # The root itself is intentionally not inserted here.
    tree_table = {}

    for _ in range(
        num_simulations
    ):
        run_search_simulation(
            root_node=root_node,
            model_interface=model_interface,
            tree_table=tree_table,
            max_rollout_depth=(
                max_rollout_depth
            ),
            c_param=c_param,
        )

    action_inference = infer_root_action(
        root_node=root_node,
        model_interface=model_interface,
        action_selection=action_selection,
        gamma=gamma,
        alpha=alpha,
        rng=rng,
    )

    return _build_plan(
        action_inference=action_inference,
        root_node=root_node,
        tree_table=tree_table,
        num_simulations=num_simulations,
    )


def _build_plan(
    action_inference: RootActionInference,
    root_node: SearchTreeNode,
    tree_table,
    num_simulations: int,
) -> MCTSPlan:
    """Assemble the public planning result."""
    return MCTSPlan(
        selected_action=(
            action_inference.selected_action
        ),
        policy_posterior=(
            action_inference.full_policy_posterior.copy()
        ),
        action_values=(
            action_inference.full_action_values.copy()
        ),
        selection_probabilities=(
            action_inference.selection_probabilities.copy()
        ),
        available_actions=(
            action_inference.available_actions
        ),
        root_node=root_node,
        tree_table=tree_table,
        num_simulations=num_simulations,
    )


def _validate_positive_integer(
    value,
    name: str,
) -> None:
    """Require an integer greater than zero."""
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
    ):
        raise TypeError(
            f"{name} must be an integer"
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero"
        )


def _validate_nonnegative_integer(
    value,
    name: str,
) -> None:
    """Require an integer greater than or equal to zero."""
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
    ):
        raise TypeError(
            f"{name} must be an integer"
        )

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )
