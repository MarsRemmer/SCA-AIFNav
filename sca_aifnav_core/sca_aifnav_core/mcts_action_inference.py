"""Final action inference from Monte Carlo tree-search statistics."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sca_aifnav_core.mcts_model_interface import (
    MCTSModelInterface,
)
from sca_aifnav_core.mcts_node import (
    SearchTreeNode,
)


POLICY_PRECISION = 16.0
ACTION_PRECISION = 16.0
DEFAULT_ACTION_SELECTION = "stochastic"
LOG_EPSILON = 1e-16
TIE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class RootActionInference:
    """Summarize action inference from root-node statistics."""

    selected_action: Optional[int]
    available_actions: Tuple[int, ...]
    action_values: np.ndarray
    policy_posterior: np.ndarray
    selection_probabilities: np.ndarray
    full_action_values: np.ndarray
    full_policy_posterior: np.ndarray


def infer_root_action(
    root_node: SearchTreeNode,
    model_interface: MCTSModelInterface,
    action_selection: str = DEFAULT_ACTION_SELECTION,
    gamma: float = POLICY_PRECISION,
    alpha: float = ACTION_PRECISION,
    rng=None,
) -> RootActionInference:
    """
    Infer the next action from the average rewards of root children.

    Each root child acts as a one-step policy. Average rewards are converted
    into a posterior over policies and then into an action choice.
    """
    all_actions = tuple(
        model_interface.get_possible_actions()
    )

    full_action_values = np.zeros(
        len(all_actions),
        dtype=float,
    )

    full_policy_posterior = np.zeros(
        len(all_actions),
        dtype=float,
    )

    if not root_node.children:
        return RootActionInference(
            selected_action=None,
            available_actions=(),
            action_values=np.array(
                [],
                dtype=float,
            ),
            policy_posterior=np.array(
                [],
                dtype=float,
            ),
            selection_probabilities=np.array(
                [],
                dtype=float,
            ),
            full_action_values=full_action_values,
            full_policy_posterior=(
                full_policy_posterior
            ),
        )

    available_actions = []
    action_values = []

    for action_id, child in root_node.children.items():
        average_reward = (
            child.average_reward()
        )

        # Uncharted tree nodes are neutralized by the baseline planner.
        if child.place_id < 0:
            average_reward = 0.0

        available_actions.append(
            action_id
        )

        action_values.append(
            average_reward
        )

    available_actions = tuple(
        available_actions
    )

    action_values = np.asarray(
        action_values,
        dtype=float,
    )

    policy_posterior = (
        _policy_posterior(
            action_values=action_values,
            gamma=gamma,
        )
    )

    selected_index, selection_probabilities = (
        _select_action_index(
            policy_posterior=policy_posterior,
            action_selection=action_selection,
            alpha=alpha,
            rng=rng,
        )
    )

    selected_action = available_actions[
        selected_index
    ]

    action_to_full_index = {
        action_id: index
        for index, action_id in enumerate(
            all_actions
        )
    }

    for local_index, action_id in enumerate(
        available_actions
    ):
        full_index = action_to_full_index[
            action_id
        ]

        full_action_values[
            full_index
        ] = action_values[
            local_index
        ]

        full_policy_posterior[
            full_index
        ] = policy_posterior[
            local_index
        ]

    return RootActionInference(
        selected_action=selected_action,
        available_actions=(
            available_actions
        ),
        action_values=action_values,
        policy_posterior=(
            policy_posterior
        ),
        selection_probabilities=(
            selection_probabilities
        ),
        full_action_values=(
            full_action_values
        ),
        full_policy_posterior=(
            full_policy_posterior
        ),
    )


def _policy_posterior(
    action_values: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Convert root action values into the policy posterior q_pi."""
    uniform_prior = np.full(
        len(action_values),
        1.0 / len(action_values),
        dtype=float,
    )

    log_prior = np.log(
        uniform_prior
        + LOG_EPSILON
    )

    return _softmax(
        action_values
        * gamma
        + log_prior
    )


def _select_action_index(
    policy_posterior: np.ndarray,
    action_selection: str,
    alpha: float,
    rng=None,
):
    """Select one action index from the posterior over root actions."""
    if action_selection == "deterministic":
        probabilities = (
            policy_posterior.copy()
        )

        best_probability = np.max(
            probabilities
        )

        tied_indices = np.flatnonzero(
            np.abs(
                probabilities
                - best_probability
            )
            <= TIE_TOLERANCE
        )

        if len(tied_indices) == 1:
            return (
                int(tied_indices[0]),
                probabilities,
            )

        selected = _random_choice(
            tied_indices,
            rng=rng,
        )

        return (
            int(selected),
            probabilities,
        )

    if action_selection == "stochastic":
        log_marginal = np.log(
            policy_posterior
            + LOG_EPSILON
        )

        probabilities = _softmax(
            log_marginal
            * alpha
        )

        indices = np.arange(
            len(probabilities)
        )

        selected = _random_choice(
            indices,
            probabilities=probabilities,
            rng=rng,
        )

        return (
            int(selected),
            probabilities,
        )

    raise ValueError(
        "action_selection must be "
        "'deterministic' or 'stochastic'"
    )


def _random_choice(
    values,
    probabilities=None,
    rng=None,
):
    """Sample one value using either a supplied RNG or NumPy."""
    if rng is None:
        return np.random.choice(
            values,
            p=probabilities,
        )

    return rng.choice(
        values,
        p=probabilities,
    )


def _softmax(
    values: np.ndarray,
) -> np.ndarray:
    """Numerically stable softmax."""
    values = np.asarray(
        values,
        dtype=float,
    )

    shifted = (
        values
        - np.max(values)
    )

    exponentials = np.exp(
        shifted
    )

    return (
        exponentials
        / exponentials.sum()
    )
