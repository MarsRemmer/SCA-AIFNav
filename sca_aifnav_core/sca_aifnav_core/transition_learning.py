"""Transition learning rules for the reference baseline baseline."""

import math

import numpy as np

from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.motion_primitives import BaselineMotionSet


MIN_TRANSITION_CONCENTRATION = 0.005
DIRECT_TRANSITION_RATE = 10.0
REVERSE_TRANSITION_RATE = 7.0


def learn_transition(
    model: BaselineGenerativeModel,
    current_belief: np.ndarray,
    previous_belief: np.ndarray,
    action_id: int,
    learning_rate: float,
) -> np.ndarray:
    """
    Apply one baseline Dirichlet transition update.

    The current and previous state beliefs form an outer-product evidence
    matrix. Evidence is added to pB for the selected action, concentrations
    are lower-bounded by 0.005, and B is recomputed by normalization.
    """
    current = _validated_belief(
        current_belief,
        model.num_states,
        "current_belief",
    )

    previous = _validated_belief(
        previous_belief,
        model.num_states,
        "previous_belief",
    )

    _validate_action_id(
        action_id,
        model.num_actions,
    )

    if not math.isfinite(learning_rate):
        raise ValueError(
            "learning_rate must be finite"
        )

    evidence = np.outer(
        current,
        previous,
    )

    support = (
        model.transition_likelihood[
            :,
            :,
            action_id,
        ] > 0.0
    ).astype(float)

    updated_concentration = (
        model.transition_concentration.copy()
    )

    updated_concentration[
        :,
        :,
        action_id,
    ] += learning_rate * evidence * support

    updated_concentration = np.maximum(
        updated_concentration,
        MIN_TRANSITION_CONCENTRATION,
    )

    model.transition_concentration = (
        updated_concentration
    )

    model.transition_likelihood = (
        _normalize_transition_concentration(
            updated_concentration
        )
    )

    return model.transition_likelihood[
        :,
        :,
        action_id,
    ].copy()


def learn_bidirectional_transition(
    model: BaselineGenerativeModel,
    previous_belief: np.ndarray,
    next_belief: np.ndarray,
    action_id: int,
    motion_set: BaselineMotionSet,
    direct_rate: float = DIRECT_TRANSITION_RATE,
    reverse_rate: float = REVERSE_TRANSITION_RATE,
) -> int:
    """
    Learn one transition and its reverse using reference baseline rates.

    Return the reverse action identifier.
    """
    if not motion_set.is_directional(action_id):
        raise ValueError(
            "bidirectional learning requires a directional action"
        )

    reverse_action = motion_set.reverse_action(
        action_id
    )

    learn_transition(
        model=model,
        current_belief=next_belief,
        previous_belief=previous_belief,
        action_id=action_id,
        learning_rate=direct_rate,
    )

    learn_transition(
        model=model,
        current_belief=previous_belief,
        previous_belief=next_belief,
        action_id=reverse_action,
        learning_rate=reverse_rate,
    )

    return reverse_action


def _normalize_transition_concentration(
    concentration: np.ndarray,
) -> np.ndarray:
    """
    Normalize pB over possible next states.

    Axis 0 represents the next-state dimension in B.
    """
    denominator = concentration.sum(
        axis=0,
        keepdims=True,
    )

    if np.any(denominator <= 0.0):
        raise ValueError(
            "transition concentration cannot be normalized"
        )

    return concentration / denominator


def _validated_belief(
    belief: np.ndarray,
    num_states: int,
    name: str,
) -> np.ndarray:
    """Return a validated one-dimensional state belief."""
    result = np.asarray(
        belief,
        dtype=float,
    )

    if result.shape != (num_states,):
        raise ValueError(
            f"{name} shape must match number of states"
        )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must contain finite values"
        )

    return result.copy()


def _validate_action_id(
    action_id: int,
    num_actions: int,
) -> None:
    """Validate a discrete action identifier."""
    if isinstance(action_id, bool) or not isinstance(
        action_id,
        int,
    ):
        raise TypeError(
            "action_id must be an integer"
        )

    if not 0 <= action_id < num_actions:
        raise ValueError(
            "action_id is out of range"
        )
