"""Backward propagation of state preferences through learned transitions."""

import numpy as np


MIN_CERTAINTY_THRESHOLD = 0.15
DEFAULT_INDUCTIVE_HORIZON = 4


def inductive_preference(
    current_belief: np.ndarray,
    predicted_belief: np.ndarray,
    transition_likelihood: np.ndarray,
    preferred_states: np.ndarray,
    lookahead_horizon: int = DEFAULT_INDUCTIVE_HORIZON,
) -> float:
    """
    Evaluate whether a predicted state moves toward preferred states.

    Preferred-state membership is propagated backward through sufficiently
    certain transitions. If the current most likely state can reach that
    backward preference set within the lookahead horizon, the predicted
    belief receives the corresponding inductive preference value.
    """
    current_belief = _validated_vector(
        current_belief,
        "current_belief",
    )

    predicted_belief = _validated_vector(
        predicted_belief,
        "predicted_belief",
    )

    preferred_states = _validated_vector(
        preferred_states,
        "preferred_states",
    )

    transition_likelihood = np.asarray(
        transition_likelihood,
        dtype=float,
    )

    if transition_likelihood.ndim != 3:
        raise ValueError(
            "transition_likelihood must have shape "
            "[next_state, current_state, action]"
        )

    num_states = transition_likelihood.shape[0]

    if transition_likelihood.shape[1] != num_states:
        raise ValueError(
            "transition_likelihood must have equal "
            "next-state and current-state dimensions"
        )

    if current_belief.size != num_states:
        raise ValueError(
            "current_belief size must match transition states"
        )

    if predicted_belief.size != num_states:
        raise ValueError(
            "predicted_belief size must match transition states"
        )

    if preferred_states.size != num_states:
        raise ValueError(
            "preferred_states size must match transition states"
        )

    if isinstance(lookahead_horizon, bool):
        raise TypeError(
            "lookahead_horizon must be an integer"
        )

    if not isinstance(
        lookahead_horizon,
        (int, np.integer),
    ):
        raise TypeError(
            "lookahead_horizon must be an integer"
        )

    if lookahead_horizon < 0:
        raise ValueError(
            "lookahead_horizon must be non-negative"
        )

    current_state = int(
        np.argmax(current_belief)
    )

    outgoing_from_current = (
        transition_likelihood[
            :,
            current_state,
            :,
        ]
    )

    median = np.median(
        outgoing_from_current
    )

    above_median = outgoing_from_current[
        outgoing_from_current > median
    ]

    if above_median.size == 0:
        certainty_threshold = (
            MIN_CERTAINTY_THRESHOLD
        )
    else:
        certainty_threshold = max(
            float(
                np.mean(
                    above_median
                )
            ),
            MIN_CERTAINTY_THRESHOLD,
        )

    certain_transitions = (
        transition_likelihood
        > certainty_threshold
    ).astype(float)

    backward_preferences = [
        preferred_states.copy()
    ]

    found_path = False
    reward_layer = 0

    for depth in range(
        lookahead_horizon
    ):
        previous_layer = (
            backward_preferences[-1]
        )

        reachable_by_action = (
            certain_transitions.T.dot(
                previous_layer
            )
            > 0.0
        ).astype(float)

        next_layer = np.max(
            reachable_by_action,
            axis=0,
        )

        backward_preferences.append(
            next_layer
        )

        if (
            next_layer[current_state]
            >= current_belief[current_state]
        ):
            reward_layer = depth
            found_path = True
            break

    if not found_path:
        return 0.0

    return float(
        np.log(
            certainty_threshold
        )
        * backward_preferences[
            reward_layer
        ].dot(
            predicted_belief
        )
    )


def inductive_bonus(
    current_belief: np.ndarray,
    predicted_belief: np.ndarray,
    transition_likelihood: np.ndarray,
    preferred_states: np.ndarray,
    lookahead_horizon: int = DEFAULT_INDUCTIVE_HORIZON,
) -> float:
    """
    Return the positive MCTS contribution derived from inductive preference.

    The baseline MCTS subtracts the inductive-preference value, so this helper
    exposes that final contribution directly.
    """
    return -inductive_preference(
        current_belief=current_belief,
        predicted_belief=predicted_belief,
        transition_likelihood=(
            transition_likelihood
        ),
        preferred_states=preferred_states,
        lookahead_horizon=(
            lookahead_horizon
        ),
    )


def _validated_vector(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    """Return one finite one-dimensional floating-point vector."""
    result = np.asarray(
        values,
        dtype=float,
    )

    if result.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional"
        )

    if result.size == 0:
        raise ValueError(
            f"{name} must not be empty"
        )

    if not np.all(
        np.isfinite(result)
    ):
        raise ValueError(
            f"{name} must contain only finite values"
        )

    return result
