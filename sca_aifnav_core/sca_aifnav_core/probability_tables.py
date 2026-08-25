"""Probability-table utilities matching the AIMAPP V5 baseline."""


import numpy as np


UNKNOWN_LIKELIHOOD_WEIGHT = 0.001
UNEXPLORED_TRANSITION_WEIGHT = 0.05


def create_likelihood_table(
    num_observations: int,
    num_states: int,
) -> np.ndarray:
    """Create a uniform V5 observation likelihood table."""
    _validate_positive_count(
        num_observations,
        "num_observations",
    )
    _validate_positive_count(
        num_states,
        "num_states",
    )

    return np.full(
        (num_observations, num_states),
        1.0 / num_observations,
        dtype=float,
    )


def create_transition_table(
    num_states: int,
    num_actions: int,
) -> np.ndarray:
    """Create a uniform V5 transition table."""
    _validate_positive_count(
        num_states,
        "num_states",
    )
    _validate_positive_count(
        num_actions,
        "num_actions",
    )

    return np.full(
        (
            num_states,
            num_states,
            num_actions,
        ),
        1.0 / num_states,
        dtype=float,
    )


def expand_likelihood_table(
    table: np.ndarray,
    add_observations: int = 0,
    add_states: int = 0,
    null_probability: bool = True,
) -> np.ndarray:
    """
    Expand one V5 observation likelihood table.

    Existing entries are preserved exactly. Newly introduced entries use
    the V5 small unknown weight when null_probability is enabled.
    """
    _validate_likelihood_table(table)
    _validate_non_negative_count(
        add_observations,
        "add_observations",
    )
    _validate_non_negative_count(
        add_states,
        "add_states",
    )

    old_observations, old_states = table.shape

    new_observations = (
        old_observations + add_observations
    )
    new_states = old_states + add_states

    expanded = create_likelihood_table(
        num_observations=new_observations,
        num_states=new_states,
    )

    if null_probability:
        expanded[:] = UNKNOWN_LIKELIHOOD_WEIGHT
    else:
        old_uniform = 1.0 / old_observations
        new_uniform = 1.0 / new_observations

        expanded[
            expanded == old_uniform
        ] = new_uniform

        expanded[
            expanded == 1.0 / (old_observations * 2.0)
        ] = 1.0 / (new_observations * 2.0)

        expanded[:, :old_states] = (
            1.0 / (new_observations * 2.0)
        )

    expanded[
        :old_observations,
        :old_states,
    ] = table

    return expanded


def expand_transition_table(
    table: np.ndarray,
    add_states: int = 1,
    alter_weights: bool = True,
) -> np.ndarray:
    """
    Expand one V5 transition table with additional hidden states.

    Existing entries are copied first. When alter_weights is enabled,
    unexplored uniform entries are replaced by the V5 weight 0.05.
    """
    _validate_transition_table(table)
    _validate_non_negative_count(
        add_states,
        "add_states",
    )

    if add_states == 0:
        return table.copy()

    old_states = table.shape[0]
    num_actions = table.shape[2]
    new_states = old_states + add_states

    old_uniform = 1.0 / old_states
    new_uniform = 1.0 / new_states

    expanded = create_transition_table(
        num_states=new_states,
        num_actions=num_actions,
    )

    expanded[
        :old_states,
        :old_states,
        :,
    ] = table

    if alter_weights:
        expanded[
            expanded == old_uniform
        ] = UNEXPLORED_TRANSITION_WEIGHT

        expanded[
            expanded == new_uniform
        ] = UNEXPLORED_TRANSITION_WEIGHT

    return expanded


def _validate_likelihood_table(
    table: np.ndarray,
) -> None:
    """Validate an observation likelihood table."""
    if not isinstance(table, np.ndarray):
        raise TypeError("table must be a numpy array")

    if table.ndim != 2:
        raise ValueError(
            "likelihood table must have two dimensions"
        )

    if not np.all(np.isfinite(table)):
        raise ValueError(
            "likelihood table must contain finite values"
        )


def _validate_transition_table(
    table: np.ndarray,
) -> None:
    """Validate a transition probability table."""
    if not isinstance(table, np.ndarray):
        raise TypeError("table must be a numpy array")

    if table.ndim != 3:
        raise ValueError(
            "transition table must have three dimensions"
        )

    if table.shape[0] != table.shape[1]:
        raise ValueError(
            "transition state dimensions must be equal"
        )

    if not np.all(np.isfinite(table)):
        raise ValueError(
            "transition table must contain finite values"
        )


def _validate_positive_count(
    value: int,
    name: str,
) -> None:
    """Validate a strictly positive integer count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative_count(
    value: int,
    name: str,
) -> None:
    """Validate a non-negative integer count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )
