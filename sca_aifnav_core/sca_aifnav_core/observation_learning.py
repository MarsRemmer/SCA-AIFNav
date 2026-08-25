"""Observation-model learning for the AIMAPP V5 baseline."""

import math

import numpy as np

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)


OBSERVATION_LEARNING_RATE = 5.0


def learn_sensory_observation(
    model: BaselineGenerativeModel,
    observation_id: int,
    state_belief: np.ndarray,
    learning_rate: float = OBSERVATION_LEARNING_RATE,
) -> np.ndarray:
    """
    Update the sensory pA/A distribution using one observation.

    AIMAPP V5 uses a Dirichlet pseudo-count update with learning rate 5.
    Only entries with existing non-zero support in A may be learned.
    """
    belief = _validated_belief(
        state_belief,
        model.num_states,
    )

    _validate_observation(
        observation_id,
        model.sensory_observations,
    )

    _validate_learning_rate(
        learning_rate,
    )

    observation = np.zeros(
        model.sensory_observations,
        dtype=float,
    )

    observation[observation_id] = 1.0

    evidence = np.outer(
        observation,
        belief,
    )

    support = (
        model.sensory_likelihood > 0.0
    ).astype(float)

    model.sensory_concentration = (
        model.sensory_concentration
        + learning_rate
        * evidence
        * support
    )

    model.sensory_likelihood = (
        _normalize_observation_concentration(
            model.sensory_concentration
        )
    )

    return model.sensory_likelihood.copy()


def learn_place_observation(
    model: BaselineGenerativeModel,
    place_id: int,
    state_belief: np.ndarray,
    learning_rate: float = OBSERVATION_LEARNING_RATE,
) -> np.ndarray:
    """
    Update the cognitive-place pA/A distribution.

    The rule is identical to sensory learning but is applied to the
    discrete place-observation modality.
    """
    belief = _validated_belief(
        state_belief,
        model.num_states,
    )

    _validate_observation(
        place_id,
        model.place_observations,
    )

    _validate_learning_rate(
        learning_rate,
    )

    observation = np.zeros(
        model.place_observations,
        dtype=float,
    )

    observation[place_id] = 1.0

    evidence = np.outer(
        observation,
        belief,
    )

    support = (
        model.place_likelihood > 0.0
    ).astype(float)

    model.place_concentration = (
        model.place_concentration
        + learning_rate
        * evidence
        * support
    )

    model.place_likelihood = (
        _normalize_observation_concentration(
            model.place_concentration
        )
    )

    return model.place_likelihood.copy()


def learn_multimodal_observation(
    model: BaselineGenerativeModel,
    sensory_observation: int,
    place_observation: int,
    state_belief: np.ndarray,
    learning_rate: float = OBSERVATION_LEARNING_RATE,
) -> None:
    """Apply the same V5 pA update to both observation modalities."""
    learn_sensory_observation(
        model=model,
        observation_id=sensory_observation,
        state_belief=state_belief,
        learning_rate=learning_rate,
    )

    learn_place_observation(
        model=model,
        place_id=place_observation,
        state_belief=state_belief,
        learning_rate=learning_rate,
    )


def _normalize_observation_concentration(
    concentration: np.ndarray,
) -> np.ndarray:
    """Normalize Dirichlet parameters over observation outcomes."""
    denominator = concentration.sum(
        axis=0,
        keepdims=True,
    )

    if np.any(denominator <= 0.0):
        raise ValueError(
            "observation concentration cannot be normalized"
        )

    return concentration / denominator


def _validated_belief(
    belief: np.ndarray,
    num_states: int,
) -> np.ndarray:
    """Validate one hidden-state posterior."""
    result = np.asarray(
        belief,
        dtype=float,
    )

    if result.shape != (num_states,):
        raise ValueError(
            "state_belief shape must match number of states"
        )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "state_belief must contain finite values"
        )

    return result.copy()


def _validate_observation(
    observation_id: int,
    observation_count: int,
) -> None:
    """Validate one discrete observation identifier."""
    if (
        isinstance(observation_id, bool)
        or not isinstance(observation_id, int)
    ):
        raise TypeError(
            "observation_id must be an integer"
        )

    if not 0 <= observation_id < observation_count:
        raise ValueError(
            "observation_id is out of range"
        )


def _validate_learning_rate(
    learning_rate: float,
) -> None:
    """Validate the pA learning rate."""
    if not math.isfinite(learning_rate):
        raise ValueError(
            "learning_rate must be finite"
        )
