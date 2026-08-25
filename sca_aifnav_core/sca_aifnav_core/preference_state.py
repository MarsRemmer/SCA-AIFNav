"""Observation and state preferences for the baseline."""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.probability_tables import (
    expand_likelihood_table,
)


PREFERRED_STATE_THRESHOLD = 0.45
NO_PREFERENCE = -1


@dataclass(frozen=True)
class PreferenceSnapshot:
    """Immutable summary of the current baseline preferences."""

    sensory: np.ndarray
    place: np.ndarray
    preferred_states: np.ndarray
    preferred_observations: Tuple[int, int]


class BaselinePreferenceState:
    """
    Store baseline observation preferences C and preferred states Cs.

    C expresses desired observations. Cs maps those preferences back into
    hidden-state space using the current observation model A.
    """

    def __init__(
        self,
        model: BaselineGenerativeModel,
    ) -> None:
        self.sensory = np.zeros(
            model.sensory_observations,
            dtype=float,
        )

        self.place = np.zeros(
            model.place_observations,
            dtype=float,
        )

        self.preferred_observations = (
            NO_PREFERENCE,
            NO_PREFERENCE,
        )

        self.preferred_states = np.zeros(
            model.num_states,
            dtype=float,
        )

    def update(
        self,
        model: BaselineGenerativeModel,
        sensory_observation: int = NO_PREFERENCE,
        place_observation: int = NO_PREFERENCE,
        preference_weight: float = 1.0,
    ) -> PreferenceSnapshot:
        """Set baseline observation preferences and infer preferred states."""
        _validate_preference_id(
            sensory_observation,
            "sensory_observation",
        )

        _validate_preference_id(
            place_observation,
            "place_observation",
        )

        if not math.isfinite(preference_weight):
            raise ValueError(
                "preference_weight must be finite"
            )

        _ensure_preference_observation_capacity(
            model=model,
            sensory_observation=sensory_observation,
            place_observation=place_observation,
        )

        self.sensory = np.zeros(
            model.sensory_observations,
            dtype=float,
        )

        self.place = np.zeros(
            model.place_observations,
            dtype=float,
        )

        if sensory_observation >= 0:
            self.sensory[
                sensory_observation
            ] = preference_weight

        if place_observation >= 0:
            self.place[
                place_observation
            ] = preference_weight

        self.preferred_observations = (
            sensory_observation,
            place_observation,
        )

        self.preferred_states = (
            _preferred_state_vector(
                model=model,
                preferred_observations=(
                    self.preferred_observations
                ),
            )
        )

        return self.snapshot()

    def clear(
        self,
        model: BaselineGenerativeModel,
    ) -> PreferenceSnapshot:
        """Remove all observation and hidden-state preferences."""
        self.sensory = np.zeros(
            model.sensory_observations,
            dtype=float,
        )

        self.place = np.zeros(
            model.place_observations,
            dtype=float,
        )

        self.preferred_observations = (
            NO_PREFERENCE,
            NO_PREFERENCE,
        )

        self.preferred_states = np.zeros(
            model.num_states,
            dtype=float,
        )

        return self.snapshot()

    def sync_dimensions(
        self,
        model: BaselineGenerativeModel,
    ) -> bool:
        """
        Extend C after the generative model gains observations or states.

        Return True when at least one preference dimension changed.
        """
        dimension_changed = False

        if len(self.sensory) < model.sensory_observations:
            self.sensory = np.append(
                self.sensory,
                np.zeros(
                    model.sensory_observations
                    - len(self.sensory)
                ),
            )

            dimension_changed = True

        if len(self.place) < model.place_observations:
            self.place = np.append(
                self.place,
                np.zeros(
                    model.place_observations
                    - len(self.place)
                ),
            )

            dimension_changed = True

        if len(self.preferred_states) < model.num_states:
            self.preferred_states = np.append(
                self.preferred_states,
                np.zeros(
                    model.num_states
                    - len(self.preferred_states)
                ),
            )

            dimension_changed = True

        if dimension_changed:
            self.preferred_states = (
                _preferred_state_vector(
                    model=model,
                    preferred_observations=(
                        self.preferred_observations
                    ),
                )
            )

        return dimension_changed

    def snapshot(self) -> PreferenceSnapshot:
        """Return independent copies of current preference arrays."""
        return PreferenceSnapshot(
            sensory=self.sensory.copy(),
            place=self.place.copy(),
            preferred_states=(
                self.preferred_states.copy()
            ),
            preferred_observations=(
                self.preferred_observations
            ),
        )


def _preferred_state_vector(
    model: BaselineGenerativeModel,
    preferred_observations: Tuple[int, int],
) -> np.ndarray:
    """Map baseline preferred observations into hidden-state preference Cs."""
    recurrence = {}

    modalities = (
        model.sensory_likelihood,
        model.place_likelihood,
    )

    for modality, observation_id in enumerate(
        preferred_observations
    ):
        if observation_id < 0:
            continue

        likelihood = modalities[
            modality
        ][observation_id, :]

        standout_states = np.where(
            likelihood
            >= PREFERRED_STATE_THRESHOLD
        )[0]

        for state_id in standout_states:
            recurrence[state_id] = (
                recurrence.get(
                    state_id,
                    0,
                )
                + 1
            )

    result = np.zeros(
        model.num_states,
        dtype=float,
    )

    if not recurrence:
        return result

    highest_recurrence = max(
        recurrence.values()
    )

    for state_id, count in recurrence.items():
        if count == highest_recurrence:
            result[state_id] = 1.0

    return result


def _ensure_preference_observation_capacity(
    model: BaselineGenerativeModel,
    sensory_observation: int,
    place_observation: int,
) -> None:
    """
    Reproduce the A/pA preparation performed by baseline update_preference.

    baseline calls update_A_dim_given_obs with null_probability=False and then
    rebuilds the Dirichlet concentrations from ones.
    """
    sensory_addition = 0

    if sensory_observation >= 0:
        sensory_addition = max(
            0,
            sensory_observation
            + 1
            - model.sensory_observations,
        )

    place_addition = 0

    if place_observation >= 0:
        place_addition = max(
            0,
            place_observation
            + 1
            - model.place_observations,
        )

    if sensory_addition > 0:
        model.sensory_likelihood = (
            expand_likelihood_table(
                model.sensory_likelihood,
                add_observations=(
                    sensory_addition
                ),
                null_probability=False,
            )
        )

    if place_addition > 0:
        model.place_likelihood = (
            expand_likelihood_table(
                model.place_likelihood,
                add_observations=(
                    place_addition
                ),
                null_probability=False,
            )
        )

    # baseline recreates pA inside update_A_dim_given_obs.
    model.sensory_concentration = np.ones_like(
        model.sensory_likelihood
    )

    model.place_concentration = np.ones_like(
        model.place_likelihood
    )


def _validate_preference_id(
    observation_id: int,
    name: str,
) -> None:
    """Validate a baseline observation preference identifier."""
    if (
        isinstance(observation_id, bool)
        or not isinstance(observation_id, int)
    ):
        raise TypeError(
            f"{name} must be an integer"
        )

    if observation_id < NO_PREFERENCE:
        raise ValueError(
            f"{name} must be -1 or non-negative"
        )
