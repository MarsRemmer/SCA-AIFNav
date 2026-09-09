"""Real observation and transition updates for the baseline."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.probability_tables import (
    expand_likelihood_table,
)
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.observation_learning import (
    learn_multimodal_observation,
)
from sca_aifnav_core.transition_learning import (
    DIRECT_TRANSITION_RATE,
    REVERSE_TRANSITION_RATE,
    learn_transition,
)


@dataclass(frozen=True)
class RealExperienceResult:
    """Summarize one baseline real state-observation update."""

    preliminary_belief: np.ndarray
    learning_belief: np.ndarray
    posterior_belief: np.ndarray
    transition_updated: bool
    reverse_transition_updated: bool
    reverse_action_id: Optional[int]


def update_real_experience(
    model: BaselineGenerativeModel,
    sensory_observation: int,
    place_observation: int,
    action_id: int,
    previous_belief: Optional[np.ndarray],
    motion_set: BaselineMotionSet,
    observation_prepared: bool = False,
) -> RealExperienceResult:
    """
    Apply the baseline real-observation update sequence.

    Real state inference first predicts through B using the executed action,
    then conditions that predicted prior on the current observations.

    The preliminary inference does not replace the stored belief. After B
    learning, a second inference from the previous stored belief becomes the
    learning belief. That belief is saved before A learning, matching the
    reference infer_states(save_hist=True) lifecycle. The final posterior is
    therefore predicted from the learning belief using the updated B and A.
    """
    if not observation_prepared:
        prepare_real_observation_dimensions(
            model=model,
            sensory_observation=sensory_observation,
            place_observation=place_observation,
        )

    reference_belief = _align_belief_dimension(
        model.state_belief,
        model.num_states,
    )

    preliminary = _infer_from_action_prior(
        model=model,
        sensory_observation=sensory_observation,
        place_observation=place_observation,
        action_id=action_id,
        reference_belief=reference_belief,
    )

    transition_updated = False
    reverse_updated = False
    reverse_action_id = None

    if previous_belief is not None:
        previous = _align_belief_dimension(
            previous_belief,
            model.num_states,
        )

        learn_transition(
            model=model,
            current_belief=preliminary,
            previous_belief=previous,
            action_id=action_id,
            learning_rate=DIRECT_TRANSITION_RATE,
        )

        transition_updated = True

        if (
            np.argmax(previous)
            != np.argmax(preliminary)
        ):
            reverse_action_id = (
                motion_set.reverse_action(
                    action_id
                )
            )

            learn_transition(
                model=model,
                current_belief=previous,
                previous_belief=preliminary,
                action_id=reverse_action_id,
                learning_rate=REVERSE_TRANSITION_RATE,
            )

            reverse_updated = True

    # AIMAPP calls infer_states(obs) here after updating B.
    # The preliminary inference was not saved, so this inference still
    # starts from the belief stored before the current physical update.
    learning_belief = _infer_from_action_prior(
        model=model,
        sensory_observation=sensory_observation,
        place_observation=place_observation,
        action_id=action_id,
        reference_belief=reference_belief,
    )

    # infer_states(obs) uses save_hist=True in update_believes_with_obs(),
    # so AIMAPP makes this the current qs before updating A.
    model.state_belief = (
        learning_belief.copy()
    )

    learn_multimodal_observation(
        model=model,
        sensory_observation=sensory_observation,
        place_observation=place_observation,
        state_belief=learning_belief,
    )

    # agent_step_update() then calls infer_states() again. At this point
    # get_belief_over_states() returns the saved learning belief above.
    posterior = _infer_from_action_prior(
        model=model,
        sensory_observation=sensory_observation,
        place_observation=place_observation,
        action_id=action_id,
        reference_belief=learning_belief,
    )

    model.state_belief = posterior.copy()

    return RealExperienceResult(
        preliminary_belief=preliminary,
        learning_belief=learning_belief,
        posterior_belief=posterior,
        transition_updated=transition_updated,
        reverse_transition_updated=reverse_updated,
        reverse_action_id=reverse_action_id,
    )


def prepare_real_observation_dimensions(
    model: BaselineGenerativeModel,
    sensory_observation: int,
    place_observation: int,
) -> None:
    """
    Expand observation dimensions before inference.

    The reference runtime updates observation-model dimensions before
    attempting inference with a newly observed visual or place identifier.
    Visual observations use the small unknown likelihood for new rows,
    whereas place observations use the ordinary observation expansion.
    """
    if (
        isinstance(sensory_observation, bool)
        or not isinstance(sensory_observation, int)
    ):
        raise TypeError(
            "sensory_observation must be an integer"
        )

    if (
        isinstance(place_observation, bool)
        or not isinstance(place_observation, int)
    ):
        raise TypeError(
            "place_observation must be an integer"
        )

    if sensory_observation < 0:
        raise ValueError(
            "sensory_observation must be non-negative"
        )

    if place_observation < 0:
        raise ValueError(
            "place_observation must be non-negative"
        )

    sensory_addition = max(
        0,
        sensory_observation
        + 1
        - model.sensory_observations,
    )

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
                add_observations=sensory_addition,
                null_probability=True,
            )
        )

    if place_addition > 0:
        model.place_likelihood = (
            expand_likelihood_table(
                model.place_likelihood,
                add_observations=place_addition,
                null_probability=False,
            )
        )

    # Rebuild the Dirichlet parameters from the current likelihood
    # tables after checking observation dimensions.
    model.sensory_concentration = (
        model.sensory_likelihood.copy()
    )

    model.place_concentration = (
        model.place_likelihood.copy()
    )


def _infer_from_action_prior(
    model: BaselineGenerativeModel,
    sensory_observation: int,
    place_observation: int,
    action_id: int,
    reference_belief: np.ndarray,
) -> np.ndarray:
    """Infer one posterior from the executed-action predicted prior."""
    reference = _align_belief_dimension(
        reference_belief,
        model.num_states,
    )

    action_prior = model.predicted_state_prior(
        action_id=action_id,
        belief=reference,
    )

    return model.infer_state_belief(
        sensory_observation=sensory_observation,
        place_observation=place_observation,
        prior=action_prior,
    )


def _align_belief_dimension(
    belief: np.ndarray,
    num_states: int,
) -> np.ndarray:
    """Pad an older physical belief after cognitive state growth."""
    result = np.asarray(
        belief,
        dtype=float,
    )

    if result.ndim != 1:
        raise ValueError(
            "previous_belief must be one-dimensional"
        )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "previous_belief must contain finite values"
        )

    if len(result) > num_states:
        raise ValueError(
            "previous_belief has more entries than model states"
        )

    missing = num_states - len(result)

    if missing == 0:
        return result.copy()

    return np.append(
        result,
        np.zeros(missing),
    )
