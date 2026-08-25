"""Real observation and transition updates for the baseline."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from sca_aifnav_core.generative_model import BaselineGenerativeModel
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
) -> RealExperienceResult:
    """
    Apply the baseline real-observation update sequence.

    Real observation inference uses a uniform state prior because baseline calls
    infer_states without an action in agent_step_update. Transition learning
    then compares that inferred belief with the stored previous belief.
    """
    preliminary = _infer_from_uniform_prior(
        model=model,
        sensory_observation=sensory_observation,
        place_observation=place_observation,
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

    learning_belief = _infer_from_uniform_prior(
        model=model,
        sensory_observation=sensory_observation,
        place_observation=place_observation,
    )

    learn_multimodal_observation(
        model=model,
        sensory_observation=sensory_observation,
        place_observation=place_observation,
        state_belief=learning_belief,
    )

    posterior = _infer_from_uniform_prior(
        model=model,
        sensory_observation=sensory_observation,
        place_observation=place_observation,
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


def _infer_from_uniform_prior(
    model: BaselineGenerativeModel,
    sensory_observation: int,
    place_observation: int,
) -> np.ndarray:
    """Infer baseline real state belief using its uniform D prior."""
    uniform_prior = np.full(
        model.num_states,
        1.0 / model.num_states,
        dtype=float,
    )

    return model.infer_state_belief(
        sensory_observation=sensory_observation,
        place_observation=place_observation,
        prior=uniform_prior,
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
