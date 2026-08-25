"""Imagined cognitive links for the baseline."""

from dataclasses import dataclass

import numpy as np

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.cognitive_growth import (
    HypotheticalStateResult,
    create_hypothetical_state,
)
from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory
from sca_aifnav_core.transition_learning import (
    learn_bidirectional_transition,
)


IMAGINED_DIRECT_RATE = 5.0
IMAGINED_REVERSE_RATE = 3.0


@dataclass(frozen=True)
class ImaginedLinkResult:
    """Describe one directly imagined baseline cognitive transition."""

    hypothetical: HypotheticalStateResult
    reverse_action_id: int
    previous_belief: np.ndarray


def create_and_link_hypothetical_state(
    state: CognitiveOdomState,
    action_id: int,
    memory: BaselinePlaceMemory,
    model: BaselineGenerativeModel,
    motion_set: BaselineMotionSet,
    robot_dimension: float = 0.25,
    state_step: int = 1,
    reference_belief=None,
) -> ImaginedLinkResult:
    """
    Create or reuse a ghost state and learn its direct imagined link.

    This function represents the positive direct-link branch of the baseline.
    Obstacle gating is intentionally handled by a later layer.
    """
    if reference_belief is None:
        physical_belief = model.state_belief.copy()
    else:
        physical_belief = np.asarray(
            reference_belief,
            dtype=float,
        ).copy()

    hypothetical = create_hypothetical_state(
        state=state,
        action_id=action_id,
        memory=memory,
        model=model,
        motion_set=motion_set,
        robot_dimension=robot_dimension,
        state_step=state_step,
        reference_belief=physical_belief,
    )

    aligned_physical_belief = _align_belief_dimension(
        physical_belief,
        model.num_states,
    )

    reverse_action_id = learn_bidirectional_transition(
        model=model,
        previous_belief=aligned_physical_belief,
        next_belief=hypothetical.posterior,
        action_id=action_id,
        motion_set=motion_set,
        direct_rate=IMAGINED_DIRECT_RATE,
        reverse_rate=IMAGINED_REVERSE_RATE,
    )

    return ImaginedLinkResult(
        hypothetical=hypothetical,
        reverse_action_id=reverse_action_id,
        previous_belief=aligned_physical_belief,
    )


def _align_belief_dimension(
    belief: np.ndarray,
    num_states: int,
) -> np.ndarray:
    """Pad an older belief with zeros after baseline model growth."""
    result = np.asarray(
        belief,
        dtype=float,
    )

    if result.ndim != 1:
        raise ValueError(
            "belief must be one-dimensional"
        )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "belief must contain finite values"
        )

    if len(result) > num_states:
        raise ValueError(
            "belief has more entries than model states"
        )

    missing_states = num_states - len(result)

    if missing_states == 0:
        return result.copy()

    return np.append(
        result,
        np.zeros(missing_states),
    )
