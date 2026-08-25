"""Cognitive-map growth operations for the reference baseline baseline."""

from dataclasses import dataclass
import math

import numpy as np

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.cognitive_projection import project_action_position
from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory


@dataclass(frozen=True)
class HypotheticalStateResult:
    """Describe one baseline hypothetical cognitive-state projection."""

    position: Point2D
    place_id: int
    requested_distance: float
    place_created: bool
    model_expanded: bool
    posterior: np.ndarray


def node_step_distance(
    influence_radius: float,
    robot_dimension: float,
    state_step: int = 1,
) -> float:
    """
    Return the baseline distance used for a cognitive node step.

    reference baseline uses:
        influence_radius * state_step + robot_dimension / 2
    """
    if not math.isfinite(influence_radius):
        raise ValueError(
            "influence_radius must be finite"
        )

    if influence_radius <= 0.0:
        raise ValueError(
            "influence_radius must be positive"
        )

    if not math.isfinite(robot_dimension):
        raise ValueError(
            "robot_dimension must be finite"
        )

    if robot_dimension < 0.0:
        raise ValueError(
            "robot_dimension must be non-negative"
        )

    if isinstance(state_step, bool) or not isinstance(
        state_step,
        int,
    ):
        raise TypeError(
            "state_step must be an integer"
        )

    if state_step <= 0:
        raise ValueError(
            "state_step must be positive"
        )

    return (
        influence_radius * state_step
        + robot_dimension / 2.0
    )


def create_hypothetical_state(
    state: CognitiveOdomState,
    action_id: int,
    memory: BaselinePlaceMemory,
    model: BaselineGenerativeModel,
    motion_set: BaselineMotionSet,
    robot_dimension: float = 0.25,
    state_step: int = 1,
    reference_belief=None,
) -> HypotheticalStateResult:
    """
    Create or reuse one reachable baseline hypothetical cognitive state.

    This function assumes the action direction is reachable. Obstacle
    rejection and transition learning are handled by later layers.
    """
    primitive = motion_set.action(action_id)

    if primitive.is_stationary:
        raise ValueError(
            "stationary action does not create a ghost state"
        )

    requested_distance = node_step_distance(
        influence_radius=memory.influence_radius,
        robot_dimension=robot_dimension,
        state_step=state_step,
    )

    if reference_belief is None:
        reference = model.state_belief.copy()
    else:
        reference = np.asarray(
            reference_belief,
            dtype=float,
        ).copy()

    projected_position = project_action_position(
        state=state,
        action_id=action_id,
        memory=memory,
        motion_set=motion_set,
        ideal_distance=requested_distance,
    )

    rounded_position = Point2D(
        x=round(projected_position.x, 2),
        y=round(projected_position.y, 2),
    )

    existing_place_id = memory.find_match(
        rounded_position
    )

    if existing_place_id >= 0:
        place_id = existing_place_id
        place_created = False
        model_expanded = False
    else:
        place_id = memory.resolve_place(
            rounded_position
        )
        place_created = True

        model_expanded = (
            model.register_place_observation(
                place_id
            )
        )

    aligned_reference = _align_belief_dimension(
        reference,
        model.num_states,
    )

    predicted_prior = model.predicted_state_prior(
        action_id=action_id,
        belief=aligned_reference,
    )

    posterior = model.infer_state_belief(
        place_observation=place_id,
        prior=predicted_prior,
    )

    return HypotheticalStateResult(
        position=memory.place(place_id),
        place_id=place_id,
        requested_distance=requested_distance,
        place_created=place_created,
        model_expanded=model_expanded,
        posterior=posterior,
    )


def _align_belief_dimension(
    belief: np.ndarray,
    num_states: int,
) -> np.ndarray:
    """Extend a previous baseline belief with zero-weight new states."""
    if belief.ndim != 1:
        raise ValueError(
            "belief must be one-dimensional"
        )

    if len(belief) > num_states:
        raise ValueError(
            "belief has more entries than model states"
        )

    if not np.all(np.isfinite(belief)):
        raise ValueError(
            "belief must contain finite values"
        )

    missing_states = num_states - len(belief)

    if missing_states == 0:
        return belief.copy()

    return np.append(
        belief,
        np.zeros(missing_states),
    )
