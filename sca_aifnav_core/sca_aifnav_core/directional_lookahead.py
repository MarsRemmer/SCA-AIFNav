"""Directional multi-step cognitive lookahead for reference baseline."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.cognitive_growth import (
    create_hypothetical_state,
    node_step_distance,
)
from sca_aifnav_core.cognitive_projection import (
    position_in_action_sector,
)
from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.obstacle_evidence import (
    apply_direct_imagined_evidence,
    reinforce_blocked_self_loop,
)
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory


DEFAULT_LOOKAHEAD_STEPS = 3


@dataclass(frozen=True)
class LookaheadNode:
    """Describe one generated or reused node in a directional chain."""

    state_step: int
    obstacle_threshold: float
    place_id: int
    position: object
    place_created: bool
    model_expanded: bool
    posterior: np.ndarray


@dataclass(frozen=True)
class DirectionalLookaheadResult:
    """Summarize one baseline directional lookahead chain."""

    action_id: int
    nodes: Tuple[LookaheadNode, ...]
    deep_direct_links_updated: int
    stopped_by_obstacle: bool
    obstacle_stop_step: Optional[int]
    unknown_obstacle: bool


def grow_directional_lookahead(
    state: CognitiveOdomState,
    action_id: int,
    obstacle_distance: float,
    memory: BaselinePlaceMemory,
    model: BaselineGenerativeModel,
    motion_set: BaselineMotionSet,
    robot_dimension: float = 0.25,
    max_steps: int = DEFAULT_LOOKAHEAD_STEPS,
    reference_belief=None,
) -> DirectionalLookaheadResult:
    """
    Grow one baseline cognitive chain along a directional action.

    Obstacle thresholds grow with state_step, while every incremental
    spatial projection remains one base node step from the previous node.
    """
    if not motion_set.is_directional(action_id):
        raise ValueError(
            "lookahead requires a directional action"
        )

    _validate_max_steps(max_steps)

    distance = float(obstacle_distance)

    if math.isinf(distance) and distance < 0.0:
        raise ValueError(
            "obstacle_distance cannot be negative infinity"
        )

    base_step_distance = node_step_distance(
        influence_radius=memory.influence_radius,
        robot_dimension=robot_dimension,
        state_step=1,
    )

    if reference_belief is None:
        previous_belief = model.state_belief.copy()
    else:
        previous_belief = np.asarray(
            reference_belief,
            dtype=float,
        ).copy()

    physical_origin = state.position
    current_reference_state = state

    nodes = []
    deep_direct_links_updated = 0

    stopped_by_obstacle = False
    obstacle_stop_step = None
    unknown_obstacle = math.isnan(distance)

    pose_in_action_range = True

    for state_step in range(
        1,
        max_steps + 1,
    ):
        obstacle_threshold = node_step_distance(
            influence_radius=memory.influence_radius,
            robot_dimension=robot_dimension,
            state_step=state_step,
        )

        if math.isnan(distance):
            continue

        if distance <= obstacle_threshold:
            stopped_by_obstacle = True
            obstacle_stop_step = state_step

            if state_step == 1:
                physical_belief = _align_belief_dimension(
                    previous_belief,
                    model.num_states,
                )

                reinforce_blocked_self_loop(
                    model=model,
                    belief=physical_belief,
                    action_id=action_id,
                    motion_set=motion_set,
                )

            break

        if not pose_in_action_range:
            break

        previous_belief = _align_belief_dimension(
            previous_belief,
            model.num_states,
        )

        hypothetical = create_hypothetical_state(
            state=current_reference_state,
            action_id=action_id,
            memory=memory,
            model=model,
            motion_set=motion_set,
            robot_dimension=robot_dimension,
            state_step=1,
            reference_belief=previous_belief,
        )

        previous_belief = _align_belief_dimension(
            previous_belief,
            model.num_states,
        )

        if state_step > 1:
            direct_pose_in_range = (
                position_in_action_sector(
                    origin=current_reference_state.position,
                    candidate=hypothetical.position,
                    action_id=action_id,
                    influence_radius=memory.influence_radius,
                    motion_set=motion_set,
                )
            )

            if direct_pose_in_range:
                apply_direct_imagined_evidence(
                    model=model,
                    previous_belief=previous_belief,
                    next_belief=hypothetical.posterior,
                    action_id=action_id,
                    motion_set=motion_set,
                    reachable=True,
                )

                deep_direct_links_updated += 1

        nodes.append(
            LookaheadNode(
                state_step=state_step,
                obstacle_threshold=obstacle_threshold,
                place_id=hypothetical.place_id,
                position=hypothetical.position,
                place_created=hypothetical.place_created,
                model_expanded=hypothetical.model_expanded,
                posterior=hypothetical.posterior,
            )
        )

        previous_belief = hypothetical.posterior.copy()

        current_reference_state = CognitiveOdomState(
            position=hypothetical.position,
            travel_heading_rad=state.travel_heading_rad,
        )

        pose_in_action_range = (
            position_in_action_sector(
                origin=physical_origin,
                candidate=hypothetical.position,
                action_id=action_id,
                influence_radius=obstacle_threshold,
                motion_set=motion_set,
            )
        )

        if base_step_distance <= 0.0:
            raise RuntimeError(
                "base node step distance must be positive"
            )

    return DirectionalLookaheadResult(
        action_id=action_id,
        nodes=tuple(nodes),
        deep_direct_links_updated=deep_direct_links_updated,
        stopped_by_obstacle=stopped_by_obstacle,
        obstacle_stop_step=obstacle_stop_step,
        unknown_obstacle=unknown_obstacle,
    )


def _align_belief_dimension(
    belief: np.ndarray,
    num_states: int,
) -> np.ndarray:
    """Pad a baseline belief with zero-weight newly created states."""
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

    missing = num_states - len(result)

    if missing == 0:
        return result.copy()

    return np.append(
        result,
        np.zeros(missing),
    )


def _validate_max_steps(
    max_steps: int,
) -> None:
    """Validate the directional lookahead depth."""
    if isinstance(max_steps, bool) or not isinstance(
        max_steps,
        int,
    ):
        raise TypeError(
            "max_steps must be an integer"
        )

    if max_steps <= 0:
        raise ValueError(
            "max_steps must be positive"
        )
