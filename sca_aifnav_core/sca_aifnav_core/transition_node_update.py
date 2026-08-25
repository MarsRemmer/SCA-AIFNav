"""Multi-step transition-node update for the baseline."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.cognitive_growth import node_step_distance
from sca_aifnav_core.directional_lookahead import (
    DEFAULT_LOOKAHEAD_STEPS,
    LookaheadNode,
    grow_directional_lookahead,
)
from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.local_map_update import (
    update_imagined_neighborhood,
)
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory


@dataclass(frozen=True)
class DirectionTransitionUpdate:
    """Summarize one direction of the baseline transition-node sweep."""

    action_id: int
    obstacle_distance: float
    nodes: Tuple[LookaheadNode, ...]
    stopped_by_obstacle: bool
    obstacle_stop_step: Optional[int]
    unknown_obstacle: bool
    deep_direct_links_updated: int
    neighborhood_direct_updated: bool
    lateral_links_updated: int


@dataclass(frozen=True)
class TransitionNodeUpdate:
    """Summarize one complete baseline transition-node sweep."""

    directions: Tuple[DirectionTransitionUpdate, ...]
    deep_direct_links_updated: int
    neighborhood_direct_links_updated: int
    lateral_links_updated: int


def update_cognitive_transition_nodes(
    state: CognitiveOdomState,
    obstacle_distances,
    memory: BaselinePlaceMemory,
    model: BaselineGenerativeModel,
    motion_set: BaselineMotionSet,
    robot_dimension: float = 0.25,
    max_steps: int = DEFAULT_LOOKAHEAD_STEPS,
) -> TransitionNodeUpdate:
    """
    Reproduce the main structure of baseline transition-node growth.

    Each directional action first grows a straight hypothetical chain.
    The physical first-layer neighborhood is then updated for direct and
    lateral imagined transition evidence.
    """
    distances = _validated_obstacle_distances(
        obstacle_distances,
        motion_set.DIRECTION_COUNT,
    )

    base_step_distance = node_step_distance(
        influence_radius=memory.influence_radius,
        robot_dimension=robot_dimension,
        state_step=1,
    )

    physical_belief = model.state_belief.copy()

    direction_results = []

    deep_direct_total = 0
    neighborhood_direct_total = 0
    lateral_total = 0

    for action_id in range(
        motion_set.DIRECTION_COUNT
    ):
        physical_belief = _align_belief_dimension(
            physical_belief,
            model.num_states,
        )

        chain = grow_directional_lookahead(
            state=state,
            action_id=action_id,
            obstacle_distance=distances[action_id],
            memory=memory,
            model=model,
            motion_set=motion_set,
            robot_dimension=robot_dimension,
            max_steps=max_steps,
            reference_belief=physical_belief,
        )

        physical_belief = _align_belief_dimension(
            physical_belief,
            model.num_states,
        )

        direct_updated, lateral_updated = (
            update_imagined_neighborhood(
                state=state,
                action_id=action_id,
                obstacle_distances=distances,
                step_distance=base_step_distance,
                physical_belief=physical_belief,
                memory=memory,
                model=model,
                motion_set=motion_set,
            )
        )

        deep_direct_total += (
            chain.deep_direct_links_updated
        )

        neighborhood_direct_total += int(
            direct_updated
        )

        lateral_total += lateral_updated

        direction_results.append(
            DirectionTransitionUpdate(
                action_id=action_id,
                obstacle_distance=distances[
                    action_id
                ],
                nodes=chain.nodes,
                stopped_by_obstacle=(
                    chain.stopped_by_obstacle
                ),
                obstacle_stop_step=(
                    chain.obstacle_stop_step
                ),
                unknown_obstacle=(
                    chain.unknown_obstacle
                ),
                deep_direct_links_updated=(
                    chain.deep_direct_links_updated
                ),
                neighborhood_direct_updated=(
                    direct_updated
                ),
                lateral_links_updated=(
                    lateral_updated
                ),
            )
        )

    return TransitionNodeUpdate(
        directions=tuple(direction_results),
        deep_direct_links_updated=(
            deep_direct_total
        ),
        neighborhood_direct_links_updated=(
            neighborhood_direct_total
        ),
        lateral_links_updated=lateral_total,
    )


def _align_belief_dimension(
    belief: np.ndarray,
    num_states: int,
) -> np.ndarray:
    """Pad the physical belief after cognitive model growth."""
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


def _validated_obstacle_distances(
    obstacle_distances,
    expected_count: int,
) -> Tuple[float, ...]:
    """Validate one obstacle range per directional action."""
    values = tuple(obstacle_distances)

    if len(values) != expected_count:
        raise ValueError(
            "obstacle_distances must match directional action count"
        )

    result = []

    for distance in values:
        value = float(distance)

        if math.isinf(value) and value < 0.0:
            raise ValueError(
                "obstacle distance cannot be negative infinity"
            )

        result.append(value)

    return tuple(result)
