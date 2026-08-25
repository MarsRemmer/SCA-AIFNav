"""First-layer local cognitive-map update for reference baseline."""

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
    project_action_position,
)
from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.lateral_linking import (
    apply_lateral_imagined_evidence,
    lateral_action_between,
)
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.obstacle_evidence import (
    apply_direct_imagined_evidence,
    reinforce_blocked_self_loop,
)
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory


NEIGHBOR_ACTION_JUMP = 2


@dataclass(frozen=True)
class FirstLayerActionUpdate:
    """Summarize one directional update of the local cognitive map."""

    action_id: int
    obstacle_distance: float
    status: str
    place_id: Optional[int]
    place_created: bool
    model_expanded: bool


@dataclass(frozen=True)
class FirstLayerMapUpdate:
    """Summarize one complete first-layer baseline directional sweep."""

    actions: Tuple[FirstLayerActionUpdate, ...]
    direct_links_updated: int
    lateral_links_updated: int


def update_first_layer_local_map(
    state: CognitiveOdomState,
    obstacle_distances,
    memory: BaselinePlaceMemory,
    model: BaselineGenerativeModel,
    motion_set: BaselineMotionSet,
    robot_dimension: float = 0.25,
) -> FirstLayerMapUpdate:
    """
    Update the first layer of the local baseline cognitive map.

    All directional actions are inspected in order. Obstacle information
    determines whether the model reinforces a self-loop or creates/reuses
    a hypothetical place. Direct and lateral imagined links are then
    updated using the local neighborhood around that action.
    """
    distances = _validated_obstacle_distances(
        obstacle_distances,
        motion_set.DIRECTION_COUNT,
    )

    step_distance = node_step_distance(
        influence_radius=memory.influence_radius,
        robot_dimension=robot_dimension,
        state_step=1,
    )

    physical_belief = model.state_belief.copy()

    action_results = []
    direct_links_updated = 0
    lateral_links_updated = 0

    for action_id in range(
        motion_set.DIRECTION_COUNT
    ):
        distance = distances[action_id]

        physical_belief = _align_belief_dimension(
            physical_belief,
            model.num_states,
        )

        if math.isnan(distance):
            status = "unknown"
            place_id = None
            place_created = False
            model_expanded = False

        elif distance <= step_distance:
            reinforce_blocked_self_loop(
                model=model,
                belief=physical_belief,
                action_id=action_id,
                motion_set=motion_set,
            )

            status = "blocked"
            place_id = None
            place_created = False
            model_expanded = False

        else:
            hypothetical = create_hypothetical_state(
                state=state,
                action_id=action_id,
                memory=memory,
                model=model,
                motion_set=motion_set,
                robot_dimension=robot_dimension,
                state_step=1,
                reference_belief=physical_belief,
            )

            place_id = hypothetical.place_id
            place_created = (
                hypothetical.place_created
            )
            model_expanded = (
                hypothetical.model_expanded
            )

            if place_created:
                status = "created"
            else:
                status = "reused"

        physical_belief = _align_belief_dimension(
            physical_belief,
            model.num_states,
        )

        direct_updated, lateral_updated = (
            update_imagined_neighborhood(
                state=state,
                action_id=action_id,
                obstacle_distances=distances,
                step_distance=step_distance,
                physical_belief=physical_belief,
                memory=memory,
                model=model,
                motion_set=motion_set,
            )
        )

        direct_links_updated += int(
            direct_updated
        )
        lateral_links_updated += lateral_updated

        action_results.append(
            FirstLayerActionUpdate(
                action_id=action_id,
                obstacle_distance=distance,
                status=status,
                place_id=place_id,
                place_created=place_created,
                model_expanded=model_expanded,
            )
        )

    return FirstLayerMapUpdate(
        actions=tuple(action_results),
        direct_links_updated=direct_links_updated,
        lateral_links_updated=lateral_links_updated,
    )


def update_imagined_neighborhood(
    state: CognitiveOdomState,
    action_id: int,
    obstacle_distances: Tuple[float, ...],
    step_distance: float,
    physical_belief: np.ndarray,
    memory: BaselinePlaceMemory,
    model: BaselineGenerativeModel,
    motion_set: BaselineMotionSet,
) -> Tuple[bool, int]:
    """Update direct and lateral imagined links around one action."""
    main_place_id = _existing_projected_place(
        state=state,
        action_id=action_id,
        step_distance=step_distance,
        memory=memory,
        motion_set=motion_set,
    )

    if main_place_id is None:
        return False, 0

    main_position = memory.place(
        main_place_id
    )

    if main_position is None:
        raise RuntimeError(
            "main place disappeared during map update"
        )

    direct_updated = False
    lateral_count = 0
    previous_lateral_action = None

    for offset in (
        NEIGHBOR_ACTION_JUMP,
        1,
        0,
        -1,
        -NEIGHBOR_ACTION_JUMP,
    ):
        adjacent_action = (
            action_id + offset
        ) % motion_set.DIRECTION_COUNT

        adjacent_place_id = (
            _existing_projected_place(
                state=state,
                action_id=adjacent_action,
                step_distance=step_distance,
                memory=memory,
                motion_set=motion_set,
            )
        )

        if adjacent_place_id is None:
            continue

        if (
            offset != 0
            and adjacent_place_id == main_place_id
        ):
            continue

        adjacent_position = memory.place(
            adjacent_place_id
        )

        if adjacent_position is None:
            raise RuntimeError(
                "adjacent place disappeared during map update"
            )

        aligned_physical = (
            _align_belief_dimension(
                physical_belief,
                model.num_states,
            )
        )

        adjacent_belief = _infer_place_belief(
            model=model,
            physical_belief=aligned_physical,
            place_id=adjacent_place_id,
            action_id=adjacent_action,
        )

        reachable = (
            _distance_is_clear(
                obstacle_distances[action_id],
                step_distance,
            )
            and _distance_is_clear(
                obstacle_distances[
                    adjacent_action
                ],
                step_distance,
            )
        )

        if offset == 0:
            apply_direct_imagined_evidence(
                model=model,
                previous_belief=aligned_physical,
                next_belief=adjacent_belief,
                action_id=action_id,
                motion_set=motion_set,
                reachable=reachable,
            )

            direct_updated = True
            continue

        lateral_action = lateral_action_between(
            source=main_position,
            target=adjacent_position,
            motion_set=motion_set,
        )

        if lateral_action == previous_lateral_action:
            continue

        main_belief = _infer_place_belief(
            model=model,
            physical_belief=aligned_physical,
            place_id=main_place_id,
            action_id=action_id,
        )

        lateral_result = (
            apply_lateral_imagined_evidence(
                model=model,
                source_belief=main_belief,
                target_belief=adjacent_belief,
                source_position=main_position,
                target_position=adjacent_position,
                memory=memory,
                motion_set=motion_set,
                reachable=reachable,
            )
        )

        if lateral_result is None:
            continue

        previous_lateral_action = (
            lateral_result.action_id
        )
        lateral_count += 1

    return direct_updated, lateral_count


def _existing_projected_place(
    state: CognitiveOdomState,
    action_id: int,
    step_distance: float,
    memory: BaselinePlaceMemory,
    motion_set: BaselineMotionSet,
) -> Optional[int]:
    """
    Return the place matching the baseline's action projection without creating it.

    The action projection is rounded to two decimals before fixed-radius
    place matching, matching determine_next_pose behavior.
    """
    projected = project_action_position(
        state=state,
        action_id=action_id,
        memory=memory,
        motion_set=motion_set,
        ideal_distance=step_distance,
    )

    rounded = type(projected)(
        x=round(projected.x, 2),
        y=round(projected.y, 2),
    )

    place_id = memory.find_match(
        rounded
    )

    if place_id < 0:
        return None

    return place_id


def _infer_place_belief(
    model: BaselineGenerativeModel,
    physical_belief: np.ndarray,
    place_id: int,
    action_id: int,
) -> np.ndarray:
    """Infer a hypothetical state from one place observation."""
    aligned = _align_belief_dimension(
        physical_belief,
        model.num_states,
    )

    prior = model.predicted_state_prior(
        action_id=action_id,
        belief=aligned,
    )

    return model.infer_state_belief(
        place_observation=place_id,
        prior=prior,
    )


def _distance_is_clear(
    distance: float,
    required_distance: float,
) -> bool:
    """Return whether baseline treats an obstacle range as reachable."""
    return (
        not math.isnan(distance)
        and distance > required_distance
    )


def _align_belief_dimension(
    belief: np.ndarray,
    num_states: int,
) -> np.ndarray:
    """Pad an older state belief with zero-valued new states."""
    result = np.asarray(
        belief,
        dtype=float,
    )

    if result.ndim != 1:
        raise ValueError(
            "belief must be one-dimensional"
        )

    if len(result) > num_states:
        raise ValueError(
            "belief has more entries than model states"
        )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "belief must contain finite values"
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
    """Validate one obstacle range for every directional action."""
    if len(obstacle_distances) != expected_count:
        raise ValueError(
            "obstacle_distances must match directional action count"
        )

    result = []

    for distance in obstacle_distances:
        value = float(distance)

        if math.isinf(value) and value < 0.0:
            raise ValueError(
                "obstacle distance cannot be negative infinity"
            )

        result.append(value)

    return tuple(result)
