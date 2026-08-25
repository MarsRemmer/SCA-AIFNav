"""Action-based spatial projection for the SCA-AIFNav baseline."""

import math
from typing import Optional

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory


def _angular_difference_deg(
    angle_a_deg: float,
    angle_b_deg: float,
) -> float:
    """Return the shortest signed angular difference in degrees."""
    return (
        angle_a_deg
        - angle_b_deg
        + 180.0
    ) % 360.0 - 180.0


def position_in_action_sector(
    origin: Point2D,
    candidate: Point2D,
    action_id: int,
    influence_radius: float,
    motion_set: BaselineMotionSet,
) -> bool:
    """
    Test whether a position lies inside a baseline action search sector.

    The action search region extends to twice the influence radius and
    spans the 30-degree directional sector associated with the action.
    """
    primitive = motion_set.action(action_id)

    if primitive.is_stationary:
        return candidate == origin

    distance = origin.distance_to(candidate)

    if distance == 0.0:
        return False

    if distance > 2.0 * influence_radius:
        return False

    bearing_deg = (
        math.degrees(origin.bearing_to(candidate))
        % 360.0
    )

    angular_error = _angular_difference_deg(
        bearing_deg,
        primitive.center_deg,
    )

    half_sector_deg = (
        motion_set.SECTOR_WIDTH_DEG / 2.0
    )

    return abs(angular_error) <= half_sector_deg + 1e-12


def existing_place_for_action(
    state: CognitiveOdomState,
    action_id: int,
    memory: BaselinePlaceMemory,
    motion_set: BaselineMotionSet,
) -> Optional[int]:
    """
    Return the first stored place reachable in an action sector.

    Stored places are inspected in creation order to preserve reference baseline
    behavior. The current position itself is excluded.
    """
    primitive = motion_set.action(action_id)

    if primitive.is_stationary:
        return None

    for place_id, position in enumerate(memory.places()):
        if position == state.position:
            continue

        if position_in_action_sector(
            origin=state.position,
            candidate=position,
            action_id=action_id,
            influence_radius=memory.influence_radius,
            motion_set=motion_set,
        ):
            return place_id

    return None


def project_action_position(
    state: CognitiveOdomState,
    action_id: int,
    memory: BaselinePlaceMemory,
    motion_set: BaselineMotionSet,
    ideal_distance: Optional[float] = None,
) -> Point2D:
    """
    Return the baseline spatial prediction associated with one action.

    An existing stored place inside the action search region is reused.
    Otherwise a hypothetical position is generated along the center of
    the action sector.
    """
    primitive = motion_set.action(action_id)

    if primitive.is_stationary:
        return state.position

    existing_place_id = existing_place_for_action(
        state=state,
        action_id=action_id,
        memory=memory,
        motion_set=motion_set,
    )

    if existing_place_id is not None:
        existing_position = memory.place(existing_place_id)

        if existing_position is None:
            raise RuntimeError(
                "stored place disappeared during action projection"
            )

        return existing_position

    if ideal_distance is None:
        ideal_distance = (
            memory.influence_radius
            + memory.influence_radius / 5.0
        )

    if not math.isfinite(ideal_distance):
        raise ValueError("ideal_distance must be finite")

    if ideal_distance < 0.0:
        raise ValueError(
            "ideal_distance must be non-negative"
        )

    angle_rad = primitive.center_rad

    return Point2D(
        x=state.x
        + ideal_distance * math.cos(angle_rad),
        y=state.y
        + ideal_distance * math.sin(angle_rad),
    )
