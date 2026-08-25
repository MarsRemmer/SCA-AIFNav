"""Mapping between planar geometry and baseline motion primitives."""

import math

from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.planar_geometry import PlanarPose, Point2D


def target_action(
    pose: PlanarPose,
    target: Point2D,
    motion_set: BaselineMotionSet,
) -> int:
    """
    Return the baseline action pointing toward a target.

    reference baseline defines directional actions in the global map frame.
    The robot yaw therefore does not rotate the cognitive action sectors.
    """
    global_bearing = pose.global_bearing_to(target)

    return motion_set.action_for_angle_rad(global_bearing)


def target_distance(
    pose: PlanarPose,
    target: Point2D,
) -> float:
    """Return planar distance from a pose to a target point."""
    return pose.position.distance_to(target)


def projected_target(
    pose: PlanarPose,
    action_id: int,
    distance: float,
    motion_set: BaselineMotionSet,
) -> Point2D:
    """
    Project a point along one baseline action in the global map frame.

    Directional action sectors are fixed relative to the map coordinates,
    matching the reference baseline cognitive motion model.
    """
    if not math.isfinite(distance):
        raise ValueError("distance must be finite")

    if distance < 0.0:
        raise ValueError("distance must be non-negative")

    primitive = motion_set.action(action_id)

    if primitive.is_stationary:
        return pose.position

    global_angle = primitive.center_rad

    return Point2D(
        x=pose.x + distance * math.cos(global_angle),
        y=pose.y + distance * math.sin(global_angle),
    )
