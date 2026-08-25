"""Lateral imagined links for the baseline."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from sca_aifnav_core.cognitive_projection import (
    position_in_action_sector,
)
from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory
from sca_aifnav_core.transition_learning import (
    learn_bidirectional_transition,
)


LATERAL_DIRECT_RATE = 1.0
LATERAL_REVERSE_RATE = 1.0


@dataclass(frozen=True)
class LateralLinkResult:
    """Describe one baseline lateral imagined transition."""

    action_id: int
    reverse_action_id: int
    reachable: bool


def lateral_action_between(
    source: Point2D,
    target: Point2D,
    motion_set: BaselineMotionSet,
) -> int:
    """Return the global action pointing from source to target."""
    if source == target:
        raise ValueError(
            "source and target positions must differ"
        )

    bearing = source.bearing_to(target)

    return motion_set.action_for_angle_rad(
        bearing
    )


def apply_lateral_imagined_evidence(
    model: BaselineGenerativeModel,
    source_belief: np.ndarray,
    target_belief: np.ndarray,
    source_position: Point2D,
    target_position: Point2D,
    memory: BaselinePlaceMemory,
    motion_set: BaselineMotionSet,
    reachable: bool,
) -> Optional[LateralLinkResult]:
    """
    Learn one lateral imagined transition using baseline rules.

    If source and target resolve to the same location, or if the target is
    outside the geometric action range, no transition evidence is added.
    """
    if source_position == target_position:
        return None

    action_id = lateral_action_between(
        source=source_position,
        target=target_position,
        motion_set=motion_set,
    )

    in_action_range = position_in_action_sector(
        origin=source_position,
        candidate=target_position,
        action_id=action_id,
        influence_radius=memory.influence_radius,
        motion_set=motion_set,
    )

    if not in_action_range:
        return None

    sign = 1.0 if reachable else -1.0

    reverse_action_id = learn_bidirectional_transition(
        model=model,
        previous_belief=source_belief,
        next_belief=target_belief,
        action_id=action_id,
        motion_set=motion_set,
        direct_rate=sign * LATERAL_DIRECT_RATE,
        reverse_rate=sign * LATERAL_REVERSE_RATE,
    )

    return LateralLinkResult(
        action_id=action_id,
        reverse_action_id=reverse_action_id,
        reachable=reachable,
    )
