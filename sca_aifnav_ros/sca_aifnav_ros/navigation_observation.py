"""Frozen navigation observations passed toward the pure Python core."""

from dataclasses import dataclass

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)


@dataclass(frozen=True)
class NavigationObservation:
    """Represent one completed discrete navigation observation."""

    state: CognitiveOdomState
    sensory_observation: int
    place_observation: int
    obstacle_distances: tuple
    odometry_revision: int
    scan_revision: int


def capture_navigation_observation(
    state,
    sensory_observation,
    place_observation,
    obstacle_distances,
    odometry_revision,
    scan_revision,
) -> NavigationObservation:
    """Validate and freeze one navigation observation."""
    if not isinstance(
        state,
        CognitiveOdomState,
    ):
        raise TypeError(
            "state must be a CognitiveOdomState"
        )

    for name, value in (
        (
            "sensory_observation",
            sensory_observation,
        ),
        (
            "place_observation",
            place_observation,
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value < 0:
            raise ValueError(
                f"{name} must be non-negative"
            )

    try:
        distances = tuple(
            float(distance)
            for distance in obstacle_distances
        )
    except TypeError as exc:
        raise TypeError(
            "obstacle_distances must be iterable"
        ) from exc

    if (
        len(distances)
        != BaselineMotionSet.DIRECTION_COUNT
    ):
        raise ValueError(
            "obstacle_distances must contain "
            "twelve directional values"
        )

    for name, value in (
        (
            "odometry_revision",
            odometry_revision,
        ),
        (
            "scan_revision",
            scan_revision,
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be positive"
            )

    return NavigationObservation(
        state=state,
        sensory_observation=(
            sensory_observation
        ),
        place_observation=(
            place_observation
        ),
        obstacle_distances=distances,
        odometry_revision=(
            odometry_revision
        ),
        scan_revision=(
            scan_revision
        ),
    )
