"""Navigation sensor snapshots for SCA-AIFNav."""

from dataclasses import dataclass

import numpy as np

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)


@dataclass(frozen=True)
class NavigationSensorSnapshot:
    """Store one synchronized navigation-time sensor snapshot."""

    state: CognitiveOdomState
    obstacle_distances: tuple
    image: np.ndarray

    odometry_revision: int
    scan_revision: int
    image_revision: int


def capture_sensor_snapshot(
    state: CognitiveOdomState,
    obstacle_distances,
    image: np.ndarray,
    odometry_revision: int,
    scan_revision: int,
    image_revision: int,
) -> NavigationSensorSnapshot:
    """
    Freeze the latest navigation sensor values into one snapshot.

    The snapshot does not create visual or place observation IDs. It only
    captures the latest ROS-derived sensor state for one later navigation
    update.
    """
    if not isinstance(
        state,
        CognitiveOdomState,
    ):
        raise TypeError(
            "state must be CognitiveOdomState"
        )

    if obstacle_distances is None:
        raise ValueError(
            "obstacle_distances are required"
        )

    distances = tuple(
        float(distance)
        for distance in obstacle_distances
    )

    if len(distances) != 12:
        raise ValueError(
            "obstacle_distances must contain 12 values"
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "image must be a numpy array"
        )

    if image.ndim != 3:
        raise ValueError(
            "image must have three dimensions"
        )

    if image.shape[2] != 3:
        raise ValueError(
            "image must have three BGR channels"
        )

    revisions = (
        odometry_revision,
        scan_revision,
        image_revision,
    )

    for revision in revisions:
        if (
            isinstance(revision, bool)
            or not isinstance(
                revision,
                int,
            )
        ):
            raise TypeError(
                "sensor revisions must be integers"
            )

        if revision < 1:
            raise ValueError(
                "sensor revisions must be positive"
            )

    return NavigationSensorSnapshot(
        state=state,
        obstacle_distances=distances,
        image=image.copy(),
        odometry_revision=(
            odometry_revision
        ),
        scan_revision=(
            scan_revision
        ),
        image_revision=(
            image_revision
        ),
    )
