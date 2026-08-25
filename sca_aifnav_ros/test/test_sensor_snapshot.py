"""Tests for navigation sensor snapshots."""

import math

import numpy as np
import pytest

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.sensor_snapshot import (
    NavigationSensorSnapshot,
    capture_sensor_snapshot,
)


def cognitive_state():
    """Create one deterministic cognitive odometry state."""
    return CognitiveOdomState(
        position=Point2D(
            x=1.0,
            y=2.0,
        ),
        travel_heading_rad=0.5,
    )


def camera_image():
    """Create one deterministic BGR image."""
    return np.zeros(
        (
            4,
            6,
            3,
        ),
        dtype=np.uint8,
    )


def obstacle_distances():
    """Create twelve directional obstacle distances."""
    return [
        float(value)
        for value in range(
            1,
            13,
        )
    ]


def test_snapshot_captures_all_navigation_sensor_values():
    """One snapshot should preserve all required sensor values."""
    state = cognitive_state()

    snapshot = capture_sensor_snapshot(
        state=state,
        obstacle_distances=(
            obstacle_distances()
        ),
        image=camera_image(),
        odometry_revision=3,
        scan_revision=4,
        image_revision=5,
    )

    assert isinstance(
        snapshot,
        NavigationSensorSnapshot,
    )

    assert snapshot.state is state

    assert (
        snapshot.obstacle_distances
        == tuple(
            obstacle_distances()
        )
    )

    assert snapshot.image.shape == (
        4,
        6,
        3,
    )

    assert snapshot.odometry_revision == 3
    assert snapshot.scan_revision == 4
    assert snapshot.image_revision == 5


def test_snapshot_copies_image_data():
    """Changing the source image should not alter an existing snapshot."""
    image = camera_image()

    snapshot = capture_sensor_snapshot(
        state=cognitive_state(),
        obstacle_distances=(
            obstacle_distances()
        ),
        image=image,
        odometry_revision=1,
        scan_revision=1,
        image_revision=1,
    )

    image[
        0,
        0,
        0,
    ] = 255

    assert (
        snapshot.image[
            0,
            0,
            0,
        ]
        == 0
    )


def test_snapshot_freezes_obstacle_sequence():
    """Changing the source obstacle list should not alter the snapshot."""
    distances = obstacle_distances()

    snapshot = capture_sensor_snapshot(
        state=cognitive_state(),
        obstacle_distances=distances,
        image=camera_image(),
        odometry_revision=1,
        scan_revision=1,
        image_revision=1,
    )

    distances[0] = 99.0

    assert (
        snapshot.obstacle_distances[0]
        == pytest.approx(1.0)
    )

    assert isinstance(
        snapshot.obstacle_distances,
        tuple,
    )


def test_snapshot_preserves_nan_obstacle_distance():
    """An unobserved direction may remain NaN in the sensor snapshot."""
    distances = obstacle_distances()
    distances[4] = float("nan")

    snapshot = capture_sensor_snapshot(
        state=cognitive_state(),
        obstacle_distances=distances,
        image=camera_image(),
        odometry_revision=1,
        scan_revision=1,
        image_revision=1,
    )

    assert math.isnan(
        snapshot.obstacle_distances[
            4
        ]
    )


def test_snapshot_requires_cognitive_odometry():
    """A snapshot should reject missing cognitive odometry."""
    with pytest.raises(
        TypeError,
        match=(
            "state must be CognitiveOdomState"
        ),
    ):
        capture_sensor_snapshot(
            state=None,
            obstacle_distances=(
                obstacle_distances()
            ),
            image=camera_image(),
            odometry_revision=1,
            scan_revision=1,
            image_revision=1,
        )


def test_snapshot_requires_obstacle_distances():
    """A snapshot should reject missing obstacle information."""
    with pytest.raises(
        ValueError,
        match=(
            "obstacle_distances are required"
        ),
    ):
        capture_sensor_snapshot(
            state=cognitive_state(),
            obstacle_distances=None,
            image=camera_image(),
            odometry_revision=1,
            scan_revision=1,
            image_revision=1,
        )


def test_snapshot_requires_twelve_obstacle_directions():
    """A snapshot should require the complete twelve-direction format."""
    with pytest.raises(
        ValueError,
        match=(
            "obstacle_distances must contain 12 values"
        ),
    ):
        capture_sensor_snapshot(
            state=cognitive_state(),
            obstacle_distances=[
                1.0,
                2.0,
            ],
            image=camera_image(),
            odometry_revision=1,
            scan_revision=1,
            image_revision=1,
        )


def test_snapshot_requires_numpy_image():
    """A snapshot should reject non-array camera data."""
    with pytest.raises(
        TypeError,
        match=(
            "image must be a numpy array"
        ),
    ):
        capture_sensor_snapshot(
            state=cognitive_state(),
            obstacle_distances=(
                obstacle_distances()
            ),
            image=None,
            odometry_revision=1,
            scan_revision=1,
            image_revision=1,
        )


def test_snapshot_requires_bgr_image_shape():
    """A snapshot should require a three-channel image."""
    image = np.zeros(
        (
            4,
            6,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match=(
            "image must have three dimensions"
        ),
    ):
        capture_sensor_snapshot(
            state=cognitive_state(),
            obstacle_distances=(
                obstacle_distances()
            ),
            image=image,
            odometry_revision=1,
            scan_revision=1,
            image_revision=1,
        )


@pytest.mark.parametrize(
    "revision_name",
    [
        "odometry_revision",
        "scan_revision",
        "image_revision",
    ],
)
def test_snapshot_requires_positive_sensor_revisions(
    revision_name,
):
    """Each captured sensor stream should have been updated at least once."""
    arguments = {
        "state": cognitive_state(),
        "obstacle_distances": (
            obstacle_distances()
        ),
        "image": camera_image(),
        "odometry_revision": 1,
        "scan_revision": 1,
        "image_revision": 1,
    }

    arguments[
        revision_name
    ] = 0

    with pytest.raises(
        ValueError,
        match=(
            "sensor revisions must be positive"
        ),
    ):
        capture_sensor_snapshot(
            **arguments
        )
