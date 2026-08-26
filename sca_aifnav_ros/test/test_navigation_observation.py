"""Tests for frozen core-facing navigation observations."""

import math

import pytest

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.navigation_observation import (
    capture_navigation_observation,
)


def cognitive_state():
    """Create one deterministic cognitive state."""
    return CognitiveOdomState(
        position=Point2D(
            1.0,
            -2.0,
        ),
        travel_heading_rad=0.5,
    )


def distances():
    """Create twelve directional obstacle values."""
    return tuple(
        float(index)
        for index in range(12)
    )


def test_navigation_observation_is_frozen():
    """A valid observation should preserve core-facing values."""
    state = cognitive_state()

    observation = (
        capture_navigation_observation(
            state=state,
            sensory_observation=3,
            place_observation=2,
            obstacle_distances=distances(),
            odometry_revision=10,
            scan_revision=20,
        )
    )

    assert observation.state is state
    assert observation.sensory_observation == 3
    assert observation.place_observation == 2

    assert (
        observation.obstacle_distances
        == distances()
    )

    assert observation.odometry_revision == 10
    assert observation.scan_revision == 20


def test_obstacle_distances_are_copied_to_tuple():
    """Later changes to an input list should not alter the observation."""
    source = list(
        distances()
    )

    observation = (
        capture_navigation_observation(
            state=cognitive_state(),
            sensory_observation=0,
            place_observation=0,
            obstacle_distances=source,
            odometry_revision=1,
            scan_revision=1,
        )
    )

    source[0] = 999.0

    assert (
        observation.obstacle_distances[0]
        == 0.0
    )


def test_nan_obstacle_value_is_preserved():
    """Empty directional sectors may legitimately remain NaN."""
    source = list(
        distances()
    )

    source[4] = math.nan

    observation = (
        capture_navigation_observation(
            state=cognitive_state(),
            sensory_observation=0,
            place_observation=0,
            obstacle_distances=source,
            odometry_revision=1,
            scan_revision=1,
        )
    )

    assert math.isnan(
        observation.obstacle_distances[4]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "sensory_observation",
            True,
        ),
        (
            "sensory_observation",
            1.0,
        ),
        (
            "place_observation",
            False,
        ),
        (
            "place_observation",
            "1",
        ),
    ],
)
def test_discrete_observation_ids_must_be_integers(
    field,
    value,
):
    """Discrete visual and place observations require integer IDs."""
    arguments = {
        "state": cognitive_state(),
        "sensory_observation": 0,
        "place_observation": 0,
        "obstacle_distances": distances(),
        "odometry_revision": 1,
        "scan_revision": 1,
    }

    arguments[field] = value

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        capture_navigation_observation(
            **arguments
        )


@pytest.mark.parametrize(
    "field",
    [
        "sensory_observation",
        "place_observation",
    ],
)
def test_discrete_observation_ids_cannot_be_negative(
    field,
):
    """Discrete observation IDs should be non-negative."""
    arguments = {
        "state": cognitive_state(),
        "sensory_observation": 0,
        "place_observation": 0,
        "obstacle_distances": distances(),
        "odometry_revision": 1,
        "scan_revision": 1,
    }

    arguments[field] = -1

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        capture_navigation_observation(
            **arguments
        )


def test_exactly_twelve_obstacle_values_are_required():
    """The core action geometry requires twelve directional distances."""
    with pytest.raises(
        ValueError,
        match="twelve",
    ):
        capture_navigation_observation(
            state=cognitive_state(),
            sensory_observation=0,
            place_observation=0,
            obstacle_distances=(
                1.0,
                2.0,
            ),
            odometry_revision=1,
            scan_revision=1,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "odometry_revision",
            0,
        ),
        (
            "scan_revision",
            0,
        ),
        (
            "odometry_revision",
            -1,
        ),
        (
            "scan_revision",
            -1,
        ),
    ],
)
def test_sensor_revisions_must_be_positive(
    field,
    value,
):
    """A complete observation must originate from real sensor updates."""
    arguments = {
        "state": cognitive_state(),
        "sensory_observation": 0,
        "place_observation": 0,
        "obstacle_distances": distances(),
        "odometry_revision": 1,
        "scan_revision": 1,
    }

    arguments[field] = value

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        capture_navigation_observation(
            **arguments
        )
