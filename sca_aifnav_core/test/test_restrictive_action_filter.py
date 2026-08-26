"""Tests for current-obstacle root action filtering."""

import math

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.navigation_cycle import (
    BaselineNavigationCoordinator,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)


def coordinator_with_action_zero_target():
    """Create a simple two-place navigation model."""
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    origin_id = memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    distance = 0.65
    angle = math.radians(
        15.0
    )

    memory.resolve_place(
        Point2D(
            distance * math.cos(angle),
            distance * math.sin(angle),
        )
    )

    model = BaselineGenerativeModel(
        sensory_observations=2,
        place_observations=2,
        num_states=2,
    )

    coordinator = (
        BaselineNavigationCoordinator(
            model=model,
            memory=memory,
            motion_set=BaselineMotionSet(),
            robot_dimension=0.3,
        )
    )

    return (
        coordinator,
        origin_id,
    )


def test_runtime_minimum_distance_is_point_65():
    """Runtime geometry should use 0.5 + 0.3 / 2."""
    coordinator, _ = (
        coordinator_with_action_zero_target()
    )

    minimum_distance = (
        coordinator.memory.influence_radius
        + coordinator.learning.robot_dimension
        / 2.0
    )

    assert math.isclose(
        minimum_distance,
        0.65,
    )


def test_blocked_direction_is_not_given_to_mcts():
    """A direction below 0.65 m must not enter the root action set."""
    coordinator, origin_id = (
        coordinator_with_action_zero_target()
    )

    distances = [
        12.0
        for _ in range(12)
    ]

    distances[0] = 0.64

    actions = (
        coordinator
        .restrictive_possible_actions(
            current_place_id=origin_id,
            obstacle_distances=distances,
        )
    )

    assert 0 not in actions
    assert 12 in actions


def test_clear_known_direction_can_enter_mcts():
    """A known direction at the threshold should remain available."""
    coordinator, origin_id = (
        coordinator_with_action_zero_target()
    )

    distances = [
        0.0
        for _ in range(12)
    ]

    distances[0] = 0.65

    actions = (
        coordinator
        .restrictive_possible_actions(
            current_place_id=origin_id,
            obstacle_distances=distances,
        )
    )

    assert 0 in actions
    assert 12 in actions
