"""Tests for baseline-compatible action-based spatial projection."""

import math

import pytest

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.cognitive_projection import (
    existing_place_for_action,
    position_in_action_sector,
    project_action_position,
)
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory


@pytest.fixture
def motion_set():
    return BaselineMotionSet()


def make_state(
    x=0.0,
    y=0.0,
    heading=0.0,
):
    return CognitiveOdomState(
        position=Point2D(x, y),
        travel_heading_rad=heading,
    )


def test_position_inside_action_zero_sector(motion_set):
    candidate = Point2D(
        0.8 * math.cos(math.radians(15.0)),
        0.8 * math.sin(math.radians(15.0)),
    )

    assert position_in_action_sector(
        origin=Point2D(0.0, 0.0),
        candidate=candidate,
        action_id=0,
        influence_radius=0.5,
        motion_set=motion_set,
    )


def test_position_outside_action_sector_is_rejected(
    motion_set,
):
    candidate = Point2D(
        0.8 * math.cos(math.radians(45.0)),
        0.8 * math.sin(math.radians(45.0)),
    )

    assert not position_in_action_sector(
        origin=Point2D(0.0, 0.0),
        candidate=candidate,
        action_id=0,
        influence_radius=0.5,
        motion_set=motion_set,
    )


def test_position_beyond_twice_radius_is_rejected(
    motion_set,
):
    candidate = Point2D(
        1.01 * math.cos(math.radians(15.0)),
        1.01 * math.sin(math.radians(15.0)),
    )

    assert not position_in_action_sector(
        origin=Point2D(0.0, 0.0),
        candidate=candidate,
        action_id=0,
        influence_radius=0.5,
        motion_set=motion_set,
    )


def test_position_at_twice_radius_is_included(
    motion_set,
):
    candidate = Point2D(
        math.cos(math.radians(15.0)),
        math.sin(math.radians(15.0)),
    )

    assert position_in_action_sector(
        origin=Point2D(0.0, 0.0),
        candidate=candidate,
        action_id=0,
        influence_radius=0.5,
        motion_set=motion_set,
    )


def test_action_sector_boundary_is_included(motion_set):
    candidate = Point2D(
        0.8 * math.cos(math.radians(30.0)),
        0.8 * math.sin(math.radians(30.0)),
    )

    assert position_in_action_sector(
        origin=Point2D(0.0, 0.0),
        candidate=candidate,
        action_id=0,
        influence_radius=0.5,
        motion_set=motion_set,
    )


def test_action_eleven_wraps_across_zero_degrees(
    motion_set,
):
    candidate = Point2D(
        0.8 * math.cos(math.radians(350.0)),
        0.8 * math.sin(math.radians(350.0)),
    )

    assert position_in_action_sector(
        origin=Point2D(0.0, 0.0),
        candidate=candidate,
        action_id=11,
        influence_radius=0.5,
        motion_set=motion_set,
    )


def test_current_position_is_not_existing_transition(
    motion_set,
):
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )
    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    state = make_state()

    assert existing_place_for_action(
        state,
        0,
        memory,
        motion_set,
    ) is None


def test_existing_place_in_action_sector_is_reused(
    motion_set,
):
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    existing = Point2D(
        0.8 * math.cos(math.radians(15.0)),
        0.8 * math.sin(math.radians(15.0)),
    )
    existing_id = memory.resolve_place(existing)

    state = make_state()

    assert existing_place_for_action(
        state,
        0,
        memory,
        motion_set,
    ) == existing_id

    result = project_action_position(
        state,
        0,
        memory,
        motion_set,
    )

    assert result == existing


def test_baseline_uses_first_matching_place_not_nearest(
    motion_set,
):
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    first_match = Point2D(1.0, 0.0)
    nearer_match = Point2D(0.5, 0.0)

    first_id = memory.resolve_place(first_match)
    second_id = memory.resolve_place(nearer_match)

    assert first_id == 1
    assert second_id == 2

    state = make_state()

    assert existing_place_for_action(
        state,
        0,
        memory,
        motion_set,
    ) == first_id

    assert project_action_position(
        state,
        0,
        memory,
        motion_set,
    ) == first_match


def test_missing_place_generates_default_hypothetical_pose(
    motion_set,
):
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    state = make_state()

    result = project_action_position(
        state,
        0,
        memory,
        motion_set,
    )

    expected_distance = 0.6

    assert result.x == pytest.approx(
        expected_distance
        * math.cos(math.radians(15.0))
    )
    assert result.y == pytest.approx(
        expected_distance
        * math.sin(math.radians(15.0))
    )


def test_default_hypothetical_distance_is_1_point_2_radius(
    motion_set,
):
    memory = BaselinePlaceMemory(
        influence_radius=1.0
    )

    state = make_state()

    result = project_action_position(
        state,
        3,
        memory,
        motion_set,
    )

    assert state.position.distance_to(result) == pytest.approx(
        1.2
    )


def test_custom_ideal_distance_is_respected(motion_set):
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    state = make_state()

    result = project_action_position(
        state,
        0,
        memory,
        motion_set,
        ideal_distance=2.5,
    )

    assert state.position.distance_to(result) == pytest.approx(
        2.5
    )


def test_projection_uses_global_action_frame(
    motion_set,
):
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    state = make_state(
        heading=math.pi / 2.0
    )

    result = project_action_position(
        state,
        0,
        memory,
        motion_set,
    )

    assert math.degrees(
        state.position.bearing_to(result)
    ) == pytest.approx(15.0)


def test_stay_returns_current_position(motion_set):
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    state = make_state(
        x=2.0,
        y=3.0,
        heading=1.0,
    )

    result = project_action_position(
        state,
        12,
        memory,
        motion_set,
    )

    assert result == Point2D(2.0, 3.0)


def test_hypothetical_projection_does_not_modify_memory(
    motion_set,
):
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    state = make_state()

    assert len(memory) == 0

    project_action_position(
        state,
        0,
        memory,
        motion_set,
    )

    assert len(memory) == 0


@pytest.mark.parametrize(
    "invalid_distance",
    [
        -1.0,
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_invalid_ideal_distance_is_rejected(
    motion_set,
    invalid_distance,
):
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    state = make_state()

    with pytest.raises(ValueError):
        project_action_position(
            state,
            0,
            memory,
            motion_set,
            ideal_distance=invalid_distance,
        )
