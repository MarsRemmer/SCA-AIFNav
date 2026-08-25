"""Tests for the first-layer V5 local cognitive-map update."""

import math

import numpy as np
import pytest

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.local_map_update import (
    NEIGHBOR_ACTION_JUMP,
    update_first_layer_local_map,
)
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory


@pytest.fixture
def motion_set():
    return BaselineMotionSet()


@pytest.fixture
def memory():
    result = BaselinePlaceMemory(
        influence_radius=0.5
    )
    result.resolve_place(
        Point2D(0.0, 0.0)
    )
    return result


@pytest.fixture
def model():
    return BaselineGenerativeModel()


@pytest.fixture
def origin_state():
    return CognitiveOdomState(
        position=Point2D(0.0, 0.0),
        travel_heading_rad=0.0,
    )


def test_neighbor_action_jump_matches_v5():
    assert NEIGHBOR_ACTION_JUMP == 2


def test_one_result_is_returned_per_direction(
    motion_set,
    memory,
    model,
    origin_state,
):
    obstacles = [0.0] * 12

    result = update_first_layer_local_map(
        state=origin_state,
        obstacle_distances=obstacles,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    assert len(result.actions) == 12


def test_all_blocked_directions_create_no_ghosts(
    motion_set,
    memory,
    model,
    origin_state,
):
    obstacles = [0.0] * 12

    result = update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert len(memory) == 1

    assert all(
        item.status == "blocked"
        for item in result.actions
    )


def test_blocked_action_strengthens_self_loop(
    motion_set,
    memory,
    model,
    origin_state,
):
    obstacles = [0.0] * 12

    update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert model.transition_likelihood[
        0,
        0,
        0,
    ] > 0.95


def test_clear_action_zero_creates_first_ghost(
    motion_set,
    memory,
    model,
    origin_state,
):
    obstacles = [0.0] * 12
    obstacles[0] = 5.0

    result = update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    action_zero = result.actions[0]

    assert action_zero.status == "created"
    assert action_zero.place_id == 1

    assert memory.place(1) == Point2D(
        0.60,
        0.16,
    )


def test_clear_action_zero_gets_direct_imagined_link(
    motion_set,
    memory,
    model,
    origin_state,
):
    obstacles = [0.0] * 12
    obstacles[0] = 5.0

    update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert model.transition_likelihood[
        1,
        0,
        0,
    ] > 0.90


def test_action_one_can_reuse_action_zero_place(
    motion_set,
    memory,
    model,
    origin_state,
):
    obstacles = [0.0] * 12
    obstacles[0] = 5.0
    obstacles[1] = 5.0

    result = update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert result.actions[0].place_id == 1
    assert result.actions[1].place_id == 1

    assert result.actions[1].status == "reused"


def test_clear_actions_zero_and_two_create_two_ghosts(
    motion_set,
    memory,
    model,
    origin_state,
):
    obstacles = [0.0] * 12
    obstacles[0] = 5.0
    obstacles[2] = 5.0

    result = update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert result.actions[0].place_id == 1
    assert result.actions[2].place_id == 2

    assert memory.place(1) == Point2D(
        0.60,
        0.16,
    )

    assert memory.place(2) == Point2D(
        0.16,
        0.60,
    )

    assert model.num_states == 3


def test_clear_neighbor_actions_gain_lateral_link(
    motion_set,
    memory,
    model,
    origin_state,
):
    obstacles = [0.0] * 12
    obstacles[0] = 5.0
    obstacles[1] = 5.0
    obstacles[2] = 5.0

    result = update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert result.lateral_links_updated >= 1

    assert model.transition_likelihood[
        2,
        1,
        4,
    ] > 0.90


def test_blocked_known_projection_gets_negative_direct_evidence(
    motion_set,
    memory,
    model,
    origin_state,
):
    memory.resolve_place(
        Point2D(0.60, 0.16)
    )
    model.register_place_observation(1)

    obstacles = [0.0] * 12

    update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert model.transition_likelihood[
        1,
        0,
        0,
    ] < 0.01


def test_nan_obstacle_does_not_create_new_place(
    motion_set,
    memory,
    model,
    origin_state,
):
    obstacles = [0.0] * 12
    obstacles[0] = math.nan

    result = update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert result.actions[0].status == "unknown"
    assert result.actions[0].place_id is None
    assert len(memory) == 1


def test_physical_belief_is_not_replaced_by_sweep(
    motion_set,
    memory,
    model,
    origin_state,
):
    physical = model.state_belief.copy()

    obstacles = [0.0] * 12
    obstacles[0] = 5.0
    obstacles[2] = 5.0

    update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    np.testing.assert_allclose(
        model.state_belief[:2],
        physical,
    )

    assert model.state_belief[2] == pytest.approx(
        0.0
    )


def test_wrong_obstacle_count_is_rejected(
    motion_set,
    memory,
    model,
    origin_state,
):
    with pytest.raises(ValueError):
        update_first_layer_local_map(
            origin_state,
            [1.0] * 11,
            memory,
            model,
            motion_set,
        )


def test_blocked_middle_direction_weakens_lateral_link(
    motion_set,
    memory,
    model,
    origin_state,
):
    obstacles = [0.0] * 12

    obstacles[0] = 5.0
    obstacles[2] = 5.0

    result = update_first_layer_local_map(
        origin_state,
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert result.actions[0].place_id == 1
    assert result.actions[1].status == "blocked"
    assert result.actions[2].place_id == 2

    assert result.lateral_links_updated >= 1

    assert model.transition_likelihood[
        2,
        1,
        4,
    ] < 1.0 / 3.0
