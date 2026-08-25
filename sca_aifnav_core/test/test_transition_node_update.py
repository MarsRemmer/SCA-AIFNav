"""Tests for the assembled baseline transition-node update."""

import math

import numpy as np
import pytest

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.local_map_update import (
    update_first_layer_local_map,
)
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory
from sca_aifnav_core.transition_node_update import (
    update_cognitive_transition_nodes,
)


@pytest.fixture
def motion_set():
    return BaselineMotionSet()


def make_memory():
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    return memory


def make_state():
    return CognitiveOdomState(
        position=Point2D(0.0, 0.0),
        travel_heading_rad=0.0,
    )


def test_sweep_returns_all_twelve_directions(
    motion_set,
):
    memory = make_memory()
    model = BaselineGenerativeModel()

    result = update_cognitive_transition_nodes(
        state=make_state(),
        obstacle_distances=[0.0] * 12,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    assert len(result.directions) == 12


def test_all_blocked_creates_no_ghost_nodes(
    motion_set,
):
    memory = make_memory()
    model = BaselineGenerativeModel()

    result = update_cognitive_transition_nodes(
        make_state(),
        [0.0] * 12,
        memory,
        model,
        motion_set,
    )

    assert len(memory) == 1

    assert all(
        len(direction.nodes) == 0
        for direction in result.directions
    )

    assert all(
        direction.obstacle_stop_step == 1
        for direction in result.directions
    )


def test_blocked_direction_strengthens_self_loop(
    motion_set,
):
    memory = make_memory()
    model = BaselineGenerativeModel()

    update_cognitive_transition_nodes(
        make_state(),
        [0.0] * 12,
        memory,
        model,
        motion_set,
    )

    assert model.transition_likelihood[
        0,
        0,
        0,
    ] > 0.95


def test_one_clear_direction_grows_three_nodes(
    motion_set,
):
    memory = make_memory()
    model = BaselineGenerativeModel()

    obstacles = [0.0] * 12
    obstacles[0] = 5.0

    result = update_cognitive_transition_nodes(
        make_state(),
        obstacles,
        memory,
        model,
        motion_set,
    )

    action_zero = result.directions[0]

    assert len(action_zero.nodes) == 3

    assert action_zero.nodes[0].position == Point2D(
        0.60,
        0.16,
    )

    assert action_zero.nodes[1].position == Point2D(
        1.20,
        0.32,
    )

    assert action_zero.nodes[2].position == Point2D(
        1.80,
        0.48,
    )


def test_one_clear_direction_has_two_deep_links(
    motion_set,
):
    memory = make_memory()
    model = BaselineGenerativeModel()

    obstacles = [0.0] * 12
    obstacles[0] = 5.0

    result = update_cognitive_transition_nodes(
        make_state(),
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert (
        result.directions[
            0
        ].deep_direct_links_updated
        == 2
    )

    assert result.deep_direct_links_updated == 2


def test_first_and_deep_edges_are_all_learned(
    motion_set,
):
    memory = make_memory()
    model = BaselineGenerativeModel()

    obstacles = [0.0] * 12
    obstacles[0] = 5.0

    update_cognitive_transition_nodes(
        make_state(),
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

    assert model.transition_likelihood[
        2,
        1,
        0,
    ] > 0.90

    assert model.transition_likelihood[
        3,
        2,
        0,
    ] > 0.90


def test_one_meter_obstacle_stops_at_second_step(
    motion_set,
):
    memory = make_memory()
    model = BaselineGenerativeModel()

    obstacles = [0.0] * 12
    obstacles[0] = 1.0

    result = update_cognitive_transition_nodes(
        make_state(),
        obstacles,
        memory,
        model,
        motion_set,
    )

    action_zero = result.directions[0]

    assert len(action_zero.nodes) == 1
    assert action_zero.obstacle_stop_step == 2


def test_nan_direction_creates_no_chain(
    motion_set,
):
    memory = make_memory()
    model = BaselineGenerativeModel()

    obstacles = [0.0] * 12
    obstacles[0] = math.nan

    result = update_cognitive_transition_nodes(
        make_state(),
        obstacles,
        memory,
        model,
        motion_set,
    )

    assert result.directions[0].unknown_obstacle
    assert len(result.directions[0].nodes) == 0


def test_depth_one_matches_existing_first_layer_update(
    motion_set,
):
    obstacles = [0.0] * 12
    obstacles[0] = 5.0
    obstacles[1] = 5.0
    obstacles[2] = 5.0

    full_memory = make_memory()
    full_model = BaselineGenerativeModel()

    update_cognitive_transition_nodes(
        state=make_state(),
        obstacle_distances=obstacles,
        memory=full_memory,
        model=full_model,
        motion_set=motion_set,
        max_steps=1,
    )

    first_memory = make_memory()
    first_model = BaselineGenerativeModel()

    update_first_layer_local_map(
        state=make_state(),
        obstacle_distances=obstacles,
        memory=first_memory,
        model=first_model,
        motion_set=motion_set,
    )

    assert full_memory.places() == (
        first_memory.places()
    )

    np.testing.assert_allclose(
        full_model.transition_concentration,
        first_model.transition_concentration,
    )

    np.testing.assert_allclose(
        full_model.transition_likelihood,
        first_model.transition_likelihood,
    )


def test_physical_belief_is_only_padded(
    motion_set,
):
    memory = make_memory()
    model = BaselineGenerativeModel()

    physical = model.state_belief.copy()

    obstacles = [0.0] * 12
    obstacles[0] = 5.0

    update_cognitive_transition_nodes(
        make_state(),
        obstacles,
        memory,
        model,
        motion_set,
    )

    np.testing.assert_allclose(
        model.state_belief[:2],
        physical,
    )

    np.testing.assert_allclose(
        model.state_belief[2:],
        np.zeros(2),
    )


def test_wrong_obstacle_count_is_rejected(
    motion_set,
):
    with pytest.raises(ValueError):
        update_cognitive_transition_nodes(
            state=make_state(),
            obstacle_distances=[1.0] * 11,
            memory=make_memory(),
            model=BaselineGenerativeModel(),
            motion_set=motion_set,
        )
