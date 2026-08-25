"""Tests for V5-compatible directional cognitive lookahead."""

import math

import numpy as np
import pytest

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.directional_lookahead import (
    DEFAULT_LOOKAHEAD_STEPS,
    grow_directional_lookahead,
)
from sca_aifnav_core.generative_model import BaselineGenerativeModel
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


def test_default_lookahead_depth_matches_v5():
    assert DEFAULT_LOOKAHEAD_STEPS == 3


def test_clear_action_creates_three_nodes(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        state=origin_state,
        action_id=0,
        obstacle_distance=5.0,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    assert len(result.nodes) == 3


def test_three_node_positions_use_fixed_increment(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        5.0,
        memory,
        model,
        motion_set,
    )

    assert result.nodes[0].position == Point2D(
        0.60,
        0.16,
    )

    assert result.nodes[1].position == Point2D(
        1.20,
        0.32,
    )

    assert result.nodes[2].position == Point2D(
        1.80,
        0.48,
    )


def test_obstacle_thresholds_grow_with_depth(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        5.0,
        memory,
        model,
        motion_set,
    )

    thresholds = [
        node.obstacle_threshold
        for node in result.nodes
    ]

    np.testing.assert_allclose(
        thresholds,
        np.array(
            [
                0.625,
                1.125,
                1.625,
            ]
        ),
    )


def test_first_node_uses_reserved_state(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        5.0,
        memory,
        model,
        motion_set,
    )

    assert result.nodes[0].place_id == 1
    assert not result.nodes[0].model_expanded


def test_second_node_expands_model(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        5.0,
        memory,
        model,
        motion_set,
    )

    assert result.nodes[1].place_id == 2
    assert result.nodes[1].model_expanded


def test_third_node_expands_model_again(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        5.0,
        memory,
        model,
        motion_set,
    )

    assert result.nodes[2].place_id == 3
    assert result.nodes[2].model_expanded

    assert model.num_states == 4


def test_deeper_nodes_get_direct_imagined_links(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        5.0,
        memory,
        model,
        motion_set,
    )

    assert result.deep_direct_links_updated == 2


def test_second_layer_link_is_strong(
    motion_set,
    memory,
    model,
    origin_state,
):
    grow_directional_lookahead(
        origin_state,
        0,
        5.0,
        memory,
        model,
        motion_set,
    )

    assert model.transition_likelihood[
        2,
        1,
        0,
    ] > 0.90


def test_obstacle_at_one_meter_allows_only_first_node(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        1.0,
        memory,
        model,
        motion_set,
    )

    assert len(result.nodes) == 1
    assert result.stopped_by_obstacle
    assert result.obstacle_stop_step == 2


def test_obstacle_at_one_point_two_allows_two_nodes(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        1.2,
        memory,
        model,
        motion_set,
    )

    assert len(result.nodes) == 2
    assert result.stopped_by_obstacle
    assert result.obstacle_stop_step == 3


def test_boundary_zero_point_625_is_blocked(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        0.625,
        memory,
        model,
        motion_set,
    )

    assert len(result.nodes) == 0
    assert result.obstacle_stop_step == 1

    assert model.transition_likelihood[
        0,
        0,
        0,
    ] > 0.95


def test_just_above_first_threshold_creates_one_node(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        0.626,
        memory,
        model,
        motion_set,
    )

    assert len(result.nodes) == 1
    assert result.obstacle_stop_step == 2


def test_nan_obstacle_creates_no_nodes(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        math.nan,
        memory,
        model,
        motion_set,
    )

    assert len(result.nodes) == 0
    assert result.unknown_obstacle
    assert not result.stopped_by_obstacle


def test_physical_belief_is_only_padded_by_growth(
    motion_set,
    memory,
    model,
    origin_state,
):
    physical = model.state_belief.copy()

    grow_directional_lookahead(
        origin_state,
        0,
        5.0,
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


def test_custom_depth_two_stops_after_two_nodes(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = grow_directional_lookahead(
        origin_state,
        0,
        5.0,
        memory,
        model,
        motion_set,
        max_steps=2,
    )

    assert len(result.nodes) == 2


def test_stationary_action_is_rejected(
    motion_set,
    memory,
    model,
    origin_state,
):
    with pytest.raises(ValueError):
        grow_directional_lookahead(
            origin_state,
            12,
            5.0,
            memory,
            model,
            motion_set,
        )


@pytest.mark.parametrize(
    "invalid_depth",
    [
        0,
        -1,
    ],
)
def test_invalid_depth_is_rejected(
    invalid_depth,
    motion_set,
    memory,
    model,
    origin_state,
):
    with pytest.raises(ValueError):
        grow_directional_lookahead(
            origin_state,
            0,
            5.0,
            memory,
            model,
            motion_set,
            max_steps=invalid_depth,
        )


def test_boolean_depth_is_rejected(
    motion_set,
    memory,
    model,
    origin_state,
):
    with pytest.raises(TypeError):
        grow_directional_lookahead(
            origin_state,
            0,
            5.0,
            memory,
            model,
            motion_set,
            max_steps=True,
        )
