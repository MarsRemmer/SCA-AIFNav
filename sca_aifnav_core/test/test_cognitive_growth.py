"""Tests for baseline-compatible cognitive-map growth."""

import numpy as np
import pytest

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.cognitive_growth import (
    create_hypothetical_state,
    node_step_distance,
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


def test_first_node_step_distance_matches_baseline():
    assert node_step_distance(
        influence_radius=0.5,
        robot_dimension=0.25,
        state_step=1,
    ) == pytest.approx(0.625)


def test_second_node_step_distance_matches_baseline():
    assert node_step_distance(
        influence_radius=0.5,
        robot_dimension=0.25,
        state_step=2,
    ) == pytest.approx(1.125)


def test_first_action_zero_creates_place_one(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = create_hypothetical_state(
        state=origin_state,
        action_id=0,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    assert result.place_created
    assert result.place_id == 1
    assert len(memory) == 2


def test_first_ghost_uses_baseline_rounded_position(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = create_hypothetical_state(
        state=origin_state,
        action_id=0,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    assert result.position == Point2D(
        0.60,
        0.16,
    )

    assert result.requested_distance == pytest.approx(
        0.625
    )


def test_first_ghost_uses_reserved_state_one(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = create_hypothetical_state(
        state=origin_state,
        action_id=0,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    assert not result.model_expanded
    assert model.num_states == 2

    np.testing.assert_allclose(
        result.posterior,
        np.array([0.0, 1.0]),
    )


def test_ghost_inference_does_not_replace_physical_belief(
    motion_set,
    memory,
    model,
    origin_state,
):
    original_belief = model.state_belief.copy()

    create_hypothetical_state(
        state=origin_state,
        action_id=0,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    np.testing.assert_allclose(
        model.state_belief,
        original_belief,
    )


def test_adjacent_action_can_reuse_place_one(
    motion_set,
    memory,
    model,
    origin_state,
):
    first = create_hypothetical_state(
        state=origin_state,
        action_id=0,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    second = create_hypothetical_state(
        state=origin_state,
        action_id=1,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    assert first.place_id == 1
    assert second.place_id == 1

    assert not second.place_created
    assert not second.model_expanded

    assert len(memory) == 2


def test_action_two_creates_next_distinct_place(
    motion_set,
    memory,
    model,
    origin_state,
):
    create_hypothetical_state(
        state=origin_state,
        action_id=0,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    create_hypothetical_state(
        state=origin_state,
        action_id=1,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    result = create_hypothetical_state(
        state=origin_state,
        action_id=2,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    assert result.place_created
    assert result.place_id == 2
    assert result.position == Point2D(
        0.16,
        0.60,
    )


def test_place_two_expands_model_to_three_states(
    motion_set,
    memory,
    model,
    origin_state,
):
    create_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    result = create_hypothetical_state(
        origin_state,
        2,
        memory,
        model,
        motion_set,
    )

    assert result.model_expanded
    assert model.num_states == 3

    assert model.transition_likelihood.shape == (
        3,
        3,
        13,
    )


def test_place_two_posterior_matches_baseline_low_weight_behavior(
    motion_set,
    memory,
    model,
    origin_state,
):
    create_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    result = create_hypothetical_state(
        origin_state,
        2,
        memory,
        model,
        motion_set,
    )

    expected = np.array(
        [
            0.001,
            0.001,
            1.0,
        ]
    )
    expected /= expected.sum()

    np.testing.assert_allclose(
        result.posterior,
        expected,
    )


def test_model_belief_only_extends_when_model_grows(
    motion_set,
    memory,
    model,
    origin_state,
):
    original = model.state_belief.copy()

    create_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    create_hypothetical_state(
        origin_state,
        2,
        memory,
        model,
        motion_set,
    )

    np.testing.assert_allclose(
        model.state_belief,
        np.append(original, 0.0),
    )


def test_stationary_action_cannot_create_ghost(
    motion_set,
    memory,
    model,
    origin_state,
):
    with pytest.raises(ValueError):
        create_hypothetical_state(
            origin_state,
            12,
            memory,
            model,
            motion_set,
        )


@pytest.mark.parametrize(
    "invalid_step",
    [
        0,
        -1,
    ],
)
def test_invalid_state_step_is_rejected(
    invalid_step,
):
    with pytest.raises(ValueError):
        node_step_distance(
            influence_radius=0.5,
            robot_dimension=0.25,
            state_step=invalid_step,
        )


@pytest.mark.parametrize(
    "invalid_step",
    [
        True,
        1.5,
    ],
)
def test_non_integer_state_step_is_rejected(
    invalid_step,
):
    with pytest.raises(TypeError):
        node_step_distance(
            influence_radius=0.5,
            robot_dimension=0.25,
            state_step=invalid_step,
        )
