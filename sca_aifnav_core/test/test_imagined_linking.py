"""Tests for V5-compatible imagined cognitive links."""

import numpy as np
import pytest

from sca_aifnav_core.baseline_odometry import CognitiveOdomState
from sca_aifnav_core.imagined_linking import (
    IMAGINED_DIRECT_RATE,
    IMAGINED_REVERSE_RATE,
    create_and_link_hypothetical_state,
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


def test_imagined_transition_rates_match_v5():
    assert IMAGINED_DIRECT_RATE == pytest.approx(
        5.0
    )

    assert IMAGINED_REVERSE_RATE == pytest.approx(
        3.0
    )


def test_first_imagined_link_creates_place_one(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = create_and_link_hypothetical_state(
        state=origin_state,
        action_id=0,
        memory=memory,
        model=model,
        motion_set=motion_set,
    )

    assert result.hypothetical.place_id == 1
    assert result.hypothetical.place_created
    assert result.hypothetical.position == Point2D(
        0.60,
        0.16,
    )


def test_first_imagined_link_uses_reserved_state_one(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    assert not result.hypothetical.model_expanded
    assert model.num_states == 2

    np.testing.assert_allclose(
        result.hypothetical.posterior,
        np.array([0.0, 1.0]),
    )


def test_first_imagined_link_reverse_action_is_six(
    motion_set,
    memory,
    model,
    origin_state,
):
    result = create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    assert result.reverse_action_id == 6


def test_imagined_direct_pB_matches_v5_rate(
    motion_set,
    memory,
    model,
    origin_state,
):
    physical = model.state_belief.copy()

    create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    expected = (
        0.5
        + IMAGINED_DIRECT_RATE
        * physical[0]
    )

    assert model.transition_concentration[
        1,
        0,
        0,
    ] == pytest.approx(
        expected
    )


def test_imagined_direct_link_becomes_dominant(
    motion_set,
    memory,
    model,
    origin_state,
):
    create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    assert model.transition_likelihood[
        1,
        0,
        0,
    ] > 0.90


def test_imagined_reverse_pB_matches_v5_rate(
    motion_set,
    memory,
    model,
    origin_state,
):
    physical = model.state_belief.copy()

    create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    expected = (
        0.5
        + IMAGINED_REVERSE_RATE
        * physical[0]
    )

    assert model.transition_concentration[
        0,
        1,
        6,
    ] == pytest.approx(
        expected
    )


def test_imagined_reverse_link_becomes_dominant(
    motion_set,
    memory,
    model,
    origin_state,
):
    create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    assert model.transition_likelihood[
        0,
        1,
        6,
    ] > 0.85


def test_physical_belief_is_not_replaced_by_ghost(
    motion_set,
    memory,
    model,
    origin_state,
):
    physical = model.state_belief.copy()

    create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    np.testing.assert_allclose(
        model.state_belief,
        physical,
    )


def test_previous_belief_is_preserved_in_result(
    motion_set,
    memory,
    model,
    origin_state,
):
    physical = model.state_belief.copy()

    result = create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    np.testing.assert_allclose(
        result.previous_belief,
        physical,
    )


def test_place_two_growth_pads_previous_belief(
    motion_set,
    memory,
    model,
    origin_state,
):
    create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    physical_before_growth = (
        model.state_belief.copy()
    )

    result = create_and_link_hypothetical_state(
        origin_state,
        2,
        memory,
        model,
        motion_set,
    )

    assert model.num_states == 3

    np.testing.assert_allclose(
        result.previous_belief,
        np.append(
            physical_before_growth,
            0.0,
        ),
    )


def test_place_two_imagined_link_targets_new_state(
    motion_set,
    memory,
    model,
    origin_state,
):
    create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    result = create_and_link_hypothetical_state(
        origin_state,
        2,
        memory,
        model,
        motion_set,
    )

    target_state = np.argmax(
        result.hypothetical.posterior
    )

    assert target_state == 2

    assert model.transition_likelihood[
        2,
        0,
        2,
    ] > model.transition_likelihood[
        0,
        0,
        2,
    ]


def test_repeated_imagined_evidence_strengthens_link(
    motion_set,
    memory,
    model,
    origin_state,
):
    create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    first_probability = (
        model.transition_likelihood[
            1,
            0,
            0,
        ]
    )

    create_and_link_hypothetical_state(
        origin_state,
        0,
        memory,
        model,
        motion_set,
    )

    second_probability = (
        model.transition_likelihood[
            1,
            0,
            0,
        ]
    )

    assert second_probability > first_probability


def test_stationary_action_cannot_create_imagined_link(
    motion_set,
    memory,
    model,
    origin_state,
):
    with pytest.raises(ValueError):
        create_and_link_hypothetical_state(
            origin_state,
            12,
            memory,
            model,
            motion_set,
        )
