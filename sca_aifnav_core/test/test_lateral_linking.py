"""Tests for baseline-compatible lateral imagined links."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.lateral_linking import (
    LATERAL_DIRECT_RATE,
    LATERAL_REVERSE_RATE,
    apply_lateral_imagined_evidence,
    lateral_action_between,
)
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory
from sca_aifnav_core.transition_learning import (
    MIN_TRANSITION_CONCENTRATION,
)


@pytest.fixture
def motion_set():
    return BaselineMotionSet()


@pytest.fixture
def memory():
    return BaselinePlaceMemory(
        influence_radius=0.5
    )


def grown_model():
    """Return a baseline model expanded to three states."""
    model = BaselineGenerativeModel()

    model.register_place_observation(1)
    model.register_place_observation(2)

    return model


def source_position():
    return Point2D(0.60, 0.16)


def target_position():
    return Point2D(0.16, 0.60)


def source_belief():
    return np.array(
        [0.0, 1.0, 0.0]
    )


def target_belief():
    return np.array(
        [0.0, 0.0, 1.0]
    )


def test_lateral_rates_match_baseline():
    assert LATERAL_DIRECT_RATE == pytest.approx(
        1.0
    )

    assert LATERAL_REVERSE_RATE == pytest.approx(
        1.0
    )


def test_lateral_geometry_selects_action_four(
    motion_set,
):
    action = lateral_action_between(
        source_position(),
        target_position(),
        motion_set,
    )

    assert action == 4


def test_lateral_reverse_action_is_ten(
    motion_set,
    memory,
):
    model = grown_model()

    result = apply_lateral_imagined_evidence(
        model=model,
        source_belief=source_belief(),
        target_belief=target_belief(),
        source_position=source_position(),
        target_position=target_position(),
        memory=memory,
        motion_set=motion_set,
        reachable=True,
    )

    assert result is not None
    assert result.action_id == 4
    assert result.reverse_action_id == 10


def test_positive_lateral_link_adds_one(
    motion_set,
    memory,
):
    model = grown_model()

    apply_lateral_imagined_evidence(
        model=model,
        source_belief=source_belief(),
        target_belief=target_belief(),
        source_position=source_position(),
        target_position=target_position(),
        memory=memory,
        motion_set=motion_set,
        reachable=True,
    )

    assert model.transition_concentration[
        2,
        1,
        4,
    ] == pytest.approx(
        1.05
    )


def test_positive_reverse_lateral_link_adds_one(
    motion_set,
    memory,
):
    model = grown_model()

    apply_lateral_imagined_evidence(
        model=model,
        source_belief=source_belief(),
        target_belief=target_belief(),
        source_position=source_position(),
        target_position=target_position(),
        memory=memory,
        motion_set=motion_set,
        reachable=True,
    )

    assert model.transition_concentration[
        1,
        2,
        10,
    ] == pytest.approx(
        1.05
    )


def test_positive_lateral_link_becomes_dominant(
    motion_set,
    memory,
):
    model = grown_model()

    apply_lateral_imagined_evidence(
        model=model,
        source_belief=source_belief(),
        target_belief=target_belief(),
        source_position=source_position(),
        target_position=target_position(),
        memory=memory,
        motion_set=motion_set,
        reachable=True,
    )

    assert model.transition_likelihood[
        2,
        1,
        4,
    ] > 0.90


def test_negative_lateral_link_hits_baseline_floor(
    motion_set,
    memory,
):
    model = grown_model()

    apply_lateral_imagined_evidence(
        model=model,
        source_belief=source_belief(),
        target_belief=target_belief(),
        source_position=source_position(),
        target_position=target_position(),
        memory=memory,
        motion_set=motion_set,
        reachable=False,
    )

    assert model.transition_concentration[
        2,
        1,
        4,
    ] == pytest.approx(
        MIN_TRANSITION_CONCENTRATION
    )


def test_negative_reverse_lateral_link_hits_floor(
    motion_set,
    memory,
):
    model = grown_model()

    apply_lateral_imagined_evidence(
        model=model,
        source_belief=source_belief(),
        target_belief=target_belief(),
        source_position=source_position(),
        target_position=target_position(),
        memory=memory,
        motion_set=motion_set,
        reachable=False,
    )

    assert model.transition_concentration[
        1,
        2,
        10,
    ] == pytest.approx(
        MIN_TRANSITION_CONCENTRATION
    )


def test_positive_then_negative_returns_to_uniform_weight(
    motion_set,
    memory,
):
    model = grown_model()

    apply_lateral_imagined_evidence(
        model=model,
        source_belief=source_belief(),
        target_belief=target_belief(),
        source_position=source_position(),
        target_position=target_position(),
        memory=memory,
        motion_set=motion_set,
        reachable=True,
    )

    before = model.transition_likelihood[
        2,
        1,
        4,
    ]

    apply_lateral_imagined_evidence(
        model=model,
        source_belief=source_belief(),
        target_belief=target_belief(),
        source_position=source_position(),
        target_position=target_position(),
        memory=memory,
        motion_set=motion_set,
        reachable=False,
    )

    after = model.transition_likelihood[
        2,
        1,
        4,
    ]

    assert before > 0.90
    assert after == pytest.approx(
        1.0 / 3.0
    )


def test_same_position_does_not_create_lateral_link(
    motion_set,
    memory,
):
    model = grown_model()

    before = (
        model.transition_concentration.copy()
    )

    result = apply_lateral_imagined_evidence(
        model=model,
        source_belief=source_belief(),
        target_belief=target_belief(),
        source_position=source_position(),
        target_position=source_position(),
        memory=memory,
        motion_set=motion_set,
        reachable=True,
    )

    assert result is None

    np.testing.assert_allclose(
        model.transition_concentration,
        before,
    )


def test_target_beyond_action_range_is_not_linked(
    motion_set,
    memory,
):
    model = grown_model()

    far_target = Point2D(
        3.0,
        3.0,
    )

    before = (
        model.transition_concentration.copy()
    )

    result = apply_lateral_imagined_evidence(
        model=model,
        source_belief=source_belief(),
        target_belief=target_belief(),
        source_position=source_position(),
        target_position=far_target,
        memory=memory,
        motion_set=motion_set,
        reachable=True,
    )

    assert result is None

    np.testing.assert_allclose(
        model.transition_concentration,
        before,
    )
