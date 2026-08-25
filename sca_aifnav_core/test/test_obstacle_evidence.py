"""Tests for V5-compatible obstacle transition evidence."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.imagined_linking import (
    IMAGINED_DIRECT_RATE,
    IMAGINED_REVERSE_RATE,
)
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.obstacle_evidence import (
    BLOCKED_SELF_LOOP_RATE,
    UNREACHABLE_DIRECT_RATE,
    UNREACHABLE_REVERSE_RATE,
    apply_direct_imagined_evidence,
    discourage_known_unreachable_link,
    reinforce_blocked_self_loop,
)
from sca_aifnav_core.transition_learning import (
    MIN_TRANSITION_CONCENTRATION,
    learn_bidirectional_transition,
)


@pytest.fixture
def model():
    return BaselineGenerativeModel()


@pytest.fixture
def motion_set():
    return BaselineMotionSet()


def test_obstacle_learning_constants_match_v5():
    assert BLOCKED_SELF_LOOP_RATE == pytest.approx(
        10.0
    )

    assert UNREACHABLE_DIRECT_RATE == pytest.approx(
        -10.0
    )

    assert UNREACHABLE_REVERSE_RATE == pytest.approx(
        -7.0
    )


def test_blocked_action_reinforces_self_loop(
    model,
    motion_set,
):
    belief = np.array([1.0, 0.0])

    reinforce_blocked_self_loop(
        model=model,
        belief=belief,
        action_id=0,
        motion_set=motion_set,
    )

    assert model.transition_concentration[
        0,
        0,
        0,
    ] == pytest.approx(
        10.5
    )


def test_blocked_self_loop_becomes_dominant(
    model,
    motion_set,
):
    belief = np.array([1.0, 0.0])

    reinforce_blocked_self_loop(
        model,
        belief,
        0,
        motion_set,
    )

    assert model.transition_likelihood[
        0,
        0,
        0,
    ] == pytest.approx(
        10.5 / 11.0
    )


def test_blocked_action_does_not_strengthen_other_state(
    model,
    motion_set,
):
    belief = np.array([1.0, 0.0])

    reinforce_blocked_self_loop(
        model,
        belief,
        0,
        motion_set,
    )

    assert model.transition_likelihood[
        1,
        0,
        0,
    ] == pytest.approx(
        0.5 / 11.0
    )


def test_known_unreachable_direct_link_is_reduced(
    model,
    motion_set,
):
    state0 = np.array([1.0, 0.0])
    state1 = np.array([0.0, 1.0])

    discourage_known_unreachable_link(
        model=model,
        current_belief=state0,
        unreachable_belief=state1,
        action_id=0,
        motion_set=motion_set,
    )

    assert model.transition_concentration[
        1,
        0,
        0,
    ] == pytest.approx(
        MIN_TRANSITION_CONCENTRATION
    )


def test_known_unreachable_reverse_link_is_reduced(
    model,
    motion_set,
):
    state0 = np.array([1.0, 0.0])
    state1 = np.array([0.0, 1.0])

    reverse = discourage_known_unreachable_link(
        model=model,
        current_belief=state0,
        unreachable_belief=state1,
        action_id=0,
        motion_set=motion_set,
    )

    assert reverse == 6

    assert model.transition_concentration[
        0,
        1,
        6,
    ] == pytest.approx(
        MIN_TRANSITION_CONCENTRATION
    )


def test_negative_real_evidence_can_undo_positive_real_evidence(
    model,
    motion_set,
):
    state0 = np.array([1.0, 0.0])
    state1 = np.array([0.0, 1.0])

    learn_bidirectional_transition(
        model=model,
        previous_belief=state0,
        next_belief=state1,
        action_id=0,
        motion_set=motion_set,
        direct_rate=10.0,
        reverse_rate=7.0,
    )

    before = model.transition_likelihood[
        1,
        0,
        0,
    ]

    discourage_known_unreachable_link(
        model=model,
        current_belief=state0,
        unreachable_belief=state1,
        action_id=0,
        motion_set=motion_set,
    )

    after = model.transition_likelihood[
        1,
        0,
        0,
    ]

    assert before > 0.95
    assert after == pytest.approx(0.5)
    assert after < before


def test_reachable_imagined_edge_uses_positive_five(
    model,
    motion_set,
):
    state0 = np.array([1.0, 0.0])
    state1 = np.array([0.0, 1.0])

    apply_direct_imagined_evidence(
        model=model,
        previous_belief=state0,
        next_belief=state1,
        action_id=0,
        motion_set=motion_set,
        reachable=True,
    )

    assert model.transition_concentration[
        1,
        0,
        0,
    ] == pytest.approx(
        0.5 + IMAGINED_DIRECT_RATE
    )


def test_reachable_imagined_reverse_uses_positive_three(
    model,
    motion_set,
):
    state0 = np.array([1.0, 0.0])
    state1 = np.array([0.0, 1.0])

    apply_direct_imagined_evidence(
        model=model,
        previous_belief=state0,
        next_belief=state1,
        action_id=0,
        motion_set=motion_set,
        reachable=True,
    )

    assert model.transition_concentration[
        0,
        1,
        6,
    ] == pytest.approx(
        0.5 + IMAGINED_REVERSE_RATE
    )


def test_blocked_imagined_edge_uses_negative_five(
    model,
    motion_set,
):
    state0 = np.array([1.0, 0.0])
    state1 = np.array([0.0, 1.0])

    apply_direct_imagined_evidence(
        model=model,
        previous_belief=state0,
        next_belief=state1,
        action_id=0,
        motion_set=motion_set,
        reachable=False,
    )

    assert model.transition_concentration[
        1,
        0,
        0,
    ] == pytest.approx(
        MIN_TRANSITION_CONCENTRATION
    )


def test_blocked_imagined_reverse_uses_negative_three(
    model,
    motion_set,
):
    state0 = np.array([1.0, 0.0])
    state1 = np.array([0.0, 1.0])

    apply_direct_imagined_evidence(
        model=model,
        previous_belief=state0,
        next_belief=state1,
        action_id=0,
        motion_set=motion_set,
        reachable=False,
    )

    assert model.transition_concentration[
        0,
        1,
        6,
    ] == pytest.approx(
        MIN_TRANSITION_CONCENTRATION
    )


def test_positive_imagined_link_can_be_weakened_later(
    model,
    motion_set,
):
    state0 = np.array([1.0, 0.0])
    state1 = np.array([0.0, 1.0])

    apply_direct_imagined_evidence(
        model,
        state0,
        state1,
        0,
        motion_set,
        True,
    )

    before = model.transition_likelihood[
        1,
        0,
        0,
    ]

    apply_direct_imagined_evidence(
        model,
        state0,
        state1,
        0,
        motion_set,
        False,
    )

    after = model.transition_likelihood[
        1,
        0,
        0,
    ]

    assert before > 0.90
    assert after == pytest.approx(0.5)
    assert after < before


def test_stay_is_rejected_for_obstacle_evidence(
    model,
    motion_set,
):
    with pytest.raises(ValueError):
        reinforce_blocked_self_loop(
            model=model,
            belief=np.array([1.0, 0.0]),
            action_id=12,
            motion_set=motion_set,
        )
