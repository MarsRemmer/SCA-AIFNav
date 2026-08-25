"""Tests for baseline-compatible transition learning."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.transition_learning import (
    DIRECT_TRANSITION_RATE,
    MIN_TRANSITION_CONCENTRATION,
    REVERSE_TRANSITION_RATE,
    learn_bidirectional_transition,
    learn_transition,
)


def test_baseline_learning_constants():
    assert DIRECT_TRANSITION_RATE == pytest.approx(
        10.0
    )
    assert REVERSE_TRANSITION_RATE == pytest.approx(
        7.0
    )
    assert MIN_TRANSITION_CONCENTRATION == pytest.approx(
        0.005
    )


def test_forward_one_hot_transition_updates_pB():
    model = BaselineGenerativeModel()

    previous = np.array([1.0, 0.0])
    current = np.array([0.0, 1.0])

    learn_transition(
        model=model,
        current_belief=current,
        previous_belief=previous,
        action_id=0,
        learning_rate=10.0,
    )

    expected = np.array(
        [
            [0.5, 0.5],
            [10.5, 0.5],
        ]
    )

    np.testing.assert_allclose(
        model.transition_concentration[
            :,
            :,
            0,
        ],
        expected,
    )


def test_forward_transition_normalizes_to_B():
    model = BaselineGenerativeModel()

    learn_transition(
        model=model,
        current_belief=np.array([0.0, 1.0]),
        previous_belief=np.array([1.0, 0.0]),
        action_id=0,
        learning_rate=10.0,
    )

    expected = np.array(
        [
            [0.5 / 11.0, 0.5],
            [10.5 / 11.0, 0.5],
        ]
    )

    np.testing.assert_allclose(
        model.transition_likelihood[
            :,
            :,
            0,
        ],
        expected,
    )


def test_transition_columns_remain_normalized():
    model = BaselineGenerativeModel()

    learn_transition(
        model=model,
        current_belief=np.array([0.0, 1.0]),
        previous_belief=np.array([1.0, 0.0]),
        action_id=0,
        learning_rate=10.0,
    )

    np.testing.assert_allclose(
        model.transition_likelihood.sum(axis=0),
        np.ones(
            (
                model.num_states,
                model.num_actions,
            )
        ),
    )


def test_other_action_concentrations_are_unchanged():
    model = BaselineGenerativeModel()

    before = model.transition_concentration[
        :,
        :,
        1,
    ].copy()

    learn_transition(
        model=model,
        current_belief=np.array([0.0, 1.0]),
        previous_belief=np.array([1.0, 0.0]),
        action_id=0,
        learning_rate=10.0,
    )

    np.testing.assert_allclose(
        model.transition_concentration[
            :,
            :,
            1,
        ],
        before,
    )


def test_negative_learning_uses_baseline_floor():
    model = BaselineGenerativeModel()

    learn_transition(
        model=model,
        current_belief=np.array([0.0, 1.0]),
        previous_belief=np.array([1.0, 0.0]),
        action_id=0,
        learning_rate=-10.0,
    )

    assert model.transition_concentration[
        1,
        0,
        0,
    ] == pytest.approx(
        MIN_TRANSITION_CONCENTRATION
    )


def test_negative_learning_is_normalized_after_floor():
    model = BaselineGenerativeModel()

    learn_transition(
        model=model,
        current_belief=np.array([0.0, 1.0]),
        previous_belief=np.array([1.0, 0.0]),
        action_id=0,
        learning_rate=-10.0,
    )

    expected_target = (
        MIN_TRANSITION_CONCENTRATION
        / (
            0.5
            + MIN_TRANSITION_CONCENTRATION
        )
    )

    assert model.transition_likelihood[
        1,
        0,
        0,
    ] == pytest.approx(
        expected_target
    )


def test_bidirectional_learning_uses_reverse_action():
    model = BaselineGenerativeModel()
    motion_set = BaselineMotionSet()

    reverse = learn_bidirectional_transition(
        model=model,
        previous_belief=np.array([1.0, 0.0]),
        next_belief=np.array([0.0, 1.0]),
        action_id=0,
        motion_set=motion_set,
    )

    assert reverse == 6


def test_bidirectional_forward_rate_is_ten():
    model = BaselineGenerativeModel()
    motion_set = BaselineMotionSet()

    learn_bidirectional_transition(
        model=model,
        previous_belief=np.array([1.0, 0.0]),
        next_belief=np.array([0.0, 1.0]),
        action_id=0,
        motion_set=motion_set,
    )

    assert model.transition_concentration[
        1,
        0,
        0,
    ] == pytest.approx(
        10.5
    )


def test_bidirectional_reverse_rate_is_seven():
    model = BaselineGenerativeModel()
    motion_set = BaselineMotionSet()

    learn_bidirectional_transition(
        model=model,
        previous_belief=np.array([1.0, 0.0]),
        next_belief=np.array([0.0, 1.0]),
        action_id=0,
        motion_set=motion_set,
    )

    assert model.transition_concentration[
        0,
        1,
        6,
    ] == pytest.approx(
        7.5
    )


def test_reverse_transition_probability_becomes_high():
    model = BaselineGenerativeModel()
    motion_set = BaselineMotionSet()

    learn_bidirectional_transition(
        model=model,
        previous_belief=np.array([1.0, 0.0]),
        next_belief=np.array([0.0, 1.0]),
        action_id=0,
        motion_set=motion_set,
    )

    assert model.transition_likelihood[
        0,
        1,
        6,
    ] == pytest.approx(
        7.5 / 8.0
    )


def test_learning_does_not_change_state_belief():
    model = BaselineGenerativeModel()

    original = model.state_belief.copy()

    learn_transition(
        model=model,
        current_belief=np.array([0.0, 1.0]),
        previous_belief=np.array([1.0, 0.0]),
        action_id=0,
        learning_rate=10.0,
    )

    np.testing.assert_allclose(
        model.state_belief,
        original,
    )


def test_stay_is_rejected_for_bidirectional_learning():
    model = BaselineGenerativeModel()
    motion_set = BaselineMotionSet()

    with pytest.raises(ValueError):
        learn_bidirectional_transition(
            model=model,
            previous_belief=np.array([1.0, 0.0]),
            next_belief=np.array([1.0, 0.0]),
            action_id=12,
            motion_set=motion_set,
        )


def test_invalid_belief_shape_is_rejected():
    model = BaselineGenerativeModel()

    with pytest.raises(ValueError):
        learn_transition(
            model=model,
            current_belief=np.array([1.0]),
            previous_belief=np.array([1.0, 0.0]),
            action_id=0,
            learning_rate=10.0,
        )


def test_invalid_action_is_rejected():
    model = BaselineGenerativeModel()

    with pytest.raises(ValueError):
        learn_transition(
            model=model,
            current_belief=np.array([0.0, 1.0]),
            previous_belief=np.array([1.0, 0.0]),
            action_id=13,
            learning_rate=10.0,
        )
