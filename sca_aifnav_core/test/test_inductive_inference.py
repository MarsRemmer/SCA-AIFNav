"""Tests for backward propagation of state preferences."""

import numpy as np
import pytest

from sca_aifnav_core.inductive_inference import (
    DEFAULT_INDUCTIVE_HORIZON,
    MIN_CERTAINTY_THRESHOLD,
    inductive_bonus,
    inductive_preference,
)


def transition_model(
    num_states,
    num_actions=2,
):
    matrix = np.zeros(
        (
            num_states,
            num_states,
            num_actions,
        ),
        dtype=float,
    )

    for state in range(
        num_states
    ):
        matrix[
            state,
            state,
            :,
        ] = 0.1

    return matrix


def one_hot(
    size,
    index,
):
    result = np.zeros(
        size,
        dtype=float,
    )

    result[index] = 1.0

    return result


def test_constants_match_baseline():
    assert MIN_CERTAINTY_THRESHOLD == pytest.approx(
        0.15
    )

    assert DEFAULT_INDUCTIVE_HORIZON == 4


def test_no_reachable_preference_returns_zero():
    B = transition_model(
        3
    )

    current = one_hot(
        3,
        0,
    )

    predicted = one_hot(
        3,
        1,
    )

    preferred = one_hot(
        3,
        2,
    )

    result = inductive_preference(
        current_belief=current,
        predicted_belief=predicted,
        transition_likelihood=B,
        preferred_states=preferred,
    )

    assert result == pytest.approx(
        0.0
    )


def test_one_step_goal_path_rewards_goal_prediction():
    B = transition_model(
        2
    )

    B[
        1,
        0,
        0,
    ] = 0.9

    B[
        1,
        0,
        1,
    ] = 0.8

    current = one_hot(
        2,
        0,
    )

    predicted = one_hot(
        2,
        1,
    )

    preferred = one_hot(
        2,
        1,
    )

    result = inductive_preference(
        current_belief=current,
        predicted_belief=predicted,
        transition_likelihood=B,
        preferred_states=preferred,
    )

    assert result < 0.0


def test_inductive_bonus_is_negative_of_preference():
    B = transition_model(
        2
    )

    B[
        1,
        0,
        0,
    ] = 0.9
    B[
        1,
        0,
        1,
    ] = 0.8

    current = one_hot(
        2,
        0,
    )

    predicted = one_hot(
        2,
        1,
    )

    preferred = one_hot(
        2,
        1,
    )

    preference = inductive_preference(
        current_belief=current,
        predicted_belief=predicted,
        transition_likelihood=B,
        preferred_states=preferred,
    )

    bonus = inductive_bonus(
        current_belief=current,
        predicted_belief=predicted,
        transition_likelihood=B,
        preferred_states=preferred,
    )

    assert bonus == pytest.approx(
        -preference
    )

    assert bonus > 0.0


def test_two_step_path_prefers_intermediate_state():
    B = transition_model(
        3
    )

    B[
        1,
        0,
        0,
    ] = 0.9

    B[
        2,
        1,
        0,
    ] = 0.9

    current = one_hot(
        3,
        0,
    )

    toward_goal = one_hot(
        3,
        1,
    )

    away_from_goal = one_hot(
        3,
        0,
    )

    preferred = one_hot(
        3,
        2,
    )

    toward_bonus = inductive_bonus(
        current_belief=current,
        predicted_belief=toward_goal,
        transition_likelihood=B,
        preferred_states=preferred,
    )

    away_bonus = inductive_bonus(
        current_belief=current,
        predicted_belief=away_from_goal,
        transition_likelihood=B,
        preferred_states=preferred,
    )

    assert toward_bonus > away_bonus


def test_path_outside_horizon_has_no_influence():
    B = transition_model(
        4
    )

    B[
        1,
        0,
        0,
    ] = 0.9

    B[
        2,
        1,
        0,
    ] = 0.9

    B[
        3,
        2,
        0,
    ] = 0.9

    current = one_hot(
        4,
        0,
    )

    predicted = one_hot(
        4,
        1,
    )

    preferred = one_hot(
        4,
        3,
    )

    result = inductive_bonus(
        current_belief=current,
        predicted_belief=predicted,
        transition_likelihood=B,
        preferred_states=preferred,
        lookahead_horizon=2,
    )

    assert result == pytest.approx(
        0.0
    )


def test_horizon_three_can_reach_three_step_goal():
    B = transition_model(
        4
    )

    B[
        1,
        0,
        0,
    ] = 0.9

    B[
        2,
        1,
        0,
    ] = 0.9

    B[
        3,
        2,
        0,
    ] = 0.9

    current = one_hot(
        4,
        0,
    )

    predicted = one_hot(
        4,
        1,
    )

    preferred = one_hot(
        4,
        3,
    )

    result = inductive_bonus(
        current_belief=current,
        predicted_belief=predicted,
        transition_likelihood=B,
        preferred_states=preferred,
        lookahead_horizon=3,
    )

    assert result > 0.0


def test_uncertain_transition_below_threshold_is_ignored():
    B = transition_model(
        2
    )

    B[
        1,
        0,
        0,
    ] = 0.14

    current = one_hot(
        2,
        0,
    )

    predicted = one_hot(
        2,
        1,
    )

    preferred = one_hot(
        2,
        1,
    )

    result = inductive_bonus(
        current_belief=current,
        predicted_belief=predicted,
        transition_likelihood=B,
        preferred_states=preferred,
    )

    assert result == pytest.approx(
        0.0
    )


def test_soft_predicted_belief_scales_bonus():
    B = transition_model(
        2
    )

    B[
        1,
        0,
        0,
    ] = 0.9
    B[
        1,
        0,
        1,
    ] = 0.8

    current = one_hot(
        2,
        0,
    )

    preferred = one_hot(
        2,
        1,
    )

    certain_prediction = np.array(
        [0.0, 1.0]
    )

    mixed_prediction = np.array(
        [0.5, 0.5]
    )

    certain_bonus = inductive_bonus(
        current_belief=current,
        predicted_belief=(
            certain_prediction
        ),
        transition_likelihood=B,
        preferred_states=preferred,
    )

    mixed_bonus = inductive_bonus(
        current_belief=current,
        predicted_belief=(
            mixed_prediction
        ),
        transition_likelihood=B,
        preferred_states=preferred,
    )

    assert certain_bonus > mixed_bonus


def test_input_arrays_are_not_modified():
    B = transition_model(
        2
    )

    B[
        1,
        0,
        0,
    ] = 0.9

    current = one_hot(
        2,
        0,
    )

    predicted = one_hot(
        2,
        1,
    )

    preferred = one_hot(
        2,
        1,
    )

    B_before = B.copy()
    current_before = current.copy()
    predicted_before = predicted.copy()
    preferred_before = preferred.copy()

    inductive_bonus(
        current_belief=current,
        predicted_belief=predicted,
        transition_likelihood=B,
        preferred_states=preferred,
    )

    np.testing.assert_allclose(
        B,
        B_before,
    )

    np.testing.assert_allclose(
        current,
        current_before,
    )

    np.testing.assert_allclose(
        predicted,
        predicted_before,
    )

    np.testing.assert_allclose(
        preferred,
        preferred_before,
    )


def test_invalid_transition_shape_is_rejected():
    with pytest.raises(
        ValueError
    ):
        inductive_bonus(
            current_belief=np.array(
                [1.0, 0.0]
            ),
            predicted_belief=np.array(
                [0.0, 1.0]
            ),
            transition_likelihood=np.eye(
                2
            ),
            preferred_states=np.array(
                [0.0, 1.0]
            ),
        )


def test_single_above_median_transition_equal_to_threshold_is_excluded():
    B = transition_model(
        2
    )

    # With this small synthetic model, 0.9 is the only transition above
    # the median. Its value therefore also becomes the certainty threshold.
    B[
        1,
        0,
        0,
    ] = 0.9

    current = one_hot(
        2,
        0,
    )

    predicted = one_hot(
        2,
        1,
    )

    preferred = one_hot(
        2,
        1,
    )

    result = inductive_bonus(
        current_belief=current,
        predicted_belief=predicted,
        transition_likelihood=B,
        preferred_states=preferred,
    )

    # Certain transitions use a strict ">" threshold comparison.
    assert result == pytest.approx(
        0.0
    )
