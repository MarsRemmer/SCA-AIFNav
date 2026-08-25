"""Tests for V5-compatible one-step action evaluation."""

import numpy as np
import pytest

from sca_aifnav_core.action_evaluation import (
    evaluate_action,
    expected_utility,
    predict_action_state,
    predict_observations,
    state_information_gain,
)
from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.preference_state import (
    BaselinePreferenceState,
)


def deterministic_model():
    """Return a simple two-state deterministic observation model."""
    model = BaselineGenerativeModel()

    model.sensory_likelihood = np.eye(2)
    model.place_likelihood = np.eye(2)

    return model


def test_action_prediction_uses_B():
    model = BaselineGenerativeModel()

    model.transition_likelihood[
        :,
        :,
        0,
    ] = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    predicted = predict_action_state(
        model=model,
        state_belief=np.array(
            [1.0, 0.0]
        ),
        action_id=0,
    )

    np.testing.assert_allclose(
        predicted,
        np.array([0.0, 1.0]),
    )


def test_expected_observation_uses_A():
    model = deterministic_model()

    observations = predict_observations(
        model=model,
        state_belief=np.array(
            [0.25, 0.75]
        ),
    )

    np.testing.assert_allclose(
        observations.sensory,
        np.array([0.25, 0.75]),
    )

    np.testing.assert_allclose(
        observations.place,
        np.array([0.25, 0.75]),
    )


def test_no_preference_has_zero_utility():
    model = deterministic_model()

    preference = BaselinePreferenceState(
        model
    )

    observations = predict_observations(
        model=model,
        state_belief=np.array(
            [0.0, 1.0]
        ),
    )

    utility = expected_utility(
        expected_observations=observations,
        preferences=preference,
    )

    assert utility == pytest.approx(
        0.0
    )


def test_preferred_place_has_expected_log_utility():
    model = deterministic_model()

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        place_observation=1,
        preference_weight=2.0,
    )

    observations = predict_observations(
        model=model,
        state_belief=np.array(
            [0.0, 1.0]
        ),
    )

    utility = expected_utility(
        expected_observations=observations,
        preferences=preference,
    )

    expected = np.log(
        np.exp(2.0)
        / (
            1.0
            + np.exp(2.0)
        )
    )

    assert utility == pytest.approx(
        expected
    )


def test_nonpreferred_place_has_lower_utility():
    model = deterministic_model()

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        place_observation=1,
        preference_weight=2.0,
    )

    preferred = expected_utility(
        predict_observations(
            model,
            np.array([0.0, 1.0]),
        ),
        preference,
    )

    nonpreferred = expected_utility(
        predict_observations(
            model,
            np.array([1.0, 0.0]),
        ),
        preference,
    )

    assert preferred > nonpreferred


def test_certain_state_has_zero_information_gain():
    model = deterministic_model()

    information_gain = state_information_gain(
        model=model,
        state_belief=np.array(
            [1.0, 0.0]
        ),
    )

    assert information_gain == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_uncertain_state_with_informative_A_has_ln_two_gain():
    model = deterministic_model()

    information_gain = state_information_gain(
        model=model,
        state_belief=np.array(
            [0.5, 0.5]
        ),
    )

    assert information_gain == pytest.approx(
        np.log(2.0),
        abs=1e-6,
    )


def test_ambiguous_A_has_zero_information_gain():
    model = BaselineGenerativeModel()

    ambiguous = np.full(
        (2, 2),
        0.5,
    )

    model.sensory_likelihood = (
        ambiguous.copy()
    )

    model.place_likelihood = (
        ambiguous.copy()
    )

    information_gain = state_information_gain(
        model=model,
        state_belief=np.array(
            [0.5, 0.5]
        ),
    )

    assert information_gain == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_action_score_is_sum_of_enabled_terms():
    model = deterministic_model()

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        place_observation=1,
        preference_weight=2.0,
    )

    result = evaluate_action(
        model=model,
        preferences=preference,
        action_id=0,
        state_belief=np.array(
            [0.5, 0.5]
        ),
    )

    assert result.score == pytest.approx(
        result.expected_utility
        + result.state_information_gain
    )


def test_goal_action_scores_above_wrong_action():
    model = deterministic_model()

    model.transition_likelihood[
        :,
        :,
        0,
    ] = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    model.transition_likelihood[
        :,
        :,
        1,
    ] = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        place_observation=1,
        preference_weight=2.0,
    )

    goal_action = evaluate_action(
        model=model,
        preferences=preference,
        action_id=0,
        state_belief=np.array(
            [1.0, 0.0]
        ),
        use_state_information_gain=False,
    )

    wrong_action = evaluate_action(
        model=model,
        preferences=preference,
        action_id=1,
        state_belief=np.array(
            [1.0, 0.0]
        ),
        use_state_information_gain=False,
    )

    assert goal_action.score > wrong_action.score


def test_epistemic_action_prefers_uncertain_future():
    model = deterministic_model()

    model.transition_likelihood[
        :,
        :,
        0,
    ] = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    model.transition_likelihood[
        :,
        :,
        1,
    ] = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )

    preference = BaselinePreferenceState(
        model
    )

    exploratory = evaluate_action(
        model=model,
        preferences=preference,
        action_id=0,
        state_belief=np.array(
            [1.0, 0.0]
        ),
        use_utility=False,
    )

    certain = evaluate_action(
        model=model,
        preferences=preference,
        action_id=1,
        state_belief=np.array(
            [1.0, 0.0]
        ),
        use_utility=False,
    )

    assert exploratory.score > certain.score

    assert exploratory.score == pytest.approx(
        np.log(2.0),
        abs=1e-6,
    )

    assert certain.score == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_invalid_state_belief_is_rejected():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    with pytest.raises(ValueError):
        evaluate_action(
            model=model,
            preferences=preference,
            action_id=0,
            state_belief=np.array(
                [1.0]
            ),
        )
