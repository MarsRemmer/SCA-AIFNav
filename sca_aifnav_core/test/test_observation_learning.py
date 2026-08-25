"""Tests for baseline-compatible observation learning."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.observation_learning import (
    OBSERVATION_LEARNING_RATE,
    learn_multimodal_observation,
    learn_place_observation,
    learn_sensory_observation,
)


def test_observation_learning_rate_matches_baseline():
    assert OBSERVATION_LEARNING_RATE == pytest.approx(
        5.0
    )


def test_sensory_learning_adds_five_pseudocounts():
    model = BaselineGenerativeModel()

    belief = np.array([0.0, 1.0])

    learn_sensory_observation(
        model=model,
        observation_id=1,
        state_belief=belief,
    )

    assert model.sensory_concentration[
        1,
        1,
    ] == pytest.approx(
        6.0
    )


def test_sensory_learning_normalizes_A():
    model = BaselineGenerativeModel()

    learn_sensory_observation(
        model=model,
        observation_id=1,
        state_belief=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(
        model.sensory_likelihood.sum(axis=0),
        np.ones(model.num_states),
    )


def test_learned_observation_becomes_likely():
    model = BaselineGenerativeModel()

    learn_sensory_observation(
        model=model,
        observation_id=1,
        state_belief=np.array([0.0, 1.0]),
    )

    assert model.sensory_likelihood[
        1,
        1,
    ] == pytest.approx(
        6.0 / 7.0
    )


def test_zero_support_entry_is_not_revived():
    model = BaselineGenerativeModel()

    before = model.sensory_concentration[
        1,
        0,
    ]

    learn_sensory_observation(
        model=model,
        observation_id=1,
        state_belief=np.array([1.0, 0.0]),
    )

    assert model.sensory_concentration[
        1,
        0,
    ] == pytest.approx(
        before
    )


def test_place_learning_uses_same_rate():
    model = BaselineGenerativeModel()
    model.register_place_observation(1)

    learn_place_observation(
        model=model,
        place_id=1,
        state_belief=np.array([0.0, 1.0]),
    )

    assert model.place_concentration[
        1,
        1,
    ] == pytest.approx(
        6.0
    )


def test_place_deterministic_zero_support_is_preserved():
    model = BaselineGenerativeModel()
    model.register_place_observation(1)

    learn_place_observation(
        model=model,
        place_id=0,
        state_belief=np.array([0.0, 1.0]),
    )

    assert model.place_concentration[
        0,
        1,
    ] == pytest.approx(
        1.0
    )


def test_multimodal_learning_updates_both_pA_arrays():
    model = BaselineGenerativeModel()
    model.register_place_observation(1)

    learn_multimodal_observation(
        model=model,
        sensory_observation=1,
        place_observation=1,
        state_belief=np.array([0.0, 1.0]),
    )

    assert model.sensory_concentration[
        1,
        1,
    ] == pytest.approx(
        6.0
    )

    assert model.place_concentration[
        1,
        1,
    ] == pytest.approx(
        6.0
    )


def test_learning_does_not_change_B():
    model = BaselineGenerativeModel()

    before_B = (
        model.transition_likelihood.copy()
    )

    learn_sensory_observation(
        model=model,
        observation_id=1,
        state_belief=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(
        model.transition_likelihood,
        before_B,
    )


def test_learning_does_not_replace_state_belief():
    model = BaselineGenerativeModel()

    before = model.state_belief.copy()

    learn_sensory_observation(
        model=model,
        observation_id=1,
        state_belief=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(
        model.state_belief,
        before,
    )


def test_invalid_sensory_observation_is_rejected():
    model = BaselineGenerativeModel()

    with pytest.raises(ValueError):
        learn_sensory_observation(
            model=model,
            observation_id=2,
            state_belief=np.array([1.0, 0.0]),
        )


def test_invalid_belief_shape_is_rejected():
    model = BaselineGenerativeModel()

    with pytest.raises(ValueError):
        learn_sensory_observation(
            model=model,
            observation_id=0,
            state_belief=np.array([1.0]),
        )
