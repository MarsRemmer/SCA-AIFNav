"""Tests for the dynamic V5-compatible generative model."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
    INITIAL_UNKNOWN_LIKELIHOOD,
)


def test_initial_model_dimensions():
    model = BaselineGenerativeModel()

    assert model.num_states == 2
    assert model.num_actions == 13

    assert model.sensory_likelihood.shape == (2, 2)
    assert model.place_likelihood.shape == (2, 2)

    assert model.transition_likelihood.shape == (
        2,
        2,
        13,
    )


def test_runtime_sensory_likelihood_matches_v5():
    model = BaselineGenerativeModel()

    np.testing.assert_allclose(
        model.sensory_likelihood,
        np.array(
            [
                [1.0, 0.01],
                [0.0, 0.01],
            ]
        ),
    )


def test_runtime_place_likelihood_matches_v5():
    model = BaselineGenerativeModel()

    np.testing.assert_allclose(
        model.place_likelihood,
        np.array(
            [
                [1.0, 0.01],
                [0.0, 0.01],
            ]
        ),
    )


def test_initial_unknown_likelihood_is_point_zero_one():
    assert INITIAL_UNKNOWN_LIKELIHOOD == pytest.approx(
        0.01
    )


def test_initial_concentrations_remain_one():
    model = BaselineGenerativeModel()

    np.testing.assert_allclose(
        model.sensory_concentration,
        1.0,
    )

    np.testing.assert_allclose(
        model.place_concentration,
        1.0,
    )


def test_initial_state_belief_favors_start_state():
    model = BaselineGenerativeModel()

    expected = np.array(
        [
            1.0,
            0.0001,
        ]
    )
    expected /= expected.sum()

    np.testing.assert_allclose(
        model.state_belief,
        expected,
    )

    assert model.state_belief[0] > 0.999


def test_non_stationary_transition_starts_uniform():
    model = BaselineGenerativeModel()

    np.testing.assert_allclose(
        model.transition_likelihood[:, :, 0],
        0.5,
    )


def test_stationary_action_is_identity():
    model = BaselineGenerativeModel()

    np.testing.assert_allclose(
        model.transition_likelihood[:, :, 12],
        np.eye(2),
    )


def test_transition_concentration_remains_initial_prior():
    model = BaselineGenerativeModel()

    np.testing.assert_allclose(
        model.transition_concentration,
        0.5,
    )


def test_partial_place_inference_identifies_state_one():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)

    prior = model.predicted_state_prior(
        action_id=0,
    )

    posterior = model.infer_state_belief(
        place_observation=1,
        prior=prior,
    )

    np.testing.assert_allclose(
        posterior,
        np.array([0.0, 1.0]),
    )


def test_register_existing_place_does_not_expand_state():
    model = BaselineGenerativeModel()

    expanded = model.register_place_observation(1)

    assert not expanded
    assert model.num_states == 2


def test_existing_place_becomes_deterministic():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)

    np.testing.assert_allclose(
        model.place_likelihood[:, 1],
        np.array([0.0, 1.0]),
    )


def test_new_place_expands_model_to_three_states():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)
    expanded = model.register_place_observation(2)

    assert expanded
    assert model.num_states == 3

    assert model.sensory_likelihood.shape == (
        2,
        3,
    )

    assert model.place_likelihood.shape == (
        3,
        3,
    )

    assert model.transition_likelihood.shape == (
        3,
        3,
        13,
    )


def test_new_place_column_is_deterministic():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)
    model.register_place_observation(2)

    np.testing.assert_allclose(
        model.place_likelihood[:, 2],
        np.array([0.0, 0.0, 1.0]),
    )


def test_new_sensory_state_uses_expansion_weight():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)
    model.register_place_observation(2)

    np.testing.assert_allclose(
        model.sensory_likelihood[:, 2],
        np.array([0.001, 0.001]),
    )


def test_register_place_resets_pA_concentrations():
    model = BaselineGenerativeModel()

    model.sensory_concentration[:] = 7.0
    model.place_concentration[:] = 9.0

    model.register_place_observation(1)

    np.testing.assert_allclose(
        model.sensory_concentration,
        1.0,
    )

    np.testing.assert_allclose(
        model.place_concentration,
        1.0,
    )


def test_transition_tables_expand_with_state():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)
    model.register_place_observation(2)

    assert model.transition_likelihood.shape == (
        3,
        3,
        13,
    )

    assert model.transition_concentration.shape == (
        3,
        3,
        13,
    )


def test_nonstationary_uniform_entries_become_low_weight():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)
    model.register_place_observation(2)

    np.testing.assert_allclose(
        model.transition_likelihood[:, :, 0],
        0.05,
    )


def test_state_belief_is_extended_with_zero():
    model = BaselineGenerativeModel()

    original = model.state_belief.copy()

    model.register_place_observation(1)
    model.register_place_observation(2)

    np.testing.assert_allclose(
        model.state_belief,
        np.append(original, 0.0),
    )


def test_place_two_inference_identifies_new_state():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)
    model.register_place_observation(2)

    prior = model.predicted_state_prior(
        action_id=0,
    )

    posterior = model.infer_state_belief(
        place_observation=2,
        prior=prior,
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
        posterior,
        expected,
    )

    assert posterior[2] > 0.998


def test_stationary_transition_can_be_restored_after_expansion():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)
    model.register_place_observation(2)

    model.enforce_stationary_transition()

    np.testing.assert_allclose(
        model.transition_likelihood[:, :, 12],
        np.eye(3),
    )


def test_register_same_new_place_twice_does_not_expand_twice():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)

    first = model.register_place_observation(2)
    second = model.register_place_observation(2)

    assert first
    assert not second
    assert model.num_states == 3


@pytest.mark.parametrize(
    "invalid_place_id",
    [-1, -10],
)
def test_negative_place_id_is_rejected(
    invalid_place_id,
):
    model = BaselineGenerativeModel()

    with pytest.raises(ValueError):
        model.register_place_observation(
            invalid_place_id
        )


@pytest.mark.parametrize(
    "invalid_place_id",
    [True, 1.5, "2", None],
)
def test_non_integer_place_id_is_rejected(
    invalid_place_id,
):
    model = BaselineGenerativeModel()

    with pytest.raises(TypeError):
        model.register_place_observation(
            invalid_place_id
        )
