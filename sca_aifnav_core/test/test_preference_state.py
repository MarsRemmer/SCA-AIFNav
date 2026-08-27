"""Tests for baseline-compatible observation and state preferences."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.preference_state import (
    NO_PREFERENCE,
    PREFERRED_STATE_THRESHOLD,
    BaselinePreferenceState,
)


def test_preference_constants_match_baseline():
    assert NO_PREFERENCE == -1

    assert PREFERRED_STATE_THRESHOLD == pytest.approx(
        0.45
    )


def test_preferences_start_at_zero():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    np.testing.assert_allclose(
        preference.sensory,
        np.zeros(2),
    )

    np.testing.assert_allclose(
        preference.place,
        np.zeros(2),
    )

    np.testing.assert_allclose(
        preference.preferred_states,
        np.zeros(2),
    )


def test_sensory_preference_is_weighted_one_hot():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        sensory_observation=0,
        preference_weight=3.0,
    )

    np.testing.assert_allclose(
        preference.sensory,
        np.array([3.0, 0.0]),
    )


def test_negative_id_means_no_preference():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        sensory_observation=-1,
        place_observation=-1,
    )

    np.testing.assert_allclose(
        preference.sensory,
        np.zeros(2),
    )

    np.testing.assert_allclose(
        preference.place,
        np.zeros(2),
    )


def test_initial_obs_zero_prefers_state_zero():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        sensory_observation=0,
    )

    np.testing.assert_allclose(
        preference.preferred_states,
        np.array([1.0, 0.0]),
    )


def test_unknown_sensory_goal_has_no_preferred_state():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        sensory_observation=1,
    )

    np.testing.assert_allclose(
        preference.preferred_states,
        np.zeros(2),
    )


def test_known_place_one_prefers_state_one():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        place_observation=1,
    )

    np.testing.assert_allclose(
        preference.preferred_states,
        np.array([0.0, 1.0]),
    )


def test_disagreeing_modalities_create_tied_states():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        sensory_observation=0,
        place_observation=1,
    )

    np.testing.assert_allclose(
        preference.preferred_states,
        np.array([1.0, 1.0]),
    )


def test_setting_preference_rebuilds_pA_from_likelihood():
    model = BaselineGenerativeModel()

    model.sensory_concentration[
        0,
        0,
    ] = 9.0

    model.place_concentration[
        0,
        0,
    ] = 7.0

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        sensory_observation=0,
    )

    np.testing.assert_allclose(
        model.sensory_concentration,
        model.sensory_likelihood,
    )

    np.testing.assert_allclose(
        model.place_concentration,
        model.place_likelihood,
    )


def test_new_preferred_sensory_observation_expands_A():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        sensory_observation=2,
    )

    assert model.sensory_observations == 3
    assert preference.sensory.shape == (3,)

    assert preference.sensory[2] == pytest.approx(
        1.0
    )


def test_place_preference_expands_C_after_map_growth():
    model = BaselineGenerativeModel()

    model.register_place_observation(1)

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        place_observation=1,
    )

    model.register_place_observation(2)

    changed = preference.sync_dimensions(
        model
    )

    assert changed
    assert preference.place.shape == (3,)
    assert preference.preferred_states.shape == (
        3,
    )

    np.testing.assert_allclose(
        preference.preferred_states,
        np.array(
            [
                0.0,
                1.0,
                0.0,
            ]
        ),
    )


def test_sync_without_growth_does_nothing():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    changed = preference.sync_dimensions(
        model
    )

    assert not changed


def test_clear_removes_preferences():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    preference.update(
        model=model,
        sensory_observation=0,
        preference_weight=2.0,
    )

    preference.clear(
        model
    )

    np.testing.assert_allclose(
        preference.sensory,
        np.zeros(2),
    )

    np.testing.assert_allclose(
        preference.place,
        np.zeros(2),
    )

    np.testing.assert_allclose(
        preference.preferred_states,
        np.zeros(2),
    )


def test_invalid_preference_id_is_rejected():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    with pytest.raises(ValueError):
        preference.update(
            model=model,
            sensory_observation=-2,
        )


def test_boolean_preference_id_is_rejected():
    model = BaselineGenerativeModel()

    preference = BaselinePreferenceState(
        model
    )

    with pytest.raises(TypeError):
        preference.update(
            model=model,
            sensory_observation=True,
        )
