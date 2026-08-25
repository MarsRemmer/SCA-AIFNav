"""Tests for V5-compatible real experience updates."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.real_experience import (
    update_real_experience,
)


@pytest.fixture
def motion_set():
    return BaselineMotionSet()


def prepared_model():
    """Create a model with real place 1 available."""
    model = BaselineGenerativeModel()
    model.register_place_observation(1)
    return model


def test_first_real_update_can_skip_transition_history(
    motion_set,
):
    model = prepared_model()

    result = update_real_experience(
        model=model,
        sensory_observation=1,
        place_observation=1,
        action_id=0,
        previous_belief=None,
        motion_set=motion_set,
    )

    assert not result.transition_updated
    assert not result.reverse_transition_updated


def test_preliminary_inference_identifies_place_one(
    motion_set,
):
    model = prepared_model()

    result = update_real_experience(
        model=model,
        sensory_observation=1,
        place_observation=1,
        action_id=0,
        previous_belief=None,
        motion_set=motion_set,
    )

    np.testing.assert_allclose(
        result.preliminary_belief,
        np.array([0.0, 1.0]),
    )


def test_real_transition_uses_rate_ten(
    motion_set,
):
    model = prepared_model()

    previous = np.array([1.0, 0.0])

    update_real_experience(
        model=model,
        sensory_observation=1,
        place_observation=1,
        action_id=0,
        previous_belief=previous,
        motion_set=motion_set,
    )

    assert model.transition_concentration[
        1,
        0,
        0,
    ] == pytest.approx(
        10.5
    )


def test_real_reverse_transition_uses_rate_seven(
    motion_set,
):
    model = prepared_model()

    previous = np.array([1.0, 0.0])

    result = update_real_experience(
        model=model,
        sensory_observation=1,
        place_observation=1,
        action_id=0,
        previous_belief=previous,
        motion_set=motion_set,
    )

    assert result.reverse_transition_updated
    assert result.reverse_action_id == 6

    assert model.transition_concentration[
        0,
        1,
        6,
    ] == pytest.approx(
        7.5
    )


def test_same_most_likely_state_skips_reverse_update(
    motion_set,
):
    model = BaselineGenerativeModel()

    previous = np.array([1.0, 0.0])

    result = update_real_experience(
        model=model,
        sensory_observation=0,
        place_observation=0,
        action_id=0,
        previous_belief=previous,
        motion_set=motion_set,
    )

    assert result.transition_updated
    assert not result.reverse_transition_updated
    assert result.reverse_action_id is None


def test_real_observation_updates_sensory_pA(
    motion_set,
):
    model = prepared_model()

    update_real_experience(
        model=model,
        sensory_observation=1,
        place_observation=1,
        action_id=0,
        previous_belief=np.array([1.0, 0.0]),
        motion_set=motion_set,
    )

    assert model.sensory_concentration[
        1,
        1,
    ] == pytest.approx(
        6.0
    )


def test_real_observation_updates_place_pA(
    motion_set,
):
    model = prepared_model()

    update_real_experience(
        model=model,
        sensory_observation=1,
        place_observation=1,
        action_id=0,
        previous_belief=np.array([1.0, 0.0]),
        motion_set=motion_set,
    )

    assert model.place_concentration[
        1,
        1,
    ] == pytest.approx(
        6.0
    )


def test_final_posterior_is_saved_to_model(
    motion_set,
):
    model = prepared_model()

    result = update_real_experience(
        model=model,
        sensory_observation=1,
        place_observation=1,
        action_id=0,
        previous_belief=np.array([1.0, 0.0]),
        motion_set=motion_set,
    )

    np.testing.assert_allclose(
        model.state_belief,
        result.posterior_belief,
    )


def test_state_growth_pads_previous_belief(
    motion_set,
):
    model = prepared_model()
    model.register_place_observation(2)

    previous = np.array([1.0, 0.0])

    result = update_real_experience(
        model=model,
        sensory_observation=1,
        place_observation=2,
        action_id=2,
        previous_belief=previous,
        motion_set=motion_set,
    )

    assert result.transition_updated
    assert model.num_states == 3

    assert model.transition_concentration[
        2,
        0,
        2,
    ] > 0.05


def test_real_update_changes_B_and_A_together(
    motion_set,
):
    model = prepared_model()

    before_B = (
        model.transition_concentration.copy()
    )

    before_A = (
        model.sensory_concentration.copy()
    )

    update_real_experience(
        model=model,
        sensory_observation=1,
        place_observation=1,
        action_id=0,
        previous_belief=np.array([1.0, 0.0]),
        motion_set=motion_set,
    )

    assert not np.allclose(
        model.transition_concentration,
        before_B,
    )

    assert not np.allclose(
        model.sensory_concentration,
        before_A,
    )


def test_invalid_previous_belief_is_rejected(
    motion_set,
):
    model = prepared_model()

    with pytest.raises(ValueError):
        update_real_experience(
            model=model,
            sensory_observation=1,
            place_observation=1,
            action_id=0,
            previous_belief=np.array(
                [1.0, 0.0, 0.0]
            ),
            motion_set=motion_set,
        )
