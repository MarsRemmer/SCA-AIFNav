"""Tests for AIMAPP-compatible parameter information gain."""

import numpy as np
import pytest

from sca_aifnav_core.parameter_information_gain import (
    pA_information_gain,
    pB_information_gain,
    parameter_information_gain,
    spm_wnorm,
)


def test_spm_wnorm_matches_aimapp():
    """Match AIMAPP pymdp.maths.spm_wnorm exactly."""
    concentration = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    actual = spm_wnorm(
        concentration
    )

    expected = np.array(
        [
            [-0.75, -1.0 / 3.0],
            [-1.0 / 12.0, -1.0 / 12.0],
        ]
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-15,
    )


def test_pA_information_gain_matches_aimapp_numeric_reference():
    """Match AIMAPP calc_pA_info_gain for two observation modalities."""
    sensory_concentration = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    place_concentration = np.array(
        [
            [2.0, 1.0],
            [1.0, 3.0],
        ]
    )

    predicted_state = np.array(
        [0.6, 0.4]
    )

    expected_sensory = np.array(
        [0.7, 0.3]
    )

    expected_place = np.array(
        [0.25, 0.75]
    )

    actual = pA_information_gain(
        concentrations=(
            sensory_concentration,
            place_concentration,
        ),
        expected_observations=(
            expected_sensory,
            expected_place,
        ),
        predicted_state=predicted_state,
    )

    assert actual == pytest.approx(
        0.8583333333333334,
        rel=0.0,
        abs=1e-15,
    )


def test_pA_information_gain_preserves_aimapp_support_mask():
    """Preserve AIMAPP zero-parameter support masking."""
    concentration = np.eye(2)

    actual = pA_information_gain(
        concentrations=(
            concentration,
        ),
        expected_observations=(
            np.array([0.5, 0.5]),
        ),
        predicted_state=np.array(
            [0.5, 0.5]
        ),
    )

    assert actual == pytest.approx(
        0.0,
        abs=1e-15,
    )


def test_pB_information_gain_matches_aimapp_numeric_reference():
    """Match AIMAPP calc_pB_info_gain for one action."""
    transition_concentration = np.zeros(
        (2, 2, 2),
        dtype=float,
    )

    transition_concentration[
        :,
        :,
        0,
    ] = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    transition_concentration[
        :,
        :,
        1,
    ] = np.array(
        [
            [2.0, 5.0],
            [4.0, 1.0],
        ]
    )

    previous_state = np.array(
        [0.55, 0.45]
    )

    predicted_state = np.array(
        [0.6, 0.4]
    )

    actual = pB_information_gain(
        transition_concentration=(
            transition_concentration
        ),
        predicted_state=predicted_state,
        previous_state=previous_state,
        action_id=1,
    )

    assert actual == pytest.approx(
        0.2873333333333334,
        rel=0.0,
        abs=1e-15,
    )


def test_total_parameter_information_gain_matches_aimapp_sum():
    """AIMAPP infer_param_info_gain adds pA and pB information gain."""
    sensory_concentration = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    place_concentration = np.array(
        [
            [2.0, 1.0],
            [1.0, 3.0],
        ]
    )

    transition_concentration = np.zeros(
        (2, 2, 2),
        dtype=float,
    )

    transition_concentration[
        :,
        :,
        1,
    ] = np.array(
        [
            [2.0, 5.0],
            [4.0, 1.0],
        ]
    )

    predicted_state = np.array(
        [0.6, 0.4]
    )

    previous_state = np.array(
        [0.55, 0.45]
    )

    actual = parameter_information_gain(
        observation_concentrations=(
            sensory_concentration,
            place_concentration,
        ),
        expected_observations=(
            np.array([0.7, 0.3]),
            np.array([0.25, 0.75]),
        ),
        transition_concentration=(
            transition_concentration
        ),
        predicted_state=predicted_state,
        previous_state=previous_state,
        action_id=1,
    )

    assert actual == pytest.approx(
        1.1456666666666668,
        rel=0.0,
        abs=1e-15,
    )
