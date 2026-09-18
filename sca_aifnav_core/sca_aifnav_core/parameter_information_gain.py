"""AIMAPP-compatible parameter information gain."""

import numpy as np


EPSILON = 1e-16


def spm_wnorm(
    concentration: np.ndarray,
) -> np.ndarray:
    """
    Reproduce AIMAPP/pymdp ``spm_wnorm``.

    AIMAPP computes:

        A = A + 1e-16
        norm = 1 / sum(A, axis=0)
        avg = 1 / A
        wA = norm - avg
    """
    values = np.asarray(
        concentration,
        dtype=float,
    )

    values = values + EPSILON

    norm = np.divide(
        1.0,
        np.sum(
            values,
            axis=0,
        ),
    )

    average = np.divide(
        1.0,
        values,
    )

    return norm - average


def pA_information_gain(
    concentrations,
    expected_observations,
    predicted_state: np.ndarray,
) -> float:
    """
    Reproduce AIMAPP ``calc_pA_info_gain`` for one hidden-state factor.

    ``concentrations`` contains one Dirichlet parameter matrix per
    observation modality.
    """
    if len(concentrations) != len(
        expected_observations
    ):
        raise ValueError(
            "concentrations and expected_observations "
            "must contain the same number of modalities"
        )

    state = np.asarray(
        predicted_state,
        dtype=float,
    )

    information_gain = 0.0

    for concentration, observation in zip(
        concentrations,
        expected_observations,
    ):
        concentration = np.asarray(
            concentration,
            dtype=float,
        )

        observation = np.asarray(
            observation,
            dtype=float,
        )

        weighted_parameter = (
            spm_wnorm(
                concentration
            )
            * (
                concentration > 0.0
            ).astype(float)
        )

        expected_parameter = (
            weighted_parameter
            @ state
        )

        information_gain -= float(
            observation.dot(
                expected_parameter[
                    :,
                    np.newaxis,
                ]
            )
        )

    return information_gain


def pB_information_gain(
    transition_concentration: np.ndarray,
    predicted_state: np.ndarray,
    previous_state: np.ndarray,
    action_id: int,
) -> float:
    """Reproduce AIMAPP ``calc_pB_info_gain`` for one factor and action."""
    concentration = np.asarray(
        transition_concentration,
        dtype=float,
    )

    predicted = np.asarray(
        predicted_state,
        dtype=float,
    )

    previous = np.asarray(
        previous_state,
        dtype=float,
    )

    weighted_parameter = (
        spm_wnorm(
            concentration
        )[
            :,
            :,
            action_id,
        ]
        * (
            concentration[
                :,
                :,
                action_id,
            ]
            > 0.0
        ).astype(float)
    )

    information_gain = -float(
        predicted.dot(
            weighted_parameter.dot(
                previous
            )
        )
    )

    return information_gain


def parameter_information_gain(
    observation_concentrations,
    expected_observations,
    transition_concentration: np.ndarray,
    predicted_state: np.ndarray,
    previous_state: np.ndarray,
    action_id: int,
) -> float:
    """
    Reproduce AIMAPP ``infer_param_info_gain`` for one MCTS step.

    AIMAPP sums information gain about pA and pB before the MCTS
    interface applies its separate division by 100.
    """
    observation_gain = pA_information_gain(
        concentrations=(
            observation_concentrations
        ),
        expected_observations=(
            expected_observations
        ),
        predicted_state=predicted_state,
    )

    transition_gain = pB_information_gain(
        transition_concentration=(
            transition_concentration
        ),
        predicted_state=predicted_state,
        previous_state=previous_state,
        action_id=action_id,
    )

    return observation_gain + transition_gain
