"""Tests for baseline-compatible belief history."""

import numpy as np
import pytest

from sca_aifnav_core.belief_history import (
    BeliefHistory,
)


def test_history_starts_with_initial_belief():
    history = BeliefHistory(
        np.array([0.8, 0.2])
    )

    assert len(history) == 1

    np.testing.assert_allclose(
        history.latest,
        np.array([0.8, 0.2]),
    )


def test_record_appends_belief():
    history = BeliefHistory(
        np.array([1.0, 0.0])
    )

    history.record(
        np.array([0.2, 0.8])
    )

    assert len(history) == 2

    np.testing.assert_allclose(
        history.latest,
        np.array([0.2, 0.8]),
    )


def test_history_returns_copies():
    history = BeliefHistory(
        np.array([1.0, 0.0])
    )

    latest = history.latest
    latest[0] = 0.0

    np.testing.assert_allclose(
        history.latest,
        np.array([1.0, 0.0]),
    )


def test_history_pads_after_model_growth():
    history = BeliefHistory(
        np.array([0.8, 0.2])
    )

    history.record(
        np.array([0.3, 0.7])
    )

    history.align_to_states(4)

    for belief in history.entries():
        assert belief.shape == (4,)
        np.testing.assert_allclose(
            belief[2:],
            np.zeros(2),
        )


def test_history_cannot_shrink():
    history = BeliefHistory(
        np.array([0.5, 0.5])
    )

    with pytest.raises(ValueError):
        history.align_to_states(1)


def test_non_vector_belief_is_rejected():
    with pytest.raises(ValueError):
        BeliefHistory(
            np.array(
                [
                    [1.0, 0.0],
                ]
            )
        )
