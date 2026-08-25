"""Tests for baseline-compatible probability tables."""

import numpy as np
import pytest

from sca_aifnav_core.probability_tables import (
    UNKNOWN_LIKELIHOOD_WEIGHT,
    UNEXPLORED_TRANSITION_WEIGHT,
    create_likelihood_table,
    create_transition_table,
    expand_likelihood_table,
    expand_transition_table,
)


def test_initial_likelihood_is_uniform():
    table = create_likelihood_table(
        num_observations=2,
        num_states=2,
    )

    assert table.shape == (2, 2)

    np.testing.assert_allclose(
        table,
        np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        ),
    )


def test_initial_transition_is_uniform():
    table = create_transition_table(
        num_states=2,
        num_actions=13,
    )

    assert table.shape == (2, 2, 13)

    np.testing.assert_allclose(
        table,
        0.5,
    )


def test_transition_columns_initially_sum_to_one():
    table = create_transition_table(
        num_states=3,
        num_actions=13,
    )

    np.testing.assert_allclose(
        table.sum(axis=0),
        np.ones((3, 13)),
    )


def test_likelihood_columns_initially_sum_to_one():
    table = create_likelihood_table(
        num_observations=4,
        num_states=3,
    )

    np.testing.assert_allclose(
        table.sum(axis=0),
        np.ones(3),
    )


def test_likelihood_state_expansion_preserves_old_values():
    table = np.array(
        [
            [0.9, 0.2],
            [0.1, 0.8],
        ],
        dtype=float,
    )

    expanded = expand_likelihood_table(
        table,
        add_states=1,
    )

    assert expanded.shape == (2, 3)

    np.testing.assert_allclose(
        expanded[:, :2],
        table,
    )


def test_new_likelihood_state_uses_baseline_unknown_weight():
    table = create_likelihood_table(
        num_observations=2,
        num_states=2,
    )

    expanded = expand_likelihood_table(
        table,
        add_states=1,
        null_probability=True,
    )

    np.testing.assert_allclose(
        expanded[:, 2],
        UNKNOWN_LIKELIHOOD_WEIGHT,
    )

    assert UNKNOWN_LIKELIHOOD_WEIGHT == pytest.approx(
        0.001
    )


def test_likelihood_observation_and_state_can_expand_together():
    table = create_likelihood_table(
        num_observations=2,
        num_states=2,
    )

    expanded = expand_likelihood_table(
        table,
        add_observations=1,
        add_states=1,
    )

    assert expanded.shape == (3, 3)

    np.testing.assert_allclose(
        expanded[:2, :2],
        table,
    )

    assert expanded[2, 2] == pytest.approx(
        0.001
    )


def test_transition_expansion_preserves_nonuniform_old_values():
    table = create_transition_table(
        num_states=2,
        num_actions=13,
    )

    table[0, 0, 0] = 0.9
    table[1, 0, 0] = 0.1

    expanded = expand_transition_table(
        table,
        add_states=1,
        alter_weights=True,
    )

    assert expanded.shape == (3, 3, 13)

    assert expanded[0, 0, 0] == pytest.approx(
        0.9
    )
    assert expanded[1, 0, 0] == pytest.approx(
        0.1
    )


def test_uniform_old_transition_entries_become_baseline_low_weight():
    table = create_transition_table(
        num_states=2,
        num_actions=13,
    )

    expanded = expand_transition_table(
        table,
        add_states=1,
        alter_weights=True,
    )

    assert np.all(
        expanded == UNEXPLORED_TRANSITION_WEIGHT
    )

    assert UNEXPLORED_TRANSITION_WEIGHT == pytest.approx(
        0.05
    )


def test_transition_expansion_without_weight_alteration():
    table = create_transition_table(
        num_states=2,
        num_actions=1,
    )

    expanded = expand_transition_table(
        table,
        add_states=1,
        alter_weights=False,
    )

    expected = np.array(
        [
            [[0.5], [0.5], [1.0 / 3.0]],
            [[0.5], [0.5], [1.0 / 3.0]],
            [[1.0 / 3.0], [1.0 / 3.0], [1.0 / 3.0]],
        ]
    )

    np.testing.assert_allclose(
        expanded,
        expected,
    )


def test_zero_state_expansion_returns_copy():
    table = create_transition_table(
        num_states=2,
        num_actions=13,
    )

    expanded = expand_transition_table(
        table,
        add_states=0,
    )

    np.testing.assert_allclose(
        expanded,
        table,
    )

    assert expanded is not table


@pytest.mark.parametrize(
    "invalid_count",
    [
        0,
        -1,
    ],
)
def test_invalid_initial_state_count_is_rejected(
    invalid_count,
):
    with pytest.raises(ValueError):
        create_transition_table(
            invalid_count,
            13,
        )


@pytest.mark.parametrize(
    "invalid_count",
    [
        True,
        2.5,
        "2",
    ],
)
def test_non_integer_initial_state_count_is_rejected(
    invalid_count,
):
    with pytest.raises(TypeError):
        create_transition_table(
            invalid_count,
            13,
        )


def test_invalid_likelihood_dimension_is_rejected():
    with pytest.raises(ValueError):
        expand_likelihood_table(
            np.ones((2, 2, 2)),
            add_states=1,
        )


def test_invalid_transition_dimension_is_rejected():
    with pytest.raises(ValueError):
        expand_transition_table(
            np.ones((2, 2)),
            add_states=1,
        )
