"""Tests for inductive preference integration into MCTS evaluation."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.inductive_inference import (
    inductive_bonus,
)
from sca_aifnav_core.mcts_model_interface import (
    MCTSModelInterface,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_core.preference_state import (
    BaselinePreferenceState,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)


def make_case(
    use_inductive_inference,
    inductive_horizon=4,
    add_preference=True,
):
    motion_set = BaselineMotionSet()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    model = BaselineGenerativeModel()

    model.sensory_likelihood = np.eye(
        2
    )

    model.place_likelihood = np.eye(
        2
    )

    # Construct several strong outgoing transitions so that the strict
    # certainty-threshold comparison retains the strongest transition.
    for action_id in range(
        motion_set.DIRECTION_COUNT
    ):
        model.transition_likelihood[
            :,
            0,
            action_id,
        ] = np.array(
            [0.2, 0.8]
        )

    model.transition_likelihood[
        :,
        0,
        0,
    ] = np.array(
        [0.1, 0.9]
    )

    model.enforce_stationary_transition()

    preferences = BaselinePreferenceState(
        model
    )

    if add_preference:
        preferences.update(
            model=model,
            place_observation=1,
            preference_weight=2.0,
        )

    interface = MCTSModelInterface(
        model=model,
        memory=memory,
        motion_set=motion_set,
        preferences=preferences,
        use_utility=False,
        use_state_information_gain=False,
        use_inductive_inference=(
            use_inductive_inference
        ),
        inductive_horizon=(
            inductive_horizon
        ),
    )

    return (
        model,
        preferences,
        interface,
    )


def test_disabled_inductive_term_keeps_base_score():
    (
        _,
        _,
        interface,
    ) = make_case(
        use_inductive_inference=False
    )

    result = interface.evaluate_action(
        current_belief=np.array(
            [1.0, 0.0]
        ),
        action_id=0,
    )

    assert result.score == pytest.approx(
        0.0
    )


def test_enabled_inductive_term_is_added_to_score():
    (
        model,
        preferences,
        interface,
    ) = make_case(
        use_inductive_inference=True
    )

    current_belief = np.array(
        [1.0, 0.0]
    )

    result = interface.evaluate_action(
        current_belief=current_belief,
        action_id=0,
    )

    expected_bonus = inductive_bonus(
        current_belief=current_belief,
        predicted_belief=(
            result.predicted_state
        ),
        transition_likelihood=(
            model.transition_likelihood
        ),
        preferred_states=(
            preferences.preferred_states
        ),
        lookahead_horizon=4,
    )

    assert expected_bonus > 0.0

    assert result.score == pytest.approx(
        expected_bonus
    )


def test_inductive_term_increases_goal_directed_action_value():
    (
        _,
        _,
        disabled,
    ) = make_case(
        use_inductive_inference=False
    )

    (
        _,
        _,
        enabled,
    ) = make_case(
        use_inductive_inference=True
    )

    current_belief = np.array(
        [1.0, 0.0]
    )

    without_induction = disabled.evaluate_action(
        current_belief=current_belief,
        action_id=0,
    )

    with_induction = enabled.evaluate_action(
        current_belief=current_belief,
        action_id=0,
    )

    assert (
        with_induction.score
        > without_induction.score
    )


def test_zero_inductive_horizon_adds_no_bonus():
    (
        _,
        _,
        interface,
    ) = make_case(
        use_inductive_inference=True,
        inductive_horizon=0,
    )

    result = interface.evaluate_action(
        current_belief=np.array(
            [1.0, 0.0]
        ),
        action_id=0,
    )

    assert result.score == pytest.approx(
        0.0
    )


def test_inductive_evaluation_preserves_physical_belief():
    (
        model,
        _,
        interface,
    ) = make_case(
        use_inductive_inference=True
    )

    before = model.state_belief.copy()

    interface.evaluate_action(
        current_belief=np.array(
            [1.0, 0.0]
        ),
        action_id=0,
    )

    np.testing.assert_allclose(
        model.state_belief,
        before,
    )
