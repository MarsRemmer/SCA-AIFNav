"""Tests for the complete active inference MCTS planner."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.mcts_action_inference import (
    ACTION_PRECISION,
    POLICY_PRECISION,
)
from sca_aifnav_core.mcts_model_interface import (
    MCTSModelInterface,
)
from sca_aifnav_core.mcts_planner import (
    DEFAULT_MAX_ROLLOUT_DEPTH,
    DEFAULT_MCTS_EXPLORATION,
    DEFAULT_NUM_SIMULATIONS,
    plan_mcts,
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


def make_case():
    motion_set = BaselineMotionSet()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    memory.resolve_place(
        Point2D(0.60, 0.16)
    )

    model = BaselineGenerativeModel()

    preferences = BaselinePreferenceState(
        model
    )

    interface = MCTSModelInterface(
        model=model,
        memory=memory,
        motion_set=motion_set,
        preferences=preferences,
        use_utility=True,
        use_state_information_gain=False,
        use_inductive_inference=False,
    )

    return (
        model,
        memory,
        interface,
    )


def test_planner_defaults_match_baseline():
    assert DEFAULT_NUM_SIMULATIONS == 30

    assert DEFAULT_MAX_ROLLOUT_DEPTH == 4

    assert DEFAULT_MCTS_EXPLORATION == pytest.approx(
        5.0
    )

    assert POLICY_PRECISION == pytest.approx(
        16.0
    )

    assert ACTION_PRECISION == pytest.approx(
        16.0
    )


def test_requested_number_of_simulations_updates_root():
    (
        model,
        _,
        interface,
    ) = make_case()

    result = plan_mcts(
        current_belief=model.state_belief,
        current_place_id=0,
        model_interface=interface,
        num_simulations=5,
        action_selection="deterministic",
    )

    assert result.num_simulations == 5

    assert result.root_node.visit_count == 5


def test_default_planner_runs_thirty_simulations():
    (
        model,
        _,
        interface,
    ) = make_case()

    result = plan_mcts(
        current_belief=model.state_belief,
        current_place_id=0,
        model_interface=interface,
        action_selection="deterministic",
    )

    assert result.root_node.visit_count == 30


def test_plan_returns_full_thirteen_action_vectors():
    (
        model,
        _,
        interface,
    ) = make_case()

    result = plan_mcts(
        current_belief=model.state_belief,
        current_place_id=0,
        model_interface=interface,
        num_simulations=3,
        action_selection="deterministic",
    )

    assert result.policy_posterior.shape == (
        13,
    )

    assert result.action_values.shape == (
        13,
    )


def test_selected_action_is_a_root_child():
    (
        model,
        _,
        interface,
    ) = make_case()

    result = plan_mcts(
        current_belief=model.state_belief,
        current_place_id=0,
        model_interface=interface,
        num_simulations=5,
        action_selection="deterministic",
    )

    assert result.selected_action in (
        result.root_node.children
    )


def test_root_action_constraint_can_force_stay():
    (
        model,
        _,
        interface,
    ) = make_case()

    result = plan_mcts(
        current_belief=model.state_belief,
        current_place_id=0,
        model_interface=interface,
        num_simulations=3,
        action_selection="deterministic",
        possible_actions=[12],
    )

    assert result.available_actions == (
        12,
    )

    assert result.selected_action == 12

    assert set(
        result.root_node.children
    ) == {
        12,
    }


def test_root_is_not_preinserted_into_tree_table():
    (
        model,
        _,
        interface,
    ) = make_case()

    result = plan_mcts(
        current_belief=model.state_belief,
        current_place_id=0,
        model_interface=interface,
        num_simulations=1,
        action_selection="deterministic",
        possible_actions=[12],
    )

    assert 0 in result.tree_table

    assert (
        result.tree_table[0]
        is not result.root_node
    )

    assert (
        result.tree_table[0].place_id
        == result.root_node.place_id
    )


def test_each_planning_call_gets_fresh_tree_table():
    (
        model,
        _,
        interface,
    ) = make_case()

    first = plan_mcts(
        current_belief=model.state_belief,
        current_place_id=0,
        model_interface=interface,
        num_simulations=1,
        action_selection="deterministic",
    )

    second = plan_mcts(
        current_belief=model.state_belief,
        current_place_id=0,
        model_interface=interface,
        num_simulations=1,
        action_selection="deterministic",
    )

    assert first.tree_table is not second.tree_table

    assert first.root_node is not second.root_node


def test_planning_does_not_modify_physical_belief():
    (
        model,
        _,
        interface,
    ) = make_case()

    before = model.state_belief.copy()

    plan_mcts(
        current_belief=model.state_belief,
        current_place_id=0,
        model_interface=interface,
        num_simulations=5,
        action_selection="deterministic",
    )

    np.testing.assert_allclose(
        model.state_belief,
        before,
    )


def test_seeded_stochastic_planning_returns_valid_action():
    (
        model,
        _,
        interface,
    ) = make_case()

    result = plan_mcts(
        current_belief=model.state_belief,
        current_place_id=0,
        model_interface=interface,
        num_simulations=5,
        action_selection="stochastic",
        rng=np.random.default_rng(
            7
        ),
    )

    assert result.selected_action in (
        result.root_node.children
    )


def test_zero_simulations_are_rejected():
    (
        model,
        _,
        interface,
    ) = make_case()

    with pytest.raises(
        ValueError
    ):
        plan_mcts(
            current_belief=model.state_belief,
            current_place_id=0,
            model_interface=interface,
            num_simulations=0,
        )


def test_invalid_current_place_is_rejected():
    (
        model,
        _,
        interface,
    ) = make_case()

    with pytest.raises(
        ValueError
    ):
        plan_mcts(
            current_belief=model.state_belief,
            current_place_id=99,
            model_interface=interface,
            num_simulations=1,
        )
