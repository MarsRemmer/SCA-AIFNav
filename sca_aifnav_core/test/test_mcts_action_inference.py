"""Tests for MCTS root-action inference."""

import numpy as np
import pytest

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.mcts_action_inference import (
    ACTION_PRECISION,
    DEFAULT_ACTION_SELECTION,
    POLICY_PRECISION,
    infer_root_action,
)
from sca_aifnav_core.mcts_model_interface import (
    MCTSModelInterface,
)
from sca_aifnav_core.mcts_node import (
    SearchTreeNode,
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


def make_components():
    motion_set = BaselineMotionSet()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(0.0, 0.0)
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
    )

    root = SearchTreeNode(
        state_belief=np.array(
            [1.0, 0.0]
        ),
        place_id=0,
        action_id=None,
    )

    return root, interface


def child_with_average(
    place_id,
    parent,
    average_reward,
    visits=2,
):
    child = SearchTreeNode(
        state_belief=np.array(
            [1.0, 0.0]
        ),
        place_id=place_id,
        parent=parent,
    )

    child.visit_count = visits

    child.total_reward = (
        average_reward
        * visits
    )

    return child


def test_action_inference_defaults_match_baseline():
    assert POLICY_PRECISION == pytest.approx(
        16.0
    )

    assert ACTION_PRECISION == pytest.approx(
        16.0
    )

    assert DEFAULT_ACTION_SELECTION == (
        "stochastic"
    )


def test_root_without_children_returns_no_action():
    root, interface = make_components()

    result = infer_root_action(
        root_node=root,
        model_interface=interface,
    )

    assert result.selected_action is None

    assert result.available_actions == ()

    assert result.full_action_values.shape == (
        13,
    )


def test_child_average_rewards_become_action_values():
    root, interface = make_components()

    first = child_with_average(
        place_id=1,
        parent=root,
        average_reward=0.25,
    )

    second = child_with_average(
        place_id=2,
        parent=root,
        average_reward=0.75,
    )

    root.children = {
        0: first,
        2: second,
    }

    result = infer_root_action(
        root_node=root,
        model_interface=interface,
        action_selection="deterministic",
    )

    np.testing.assert_allclose(
        result.action_values,
        np.array(
            [0.25, 0.75]
        ),
    )


def test_higher_reward_has_higher_policy_posterior():
    root, interface = make_components()

    root.children = {
        0: child_with_average(
            1,
            root,
            0.0,
        ),
        1: child_with_average(
            2,
            root,
            1.0,
        ),
    }

    result = infer_root_action(
        root_node=root,
        model_interface=interface,
        action_selection="deterministic",
    )

    assert (
        result.policy_posterior[1]
        > result.policy_posterior[0]
    )


def test_policy_posterior_uses_gamma_sixteen():
    root, interface = make_components()

    root.children = {
        0: child_with_average(
            1,
            root,
            0.0,
        ),
        1: child_with_average(
            2,
            root,
            1.0,
        ),
    }

    result = infer_root_action(
        root_node=root,
        model_interface=interface,
        action_selection="deterministic",
    )

    expected = np.exp(
        np.array(
            [0.0, 16.0]
        )
    )

    expected = (
        expected
        / expected.sum()
    )

    np.testing.assert_allclose(
        result.policy_posterior,
        expected,
    )


def test_deterministic_selection_chooses_best_action():
    root, interface = make_components()

    root.children = {
        3: child_with_average(
            1,
            root,
            -1.0,
        ),
        7: child_with_average(
            2,
            root,
            0.5,
        ),
    }

    result = infer_root_action(
        root_node=root,
        model_interface=interface,
        action_selection="deterministic",
    )

    assert result.selected_action == 7


def test_negative_place_id_is_given_zero_value():
    root, interface = make_components()

    invalid = child_with_average(
        place_id=-1,
        parent=root,
        average_reward=100.0,
    )

    valid = child_with_average(
        place_id=1,
        parent=root,
        average_reward=-1.0,
    )

    root.children = {
        0: invalid,
        1: valid,
    }

    result = infer_root_action(
        root_node=root,
        model_interface=interface,
        action_selection="deterministic",
    )

    assert result.action_values[
        0
    ] == pytest.approx(
        0.0
    )


def test_full_vectors_restore_thirteen_action_layout():
    root, interface = make_components()

    root.children = {
        2: child_with_average(
            1,
            root,
            0.25,
        ),
        12: child_with_average(
            0,
            root,
            0.50,
        ),
    }

    result = infer_root_action(
        root_node=root,
        model_interface=interface,
        action_selection="deterministic",
    )

    assert result.full_action_values.shape == (
        13,
    )

    assert result.full_policy_posterior.shape == (
        13,
    )

    assert result.full_action_values[
        2
    ] == pytest.approx(
        0.25
    )

    assert result.full_action_values[
        12
    ] == pytest.approx(
        0.50
    )

    assert result.full_action_values[
        1
    ] == pytest.approx(
        0.0
    )


def test_stochastic_selection_applies_action_precision():
    root, interface = make_components()

    root.children = {
        0: child_with_average(
            1,
            root,
            0.0,
        ),
        1: child_with_average(
            2,
            root,
            0.1,
        ),
    }

    result = infer_root_action(
        root_node=root,
        model_interface=interface,
        action_selection="stochastic",
        rng=np.random.default_rng(
            3
        ),
    )

    log_q = np.log(
        result.policy_posterior
        + 1e-16
    )

    expected = np.exp(
        log_q
        * ACTION_PRECISION
        - np.max(
            log_q
            * ACTION_PRECISION
        )
    )

    expected = (
        expected
        / expected.sum()
    )

    np.testing.assert_allclose(
        result.selection_probabilities,
        expected,
    )


def test_policy_posterior_is_normalized():
    root, interface = make_components()

    root.children = {
        0: child_with_average(
            1,
            root,
            0.1,
        ),
        1: child_with_average(
            2,
            root,
            0.2,
        ),
        2: child_with_average(
            3,
            root,
            0.3,
        ),
    }

    result = infer_root_action(
        root_node=root,
        model_interface=interface,
        action_selection="deterministic",
    )

    assert result.policy_posterior.sum() == pytest.approx(
        1.0
    )


def test_action_inference_does_not_modify_tree_statistics():
    root, interface = make_components()

    child = child_with_average(
        place_id=1,
        parent=root,
        average_reward=0.5,
        visits=4,
    )

    root.children = {
        0: child,
    }

    before = (
        child.visit_count,
        child.total_reward,
    )

    infer_root_action(
        root_node=root,
        model_interface=interface,
        action_selection="deterministic",
    )

    after = (
        child.visit_count,
        child.total_reward,
    )

    assert after == before
