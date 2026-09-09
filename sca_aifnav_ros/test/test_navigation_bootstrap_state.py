"""Exact bootstrap-state tests for the AIMAPP-compatible baseline."""

import numpy as np
import pytest

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)
from sca_aifnav_ros.navigation_core_bridge import (
    NavigationCoreBridge,
)
from sca_aifnav_ros.navigation_observation import (
    NavigationObservation,
)


def bootstrap_observation():
    """Create one deterministic fully open bootstrap observation."""
    return NavigationObservation(
        state=CognitiveOdomState(
            position=Point2D(
                0.0,
                0.0,
            ),
            travel_heading_rad=0.0,
        ),
        sensory_observation=0,
        place_observation=0,
        obstacle_distances=tuple(
            3.5
            for _ in range(12)
        ),
        odometry_revision=1,
        scan_revision=1,
    )


def bootstrapped_bridge():
    """Create a runtime-faithful bridge with physical place zero seeded."""
    memory = BaselinePlaceMemory()

    origin_id = memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    assert origin_id == 0

    bridge = NavigationCoreBridge(
        memory=memory
    )

    decision = bridge.process_observation(
        bootstrap_observation()
    )

    return bridge, decision


def test_bootstrap_preserves_physical_origin_as_place_zero():
    """The first real cognitive place should remain state/place zero."""
    bridge, _ = bootstrapped_bridge()

    assert (
        bridge.memory.place(0)
        == Point2D(
            0.0,
            0.0,
        )
    )


def test_bootstrap_state_and_place_dimensions_stay_aligned():
    """Ghost-node growth should preserve place ID equals hidden-state ID."""
    bridge, _ = bootstrapped_bridge()

    model = bridge.model
    memory = bridge.memory

    assert len(memory) > 1

    assert (
        model.num_states
        == len(memory)
    )

    assert (
        model.place_observations
        == model.num_states
    )

    for place_id in range(
        len(memory)
    ):
        assert (
            memory.place(place_id)
            is not None
        )

        assert (
            model.place_likelihood[
                place_id,
                place_id,
            ]
            == pytest.approx(1.0)
        )


def test_bootstrap_qs_remains_anchored_to_real_start_state():
    """Imagined ghost growth must not replace the real bootstrap belief."""
    bridge, _ = bootstrapped_bridge()

    belief = bridge.model.state_belief

    assert (
        belief.shape
        == (
            bridge.model.num_states,
        )
    )

    assert (
        int(
            np.argmax(
                belief
            )
        )
        == 0
    )

    assert (
        belief[0]
        > 0.999
    )

    np.testing.assert_allclose(
        belief.sum(),
        1.0,
    )


def test_bootstrap_A_and_pA_dimensions_match_after_growth():
    """Observation likelihoods and concentrations should grow together."""
    bridge, _ = bootstrapped_bridge()

    model = bridge.model

    assert (
        model.sensory_likelihood.shape
        == model.sensory_concentration.shape
    )

    assert (
        model.place_likelihood.shape
        == model.place_concentration.shape
    )

    assert (
        model.sensory_likelihood.shape[1]
        == model.num_states
    )

    assert (
        model.place_likelihood.shape
        == (
            model.num_states,
            model.num_states,
        )
    )


def test_bootstrap_B_and_pB_dimensions_match_after_growth():
    """Transition likelihood and concentration dimensions must stay aligned."""
    bridge, _ = bootstrapped_bridge()

    model = bridge.model

    expected_shape = (
        model.num_states,
        model.num_states,
        13,
    )

    assert (
        model.transition_likelihood.shape
        == expected_shape
    )

    assert (
        model.transition_concentration.shape
        == expected_shape
    )


def test_bootstrap_stationary_B_is_identity():
    """STAY should remain a deterministic self-transition after growth."""
    bridge, _ = bootstrapped_bridge()

    model = bridge.model

    np.testing.assert_allclose(
        model.transition_likelihood[
            :,
            :,
            12,
        ],
        np.eye(
            model.num_states
        ),
    )


def test_bootstrap_preferences_are_exploration_defaults():
    """Bootstrap should begin without an observation preference."""
    _, decision = bootstrapped_bridge()

    preferences = (
        decision.cycle_result
        .preferences
    )

    assert (
        preferences.preferred_observations
        == (
            -1,
            -1,
        )
    )

    np.testing.assert_allclose(
        preferences.sensory,
        0.0,
    )

    np.testing.assert_allclose(
        preferences.place,
        0.0,
    )

    np.testing.assert_allclose(
        preferences.preferred_states,
        0.0,
    )


def test_bootstrap_mcts_root_matches_current_real_state():
    """The first MCTS tree should start from place zero and current qs."""
    bridge, decision = bootstrapped_bridge()

    planning = (
        decision.cycle_result
        .planning
    )

    root = planning.root_node

    assert root.place_id == 0
    assert root.parent is None
    assert root.action_id is None

    np.testing.assert_allclose(
        root.state_belief,
        bridge.model.state_belief,
    )

    assert (
        planning.selected_action
        in planning.available_actions
    )


def test_bootstrap_uses_AIMAPP_mcts_runtime_parameters():
    """First planning should retain the fixed AIMAPP MCTS parameters."""
    bridge, decision = bootstrapped_bridge()

    coordinator = bridge.coordinator
    planning = (
        decision.cycle_result
        .planning
    )

    assert (
        coordinator.num_simulations
        == 30
    )

    assert (
        coordinator.max_rollout_depth
        == 10
    )

    assert (
        coordinator.c_param
        == pytest.approx(5.0)
    )

    assert (
        planning.num_simulations
        == 30
    )
