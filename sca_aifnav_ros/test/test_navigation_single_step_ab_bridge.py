"""Bridge-level golden test for one complete real A/B update."""

from types import SimpleNamespace

import numpy as np

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.navigation_core_bridge import (
    NavigationCoreBridge,
)
from sca_aifnav_ros.navigation_observation import (
    NavigationObservation,
)


def observation(
    x,
    y,
    sensory_observation,
    place_observation,
):
    """Create one deterministic navigation observation."""
    return NavigationObservation(
        state=CognitiveOdomState(
            position=Point2D(
                float(x),
                float(y),
            ),
            travel_heading_rad=0.0,
        ),
        sensory_observation=(
            sensory_observation
        ),
        place_observation=(
            place_observation
        ),
        obstacle_distances=tuple(
            float("nan")
            for _ in range(12)
        ),
        odometry_revision=1,
        scan_revision=1,
    )


def configured_bridge():
    """Create a deterministic two-state bridge lifecycle."""
    bridge = NavigationCoreBridge()

    source_id = bridge.memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    target_id = bridge.memory.resolve_place(
        Point2D(
            10.0,
            10.0,
        )
    )

    assert source_id == 0
    assert target_id == 1

    bridge.model.register_place_observation(
        target_id
    )

    planning_calls = []

    def restrictive_possible_actions(
        current_place_id,
        obstacle_distances,
    ):
        """Keep the bridge test independent of obstacle filtering."""
        return (
            0,
            12,
        )

    def deterministic_plan(
        current_place_id,
        possible_actions=None,
        action_selection=None,
        rng=None,
    ):
        """Select action zero first and STAY after learning."""
        call_index = len(
            planning_calls
        )

        planning_calls.append(
            current_place_id
        )

        if call_index == 0:
            selected_action = 0
        else:
            selected_action = 12

        return SimpleNamespace(
            selected_action=(
                selected_action
            ),
            available_actions=(
                tuple(
                    possible_actions
                )
                if possible_actions
                is not None
                else (
                    0,
                    12,
                )
            ),
            root_node=SimpleNamespace(
                place_id=(
                    current_place_id
                )
            ),
        )

    bridge.coordinator.restrictive_possible_actions = (
        restrictive_possible_actions
    )

    bridge.coordinator.plan_current = (
        deterministic_plan
    )

    return (
        bridge,
        planning_calls,
    )


def test_bridge_completed_action_triggers_one_real_ab_update():
    """A completed physical action should drive the next A/B update."""
    (
        bridge,
        planning_calls,
    ) = configured_bridge()

    model = bridge.model

    bootstrap_pA_sensory = (
        model.sensory_concentration.copy()
    )

    bootstrap_pA_place = (
        model.place_concentration.copy()
    )

    bootstrap_pB = (
        model.transition_concentration.copy()
    )

    first = bridge.process_observation(
        observation(
            x=0.0,
            y=0.0,
            sensory_observation=0,
            place_observation=0,
        )
    )

    # --------------------------------------------------------
    # 1. Bootstrap plans but performs no real A/B learning.
    # --------------------------------------------------------

    assert first.is_bootstrap is True
    assert first.executed_action_id is None
    assert first.next_action_id == 0

    assert (
        first.cycle_result.learning
        is None
    )

    np.testing.assert_allclose(
        model.sensory_concentration,
        bootstrap_pA_sensory,
    )

    np.testing.assert_allclose(
        model.place_concentration,
        bootstrap_pA_place,
    )

    np.testing.assert_allclose(
        model.transition_concentration,
        bootstrap_pB,
    )

    # --------------------------------------------------------
    # 2. Physical completion is recorded, but learning still
    #    waits until the consequence observation arrives.
    # --------------------------------------------------------

    pA_sensory_before = (
        model.sensory_concentration.copy()
    )

    pA_place_before = (
        model.place_concentration.copy()
    )

    pB_before = (
        model.transition_concentration.copy()
    )

    bridge.record_executed_action(
        0
    )

    assert (
        bridge.completed_action_id
        == 0
    )

    assert bridge.next_action_id is None

    np.testing.assert_allclose(
        model.sensory_concentration,
        pA_sensory_before,
    )

    np.testing.assert_allclose(
        model.place_concentration,
        pA_place_before,
    )

    np.testing.assert_allclose(
        model.transition_concentration,
        pB_before,
    )

    # --------------------------------------------------------
    # 3. The next real observation consumes action 0 and
    #    performs the complete learning cycle.
    # --------------------------------------------------------

    second = bridge.process_observation(
        observation(
            x=10.0,
            y=10.0,
            sensory_observation=1,
            place_observation=1,
        )
    )

    assert second.is_bootstrap is False

    assert (
        second.executed_action_id
        == 0
    )

    assert (
        second.cycle_result.learning
        is not None
    )

    experience = (
        second.cycle_result
        .learning
        .real_experience
    )

    assert (
        experience.transition_updated
        is True
    )

    assert (
        experience.reverse_transition_updated
        is True
    )

    assert (
        experience.reverse_action_id
        == 6
    )

    # --------------------------------------------------------
    # 4. Real pB learning occurred only for action 0 and its
    #    reverse action 6.
    # --------------------------------------------------------

    changed_transition_actions = []

    for action_id in range(
        model.num_actions
    ):
        if not np.allclose(
            model.transition_concentration[
                :,
                :,
                action_id,
            ],
            pB_before[
                :,
                :,
                action_id,
            ],
        ):
            changed_transition_actions.append(
                action_id
            )

    assert (
        changed_transition_actions
        == [
            0,
            6,
        ]
    )

    # --------------------------------------------------------
    # 5. Both observation modalities learned from the same
    #    real posterior.
    # --------------------------------------------------------

    assert not np.allclose(
        model.sensory_concentration,
        pA_sensory_before,
    )

    assert not np.allclose(
        model.place_concentration,
        pA_place_before,
    )

    assert (
        model.sensory_concentration[
            1,
            1,
        ]
        > pA_sensory_before[
            1,
            1,
        ]
    )

    assert (
        model.place_concentration[
            1,
            1,
        ]
        > pA_place_before[
            1,
            1,
        ]
    )

    # --------------------------------------------------------
    # 6. The completed action has now been consumed.
    #    A fresh action was planned from the new state.
    # --------------------------------------------------------

    assert (
        bridge.completed_action_id
        is None
    )

    assert second.next_action_id == 12
    assert bridge.next_action_id == 12

    assert planning_calls == [
        0,
        1,
    ]

    np.testing.assert_allclose(
        model.state_belief,
        experience.posterior_belief,
    )
