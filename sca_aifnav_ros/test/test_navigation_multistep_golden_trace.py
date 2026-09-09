"""Golden trace for several consecutive navigation updates."""

import math
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
    heading,
    sensory_id,
    place_id,
):
    """Create one deterministic completed navigation observation."""
    return NavigationObservation(
        state=CognitiveOdomState(
            position=Point2D(
                float(x),
                float(y),
            ),
            travel_heading_rad=float(
                heading
            ),
        ),
        sensory_observation=(
            sensory_id
        ),
        place_observation=(
            place_id
        ),
        obstacle_distances=tuple(
            float("nan")
            for _ in range(12)
        ),
        odometry_revision=1,
        scan_revision=1,
    )


def configured_bridge():
    """Create a deterministic three-place navigation trace."""
    bridge = NavigationCoreBridge()

    places = (
        Point2D(
            0.0,
            0.0,
        ),
        Point2D(
            10.0,
            0.0,
        ),
        Point2D(
            20.0,
            0.0,
        ),
    )

    for expected_id, position in enumerate(
        places
    ):
        place_id = bridge.memory.resolve_place(
            position
        )

        assert place_id == expected_id

    bridge.model.register_place_observation(
        1
    )

    bridge.model.register_place_observation(
        2
    )

    planned_actions = (
        0,
        0,
        6,
        12,
    )

    planning_calls = []

    def restrictive_possible_actions(
        current_place_id,
        obstacle_distances,
    ):
        """Expose only the actions required by the golden trace."""
        return (
            0,
            6,
            12,
        )

    def deterministic_plan(
        current_place_id,
        possible_actions=None,
        action_selection=None,
        rng=None,
    ):
        """Return the next action in the predefined trace."""
        index = len(
            planning_calls
        )

        if index >= len(
            planned_actions
        ):
            raise RuntimeError(
                "unexpected extra planning call"
            )

        selected_action = (
            planned_actions[index]
        )

        planning_calls.append(
            current_place_id
        )

        if possible_actions is None:
            available_actions = (
                0,
                6,
                12,
            )
        else:
            available_actions = tuple(
                possible_actions
            )

        assert (
            selected_action
            in available_actions
        )

        return SimpleNamespace(
            selected_action=(
                selected_action
            ),
            available_actions=(
                available_actions
            ),
            root_node=SimpleNamespace(
                place_id=(
                    current_place_id
                )
            ),
        )

    transition_targets = {
        (0, 0): 1,
        (1, 0): 2,
        (2, 6): 1,
        (1, 12): 1,
    }

    def get_next_place_id(
        current_place_id,
        action_id,
    ):
        """Resolve the exact physical target of the golden trace."""
        key = (
            current_place_id,
            action_id,
        )

        if key not in transition_targets:
            return -1

        return transition_targets[
            key
        ]

    bridge.coordinator.restrictive_possible_actions = (
        restrictive_possible_actions
    )

    bridge.coordinator.plan_current = (
        deterministic_plan
    )

    bridge.coordinator.model_interface.get_next_place_id = (
        get_next_place_id
    )

    return (
        bridge,
        planning_calls,
    )


def run_trace():
    """Execute bootstrap followed by three real physical transitions."""
    (
        bridge,
        planning_calls,
    ) = configured_bridge()

    decisions = []
    targets = []
    pB_checks = []

    bootstrap = bridge.process_observation(
        observation(
            x=0.0,
            y=0.0,
            heading=0.0,
            sensory_id=0,
            place_id=0,
        )
    )

    decisions.append(
        bootstrap
    )

    targets.append(
        bridge.resolve_planned_action_target()
    )

    real_steps = (
        (
            0,
            observation(
                x=10.0,
                y=0.0,
                heading=0.0,
                sensory_id=1,
                place_id=1,
            ),
        ),
        (
            0,
            observation(
                x=20.0,
                y=0.0,
                heading=0.0,
                sensory_id=2,
                place_id=2,
            ),
        ),
        (
            6,
            observation(
                x=10.0,
                y=0.0,
                heading=math.pi,
                sensory_id=1,
                place_id=1,
            ),
        ),
    )

    for action_id, next_observation in real_steps:
        model = bridge.model

        pB_before = (
            model.transition_concentration.copy()
        )

        previous_belief = (
            bridge.coordinator
            .learning
            .history
            .latest
            .copy()
        )

        bridge.record_executed_action(
            action_id
        )

        # Completion alone must not learn.
        np.testing.assert_allclose(
            model.transition_concentration,
            pB_before,
        )

        decision = bridge.process_observation(
            next_observation
        )

        decisions.append(
            decision
        )

        experience = (
            decision.cycle_result
            .learning
            .real_experience
        )

        expected_pB = (
            pB_before.copy()
        )

        expected_pB[
            :,
            :,
            action_id,
        ] += (
            10.0
            * np.outer(
                experience.preliminary_belief,
                previous_belief,
            )
        )

        if (
            experience.reverse_transition_updated
        ):
            reverse_action_id = (
                experience.reverse_action_id
            )

            expected_pB[
                :,
                :,
                reverse_action_id,
            ] += (
                7.0
                * np.outer(
                    previous_belief,
                    experience.preliminary_belief,
                )
            )

        np.testing.assert_allclose(
            model.transition_concentration,
            expected_pB,
        )

        pB_checks.append(
            (
                action_id,
                experience.reverse_action_id,
            )
        )

        targets.append(
            bridge.resolve_planned_action_target()
        )

    return {
        "bridge": bridge,
        "decisions": decisions,
        "targets": targets,
        "planning_calls": planning_calls,
        "pB_checks": pB_checks,
    }


def test_multistep_action_observation_trace():
    """Executed and newly planned actions must remain one step apart."""
    trace = run_trace()

    decisions = trace[
        "decisions"
    ]

    assert [
        decision.executed_action_id
        for decision in decisions
    ] == [
        None,
        0,
        0,
        6,
    ]

    assert [
        decision.next_action_id
        for decision in decisions
    ] == [
        0,
        0,
        6,
        12,
    ]

    assert [
        decision.observation.place_observation
        for decision in decisions
    ] == [
        0,
        1,
        2,
        1,
    ]


def test_multistep_planning_roots_follow_real_places():
    """Every new plan must begin from the newly observed place."""
    trace = run_trace()

    assert (
        trace["planning_calls"]
        == [
            0,
            1,
            2,
            1,
        ]
    )


def test_multistep_action_targets_match_cognitive_trace():
    """Each planned action should resolve to the expected cognitive target."""
    trace = run_trace()

    targets = trace[
        "targets"
    ]

    assert [
        target.source_place_id
        for target in targets
    ] == [
        0,
        1,
        2,
        1,
    ]

    assert [
        target.target_place_id
        for target in targets
    ] == [
        1,
        2,
        1,
        1,
    ]

    assert [
        target.action_id
        for target in targets
    ] == [
        0,
        0,
        6,
        12,
    ]


def test_multistep_qs_history_tracks_each_real_transition():
    """Each real observation should add learning and posterior beliefs."""
    trace = run_trace()

    bridge = trace[
        "bridge"
    ]

    history = (
        bridge.coordinator
        .learning
        .history
        .entries()
    )

    assert len(
        history
    ) == 7

    assert [
        int(
            np.argmax(
                belief
            )
        )
        for belief in history
    ] == [
        0,
        1,
        1,
        2,
        2,
        1,
        1,
    ]

    np.testing.assert_allclose(
        bridge.model.state_belief,
        history[-1],
    )

    np.testing.assert_allclose(
        bridge.model.state_belief.sum(),
        1.0,
    )


def test_multistep_real_B_learning_uses_correct_reverse_actions():
    """Forward and reverse B updates must follow each executed action."""
    trace = run_trace()

    assert (
        trace["pB_checks"]
        == [
            (
                0,
                6,
            ),
            (
                0,
                6,
            ),
            (
                6,
                0,
            ),
        ]
    )


def test_multistep_A_tracks_new_and_revisited_observations():
    """Both observation modalities should remain aligned after the trace."""
    trace = run_trace()

    model = trace[
        "bridge"
    ].model

    assert (
        model.sensory_observations
        == 3
    )

    assert (
        model.place_observations
        == 3
    )

    assert (
        model.num_states
        == 3
    )

    assert (
        int(
            np.argmax(
                model.sensory_likelihood[
                    1,
                    :,
                ]
            )
        )
        == 1
    )

    assert (
        int(
            np.argmax(
                model.sensory_likelihood[
                    2,
                    :,
                ]
            )
        )
        == 2
    )

    assert (
        int(
            np.argmax(
                model.place_likelihood[
                    1,
                    :,
                ]
            )
        )
        == 1
    )

    assert (
        int(
            np.argmax(
                model.place_likelihood[
                    2,
                    :,
                ]
            )
        )
        == 2
    )


def test_multistep_trace_ends_with_clean_STAY_plan():
    """After returning to place one, the next plan should be STAY."""
    trace = run_trace()

    bridge = trace[
        "bridge"
    ]

    assert (
        bridge.completed_action_id
        is None
    )

    assert (
        bridge.next_action_id
        == 12
    )

    final_target = (
        trace["targets"][-1]
    )

    assert (
        final_target.is_stationary
        is True
    )

    assert (
        final_target.source_place_id
        == final_target.target_place_id
        == 1
    )
