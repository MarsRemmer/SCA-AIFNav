"""Closed-loop navigation episode tests."""

import math

import numpy as np

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.navigation_cycle import (
    BaselineNavigationCoordinator,
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


MAX_EPISODE_STEPS = 14


def make_corridor_case():
    model = BaselineGenerativeModel()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    motion_set = BaselineMotionSet()

    root_place_id = memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    preferences = BaselinePreferenceState(
        model
    )

    coordinator = BaselineNavigationCoordinator(
        model=model,
        memory=memory,
        motion_set=motion_set,
        preferences=preferences,
        num_simulations=30,
    )

    initial_state = CognitiveOdomState(
        position=Point2D(
            0.0,
            0.0,
        ),
        travel_heading_rad=0.0,
    )

    initial_scan = np.full(
        motion_set.DIRECTION_COUNT,
        np.nan,
        dtype=float,
    )

    initial_scan[0] = 2.0

    coordinator.learning.step(
        state=initial_state,
        sensory_observation=0,
        place_observation=0,
        action_id=12,
        obstacle_distances=initial_scan,
    )

    goal_place_id = len(memory) - 1

    coordinator.set_preference(
        place_observation=goal_place_id,
        preference_weight=10.0,
    )

    return (
        model,
        memory,
        coordinator,
        root_place_id,
        goal_place_id,
    )


def physical_transition(
    current_place_id,
    action_id,
    goal_place_id,
):
    if action_id == 12:
        return current_place_id

    if action_id == 0:
        return min(
            current_place_id + 1,
            goal_place_id,
        )

    if action_id == 6:
        return max(
            current_place_id - 1,
            0,
        )

    raise AssertionError(
        f"unexpected corridor action {action_id}"
    )


def run_closed_loop():
    (
        model,
        memory,
        coordinator,
        root_place_id,
        goal_place_id,
    ) = make_corridor_case()

    unknown_scan = np.full(
        coordinator.motion_set.DIRECTION_COUNT,
        np.nan,
        dtype=float,
    )

    current_place_id = root_place_id

    plan = coordinator.plan_current(
        current_place_id=current_place_id,
        action_selection="deterministic",
    )

    visited_places = [
        current_place_id,
    ]

    executed_actions = []

    reached_goal_and_stopped = False

    for _ in range(
        MAX_EPISODE_STEPS
    ):
        action_id = plan.selected_action

        assert action_id is not None

        next_place_id = physical_transition(
            current_place_id=(
                current_place_id
            ),
            action_id=action_id,
            goal_place_id=goal_place_id,
        )

        old_position = memory.place(
            current_place_id
        )

        new_position = memory.place(
            next_place_id
        )

        if next_place_id == current_place_id:
            travel_heading = 0.0
        else:
            travel_heading = math.atan2(
                new_position.y
                - old_position.y,
                new_position.x
                - old_position.x,
            )

        state = CognitiveOdomState(
            position=new_position,
            travel_heading_rad=(
                travel_heading
            ),
        )

        result = coordinator.step_and_plan(
            state=state,
            sensory_observation=0,
            place_observation=(
                next_place_id
            ),
            executed_action_id=action_id,
            obstacle_distances=unknown_scan,
            current_place_id=(
                next_place_id
            ),
            action_selection="deterministic",
        )

        executed_actions.append(
            action_id
        )

        visited_places.append(
            next_place_id
        )

        current_place_id = (
            next_place_id
        )

        plan = result.planning

        if (
            current_place_id
            == goal_place_id
            and plan.selected_action == 12
        ):
            reached_goal_and_stopped = (
                True
            )
            break

    return {
        "model": model,
        "memory": memory,
        "coordinator": coordinator,
        "goal_place_id": goal_place_id,
        "current_place_id": (
            current_place_id
        ),
        "plan": plan,
        "visited_places": (
            visited_places
        ),
        "executed_actions": (
            executed_actions
        ),
        "reached_goal_and_stopped": (
            reached_goal_and_stopped
        ),
    }


def test_closed_loop_reaches_goal_and_prefers_stay():
    episode = run_closed_loop()

    assert (
        episode[
            "reached_goal_and_stopped"
        ]
        is True
    )

    assert (
        episode["current_place_id"]
        == episode["goal_place_id"]
    )

    assert (
        episode["plan"].selected_action
        == 12
    )


def test_closed_loop_preserves_cognitive_map_size():
    episode = run_closed_loop()

    assert len(
        episode["memory"]
    ) == 4

    assert (
        episode["model"].num_states
        == 4
    )


def test_closed_loop_uses_only_corridor_actions():
    episode = run_closed_loop()

    assert set(
        episode["executed_actions"]
    ).issubset(
        {
            0,
            6,
            12,
        }
    )

    assert 0 in (
        episode["executed_actions"]
    )


def test_closed_loop_keeps_model_belief_consistent():
    episode = run_closed_loop()

    model = episode["model"]

    coordinator = (
        episode["coordinator"]
    )

    np.testing.assert_allclose(
        model.state_belief,
        coordinator.learning.history.latest,
    )

    assert np.isclose(
        model.state_belief.sum(),
        1.0,
    )

    assert np.all(
        np.isfinite(
            model.state_belief
        )
    )
