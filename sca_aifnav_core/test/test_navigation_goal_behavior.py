"""Goal-directed end-to-end navigation behavior tests."""

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


def make_case(
    num_simulations=30,
):
    model = BaselineGenerativeModel()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    root_place_id = memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    motion_set = BaselineMotionSet()

    preferences = BaselinePreferenceState(
        model
    )

    coordinator = BaselineNavigationCoordinator(
        model=model,
        memory=memory,
        motion_set=motion_set,
        preferences=preferences,
        num_simulations=num_simulations,
    )

    state = CognitiveOdomState(
        position=Point2D(
            0.0,
            0.0,
        ),
        travel_heading_rad=0.0,
    )

    forward_clearance = np.full(
        motion_set.DIRECTION_COUNT,
        np.nan,
        dtype=float,
    )

    # Only action 0 has sufficiently long visible free space.
    forward_clearance[0] = 2.0

    return (
        model,
        memory,
        coordinator,
        state,
        forward_clearance,
        root_place_id,
    )


def grow_forward_chain(
    coordinator,
    state,
    forward_clearance,
):
    # This is an intentionally completed stationary action in the
    # synthetic test, not a substitute for a failed motion command.
    return coordinator.learning.step(
        state=state,
        sensory_observation=0,
        place_observation=0,
        action_id=12,
        obstacle_distances=(
            forward_clearance
        ),
    )


def test_plan_current_does_not_add_learning_experience():
    (
        model,
        _,
        coordinator,
        _,
        _,
        root_place_id,
    ) = make_case(
        num_simulations=3
    )

    belief_before = (
        model.state_belief.copy()
    )

    history_size_before = len(
        coordinator.learning.history.entries()
    )

    coordinator.plan_current(
        current_place_id=root_place_id,
        action_selection="deterministic",
    )

    history_size_after = len(
        coordinator.learning.history.entries()
    )

    assert (
        history_size_after
        == history_size_before
    )

    np.testing.assert_allclose(
        model.state_belief,
        belief_before,
    )


def test_before_cognitive_growth_only_stay_is_reachable():
    (
        _,
        _,
        coordinator,
        _,
        _,
        root_place_id,
    ) = make_case(
        num_simulations=3
    )

    plan = coordinator.plan_current(
        current_place_id=root_place_id,
        action_selection="deterministic",
    )

    assert plan.available_actions == (
        12,
    )

    assert plan.selected_action == 12


def test_forward_clearance_grows_three_step_cognitive_chain():
    (
        model,
        memory,
        coordinator,
        state,
        forward_clearance,
        _,
    ) = make_case(
        num_simulations=3
    )

    grow_forward_chain(
        coordinator,
        state,
        forward_clearance,
    )

    assert len(memory) == 4

    assert model.num_states == 4

    assert memory.place(1) == Point2D(
        0.6,
        0.16,
    )

    assert memory.place(2) == Point2D(
        1.2,
        0.32,
    )

    assert memory.place(3) == Point2D(
        1.8,
        0.48,
    )


def test_far_goal_is_mapped_to_a_preferred_state():
    (
        _,
        memory,
        coordinator,
        state,
        forward_clearance,
        _,
    ) = make_case(
        num_simulations=3
    )

    grow_forward_chain(
        coordinator,
        state,
        forward_clearance,
    )

    goal_place_id = len(
        memory
    ) - 1

    preference = (
        coordinator.set_preference(
            place_observation=(
                goal_place_id
            ),
            preference_weight=10.0,
        )
    )

    assert (
        preference.preferred_observations
        == (
            -1,
            goal_place_id,
        )
    )

    assert np.any(
        preference.preferred_states
        > 0.0
    )


def test_inductive_preference_improves_forward_action_score():
    (
        model,
        memory,
        coordinator,
        state,
        forward_clearance,
        _,
    ) = make_case(
        num_simulations=3
    )

    grow_forward_chain(
        coordinator,
        state,
        forward_clearance,
    )

    goal_place_id = len(
        memory
    ) - 1

    coordinator.set_preference(
        place_observation=(
            goal_place_id
        ),
        preference_weight=10.0,
    )

    current_belief = (
        model.state_belief.copy()
    )

    coordinator.model_interface.use_inductive_inference = (
        False
    )

    without_induction = (
        coordinator.model_interface.evaluate_action(
            current_belief=(
                current_belief
            ),
            action_id=0,
        )
    )

    coordinator.model_interface.use_inductive_inference = (
        True
    )

    with_induction = (
        coordinator.model_interface.evaluate_action(
            current_belief=(
                current_belief
            ),
            action_id=0,
        )
    )

    assert (
        with_induction.score
        > without_induction.score
    )


def test_far_goal_drives_mcts_toward_forward_action():
    (
        _,
        memory,
        coordinator,
        state,
        forward_clearance,
        root_place_id,
    ) = make_case()

    grow_forward_chain(
        coordinator,
        state,
        forward_clearance,
    )

    goal_place_id = len(
        memory
    ) - 1

    coordinator.set_preference(
        place_observation=(
            goal_place_id
        ),
        preference_weight=10.0,
    )

    plan = coordinator.plan_current(
        current_place_id=root_place_id,
        possible_actions=[
            0,
            12,
        ],
        action_selection="deterministic",
    )

    assert plan.available_actions == (
        0,
        12,
    )

    assert plan.selected_action == 0

    assert (
        plan.action_values[0]
        > plan.action_values[12]
    )

    assert (
        plan.policy_posterior[0]
        > plan.policy_posterior[12]
    )
