"""Tests for the assembled V5 baseline learning step."""

import numpy as np

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.baseline_step import (
    BaselineStepCoordinator,
)
from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)


def make_components():
    moves = BaselineMotionSet()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    model = BaselineGenerativeModel()

    return moves, memory, model


def test_coordinator_seeds_initial_history():
    moves, memory, model = (
        make_components()
    )

    coordinator = BaselineStepCoordinator(
        model=model,
        memory=memory,
        motion_set=moves,
    )

    assert len(coordinator.history) == 1

    np.testing.assert_allclose(
        coordinator.history.latest,
        model.state_belief,
    )


def test_real_step_adds_two_saved_inferences():
    moves, memory, model = (
        make_components()
    )

    coordinator = BaselineStepCoordinator(
        model=model,
        memory=memory,
        motion_set=moves,
    )

    state = CognitiveOdomState(
        position=Point2D(0.0, 0.0),
        travel_heading_rad=0.0,
    )

    coordinator.step(
        state=state,
        sensory_observation=0,
        place_observation=0,
        action_id=0,
        obstacle_distances=[0.0] * 12,
    )

    assert len(coordinator.history) == 3


def test_real_step_updates_transition_model():
    moves, memory, model = (
        make_components()
    )

    coordinator = BaselineStepCoordinator(
        model=model,
        memory=memory,
        motion_set=moves,
    )

    state = CognitiveOdomState(
        position=Point2D(0.0, 0.0),
        travel_heading_rad=0.0,
    )

    before = (
        model.transition_concentration.copy()
    )

    coordinator.step(
        state=state,
        sensory_observation=0,
        place_observation=0,
        action_id=0,
        obstacle_distances=[0.0] * 12,
    )

    assert not np.allclose(
        model.transition_concentration,
        before,
    )


def test_clear_direction_grows_three_ghost_nodes():
    moves, memory, model = (
        make_components()
    )

    coordinator = BaselineStepCoordinator(
        model=model,
        memory=memory,
        motion_set=moves,
    )

    state = CognitiveOdomState(
        position=Point2D(0.0, 0.0),
        travel_heading_rad=0.0,
    )

    obstacles = [0.0] * 12
    obstacles[0] = 5.0

    result = coordinator.step(
        state=state,
        sensory_observation=0,
        place_observation=0,
        action_id=12,
        obstacle_distances=obstacles,
    )

    assert len(
        result.transition_nodes
        .directions[0]
        .nodes
    ) == 3

    assert model.num_states == 4


def test_history_is_padded_after_ghost_growth():
    moves, memory, model = (
        make_components()
    )

    coordinator = BaselineStepCoordinator(
        model=model,
        memory=memory,
        motion_set=moves,
    )

    state = CognitiveOdomState(
        position=Point2D(0.0, 0.0),
        travel_heading_rad=0.0,
    )

    obstacles = [0.0] * 12
    obstacles[0] = 5.0

    coordinator.step(
        state=state,
        sensory_observation=0,
        place_observation=0,
        action_id=12,
        obstacle_distances=obstacles,
    )

    for belief in (
        coordinator.history.entries()
    ):
        assert belief.shape == (4,)


def test_stationary_B_is_identity_after_growth():
    moves, memory, model = (
        make_components()
    )

    coordinator = BaselineStepCoordinator(
        model=model,
        memory=memory,
        motion_set=moves,
    )

    state = CognitiveOdomState(
        position=Point2D(0.0, 0.0),
        travel_heading_rad=0.0,
    )

    obstacles = [0.0] * 12
    obstacles[0] = 5.0

    coordinator.step(
        state=state,
        sensory_observation=0,
        place_observation=0,
        action_id=12,
        obstacle_distances=obstacles,
    )

    np.testing.assert_allclose(
        model.transition_likelihood[
            :,
            :,
            12,
        ],
        np.eye(4),
    )


def test_second_step_uses_last_saved_posterior():
    moves, memory, model = (
        make_components()
    )

    coordinator = BaselineStepCoordinator(
        model=model,
        memory=memory,
        motion_set=moves,
    )

    state = CognitiveOdomState(
        position=Point2D(0.0, 0.0),
        travel_heading_rad=0.0,
    )

    first = coordinator.step(
        state=state,
        sensory_observation=0,
        place_observation=0,
        action_id=12,
        obstacle_distances=[0.0] * 12,
    )

    expected_previous = (
        first.final_belief.copy()
    )

    second = coordinator.step(
        state=state,
        sensory_observation=0,
        place_observation=0,
        action_id=12,
        obstacle_distances=[0.0] * 12,
    )

    np.testing.assert_allclose(
        second.previous_belief,
        expected_previous,
    )
