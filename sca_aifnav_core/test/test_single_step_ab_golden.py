"""Golden test for one complete real A/B learning step."""

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


def normalized_columns(
    values,
):
    """Normalize an observation table over outcomes."""
    return (
        values
        / values.sum(
            axis=0,
            keepdims=True,
        )
    )


def normalized_transition(
    values,
):
    """Normalize a transition table over next states."""
    return (
        values
        / values.sum(
            axis=0,
            keepdims=True,
        )
    )


def configured_step():
    """Create an isolated two-state real-learning scenario."""
    moves = BaselineMotionSet()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    source_id = memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    target_id = memory.resolve_place(
        Point2D(
            10.0,
            10.0,
        )
    )

    assert source_id == 0
    assert target_id == 1

    model = BaselineGenerativeModel()

    # Make positional observation 1 identify hidden state 1.
    model.register_place_observation(
        target_id
    )

    coordinator = BaselineStepCoordinator(
        model=model,
        memory=memory,
        motion_set=moves,
        robot_dimension=0.3,
        max_lookahead_steps=8,
    )

    return (
        moves,
        memory,
        model,
        coordinator,
    )


def test_one_real_step_exactly_updates_A_and_B():
    """One state-0 to state-1 transition should match AIMAPP learning."""
    (
        moves,
        memory,
        model,
        coordinator,
    ) = configured_step()

    action_id = 0

    reverse_action_id = (
        moves.reverse_action(
            action_id
        )
    )

    assert reverse_action_id == 6

    previous_belief = (
        coordinator.history.latest.copy()
    )

    sensory_A_before = (
        model.sensory_likelihood.copy()
    )

    place_A_before = (
        model.place_likelihood.copy()
    )

    pB_before = (
        model.transition_concentration.copy()
    )

    # NaN ranges deliberately isolate the real A/B learning from
    # hypothetical cognitive-map transition updates.
    obstacle_distances = tuple(
        float("nan")
        for _ in range(12)
    )

    result = coordinator.step(
        state=CognitiveOdomState(
            position=Point2D(
                10.0,
                10.0,
            ),
            travel_heading_rad=0.0,
        ),
        sensory_observation=1,
        place_observation=1,
        action_id=action_id,
        obstacle_distances=(
            obstacle_distances
        ),
    )

    experience = result.real_experience

    # --------------------------------------------------------
    # 1. Three-stage belief lifecycle.
    # Both modalities deterministically identify state 1.
    # --------------------------------------------------------

    expected_state_one = np.array(
        [
            0.0,
            1.0,
        ]
    )

    np.testing.assert_allclose(
        experience.preliminary_belief,
        expected_state_one,
    )

    np.testing.assert_allclose(
        experience.learning_belief,
        expected_state_one,
    )

    np.testing.assert_allclose(
        experience.posterior_belief,
        expected_state_one,
    )

    assert experience.transition_updated is True
    assert (
        experience.reverse_transition_updated
        is True
    )

    assert (
        experience.reverse_action_id
        == reverse_action_id
    )

    # --------------------------------------------------------
    # 2. Exact pB update.
    #
    # AIMAPP:
    # direct  = +10 * outer(next, previous)
    # reverse = +7  * outer(previous, next)
    # --------------------------------------------------------

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
            expected_state_one,
            previous_belief,
        )
    )

    expected_pB[
        :,
        :,
        reverse_action_id,
    ] += (
        7.0
        * np.outer(
            previous_belief,
            expected_state_one,
        )
    )

    np.testing.assert_allclose(
        model.transition_concentration,
        expected_pB,
    )

    # --------------------------------------------------------
    # 3. Exact B after pB normalization.
    # --------------------------------------------------------

    expected_B = (
        normalized_transition(
            expected_pB
        )
    )

    # STAY is restored to identity after the complete step.
    expected_B[
        :,
        :,
        12,
    ] = np.eye(
        model.num_states
    )

    np.testing.assert_allclose(
        model.transition_likelihood,
        expected_B,
    )

    # --------------------------------------------------------
    # 4. Exact pA update.
    #
    # prepare_real_observation_dimensions rebuilds pA from A.
    # Then observation learning adds:
    # +5 * one_hot(observation) outer learning_qs
    # subject to existing A support.
    # --------------------------------------------------------

    sensory_evidence = np.zeros_like(
        sensory_A_before
    )

    sensory_evidence[
        1,
        :,
    ] = (
        experience.learning_belief
    )

    sensory_support = (
        sensory_A_before
        > 0.0
    ).astype(float)

    expected_sensory_pA = (
        sensory_A_before
        + 5.0
        * sensory_evidence
        * sensory_support
    )

    place_evidence = np.zeros_like(
        place_A_before
    )

    place_evidence[
        1,
        :,
    ] = (
        experience.learning_belief
    )

    place_support = (
        place_A_before
        > 0.0
    ).astype(float)

    expected_place_pA = (
        place_A_before
        + 5.0
        * place_evidence
        * place_support
    )

    np.testing.assert_allclose(
        model.sensory_concentration,
        expected_sensory_pA,
    )

    np.testing.assert_allclose(
        model.place_concentration,
        expected_place_pA,
    )

    # --------------------------------------------------------
    # 5. Exact A after pA normalization.
    # --------------------------------------------------------

    np.testing.assert_allclose(
        model.sensory_likelihood,
        normalized_columns(
            expected_sensory_pA
        ),
    )

    np.testing.assert_allclose(
        model.place_likelihood,
        normalized_columns(
            expected_place_pA
        ),
    )

    # --------------------------------------------------------
    # 6. Saved qs lifecycle.
    # Initial qs + learning qs + posterior qs.
    # --------------------------------------------------------

    assert len(
        coordinator.history
    ) == 3

    history = (
        coordinator.history.entries()
    )

    np.testing.assert_allclose(
        history[0],
        previous_belief,
    )

    np.testing.assert_allclose(
        history[1],
        experience.learning_belief,
    )

    np.testing.assert_allclose(
        history[2],
        experience.posterior_belief,
    )

    np.testing.assert_allclose(
        model.state_belief,
        experience.posterior_belief,
    )

    # --------------------------------------------------------
    # 7. No ghost growth was allowed in this isolated test.
    # --------------------------------------------------------

    assert len(memory) == 2
    assert model.num_states == 2
