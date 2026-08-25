"""One complete learning step for the AIMAPP V5 baseline."""

from dataclasses import dataclass

import numpy as np

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.belief_history import (
    BeliefHistory,
)
from sca_aifnav_core.directional_lookahead import (
    DEFAULT_LOOKAHEAD_STEPS,
)
from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.real_experience import (
    RealExperienceResult,
    update_real_experience,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)
from sca_aifnav_core.transition_node_update import (
    TransitionNodeUpdate,
    update_cognitive_transition_nodes,
)


@dataclass(frozen=True)
class BaselineStepResult:
    """Summarize one complete V5 baseline learning step."""

    previous_belief: np.ndarray
    real_experience: RealExperienceResult
    transition_nodes: TransitionNodeUpdate
    final_belief: np.ndarray


class BaselineStepCoordinator:
    """Coordinate real learning and cognitive-map growth."""

    def __init__(
        self,
        model: BaselineGenerativeModel,
        memory: BaselinePlaceMemory,
        motion_set: BaselineMotionSet,
        robot_dimension: float = 0.25,
        max_lookahead_steps: int = (
            DEFAULT_LOOKAHEAD_STEPS
        ),
        history: BeliefHistory = None,
    ) -> None:
        self.model = model
        self.memory = memory
        self.motion_set = motion_set
        self.robot_dimension = robot_dimension
        self.max_lookahead_steps = (
            max_lookahead_steps
        )

        if history is None:
            history = BeliefHistory(
                model.state_belief
            )

        self.history = history

    def step(
        self,
        state: CognitiveOdomState,
        sensory_observation: int,
        place_observation: int,
        action_id: int,
        obstacle_distances,
    ) -> BaselineStepResult:
        """Execute one V5 learning-and-map-update cycle."""
        previous_belief = (
            self.history.latest
        )

        experience = update_real_experience(
            model=self.model,
            sensory_observation=(
                sensory_observation
            ),
            place_observation=(
                place_observation
            ),
            action_id=action_id,
            previous_belief=previous_belief,
            motion_set=self.motion_set,
        )

        # V5 saves the inference performed inside
        # update_believes_with_obs().
        self.history.record(
            experience.learning_belief
        )

        # V5 then performs and saves another posterior
        # inference after updating A.
        self.history.record(
            experience.posterior_belief
        )

        transition_nodes = (
            update_cognitive_transition_nodes(
                state=state,
                obstacle_distances=(
                    obstacle_distances
                ),
                memory=self.memory,
                model=self.model,
                motion_set=self.motion_set,
                robot_dimension=(
                    self.robot_dimension
                ),
                max_steps=(
                    self.max_lookahead_steps
                ),
            )
        )

        # Ghost growth can increase the hidden-state
        # dimension. V5 pads current and historical qs.
        self.history.align_to_states(
            self.model.num_states
        )

        # V5 restores the stationary transition after
        # all other transition-node updates.
        self.model.enforce_stationary_transition()

        final_belief = (
            self.model.state_belief.copy()
        )

        return BaselineStepResult(
            previous_belief=previous_belief,
            real_experience=experience,
            transition_nodes=transition_nodes,
            final_belief=final_belief,
        )
