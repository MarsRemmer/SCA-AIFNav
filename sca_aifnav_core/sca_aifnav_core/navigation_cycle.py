"""End-to-end coordination of learning and planning."""

from dataclasses import dataclass

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.baseline_step import (
    BaselineStepCoordinator,
    BaselineStepResult,
)
from sca_aifnav_core.directional_lookahead import (
    DEFAULT_LOOKAHEAD_STEPS,
)
from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.inductive_inference import (
    DEFAULT_INDUCTIVE_HORIZON,
)
from sca_aifnav_core.mcts_action_inference import (
    DEFAULT_ACTION_SELECTION,
)
from sca_aifnav_core.mcts_model_interface import (
    MCTSModelInterface,
)
from sca_aifnav_core.mcts_planner import (
    DEFAULT_MAX_ROLLOUT_DEPTH,
    DEFAULT_MCTS_EXPLORATION,
    DEFAULT_NUM_SIMULATIONS,
    MCTSPlan,
    plan_mcts,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.preference_state import (
    BaselinePreferenceState,
    PreferenceSnapshot,
)
from sca_aifnav_core.real_experience import (
    prepare_real_observation_dimensions,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)


@dataclass(frozen=True)
class NavigationCycleResult:
    """Summarize one complete learning-and-planning cycle."""

    learning: BaselineStepResult
    preferences: PreferenceSnapshot
    planning: MCTSPlan
    posterior_place_id: int = -1


class BaselineNavigationCoordinator:
    """Connect baseline learning, preferences, and MCTS planning."""

    def __init__(
        self,
        model: BaselineGenerativeModel,
        memory: BaselinePlaceMemory,
        motion_set: BaselineMotionSet,
        preferences: BaselinePreferenceState = None,
        robot_dimension: float = 0.25,
        max_lookahead_steps: int = DEFAULT_LOOKAHEAD_STEPS,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        max_rollout_depth: int = DEFAULT_MAX_ROLLOUT_DEPTH,
        c_param: float = DEFAULT_MCTS_EXPLORATION,
        use_utility: bool = True,
        use_state_information_gain: bool = False,
        use_inductive_inference: bool = True,
        inductive_horizon: int = DEFAULT_INDUCTIVE_HORIZON,
    ) -> None:
        self.model = model
        self.memory = memory
        self.motion_set = motion_set

        if preferences is None:
            preferences = BaselinePreferenceState(
                model
            )

        self.preferences = preferences

        self.learning = BaselineStepCoordinator(
            model=model,
            memory=memory,
            motion_set=motion_set,
            robot_dimension=robot_dimension,
            max_lookahead_steps=(
                max_lookahead_steps
            ),
        )

        self.model_interface = MCTSModelInterface(
            model=model,
            memory=memory,
            motion_set=motion_set,
            preferences=preferences,
            use_utility=use_utility,
            use_state_information_gain=(
                use_state_information_gain
            ),
            use_inductive_inference=(
                use_inductive_inference
            ),
            inductive_horizon=(
                inductive_horizon
            ),
        )

        self.num_simulations = num_simulations
        self.max_rollout_depth = (
            max_rollout_depth
        )
        self.c_param = c_param

    def set_preference(
        self,
        sensory_observation: int = -1,
        place_observation: int = -1,
        preference_weight: float = 1.0,
    ) -> PreferenceSnapshot:
        """Set a navigation preference explicitly."""
        return self.preferences.update(
            model=self.model,
            sensory_observation=(
                sensory_observation
            ),
            place_observation=(
                place_observation
            ),
            preference_weight=(
                preference_weight
            ),
        )

    def clear_preference(
        self,
    ) -> PreferenceSnapshot:
        """Remove the current navigation preference."""
        return self.preferences.clear(
            self.model
        )

    def restrictive_possible_actions(
        self,
        current_place_id: int,
        obstacle_distances,
    ):
        """
        Return currently executable root actions.

        A directional action is retained only when the latest directional
        obstacle distance reaches the first cognitive-node distance and the
        direction resolves to an existing cognitive place. STAY is always
        retained.
        """
        distances = tuple(
            float(value)
            for value in obstacle_distances
        )

        if (
            len(distances)
            != self.motion_set.DIRECTION_COUNT
        ):
            raise ValueError(
                "obstacle_distances must contain "
                "one value per directional action"
            )

        minimum_distance = (
            self.memory.influence_radius
            + self.learning.robot_dimension
            / 2.0
        )

        possible_actions = []

        for action_id in range(
            self.motion_set.DIRECTION_COUNT
        ):
            distance = distances[
                action_id
            ]

            # Using this form also rejects NaN.
            if not distance >= minimum_distance:
                continue

            next_place_id = (
                self.model_interface
                .get_next_place_id(
                    current_place_id=(
                        current_place_id
                    ),
                    action_id=action_id,
                )
            )

            if next_place_id < 0:
                continue

            possible_actions.append(
                action_id
            )

        possible_actions.append(
            self.model.stationary_action_id
        )

        return tuple(
            possible_actions
        )

    def plan_current(
        self,
        current_place_id: int,
        possible_actions=None,
        action_selection: str = (
            DEFAULT_ACTION_SELECTION
        ),
        rng=None,
    ) -> MCTSPlan:
        """
        Plan from the current learned model without adding experience.

        This entry point is used when planning is needed without a newly
        completed physical action, for example after changing a goal.
        """
        self.preferences.sync_dimensions(
            self.model
        )

        return plan_mcts(
            current_belief=(
                self.model.state_belief.copy()
            ),
            current_place_id=current_place_id,
            model_interface=(
                self.model_interface
            ),
            num_simulations=(
                self.num_simulations
            ),
            max_rollout_depth=(
                self.max_rollout_depth
            ),
            c_param=self.c_param,
            action_selection=(
                action_selection
            ),
            possible_actions=(
                possible_actions
            ),
            rng=rng,
        )

    def step_and_plan(
        self,
        state: CognitiveOdomState,
        sensory_observation: int,
        place_observation: int,
        executed_action_id: int,
        obstacle_distances,
        current_place_id=None,
        possible_actions=None,
        action_selection: str = (
            DEFAULT_ACTION_SELECTION
        ),
        rng=None,
    ) -> NavigationCycleResult:
        """
        Learn from one completed action and choose the next action.

        ``executed_action_id`` is the physical action whose consequence has
        just been observed. The returned MCTS action is the action proposed
        for the following control step.
        """
        # Prepare the newly observed A/pA dimensions before state
        # inference, then synchronize C to those dimensions.
        prepare_real_observation_dimensions(
            model=self.model,
            sensory_observation=(
                sensory_observation
            ),
            place_observation=(
                place_observation
            ),
        )

        self.preferences.sync_dimensions(
            self.model
        )

        learning_result = self.learning.step(
            state=state,
            sensory_observation=(
                sensory_observation
            ),
            place_observation=(
                place_observation
            ),
            action_id=executed_action_id,
            obstacle_distances=(
                obstacle_distances
            ),
            observation_prepared=True,
        )

        # Cognitive growth may increase observation/state dimensions.
        # Preference values themselves are not reset on every step.
        self.preferences.sync_dimensions(
            self.model
        )

        posterior_place_id = (
            self.model.get_confident_state_index(
                z_score=10.0,
                min_z_score=2.0,
                observation_count=1,
            )
        )

        preference_snapshot = (
            self.preferences.snapshot()
        )

        if posterior_place_id >= 0:
            planning_place_id = (
                posterior_place_id
            )

        elif current_place_id is None:
            planning_place_id = (
                place_observation
            )

        else:
            planning_place_id = (
                current_place_id
            )

        if possible_actions is None:
            possible_actions = (
                self.restrictive_possible_actions(
                    current_place_id=(
                        planning_place_id
                    ),
                    obstacle_distances=(
                        obstacle_distances
                    ),
                )
            )

        planning_result = self.plan_current(
            current_place_id=(
                planning_place_id
            ),
            possible_actions=(
                possible_actions
            ),
            action_selection=(
                action_selection
            ),
            rng=rng,
        )

        return NavigationCycleResult(
            learning=learning_result,
            preferences=(
                preference_snapshot
            ),
            planning=planning_result,
            posterior_place_id=(
                posterior_place_id
            ),
        )
