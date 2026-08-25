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
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)


@dataclass(frozen=True)
class NavigationCycleResult:
    """Summarize one complete learning-and-planning cycle."""

    learning: BaselineStepResult
    preferences: PreferenceSnapshot
    planning: MCTSPlan


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
        )

        # Cognitive growth may increase observation/state dimensions.
        # Preference values themselves are not reset on every step.
        self.preferences.sync_dimensions(
            self.model
        )

        preference_snapshot = (
            self.preferences.snapshot()
        )

        if current_place_id is None:
            planning_place_id = (
                place_observation
            )
        else:
            planning_place_id = (
                current_place_id
            )

        planning_result = plan_mcts(
            current_belief=(
                learning_result.final_belief
            ),
            current_place_id=(
                planning_place_id
            ),
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

        return NavigationCycleResult(
            learning=learning_result,
            preferences=(
                preference_snapshot
            ),
            planning=planning_result,
        )
